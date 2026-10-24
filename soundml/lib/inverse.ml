(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2025                                                       *)
(*    Gabriel Santamaria                                                     *)
(*                                                                           *)
(*                                                                           *)
(*  Licensed under the Apache License, Version 2.0 (the "License");          *)
(*  you may not use this file except in compliance with the License.         *)
(*  You may obtain a copy of the License at                                  *)
(*                                                                           *)
(*    http://www.apache.org/licenses/LICENSE-2.0                             *)
(*                                                                           *)
(*  Unless required by applicable law or agreed to in writing, software      *)
(*  distributed under the License is distributed on an "AS IS" BASIS,        *)
(*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *)
(*  See the License for the specific language governing permissions and      *)
(*  limitations under the License.                                           *)
(*                                                                           *)
(*****************************************************************************)

(* Feature inversion, as librosa.feature.inverse computes it, with one named
   departure: the non-negative least squares under mel_to_stft.

   mfcc_to_mel is closed-form: the lifter divided out, the orthonormal type-II
   DCT inverted (its transpose, the missing coefficients the zero padding of
   scipy's idct), and the decibels raised back to power. The first two are one
   matrix, folded per frame in a fixed order; everything is double precision,
   rounded once.

   mel_to_stft solves each frame b on its own by [proximal_steps] steps of the
   proximal-point method (Rockafellar, SIAM J. Control Optim. 14, 1976) for
   non-negative least squares: from x0 = max 0 (A+ b), the clipped
   pseudo-inverse librosa starts its L-BFGS-B from, each step goes to the unique
   x >= 0 minimising

   1/2 |A x - b|^2 + lambda/2 |x - x_k|^2

   where A is the filterbank and lambda is [ridge] times its largest squared
   column norm. As lambda vanishes the first step tends to the non-negative
   least-squares solution nearest x0 (Attouch, SIAM J. Optim. 6, 1996); the
   later ones remove what is left of lambda's pull on the fit. Where A+ b is
   already non-negative every step stays at x0, librosa's own answer.

   Each step is solved through its dual: x = max 0 (x_k + A^T v) for the
   minimiser v of the strongly convex, once differentiable

   psi v = lambda/2 |v|^2 + 1/2 |max 0 (x_k + A^T v)|^2 - b.v

   over the n_mels bands, by the generalised Newton method (Mangasarian, Optim.
   Methods Softw. 17, 2002): the Newton system on the bins F where x_k + A^T v
   is positive is lambda I + A_F A_F^T, banded — a mel filter overlaps only its
   neighbours — and factored by banded Cholesky; the step length is the exact
   minimiser of psi along the Newton direction, found by walking the breakpoints
   of the piecewise-linear derivative in order. The iteration stops when the
   gradient vanishes to rounding, when a full step lands on the minimiser of the
   quadratic piece it started in, or when a step no longer moves the iterate.

   Every frame is one sequential computation in a fixed order on its own values,
   so its bits do not depend on its neighbours, on the chunking, or on how the
   frames are spread over domains; every operation scales exactly under a power
   of two, so the fitted spectrum does too. *)

let output_shape x rows =
  let shape = Array.copy (Nx.shape x) in
  shape.(Array.length shape - 2) <- rows ;
  shape

(* [check_rank ~chunk fn ~axis x] refuses an [x] of rank below two, worded for a
   stage's chunk when [chunk] holds. *)
let check_rank ~chunk fn ~axis x =
  let nd = Nx.ndim x in
  if nd < 2 then
    invalid_arg
      ( if chunk then
          Printf.sprintf
            "%s: cannot invert a rank-%d chunk (the stage needs [...; %s; \
             frames])"
            fn nd axis
        else
          Printf.sprintf
            "%s: cannot invert a rank-%d tensor (the input must be [...; %s; \
             frames])"
            fn nd axis )

(* {1 The cepstral inverse} *)

let check_lifter fn lifter =
  Option.iter
    (fun l ->
      if not (Float.is_finite l && l >= 0.) then
        invalid_arg
          (Printf.sprintf
             "%s: cannot lifter with a coefficient of %g (lifter must be \
              finite and non-negative)"
             fn l ) )
    lifter

let check_reference fn reference =
  if not (Float.is_finite reference && reference > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot scale by a reference of %g (reference must be finite and \
          positive)"
         fn reference )

(* [lift l k] is the sinusoidal lifter weight of coefficient [k], spelled as the
   forward cepstrum spells it. *)
let lift l k =
  1. +. (l /. 2. *. Float.sin (Float.pi *. Float.of_int (k + 1) /. l))

let check_coefficients fn ~n_mels ~n_mfcc lifter =
  if n_mfcc < 1 || n_mfcc > n_mels then
    invalid_arg
      (Printf.sprintf
         "%s: cannot invert %d cepstral coefficients into %d mel bands (n_mfcc \
          must lie in [1, n_mels])"
         fn n_mfcc n_mels ) ;
  match lifter with
  | Some l when l > 0. ->
      for k = 0 to n_mfcc - 1 do
        if Float.abs (lift l k) < epsilon_float then
          invalid_arg
            (Printf.sprintf
               "%s: cannot undo the lifter %g at coefficient %d (its weight 1 \
                + (lifter / 2) sin (pi (k + 1) / lifter) vanishes)"
               fn l k )
      done
  | Some _ | None ->
      ()

(* [inverse_matrix ~n_mels ~n_mfcc lifter] is the [[n_mels; n_mfcc]] inverse of
   the forward cepstral matrix, prepared for {!Ordered.project}: row [j], column
   [k] is [2 cos (pi k (2j + 1) / (2 n_mels))] times the orthonormal scale of
   coefficient [k] ([1 / sqrt (4 n_mels)] for [k = 0], [1 / sqrt (2 n_mels)]
   otherwise), divided by its lifter weight. The coefficients past [n_mfcc] are
   the zero padding of the inverse DCT, so they simply have no column. The
   angle's numerator is reduced modulo a full turn, [4 n_mels], first. *)
let inverse_matrix ~n_mels ~n_mfcc lifter =
  let turn = 4 * n_mels in
  let w = Array.make (n_mels * n_mfcc) 0. in
  for k = 0 to n_mfcc - 1 do
    let scale =
      if k = 0 then 1. /. Float.sqrt (4. *. Float.of_int n_mels)
      else 1. /. Float.sqrt (2. *. Float.of_int n_mels)
    in
    let weight =
      match lifter with Some l when l > 0. -> lift l k | Some _ | None -> 1.
    in
    for j = 0 to n_mels - 1 do
      let a = k * ((2 * j) + 1) mod turn in
      w.((j * n_mfcc) + k) <-
        2.
        *. Float.cos (Float.pi *. Float.of_int a /. Float.of_int (2 * n_mels))
        *. scale /. weight
    done
  done ;
  Ordered.matrix (Nx.create Nx.float64 [|n_mels; n_mfcc|] w)

(* [to_power dtype ~reference db] is librosa's db_to_power of the
   double-precision log-mel [db], [reference * 10 ** (db * 0.1)] in numpy's
   order, rounded once into [dtype]. *)
let to_power (type a) (dtype : (float, a) Nx.dtype) ~reference
    (db : (float, Nx.float64_elt) Nx.t) : (float, a) Nx.t =
  let shape = Nx.shape db in
  let count = Nx.numel db in
  let grain = 4096 in
  let fill write =
    Store.read64 db (fun src ->
        Par.ranges ~n:count ~grain (fun lo hi ->
            for i = lo to hi - 1 do
              let v = Bigarray.Array1.unsafe_get src i in
              write i (reference *. Float.pow 10. (v *. 0.1))
            done ) )
  in
  match dtype with
  | Nx.Float32 ->
      Store.make32 shape (fun dst -> fill (Bigarray.Array1.unsafe_set dst))
  | Nx.Float64 ->
      Store.make64 shape (fun dst -> fill (Bigarray.Array1.unsafe_set dst))
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast dtype
        (Store.make64 shape (fun dst -> fill (Bigarray.Array1.unsafe_set dst)))

let cepstral_inverse ~reference ~n_mels matrix c =
  let dtype = Nx.dtype c in
  if Nx.numel c = 0 then Nx.zeros dtype (output_shape c n_mels)
  else to_power dtype ~reference (Ordered.project matrix c)

let mfcc_to_mel_with fn mel ?lifter ?(reference = 1.) c =
  check_lifter fn lifter ;
  check_reference fn reference ;
  check_rank ~chunk:false fn ~axis:"n_mfcc" c ;
  let n_mels = Mel.Config.n_mels mel in
  let n_mfcc = Nx.dim (Nx.ndim c - 2) c in
  check_coefficients fn ~n_mels ~n_mfcc lifter ;
  cepstral_inverse ~reference ~n_mels (inverse_matrix ~n_mels ~n_mfcc lifter) c

let mfcc_to_mel mel ?lifter ?reference c =
  mfcc_to_mel_with "mfcc_to_mel" mel ?lifter ?reference c

let mfcc_to_mel_stage mel ?(n_mfcc = 20) ?lifter ?(reference = 1.) () =
  let fn = "mfcc_to_mel_stage" in
  check_lifter fn lifter ;
  check_reference fn reference ;
  let n_mels = Mel.Config.n_mels mel in
  check_coefficients fn ~n_mels ~n_mfcc lifter ;
  let matrix = inverse_matrix ~n_mels ~n_mfcc lifter in
  Pipeline.stateless (fun c ->
      check_rank ~chunk:true fn ~axis:"n_mfcc" c ;
      let k = Nx.dim (Nx.ndim c - 2) c in
      if k <> n_mfcc then
        invalid_arg
          (Printf.sprintf
             "%s: cannot invert %d cepstral coefficients with a stage built \
              for %d"
             fn k n_mfcc ) ;
      cepstral_inverse ~reference ~n_mels matrix c )

(* {1 The non-negative least squares} *)

(* The proximal weight, relative to the largest squared column norm of the
   filterbank, and the number of proximal steps. Each step shrinks the misfit of
   a frame that some non-negative spectrum fits by about [lambda] over the
   squared singular values of the filterbank restricted to the bins in use;
   three steps at [1e-9] fit every measured spectrogram of a signal to rounding,
   a relative [2e-14], where one step leaves [1e-8], and a weight of [1e-7]
   still leaves [1e-8] after three on a band some three orders of magnitude
   below its neighbours. Where no non-negative spectrum fits, the dual grows as
   [1 / lambda] and cancels in [x_k + A^T v]: the optimality residual of such
   frames reaches [4e-6] of the frame's scale instead of rounding, and their fit
   stays within a relative [1e-9] of the optimum; a larger weight would round
   less but leave more of its pull after three steps. *)
let ridge = 1e-9

let proximal_steps = 3

(* The gradient test: a step's iteration stops once the dual gradient is this
   fraction of the scale of its terms, rounding error included. *)
let tolerance = 1e-15

(* The most Newton steps a proximal step takes; measured frames take at most
   some sixty. *)
let iterations = 1000

(* The filterbank by columns, with the pseudo-inverse and the proximal weight:
   bin [j]'s nonzero weights lie on the [span.(j)] consecutive bands from
   [lo.(j)] on ([lo.(j) = -1] and [span.(j) = 0] when no filter reaches the
   bin), stored at [weights.(j * (band + 1) + k)]. [row_sum] and [column_sum]
   are the largest absolute row and column sums, which bound the rounding of a
   product with the filterbank or its transpose. *)
type basis =
  { bands: int
  ; bins: int
  ; band: int  (** the half-bandwidth of every [A_F A_F^T] *)
  ; lo: int array
  ; span: int array
  ; weights: float array
  ; pinv: float array  (** [[bins; bands]], zero on unreached bins *)
  ; lambda: float
  ; row_sum: float
  ; column_sum: float }

(* [pseudo_inverse fb lo] is the [[bins; bands]] pseudo-inverse of the
   filterbank [fb], [V diag (1 / S) U^T] from its singular value decomposition
   with librosa's cutoff (scipy's pinv at [rtol = 1e-15]): singular values at
   most [1e-15] of the largest are dropped. Each entry adds its singular
   triplets in descending order of the value; the rows of unreached bins
   ([lo.(j) < 0]) are exactly zero, where the decomposition leaves rounding.
   Nx's decomposition runs on its own matrix product, never on a BLAS, so the
   bits are a function of the filterbank alone. *)
let pseudo_inverse fb lo =
  let u, s, vh = Nx.svd fb in
  let bands = Nx.dim 0 fb and bins = Nx.dim 1 fb and rank = Nx.dim 0 s in
  let s = Nx.to_array s and u = Nx.to_array u and vh = Nx.to_array vh in
  let cutoff = 1e-15 *. s.(0) in
  let scaled =
    Array.init (bands * rank) (fun idx ->
        let r = idx mod rank in
        if s.(r) > cutoff then u.(idx) /. s.(r) else 0. )
  in
  let q = Array.make (bins * bands) 0. in
  Par.ranges ~n:bins ~grain:8 (fun first last ->
      for j = first to last - 1 do
        if lo.(j) >= 0 then
          for i = 0 to bands - 1 do
            let acc = ref 0. in
            for r = 0 to rank - 1 do
              let vr = Array.unsafe_get vh ((r * bins) + j)
              and ur = Array.unsafe_get scaled ((i * rank) + r) in
              acc := !acc +. (vr *. ur)
            done ;
            q.((j * bands) + i) <- !acc
          done
      done ) ;
  q

let basis mel =
  let bands = Mel.Config.n_mels mel and bins = Mel.Config.bins mel in
  let fb = Mel.filterbank Nx.float64 mel in
  let a =
    Store.read64 fb (fun w ->
        Array.init (bands * bins) (fun i -> Bigarray.Array1.unsafe_get w i) )
  in
  let lo = Array.make bins (-1) and hi = Array.make bins (-1) in
  for j = 0 to bins - 1 do
    for i = 0 to bands - 1 do
      if a.((i * bins) + j) <> 0. then begin
        if lo.(j) < 0 then lo.(j) <- i ;
        hi.(j) <- i
      end
    done
  done ;
  let span =
    Array.init bins (fun j -> if lo.(j) < 0 then 0 else hi.(j) - lo.(j) + 1)
  in
  let band = Array.fold_left (fun acc s -> Int.max acc (s - 1)) 0 span in
  let stride = band + 1 in
  let weights = Array.make (bins * stride) 0. in
  let rows = Array.make bands 0. in
  let largest = ref 0. and column_sum = ref 0. in
  for j = 0 to bins - 1 do
    let energy = ref 0. and total = ref 0. in
    for k = 0 to span.(j) - 1 do
      let r = lo.(j) + k in
      let v = a.((r * bins) + j) in
      weights.((j * stride) + k) <- v ;
      let vv = v *. v in
      energy := !energy +. vv ;
      total := !total +. Float.abs v ;
      rows.(r) <- rows.(r) +. Float.abs v
    done ;
    if !energy > !largest then largest := !energy ;
    if !total > !column_sum then column_sum := !total
  done ;
  let pinv = pseudo_inverse fb lo in
  { bands
  ; bins
  ; band
  ; lo
  ; span
  ; weights
  ; pinv
  ; lambda= ridge *. !largest
  ; row_sum= Array.fold_left Float.max 0. rows
  ; column_sum= !column_sum }

(* One frame's working memory, owned by one domain. *)
type work =
  { b: float array  (** the frame, [bands] *)
  ; x0: float array  (** the start, [bins] *)
  ; s: float array  (** [x0 + A^T v], [bins] *)
  ; v: float array  (** the dual iterate, [bands] *)
  ; az: float array  (** [A max 0 s], [bands] *)
  ; g: float array  (** the dual gradient, [bands] *)
  ; d: float array  (** the Newton direction, [bands] *)
  ; e: float array  (** [A^T d], [bins] *)
  ; h: float array
        (** [[bands; band + 1]]: the Newton matrix, then its factor *)
  ; mutable cur: bool array  (** the bins where [s > 0], [bins] *)
  ; mutable prev: bool array  (** [cur] of the previous step *)
  ; at: float array  (** the line search's breakpoints, [bins] *)
  ; heap: int array  (** the breakpoints not yet crossed, [bins] *) }

let work p =
  let m () = Array.make p.bands 0. and n () = Array.make p.bins 0. in
  { b= m ()
  ; x0= n ()
  ; s= n ()
  ; v= m ()
  ; az= m ()
  ; g= m ()
  ; d= m ()
  ; e= n ()
  ; h= Array.make (p.bands * (p.band + 1)) 0.
  ; cur= Array.make p.bins false
  ; prev= Array.make p.bins false
  ; at= n ()
  ; heap= Array.make p.bins 0 }

(* [start p w] writes the clipped pseudo-inverse of the frame to [w.x0] and to
   [w.s], the dual iterate being zero. *)
let start p w =
  let m = p.bands in
  for j = 0 to p.bins - 1 do
    let x =
      if Array.unsafe_get p.lo j < 0 then 0.
      else begin
        let row = j * m in
        let acc = ref 0. in
        for i = 0 to m - 1 do
          let q = Array.unsafe_get p.pinv (row + i)
          and bi = Array.unsafe_get w.b i in
          acc := !acc +. (q *. bi)
        done ;
        if !acc > 0. then !acc else 0.
      end
    in
    Array.unsafe_set w.x0 j x ; Array.unsafe_set w.s j x
  done ;
  Array.fill w.v 0 m 0.

(* [transpose p y j] is [(A^T y) j]. *)
let transpose p y j =
  let l = Array.unsafe_get p.lo j in
  let base = j * (p.band + 1) in
  let acc = ref 0. in
  for k = 0 to Array.unsafe_get p.span j - 1 do
    let a = Array.unsafe_get p.weights (base + k)
    and yk = Array.unsafe_get y (l + k) in
    acc := !acc +. (a *. yk)
  done ;
  !acc

(* [refresh p w] recomputes [w.s = w.x0 + A^T w.v] from scratch, so no update
   drifts. *)
let refresh p w =
  for j = 0 to p.bins - 1 do
    let t = transpose p w.v j in
    Array.unsafe_set w.s j (Array.unsafe_get w.x0 j +. t)
  done

(* [project p w] writes [A max 0 s] to [w.az], each band adding its bins in
   ascending order. *)
let project p w =
  let az = w.az and stride = p.band + 1 in
  Array.fill az 0 p.bands 0. ;
  for j = 0 to p.bins - 1 do
    let z = Array.unsafe_get w.s j in
    if z > 0. then begin
      let l = Array.unsafe_get p.lo j in
      for k = 0 to Array.unsafe_get p.span j - 1 do
        let a = Array.unsafe_get p.weights ((j * stride) + k) in
        let r = l + k in
        Array.unsafe_set az r (Array.unsafe_get az r +. (a *. z))
      done
    end
  done

(* [factor p w] assembles the Newton matrix [lambda I + A_F A_F^T] over the bins
   [F] of [w.cur] in band storage — [h.(i * (band + 1) + k)] is entry [(i, i -
   k)] — and factors it in place as [L L^T]. It is [false] if a pivot is not
   positive, which the [lambda I] term rules out in exact arithmetic. *)
let factor p w =
  let m = p.bands and band = p.band in
  let stride = band + 1 and h = w.h in
  Array.fill h 0 (m * stride) 0. ;
  for i = 0 to m - 1 do
    Array.unsafe_set h (i * stride) p.lambda
  done ;
  for j = 0 to p.bins - 1 do
    if Array.unsafe_get w.cur j then begin
      let l = Array.unsafe_get p.lo j and base = j * stride in
      for r = 0 to Array.unsafe_get p.span j - 1 do
        let ar = Array.unsafe_get p.weights (base + r) in
        for c = 0 to r do
          let ac = Array.unsafe_get p.weights (base + c) in
          let idx = ((l + r) * stride) + (r - c) in
          Array.unsafe_set h idx (Array.unsafe_get h idx +. (ar *. ac))
        done
      done
    end
  done ;
  let ok = ref true in
  for row = 0 to m - 1 do
    let first = Int.max 0 (row - band) in
    for col = first to row do
      let acc = ref (Array.unsafe_get h ((row * stride) + (row - col))) in
      for k = first to col - 1 do
        let lr = Array.unsafe_get h ((row * stride) + (row - k))
        and lc = Array.unsafe_get h ((col * stride) + (col - k)) in
        acc := !acc -. (lr *. lc)
      done ;
      if col = row then
        if !acc > 0. then Array.unsafe_set h (row * stride) (Float.sqrt !acc)
        else ok := false
      else
        Array.unsafe_set h
          ((row * stride) + (row - col))
          (!acc /. Array.unsafe_get h (col * stride))
    done
  done ;
  !ok

(* [direction p w] writes the Newton direction [-(L L^T)^{-1} g] to [w.d]. *)
let direction p w =
  let m = p.bands and band = p.band in
  let stride = band + 1 and h = w.h and d = w.d in
  for i = 0 to m - 1 do
    let acc = ref (-.Array.unsafe_get w.g i) in
    for k = Int.max 0 (i - band) to i - 1 do
      let lk = Array.unsafe_get h ((i * stride) + (i - k))
      and dk = Array.unsafe_get d k in
      acc := !acc -. (lk *. dk)
    done ;
    Array.unsafe_set d i (!acc /. Array.unsafe_get h (i * stride))
  done ;
  for i = m - 1 downto 0 do
    let acc = ref (Array.unsafe_get d i) in
    for k = i + 1 to Int.min (m - 1) (i + band) do
      let lk = Array.unsafe_get h ((k * stride) + (k - i))
      and dk = Array.unsafe_get d k in
      acc := !acc -. (lk *. dk)
    done ;
    Array.unsafe_set d i (!acc /. Array.unsafe_get h (i * stride))
  done

(* The breakpoints of the line search as a binary min-heap on their position,
   ties broken by bin, so that the order they are crossed in is a function of
   the frame alone. *)
let before at a b =
  let ta = Array.unsafe_get at a and tb = Array.unsafe_get at b in
  ta < tb || (ta = tb && a < b)

let rec sift at heap size i =
  let l = (2 * i) + 1 in
  if l < size then begin
    let r = l + 1 in
    let c =
      if
        r < size
        && before at (Array.unsafe_get heap r) (Array.unsafe_get heap l)
      then r
      else l
    in
    let hc = Array.unsafe_get heap c and hi = Array.unsafe_get heap i in
    if before at hc hi then begin
      Array.unsafe_set heap i hc ;
      Array.unsafe_set heap c hi ;
      sift at heap size c
    end
  end

(* [step_length p w] is the minimiser over [t >= 0] of [phi t = psi (v + t d)].
   Its derivative

   phi' t = c0 + c1 t + sum_j e_j max 0 (s_j + t e_j)

   with [c0 = (lambda v - b).d] and [c1 = lambda |d|^2] is continuous, piecewise
   linear and non-decreasing, with a breakpoint wherever [s_j + t e_j] crosses
   zero. The walk keeps the current piece's intercept [a0] and slope [a1] — the
   bins active on it added in — and crosses breakpoints in order until the
   derivative turns non-negative; the root lies on that piece. It is [0.] if [d]
   is not a descent direction. *)
let step_length p w =
  let c0 = ref 0. and dd = ref 0. in
  for i = 0 to p.bands - 1 do
    let di = Array.unsafe_get w.d i in
    let lv = p.lambda *. Array.unsafe_get w.v i in
    let r = lv -. Array.unsafe_get w.b i in
    let rd = r *. di and d2 = di *. di in
    c0 := !c0 +. rd ;
    dd := !dd +. d2
  done ;
  let c0 = !c0 and c1 = p.lambda *. !dd in
  let a0 = ref c0 and a1 = ref c1 and active = ref 0 and count = ref 0 in
  let push j t =
    Array.unsafe_set w.at j t ;
    Array.unsafe_set w.heap !count j ;
    incr count
  in
  for j = 0 to p.bins - 1 do
    let sj = Array.unsafe_get w.s j and ej = Array.unsafe_get w.e j in
    if sj > 0. then begin
      let es = ej *. sj and ee = ej *. ej in
      a0 := !a0 +. es ;
      a1 := !a1 +. ee ;
      incr active ;
      if ej < 0. then push j (-.sj /. ej)
    end
    else if ej > 0. then
      if sj = 0. then begin
        let ee = ej *. ej in
        a1 := !a1 +. ee ;
        incr active
      end
      else push j (-.sj /. ej)
  done ;
  if not (!a0 < 0.) then 0.
  else begin
    let size = ref !count in
    for i = (!size / 2) - 1 downto 0 do
      sift w.at w.heap !size i
    done ;
    let t_lo = ref 0. and result = ref 0. and searching = ref true in
    let root_below hi =
      if !a1 > 0. then Float.min hi (Float.max !t_lo (-. !a0 /. !a1)) else !t_lo
    in
    while !searching do
      if !size = 0 then begin
        result := root_below Float.infinity ;
        searching := false
      end
      else begin
        let j = Array.unsafe_get w.heap 0 in
        let th = Array.unsafe_get w.at j in
        let rise = !a1 *. th in
        if !a0 +. rise >= 0. then begin
          result := root_below th ;
          searching := false
        end
        else begin
          decr size ;
          Array.unsafe_set w.heap 0 (Array.unsafe_get w.heap !size) ;
          sift w.at w.heap !size 0 ;
          let sj = Array.unsafe_get w.s j and ej = Array.unsafe_get w.e j in
          let es = ej *. sj and ee = ej *. ej in
          if ej > 0. then begin
            a0 := !a0 +. es ;
            a1 := !a1 +. ee ;
            incr active
          end
          else begin
            a0 := !a0 -. es ;
            a1 := !a1 -. ee ;
            decr active ;
            (* the sums restart exactly once nothing is active, and the slope
               never falls below its [lambda |d|^2] floor *)
            if !active = 0 then begin
              a0 := c0 ;
              a1 := c1
            end
            else if !a1 < c1 then a1 := c1
          end ;
          t_lo := th
        end
      end
    done ;
    !result
  end

let same a b =
  let n = Array.length a in
  let rec go j =
    j >= n || (Array.unsafe_get a j = Array.unsafe_get b j && go (j + 1))
  in
  go 0

(* [newton p w] runs one proximal step from the start in [w.x0], with [w.s =
   w.x0] and [w.v = 0], leaving the dual image [x0 + A^T v] of its solution in
   [w.s]: the solution is its positive part. The gradient test measures the
   gradient against its terms and against the rounding of [s], whose terms grow
   with the dual where the bands cannot be fitted. *)
let newton p w =
  let m = p.bands in
  let x0max = ref 0. in
  for j = 0 to p.bins - 1 do
    x0max := Float.max !x0max (Array.unsafe_get w.x0 j)
  done ;
  let running = ref true and steps = ref 0 in
  let full = ref false and stepped = ref false in
  while !running && !steps < iterations do
    incr steps ;
    project p w ;
    let gmax = ref 0. and bmax = ref 0. and amax = ref 0. and vmax = ref 0. in
    for i = 0 to m - 1 do
      let vi = Array.unsafe_get w.v i and bi = Array.unsafe_get w.b i in
      let ai = Array.unsafe_get w.az i in
      let lv = p.lambda *. vi in
      let gi = lv +. ai -. bi in
      Array.unsafe_set w.g i gi ;
      gmax := Float.max !gmax (Float.abs gi) ;
      bmax := Float.max !bmax (Float.abs bi) ;
      amax := Float.max !amax (Float.abs ai) ;
      vmax := Float.max !vmax (Float.abs vi)
    done ;
    let lv = p.lambda *. !vmax and cv = p.column_sum *. !vmax in
    let rounding = p.row_sum *. (!x0max +. cv) in
    let scale = !bmax +. !amax +. lv +. rounding in
    if scale = 0. || !gmax <= tolerance *. scale then running := false
    else begin
      for j = 0 to p.bins - 1 do
        Array.unsafe_set w.cur j (Array.unsafe_get w.s j > 0.)
      done ;
      (* A full step that kept its piece landed on that piece's minimiser, where
         the gradient of the piece, and so of [psi], is zero. *)
      if (!full && !stepped && same w.cur w.prev) || not (factor p w) then
        running := false
      else begin
        direction p w ;
        for j = 0 to p.bins - 1 do
          Array.unsafe_set w.e j (transpose p w.d j)
        done ;
        let t = step_length p w in
        let moved = ref 0. in
        for i = 0 to m - 1 do
          moved := Float.max !moved (Float.abs (t *. Array.unsafe_get w.d i))
        done ;
        if not (t > 0. && !moved > epsilon_float *. !vmax) then running := false
        else begin
          for i = 0 to m - 1 do
            let td = t *. Array.unsafe_get w.d i in
            Array.unsafe_set w.v i (Array.unsafe_get w.v i +. td)
          done ;
          refresh p w ;
          full := Float.abs (t -. 1.) <= 1e-12 ;
          stepped := true ;
          let c = w.cur in
          w.cur <- w.prev ;
          w.prev <- c
        end
      end
    end
  done

(* [restart p w] makes the solution in [w.s] the next proximal step's start. *)
let restart p w =
  for j = 0 to p.bins - 1 do
    let s = Array.unsafe_get w.s j in
    let x = if s > 0. then s else 0. in
    Array.unsafe_set w.x0 j x ; Array.unsafe_set w.s j x
  done ;
  Array.fill w.v 0 p.bands 0.

(* [solve p w] leaves in [w.s] the dual image of the last proximal step for the
   frame in [w.b]: the spectrum is its positive part. *)
let solve p w =
  start p w ;
  newton p w ;
  for _ = 2 to proximal_steps do
    restart p w ; newton p w
  done

(* [root power v] is [v ** (1 / power)]. *)
let root power =
  if Float.equal power 2. then Float.sqrt
  else if Float.equal power 1. then Fun.id
  else
    let inverse = 1. /. power in
    fun v -> Float.pow v inverse

(* [invert p ~power m] solves every frame of the non-empty mel spectrogram [m]
   in its own loop; frames are spread over domains in ranges that each own their
   working memory, which changes no bit. *)
let invert (type a) p ~power (m : (float, a) Nx.t) : (float, a) Nx.t =
  let shape = output_shape m p.bins in
  let frames = Nx.dim (Nx.ndim m - 1) m in
  let columns = Nx.numel m / p.bands in
  let bands = p.bands and bins = p.bins in
  let root = root power in
  let run read write =
    Par.ranges ~n:columns ~grain:1 (fun lo hi ->
        let w = work p in
        for col = lo to hi - 1 do
          let plane = col / frames and t = col mod frames in
          let src = (plane * bands * frames) + t in
          for i = 0 to bands - 1 do
            Array.unsafe_set w.b i (read (src + (i * frames)))
          done ;
          solve p w ;
          let dst = (plane * bins * frames) + t in
          for j = 0 to bins - 1 do
            let s = Array.unsafe_get w.s j in
            write (dst + (j * frames)) (root (if s > 0. then s else 0.))
          done
        done )
  in
  match Nx.dtype m with
  | Nx.Float32 ->
      Store.make32 shape (fun dst ->
          Store.read32 m (fun src ->
              run
                (Bigarray.Array1.unsafe_get src)
                (Bigarray.Array1.unsafe_set dst) ) )
  | Nx.Float64 ->
      Store.make64 shape (fun dst ->
          Store.read64 m (fun src ->
              run
                (Bigarray.Array1.unsafe_get src)
                (Bigarray.Array1.unsafe_set dst) ) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast (Nx.dtype m)
        (Store.make64 shape (fun dst ->
             Store.read64 (Nx.cast Nx.float64 m) (fun src ->
                 run
                   (Bigarray.Array1.unsafe_get src)
                   (Bigarray.Array1.unsafe_set dst) ) ) )

let check_power fn power =
  if not (Float.is_finite power && power > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use power = %g (power must be finite and positive)" fn
         power )

let finite (type a) (m : (float, a) Nx.t) =
  let all read count =
    let rec go i = i >= count || (Float.is_finite (read i) && go (i + 1)) in
    go 0
  in
  let count = Nx.numel m in
  match Nx.dtype m with
  | Nx.Float32 ->
      Store.read32 m (fun src -> all (Bigarray.Array1.unsafe_get src) count)
  | Nx.Float64 ->
      Store.read64 m (fun src -> all (Bigarray.Array1.unsafe_get src) count)
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Store.read64 (Nx.cast Nx.float64 m) (fun src ->
          all (Bigarray.Array1.unsafe_get src) count )

(* [check_spectrum ~chunk fn ~bands m] refuses a mel spectrogram [m] that is not
   [[...; bands; frames]] or holds a NaN or infinite value, worded for a stage's
   chunk when [chunk] holds. It runs before the filterbank's decomposition,
   which an invalid input never pays for. *)
let check_spectrum ~chunk fn ~bands m =
  check_rank ~chunk fn ~axis:"n_mels" m ;
  let got = Nx.dim (Nx.ndim m - 2) m in
  if got <> bands then
    invalid_arg
      ( if chunk then
          Printf.sprintf
            "%s: cannot invert %d mel bands with a stage built for %d" fn got
            bands
        else
          Printf.sprintf
            "%s: cannot invert %d mel bands with a filterbank of %d" fn got
            bands ) ;
  if not (finite m) then
    invalid_arg
      (Printf.sprintf "%s: cannot invert a %s holding NaN or infinite values" fn
         (if chunk then "chunk" else "mel spectrogram") )

let empty_spectrum ~bins m = Nx.zeros (Nx.dtype m) (output_shape m bins)

let mel_to_stft_with fn mel ?(power = 2.) m =
  check_power fn power ;
  check_spectrum ~chunk:false fn ~bands:(Mel.Config.n_mels mel) m ;
  if Nx.numel m = 0 then empty_spectrum ~bins:(Mel.Config.bins mel) m
  else invert (basis mel) ~power m

let mel_to_stft mel ?power m = mel_to_stft_with "mel_to_stft" mel ?power m

let mel_to_stft_stage mel ?(power = 2.) () =
  let fn = "mel_to_stft_stage" in
  check_power fn power ;
  let p = basis mel in
  Pipeline.stateless (fun m ->
      check_spectrum ~chunk:true fn ~bands:p.bands m ;
      if Nx.numel m = 0 then empty_spectrum ~bins:p.bins m
      else invert p ~power m )

(* {1 Back to audio} *)

(* [check_synthesis fn stft_config mel_config ?n_iter ?length] checks, under
   [fn]'s name and before the solve, the two configurations against each other
   and what Stft.griffin_lim would refuse only after it: a configuration that is
   not invertible, fewer than one iteration, a negative length — worded as
   Stft.griffin_lim words them. The spectrum the solve hands over always has the
   bin axis Stft.griffin_lim expects, since the two FFT sizes agree. *)
let check_synthesis fn stft_config mel_config ?n_iter ?length () =
  let stft_size = Stft.Config.fft_size stft_config in
  let mel_size = Mel.Config.fft_size mel_config in
  if stft_size <> mel_size then
    invalid_arg
      (Printf.sprintf
         "%s: cannot synthesise a %d-point STFT from a filterbank built for an \
          FFT of size %d (the two configurations must agree on fft_size)"
         fn stft_size mel_size ) ;
  Stft.check_invertible fn stft_config ;
  ( match n_iter with
  | Some n when n < 1 ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot run %d iterations (n_iter must be at least 1)" fn n )
  | Some _ | None ->
      () ) ;
  match length with
  | Some n when n < 0 ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot synthesise a signal of length %d (length must be \
            non-negative)"
           fn n )
  | Some _ | None ->
      ()

let mel_to_audio stft_config mel_config ?power ?n_iter ?length m =
  let fn = "mel_to_audio" in
  check_synthesis fn stft_config mel_config ?n_iter ?length () ;
  Stft.griffin_lim ?n_iter ?length stft_config
    (mel_to_stft_with fn mel_config ?power m)

let mfcc_to_audio stft_config mel_config ?lifter ?reference ?power ?n_iter
    ?length c =
  let fn = "mfcc_to_audio" in
  check_synthesis fn stft_config mel_config ?n_iter ?length () ;
  Option.iter (check_power fn) power ;
  let m = mfcc_to_mel_with fn mel_config ?lifter ?reference c in
  Stft.griffin_lim ?n_iter ?length stft_config
    (mel_to_stft_with fn mel_config ?power m)
