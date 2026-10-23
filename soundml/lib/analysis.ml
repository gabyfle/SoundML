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

(* Autocorrelation, Burg linear prediction and per-frame polynomial fits.

   [autocorrelate] and [lpc] read a whole row of the last axis for every value
   they produce, so they are offline only and offer no stage. [autocorrelate] is
   librosa's Wiener-Khinchin pair of real transforms at its padded length. [lpc]
   is librosa's numba kernel, operation for operation: every sum starts from
   [0.] and adds its terms in index order, every square is a product, and every
   product is bound to a name before it is added, which keeps OCaml's arm64
   backend from contracting it into a fused multiply-add the kernel never
   computes. The float64 coefficients are therefore librosa's bits.

   [poly_features] is memoryless per frame: the least-squares fit of every frame
   over one frequency grid is one linear map of the frame, precomputed once per
   grid and order as an [[order + 1; bins]] matrix — the pseudo-inverse of
   numpy.polyfit's column-scaled Vandermonde matrix, by a one-sided Jacobi SVD
   under lstsq's own rank rule, its sums compensated — and applied to every
   frame by a compensated sum ({!apply}). Each coefficient is then a fixed-order
   sum over its own frame, the same bits alone or among others, so the stage and
   the offline functions agree on every chunking. *)

let numel shape = Array.fold_left ( * ) 1 shape

let has_zero shape = Array.exists (fun d -> d = 0) shape

(* [all_finite t] is [true] when no element of [t] is NaN or infinite. *)
let all_finite (type a) (t : (float, a) Nx.t) =
  let scan32 (a : Store.f32) =
    let ok = ref true in
    for i = 0 to Bigarray.Array1.dim a - 1 do
      if not (Float.is_finite (Bigarray.Array1.unsafe_get a i)) then ok := false
    done ;
    !ok
  in
  let scan64 (a : Store.f64) =
    let ok = ref true in
    for i = 0 to Bigarray.Array1.dim a - 1 do
      if not (Float.is_finite (Bigarray.Array1.unsafe_get a i)) then ok := false
    done ;
    !ok
  in
  match Nx.dtype t with
  | Nx.Float32 ->
      Store.read32 t scan32
  | Nx.Float64 ->
      Store.read64 t scan64
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Store.read64 (Nx.cast Nx.float64 t) scan64

(* {1 Autocorrelation} *)

(* [fast_length target] is the smallest [2^a 3^b 5^c] at or above [target >= 1]:
   scipy.fft.next_fast_len's size for real input, the length librosa pads its
   transforms to — and a length Nx transforms with its native radices. *)
let fast_length target =
  let best = ref max_int in
  let p5 = ref 1 in
  while !p5 < !best do
    let p35 = ref !p5 in
    while !p35 < !best do
      let p = ref !p35 in
      while !p < target do
        p := 2 * !p
      done ;
      if !p < !best then best := !p ;
      p35 := 3 * !p35
    done ;
    p5 := 5 * !p5
  done ;
  !best

(* [power_fill src dst lo hi] writes [re *. re +. im *. im] and a zero imaginary
   part for the bins [lo, hi) of the interleaved complex128 [src], both products
   bound first, as librosa's [abs2] spells them. *)
let power_fill (src : Store.f64) (dst : Store.f64) lo hi =
  for i = lo to hi - 1 do
    let re = Bigarray.Array1.unsafe_get src (2 * i) in
    let im = Bigarray.Array1.unsafe_get src ((2 * i) + 1) in
    let rr = re *. re in
    let ii = im *. im in
    Bigarray.Array1.unsafe_set dst (2 * i) (rr +. ii) ;
    Bigarray.Array1.unsafe_set dst ((2 * i) + 1) 0.
  done

(* [lags dtype r ~n_pad ~m shape] is the first [m] values of every row of the
   [[...; n_pad]] float64 tensor [r], rounded once into [dtype], shaped
   [shape]. *)
let lags (type a) (dtype : (float, a) Nx.dtype) r ~n_pad ~m shape :
    (float, a) Nx.t =
  let rows = numel shape / m in
  let copy32 (src : Store.f64) (dst : Store.f32) =
    for row = 0 to rows - 1 do
      for k = 0 to m - 1 do
        Bigarray.Array1.unsafe_set dst
          ((row * m) + k)
          (Bigarray.Array1.unsafe_get src ((row * n_pad) + k))
      done
    done
  in
  let copy64 (src : Store.f64) (dst : Store.f64) =
    for row = 0 to rows - 1 do
      for k = 0 to m - 1 do
        Bigarray.Array1.unsafe_set dst
          ((row * m) + k)
          (Bigarray.Array1.unsafe_get src ((row * n_pad) + k))
      done
    done
  in
  let lags64 () = Store.read64 r (fun src -> Store.make64 shape (copy64 src)) in
  match dtype with
  | Nx.Float32 ->
      Store.read64 r (fun src -> Store.make32 shape (copy32 src))
  | Nx.Float64 ->
      lags64 ()
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast dtype (lags64 ())

let autocorrelate ?max_size x =
  let fn = "autocorrelate" in
  let nd = Nx.ndim x in
  if nd < 1 then
    invalid_arg
      (fn ^ ": cannot correlate a rank-zero tensor (the time axis must exist)") ;
  let n = Nx.dim (nd - 1) x in
  let m =
    match max_size with
    | None ->
        n
    | Some s ->
        if s < 1 then
          invalid_arg
            (Printf.sprintf
               "%s: cannot keep %d lags (max_size must be at least 1)" fn s ) ;
        Int.min s n
  in
  let out_shape = Array.copy (Nx.shape x) in
  out_shape.(nd - 1) <- m ;
  let dtype = Nx.dtype x in
  if has_zero out_shape then Nx.zeros dtype out_shape
  else
    (* A transform of [2n - 1] points or more keeps every lag of a row clear of
       the circular wrap. It depends on [n] alone, so a shorter [max_size] keeps
       exactly the first lags of the full result. The transforms run in double
       whatever the dtype — Nx widens float32 rows exactly — and transform the
       rows one by one, so a row's lags are the same bits alone or among
       others. *)
    let n_pad = fast_length ((2 * n) - 1) in
    let z = Nx.rfft Nx.complex128 ~axis:(-1) ~n:n_pad x in
    let half = Nx.shape z in
    let count = numel half in
    let p =
      Store.read_parts64 z (fun src ->
          Store.make_c128 half (fun dst ->
              Par.ranges ~n:count ~grain:65_536 (power_fill src dst) ) )
    in
    let r = Nx.irfft Nx.float64 ~axis:(-1) ~n:n_pad p in
    lags dtype r ~n_pad ~m out_shape

(* {1 Linear prediction} *)

(* [burg ~order ~n f b a ap] is the order-[order] Burg prediction-error filter
   of the [n] samples that [f] and [b] hold on entry — [f.(t)] is sample [t + 1]
   and [b.(t)] sample [t], for [t < n - 1] — as librosa's [__lpc] computes it
   (librosa/core/audio.py, after Marple 1980, section III): the buffer the last
   order filled, [a] or [ap], each of [order + 1] floats.

   Order [i] reads the forward errors [f.(i + t)] and the backward errors
   [b.(t)] for [t < n - 1 - i]: librosa's [f = f[1:]] and [b = b[:-1]] after
   each order move these window bounds here, not the data. The denominator is
   summed once from the signal's energy and then updated by Andersen's
   recursion, [((1 - k^2) den - b_last^2) - f_first^2], with the smallest
   positive normal double added before each division. Once the errors are
   exhausted ([i >= n - 1]) the reflection coefficient is a sum of nothing, so
   the remaining coefficients are zeros and the denominator is left as it is;
   librosa reads past its buffers there. *)
let burg ~order ~n f b a ap =
  let len = Int.max 0 (n - 1) in
  Float.Array.fill a 0 (order + 1) 0. ;
  Float.Array.fill ap 0 (order + 1) 0. ;
  Float.Array.unsafe_set a 0 1. ;
  Float.Array.unsafe_set ap 0 1. ;
  let den = ref 0. in
  for t = 0 to len - 1 do
    let ft = Float.Array.unsafe_get f t and bt = Float.Array.unsafe_get b t in
    let ff = ft *. ft in
    let bb = bt *. bt in
    let e = ff +. bb in
    den := !den +. e
  done ;
  let cur = ref a and prev = ref ap in
  for i = 0 to order - 1 do
    let w = len - i in
    let num = ref 0. in
    for t = 0 to w - 1 do
      let bt = Float.Array.unsafe_get b t
      and ft = Float.Array.unsafe_get f (i + t) in
      let p = bt *. ft in
      num := !num +. p
    done ;
    let k = !num *. -2. in
    let k = k /. (!den +. Float.min_float) in
    (* The Levinson step writes order [i]'s coefficients into the buffer that
       held order [i - 2]'s, reading order [i - 1]'s. *)
    let filled = !prev and last = !cur in
    cur := filled ;
    prev := last ;
    for j = 1 to i + 1 do
      let p = k *. Float.Array.unsafe_get last (i - j + 1) in
      Float.Array.unsafe_set filled j (Float.Array.unsafe_get last j +. p)
    done ;
    for t = 0 to w - 1 do
      let ft = Float.Array.unsafe_get f (i + t)
      and bt = Float.Array.unsafe_get b t in
      let pf = k *. bt in
      let pb = k *. ft in
      Float.Array.unsafe_set f (i + t) (ft +. pf) ;
      Float.Array.unsafe_set b t (bt +. pb)
    done ;
    if w > 0 then begin
      let kk = k *. k in
      let q = 1.0 -. kk in
      let qd = q *. !den in
      let bl = Float.Array.unsafe_get b (w - 1) in
      let bl2 = bl *. bl in
      let f0 = Float.Array.unsafe_get f i in
      let f02 = f0 *. f0 in
      let d = qd -. bl2 in
      den := d -. f02
    end
  done ;
  !cur

(* [lpc_rows ~order get ~n coeffs r0 r1] writes the coefficients of rows [r0,
   r1) into [coeffs], [order + 1] per row, reading sample [t] of row [r] as [get
   (r * n + t)]. *)
let lpc_rows ~order get ~n coeffs r0 r1 =
  let len = Int.max 0 (n - 1) in
  let f = Float.Array.make len 0. and b = Float.Array.make len 0. in
  let a = Float.Array.make (order + 1) 0.
  and ap = Float.Array.make (order + 1) 0. in
  for r = r0 to r1 - 1 do
    let off = r * n in
    for t = 0 to len - 1 do
      Float.Array.unsafe_set f t (get (off + t + 1)) ;
      Float.Array.unsafe_set b t (get (off + t))
    done ;
    let out = burg ~order ~n f b a ap in
    Float.Array.blit out 0 coeffs (r * (order + 1)) (order + 1)
  done

let lpc (type a) ~order (x : (float, a) Nx.t) : (float, a) Nx.t =
  let fn = "lpc" in
  if order < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fit a predictor of order %d (order must be at least 1)" fn
         order ) ;
  let nd = Nx.ndim x in
  if nd < 1 then
    invalid_arg
      (fn ^ ": cannot analyse a rank-zero tensor (the time axis must exist)") ;
  if not (all_finite x) then
    invalid_arg
      ( fn
      ^ ": cannot fit a signal holding NaN or infinite samples (every sample \
         must be finite)" ) ;
  let n = Nx.dim (nd - 1) x in
  let shape = Nx.shape x in
  let out_shape = Array.copy shape in
  out_shape.(nd - 1) <- order + 1 ;
  let rows = numel out_shape / (order + 1) in
  let coeffs = Float.Array.make (rows * (order + 1)) 0. in
  (* Rows are independent and each is computed alone, so the split across
     domains changes no bit. *)
  let grain = Int.max 1 (65_536 / Int.max 1 (n * order)) in
  let run get = Par.ranges ~n:rows ~grain (lpc_rows ~order get ~n coeffs) in
  let read64 x64 =
    Store.read64 x64 (fun src ->
        run (fun i -> Bigarray.Array1.unsafe_get src i) )
  in
  ( match Nx.dtype x with
  | Nx.Float32 ->
      Store.read32 x (fun src ->
          run (fun i -> Bigarray.Array1.unsafe_get src i) )
  | Nx.Float64 ->
      read64 x
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      read64 (Nx.cast Nx.float64 x) ) ;
  let fill64 (dst : Store.f64) =
    Float.Array.iteri (fun i v -> Bigarray.Array1.unsafe_set dst i v) coeffs
  in
  let result : (float, a) Nx.t =
    match Nx.dtype x with
    | Nx.Float32 ->
        Store.make32 out_shape (fun dst ->
            Float.Array.iteri
              (fun i v -> Bigarray.Array1.unsafe_set dst i v)
              coeffs )
    | Nx.Float64 ->
        Store.make64 out_shape fill64
    | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
        Nx.cast (Nx.dtype x) (Store.make64 out_shape fill64)
  in
  if not (all_finite result) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fit an order-%d predictor to this signal (a coefficient \
          overflowed: the signal is too ill-conditioned for the order, or its \
          energy or coefficients exceed the range of the dtype)"
         fn order ) ;
  result

(* {1 Polynomial features} *)

type plan =
  { order: int
  ; bins: int
  ; weights: Float.Array.t
        (* The [[order + 1; bins]] least-squares map, row-major, highest degree
           first. *) }

let check_order fn order =
  if order < 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fit a polynomial of order %d (order must be at least 0)" fn
         order )

let check_sample_rate fn sample_rate =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use a sample rate of %d Hz (sample_rate must be at least \
          1)"
         fn sample_rate )

let check_rank fn s =
  let nd = Nx.ndim s in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fit a rank-%d tensor (a spectrogram is [...; bins; \
          frames])"
         fn nd )

let no_bins fn =
  invalid_arg
    (Printf.sprintf
       "%s: cannot fit a polynomial over no bins (the spectrogram needs at \
        least one bin)"
       fn )

(* [fft_grid ~sample_rate ~fft_size bins] is bin [k] at [k * sample_rate /
   fft_size], evaluated as one reciprocal and one multiply per bin: numpy's
   rfftfreq arithmetic, which librosa's default grid is. *)
let fft_grid ~sample_rate ~fft_size bins =
  let step =
    1.0 /. (Float.of_int fft_size *. (1.0 /. Float.of_int sample_rate))
  in
  Float.Array.init bins (fun k -> Float.of_int k *. step)

(* [column_dot a ~n p q] is the dot product of columns [p] and [q] of the
   column-major matrix [a] of [n] rows, its products added in index order by
   Neumaier's compensated summation (Neumaier 1974; Higham 2002, section 4.3):
   the rounding of every addition is carried in [comp] and added back at the
   end, so the sum is within about one ulp of the exact sum of the rounded
   products whatever [n] is. A plain running sum is not: over a column of [n]
   near-equal squares its rounding grows with [n] and is systematic, and the fit
   inherits it — at order 0, whose map is [1 / bins] at every bin, a plain sum
   puts it [2.5e-13] off over [8939] bins, where numpy's fit is within [4e-15]
   of the exact mean. *)
let column_dot a ~n p q =
  let s = ref 0. and comp = ref 0. in
  for i = 0 to n - 1 do
    let x = Float.Array.unsafe_get a ((p * n) + i)
    and y = Float.Array.unsafe_get a ((q * n) + i) in
    let xy = x *. y in
    let sum = !s in
    let t = sum +. xy in
    let lost =
      if Float.abs sum >= Float.abs xy then sum -. t +. xy else xy -. t +. sum
    in
    comp := !comp +. lost ;
    s := t
  done ;
  !s +. !comp

(* [jacobi a ~n ~m r] orthogonalises the [m] columns of the column-major [n x m]
   matrix [a] in place by one-sided cyclic Jacobi rotations (Hestenes 1958),
   accumulating the rotations into the column-major [m x m] matrix [r], which
   must hold the identity on entry: on return [a] is the input times [r], with
   mutually orthogonal columns, their norms the singular values. A pair is
   rotated while the cosine of its angle exceeds the double epsilon; the sweeps
   stop once one rotates nothing, or after 60 sweeps. Columns that the others
   span shrink geometrically, sweep after sweep, towards zero. *)
let jacobi a ~n ~m r =
  let dot = column_dot a ~n in
  let rotate buf len p q c s =
    for i = 0 to len - 1 do
      let x = Float.Array.unsafe_get buf ((p * len) + i)
      and y = Float.Array.unsafe_get buf ((q * len) + i) in
      let cx = c *. x and sy = s *. y and sx = s *. x and cy = c *. y in
      Float.Array.unsafe_set buf ((p * len) + i) (cx -. sy) ;
      Float.Array.unsafe_set buf ((q * len) + i) (sx +. cy)
    done
  in
  let rotated = ref true and sweeps = ref 0 in
  while !rotated && !sweeps < 60 do
    rotated := false ;
    incr sweeps ;
    for p = 0 to m - 2 do
      for q = p + 1 to m - 1 do
        let alpha = dot p p and beta = dot q q and gamma = dot p q in
        if
          Float.abs gamma
          > epsilon_float *. (Float.sqrt alpha *. Float.sqrt beta)
        then begin
          rotated := true ;
          (* The rotation that zeroes [gamma]: [t] the tangent of its angle, the
             smaller root of [t^2 + 2 zeta t - 1 = 0]. *)
          let zeta = (beta -. alpha) /. (2. *. gamma) in
          let t =
            Float.copy_sign 1. zeta /. (Float.abs zeta +. Float.hypot 1. zeta)
          in
          let c = 1. /. Float.hypot 1. t in
          let s = c *. t in
          rotate a n p q c s ; rotate r m p q c s
        end
      done
    done
  done

(* [fit_matrix fn ~order grid] is the [[order + 1; bins]] matrix mapping a
   frame's values over [grid] to its least-squares polynomial coefficients,
   highest degree first, as numpy.polyfit sets the problem up and solves it: the
   Vandermonde matrix [V] of decreasing powers by repeated multiplication, each
   column scaled to unit norm (polyfit's equilibration, which brings the
   condition number of an order-3 fit over an audio-rate grid from about 1e12 to
   under 1e2), the scaled matrix's pseudo-inverse [R S^+ U^T] for its singular
   value decomposition [U S R^T], under lstsq's rule — singular values at or
   below [bins * eps] times the largest count as zero, which yields numpy's
   minimum-norm solution when the fit is rank-deficient — and the scaling
   undone, row [c] divided by column [c]'s norm. The decomposition is a
   one-sided Jacobi ({!jacobi}). *)
let fit_matrix fn ~order grid =
  let n = Float.Array.length grid and m = order + 1 in
  if n = 0 then no_bins fn ;
  if not (Float.Array.for_all Float.is_finite grid) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fit over a non-finite bin frequency (freqs must be finite)"
         fn ) ;
  (* [a] is [V], column-major: row [i], column [c] at [c * n + i]; column [c]
     holds the power [order - c]. *)
  let a = Float.Array.make (n * m) 0. in
  for i = 0 to n - 1 do
    let x = Float.Array.unsafe_get grid i in
    let pw = ref 1. in
    Float.Array.unsafe_set a ((order * n) + i) 1. ;
    for p = 1 to order do
      pw := !pw *. x ;
      Float.Array.unsafe_set a (((order - p) * n) + i) !pw
    done
  done ;
  let scale = Float.Array.init m (fun c -> Float.sqrt (column_dot a ~n c c)) in
  if not (Float.Array.for_all (fun s -> Float.is_finite s && s > 0.) scale) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fit a polynomial of order %d over these bin frequencies \
          (a power of the frequencies vanishes at every bin or its squares \
          overflow double precision)"
         fn order ) ;
  for c = 0 to m - 1 do
    let s = Float.Array.unsafe_get scale c in
    for i = 0 to n - 1 do
      let o = (c * n) + i in
      Float.Array.unsafe_set a o (Float.Array.unsafe_get a o /. s)
    done
  done ;
  let r = Float.Array.make (m * m) 0. in
  for c = 0 to m - 1 do
    Float.Array.unsafe_set r ((c * m) + c) 1.
  done ;
  jacobi a ~n ~m r ;
  let sigma = Float.Array.init m (fun k -> Float.sqrt (column_dot a ~n k k)) in
  let largest = Float.Array.fold_left Float.max 0. sigma in
  let floor = Float.of_int n *. epsilon_float *. largest in
  (* [P (c, j)] is [sum over the kept k, ascending, of R (c, k) / sigma k * U
     (j, k)], with [U (j, k) = a (j, k) / sigma k], over column [c]'s norm. *)
  let w = Float.Array.make (m * n) 0. in
  for c = 0 to m - 1 do
    let sc = Float.Array.unsafe_get scale c in
    for j = 0 to n - 1 do
      let acc = ref 0. in
      for k = 0 to m - 1 do
        let sk = Float.Array.unsafe_get sigma k in
        if sk > floor then begin
          let rk = Float.Array.unsafe_get r ((k * m) + c) /. sk in
          let uk = Float.Array.unsafe_get a ((k * n) + j) /. sk in
          let t = rk *. uk in
          acc := !acc +. t
        end
      done ;
      Float.Array.unsafe_set w ((c * n) + j) (!acc /. sc)
    done
  done ;
  w

let plan_of_grid fn ~order grid =
  {order; bins= Float.Array.length grid; weights= fit_matrix fn ~order grid}

let plan_of_config fn ~order ~sample_rate stft_config =
  check_order fn order ;
  check_sample_rate fn sample_rate ;
  let fft_size = Stft.Config.fft_size stft_config in
  let bins = Stft.Config.bins stft_config in
  plan_of_grid fn ~order (fft_grid ~sample_rate ~fft_size bins)

(* Frames per unit of work: a column of a unit's spectrogram block, and the
   accumulators of all its coefficients, stay in the first-level cache. *)
let tile = 64

(* [apply w ~m s] is the [[m; bins]] row-major map [w] applied to every frame of
   the non-empty [s], [[...; bins; frames]] to [[...; m; frames]], in the dtype
   of [s]. Coefficient [(i, t)] adds the products [w (i, j) *. s (j, t)], each
   bound to a name (so never fused into a multiply-add), in ascending [j] by
   cascaded summation (Ogita, Rump and Oishi 2005, algorithm Sum2): Knuth's
   TwoSum leaves the exact rounding error of each addition, the errors are
   summed apart and added back once, so the coefficient is within about an ulp
   of the exact sum of the rounded products, plus [bins * eps] squared times the
   sum of their magnitudes — the products' own rounding is the error left. A
   plain running sum is off by up to [bins * eps] of that sum, systematically so
   over a flat frame: [2.5e-13] of the mean over [8939] bins at order 0, where
   numpy's solver is within [5e-15]. A non-finite sum is kept as it is, its
   error term dropped. Each coefficient reads its own frame alone, in a fixed
   order, so its bits do not depend on the batch or on the split across domains;
   it is computed in double and rounded once into the dtype. *)
let apply (type a) w ~m (s : (float, a) Nx.t) : (float, a) Nx.t =
  let shape = Nx.shape s in
  let nd = Array.length shape in
  let k = shape.(nd - 2) and n = shape.(nd - 1) in
  let outer = numel shape / (k * n) in
  let out_shape = Array.copy shape in
  out_shape.(nd - 2) <- m ;
  let tiles = (n + tile - 1) / tile in
  let grain = Int.max 1 (65_536 / (m * k * tile)) in
  (* [load off len col] copies the [len] values of [s] from flat index [off]
     into [col]; [store off len acc] writes coefficient [i] of the [len] frames
     from [acc.(i * tile)] to flat index [off + i * n] of the result. *)
  let run load store =
    Par.ranges ~n:(outer * tiles) ~grain (fun u0 u1 ->
        let col = Float.Array.make tile 0. in
        let hi = Float.Array.make (m * tile) 0.
        and lo = Float.Array.make (m * tile) 0. in
        for u = u0 to u1 - 1 do
          let o = u / tiles and t0 = u mod tiles * tile in
          let len = Int.min tile (n - t0) in
          Float.Array.fill hi 0 (m * tile) 0. ;
          Float.Array.fill lo 0 (m * tile) 0. ;
          for j = 0 to k - 1 do
            load ((((o * k) + j) * n) + t0) len col ;
            for i = 0 to m - 1 do
              let wij = Float.Array.unsafe_get w ((i * k) + j) in
              let base = i * tile in
              for t = 0 to len - 1 do
                let x = Float.Array.unsafe_get col t in
                let p = wij *. x in
                let h = Float.Array.unsafe_get hi (base + t) in
                let sum = h +. p in
                let pv = sum -. h in
                let e = h -. (sum -. pv) +. (p -. pv) in
                Float.Array.unsafe_set hi (base + t) sum ;
                Float.Array.unsafe_set lo (base + t)
                  (Float.Array.unsafe_get lo (base + t) +. e)
              done
            done
          done ;
          for idx = 0 to (m * tile) - 1 do
            let h = Float.Array.unsafe_get hi idx in
            if Float.is_finite h then
              Float.Array.unsafe_set hi idx (h +. Float.Array.unsafe_get lo idx)
          done ;
          store ((o * m * n) + t0) len hi
        done )
  in
  let load32 (src : Store.f32) off len col =
    for t = 0 to len - 1 do
      Float.Array.unsafe_set col t (Bigarray.Array1.unsafe_get src (off + t))
    done
  in
  let store32 (dst : Store.f32) off len acc =
    for i = 0 to m - 1 do
      for t = 0 to len - 1 do
        Bigarray.Array1.unsafe_set dst
          (off + (i * n) + t)
          (Float.Array.unsafe_get acc ((i * tile) + t))
      done
    done
  in
  let load64 (src : Store.f64) off len col =
    for t = 0 to len - 1 do
      Float.Array.unsafe_set col t (Bigarray.Array1.unsafe_get src (off + t))
    done
  in
  let store64 (dst : Store.f64) off len acc =
    for i = 0 to m - 1 do
      for t = 0 to len - 1 do
        Bigarray.Array1.unsafe_set dst
          (off + (i * n) + t)
          (Float.Array.unsafe_get acc ((i * tile) + t))
      done
    done
  in
  let of64 s64 =
    Store.read64 s64 (fun src ->
        Store.make64 out_shape (fun dst -> run (load64 src) (store64 dst)) )
  in
  match Nx.dtype s with
  | Nx.Float32 ->
      Store.read32 s (fun src ->
          Store.make32 out_shape (fun dst -> run (load32 src) (store32 dst)) )
  | Nx.Float64 ->
      of64 s
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast (Nx.dtype s) (of64 (Nx.cast Nx.float64 s))

(* [fit p s] is the coefficients of every frame of [s], [[...; bins; frames]] to
   [[...; order + 1; frames]], each summed in double from its own frame in a
   fixed order and rounded once into the dtype of [s]. *)
let fit p s =
  let shape = Nx.shape s in
  if has_zero shape then begin
    (* No frames, or no signals: nothing to fit — the broadcast-consistent empty
       result. *)
    let out = Array.copy shape in
    out.(Array.length out - 2) <- p.order + 1 ;
    Nx.zeros (Nx.dtype s) out
  end
  else apply p.weights ~m:(p.order + 1) s

let poly_features stft_config ?(order = 1) ~sample_rate x =
  let fn = "poly_features" in
  let p = plan_of_config fn ~order ~sample_rate stft_config in
  if Nx.ndim x < 1 then
    invalid_arg
      (fn ^ ": cannot analyse a rank-zero tensor (the time axis must exist)") ;
  fit p (Stft.power_spectrum ~power:1. stft_config x)

let poly_features_of_spectrogram ?(order = 1) ?freqs ~sample_rate s =
  let fn = "poly_features_of_spectrogram" in
  check_order fn order ;
  check_sample_rate fn sample_rate ;
  check_rank fn s ;
  let bins = Nx.dim (Nx.ndim s - 2) s in
  let grid =
    match freqs with
    | Some f ->
        if Nx.ndim f <> 1 then
          invalid_arg
            (Printf.sprintf
               "%s: cannot use a rank-%d freqs tensor (freqs is rank-one, one \
                frequency per bin)"
               fn (Nx.ndim f) ) ;
        if Nx.dim 0 f <> bins then
          invalid_arg
            (Printf.sprintf
               "%s: cannot pair %d bin frequencies with %d bins (freqs holds \
                one frequency per bin)"
               fn (Nx.dim 0 f) bins ) ;
        Float.Array.map_from_array Fun.id (Nx.to_array (Nx.cast Nx.float64 f))
    | None ->
        if bins = 0 then no_bins fn ;
        if bins < 2 then
          invalid_arg
            (Printf.sprintf
               "%s: cannot derive bin frequencies for a %d-bin spectrogram \
                (the implied FFT size is %d; pass freqs explicitly)"
               fn bins
               (2 * (bins - 1)) ) ;
        fft_grid ~sample_rate ~fft_size:(2 * (bins - 1)) bins
  in
  fit (plan_of_grid fn ~order grid) s

let stage stft_config ?(order = 1) ~sample_rate () =
  let fn = "poly_features_stage" in
  let p = plan_of_config fn ~order ~sample_rate stft_config in
  Pipeline.stateless (fun s ->
      let nd = Nx.ndim s in
      if nd < 2 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot fit a rank-%d chunk (the stage needs [...; bins; \
              frames])"
             fn nd ) ;
      let bins = Nx.dim (nd - 2) s in
      if bins <> p.bins then
        invalid_arg
          (Printf.sprintf "%s: cannot fit %d bins with a stage built for %d" fn
             bins p.bins ) ;
      fit p s )
