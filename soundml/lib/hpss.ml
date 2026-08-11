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

(* Harmonic/percussive separation by median filtering (Fitzgerald, DAFx 2010)
   with the soft-mask generalisation of Driedger, Müller and Disch (ISMIR 2014).

   A harmonic partial is a horizontal ridge of the magnitude spectrogram [S] and
   a percussive transient a vertical one, so

   harm[b, t] = median over the k_h frames centred on t of S[b, .] perc[b, t] =
   median over the k_p bins centred on b of S[., t]

   enhance one structure and suppress the other. The two enhanced spectrograms
   drive a pair of masks — Wiener-like for finite [p], hard for [p = infinity] —
   whose margins bias the decision:

   mask_h = f (harm, m_h * perc) mask_p = f (perc, m_p * harm)

   with f (x, r) = x^p / (x^p + r^p) at finite [p] and [x > r] at [p =
   infinity], both computed on inputs rescaled by their pointwise maximum. At
   [m_h = m_p = 1] the two finite-[p] masks sum to one and the components
   partition [S]; beyond that the masks only exclude each other and the
   difference [S - (H + P)] is the residual.

   {1 The median kernels}

   Both filters are one-dimensional, so each is a sliding-window median over a
   line: the window of index [i] is [[i - k/2, i + k - 1 - k/2]] (left-biased
   for even [k]) and the value is rank [k/2] of the ascending window — the upper
   middle, never an average of two. Out-of-range indices reflect
   half-sample-symmetrically with period [2n]: index [i] maps to [refl i n], for
   every [k], however far the window overhangs.

   The window is maintained sorted. Advancing by one drops one value and admits
   one, so a binary search locates the departing value and a single shift in one
   direction reinserts the arriving one: O(k) worst case per step, a few
   contiguous moves in practice, and no allocation inside the line.

   The time filter runs along the contiguous axis: each lane is materialised
   once into its reflect-extended form ([n + k - 1] floats) and swept. The
   frequency filter runs along the strided axis, and rather than transpose it
   keeps all [frames] windows live at once ([frames * k] floats) and streams the
   source rows in storage order, so every read is contiguous. *)

(* The flat storage of a contiguous tensor, shared (not copied): the seam
   between the tensor faces and the median kernels. *)
let array1_of t = Nx_buffer.to_bigarray1 (Nx.to_buffer t)

(* {1 Line geometry} *)

(* [refl i n] is the half-sample-symmetric reflection of [i] into [[0, n)]:
   period [2n], with [-1] mapping to [0] and [n] to [n - 1]. Total for every
   integer, so windows may overhang a line by any amount. *)
let refl i n =
  let p = 2 * n in
  let j = ((i mod p) + p) mod p in
  if j < n then j else p - 1 - j

(* [sorted_insert win k] sorts the first [k] entries of [win] ascending by
   insertion — the one full sort per line, at the initial window. *)
let sorted_insert (win : float array) k =
  for i = 1 to k - 1 do
    let v = Array.unsafe_get win i in
    let j = ref (i - 1) in
    while !j >= 0 && Array.unsafe_get win !j > v do
      Array.unsafe_set win (!j + 1) (Array.unsafe_get win !j) ;
      decr j
    done ;
    Array.unsafe_set win (!j + 1) v
  done

(* [replace win base k old nw] turns the ascending run [win[base, base + k)]
   holding [old] into the ascending run holding [nw] instead: binary-search the
   leftmost slot of [old], then shift the entries between it and the slot of
   [nw] one place towards it. *)
let[@inline] replace (win : float array) base k old nw =
  let lo = ref base and hi = ref (base + k - 1) in
  while !lo < !hi do
    let m = (!lo + !hi) / 2 in
    if Array.unsafe_get win m < old then lo := m + 1 else hi := m
  done ;
  let i = ref !lo in
  if nw >= old then begin
    let stop = base + k - 1 in
    while !i < stop && Array.unsafe_get win (!i + 1) < nw do
      Array.unsafe_set win !i (Array.unsafe_get win (!i + 1)) ;
      incr i
    done ;
    Array.unsafe_set win !i nw
  end
  else begin
    while !i > base && Array.unsafe_get win (!i - 1) > nw do
      Array.unsafe_set win !i (Array.unsafe_get win (!i - 1)) ;
      decr i
    done ;
    Array.unsafe_set win !i nw
  end

(* {1 The kernels}

   One pair per element width: the buffers are monomorphic so the element
   accesses compile to direct loads and stores. *)

(* [median_rows src dst ~rows ~n ~k] writes the length-[k] running median of
   each of the [rows] contiguous lanes of [n] values. *)

let median_rows_f64
    (src : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t)
    (dst : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t)
    ~rows ~n ~k =
  let pl = k / 2 in
  let mid = k / 2 in
  let ext = Array.make (n + k - 1) 0. in
  let win = Array.make k 0. in
  for r = 0 to rows - 1 do
    let base = r * n in
    for t = 0 to n + k - 2 do
      Array.unsafe_set ext t
        (Bigarray.Array1.unsafe_get src (base + refl (t - pl) n))
    done ;
    Array.blit ext 0 win 0 k ;
    sorted_insert win k ;
    Bigarray.Array1.unsafe_set dst base (Array.unsafe_get win mid) ;
    for i = 1 to n - 1 do
      replace win 0 k
        (Array.unsafe_get ext (i - 1))
        (Array.unsafe_get ext (i + k - 1)) ;
      Bigarray.Array1.unsafe_set dst (base + i) (Array.unsafe_get win mid)
    done
  done

let median_rows_f32
    (src : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t)
    (dst : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t)
    ~rows ~n ~k =
  let pl = k / 2 in
  let mid = k / 2 in
  let ext = Array.make (n + k - 1) 0. in
  let win = Array.make k 0. in
  for r = 0 to rows - 1 do
    let base = r * n in
    for t = 0 to n + k - 2 do
      Array.unsafe_set ext t
        (Bigarray.Array1.unsafe_get src (base + refl (t - pl) n))
    done ;
    Array.blit ext 0 win 0 k ;
    sorted_insert win k ;
    Bigarray.Array1.unsafe_set dst base (Array.unsafe_get win mid) ;
    for i = 1 to n - 1 do
      replace win 0 k
        (Array.unsafe_get ext (i - 1))
        (Array.unsafe_get ext (i + k - 1)) ;
      Bigarray.Array1.unsafe_set dst (base + i) (Array.unsafe_get win mid)
    done
  done

(* [median_cols src dst ~planes ~rows ~cols ~k] writes the length-[k] running
   median down each column of [planes] consecutive [rows * cols] matrices. All
   [cols] windows advance together, one source row at a time. *)

let median_cols_f64
    (src : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t)
    (dst : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t)
    ~planes ~rows ~cols ~k =
  let pl = k / 2 in
  let mid = k / 2 in
  let wins = Array.make (cols * k) 0. in
  let scratch = Array.make k 0. in
  for plane = 0 to planes - 1 do
    let origin = plane * rows * cols in
    for j = 0 to cols - 1 do
      for t = 0 to k - 1 do
        Array.unsafe_set scratch t
          (Bigarray.Array1.unsafe_get src
             (origin + (refl (t - pl) rows * cols) + j) )
      done ;
      sorted_insert scratch k ;
      Array.blit scratch 0 wins (j * k) k ;
      Bigarray.Array1.unsafe_set dst (origin + j) (Array.unsafe_get scratch mid)
    done ;
    for r = 1 to rows - 1 do
      let out_row = origin + (refl (r - pl - 1) rows * cols) in
      let in_row = origin + (refl (r + k - 1 - pl) rows * cols) in
      let d = origin + (r * cols) in
      for j = 0 to cols - 1 do
        let base = j * k in
        replace wins base k
          (Bigarray.Array1.unsafe_get src (out_row + j))
          (Bigarray.Array1.unsafe_get src (in_row + j)) ;
        Bigarray.Array1.unsafe_set dst (d + j)
          (Array.unsafe_get wins (base + mid))
      done
    done
  done

let median_cols_f32
    (src : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t)
    (dst : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t)
    ~planes ~rows ~cols ~k =
  let pl = k / 2 in
  let mid = k / 2 in
  let wins = Array.make (cols * k) 0. in
  let scratch = Array.make k 0. in
  for plane = 0 to planes - 1 do
    let origin = plane * rows * cols in
    for j = 0 to cols - 1 do
      for t = 0 to k - 1 do
        Array.unsafe_set scratch t
          (Bigarray.Array1.unsafe_get src
             (origin + (refl (t - pl) rows * cols) + j) )
      done ;
      sorted_insert scratch k ;
      Array.blit scratch 0 wins (j * k) k ;
      Bigarray.Array1.unsafe_set dst (origin + j) (Array.unsafe_get scratch mid)
    done ;
    for r = 1 to rows - 1 do
      let out_row = origin + (refl (r - pl - 1) rows * cols) in
      let in_row = origin + (refl (r + k - 1 - pl) rows * cols) in
      let d = origin + (r * cols) in
      for j = 0 to cols - 1 do
        let base = j * k in
        replace wins base k
          (Bigarray.Array1.unsafe_get src (out_row + j))
          (Bigarray.Array1.unsafe_get src (in_row + j)) ;
        Bigarray.Array1.unsafe_set dst (d + j)
          (Array.unsafe_get wins (base + mid))
      done
    done
  done

(* {1 Tensor faces of the kernels}

   The last two axes are the bin and frame axes; leading axes are independent
   planes. Both faces read a contiguous copy of the input and write a fresh
   output of the same shape. *)

let median_time : type a. (float, a) Nx.t -> k:int -> (float, a) Nx.t =
 fun s ~k ->
  let shape = Nx.shape s in
  let n = shape.(Array.length shape - 1) in
  let rows = if n = 0 then 0 else Nx.numel s / n in
  let out = Nx.empty (Nx.dtype s) shape in
  if rows > 0 then begin
    let src = array1_of (Nx.contiguous s) in
    let dst = array1_of out in
    match Nx.dtype s with
    | Nx.Float64 ->
        median_rows_f64 src dst ~rows ~n ~k
    | Nx.Float32 ->
        median_rows_f32 src dst ~rows ~n ~k
    | _ ->
        assert false
  end ;
  out

let median_freq : type a. (float, a) Nx.t -> k:int -> (float, a) Nx.t =
 fun s ~k ->
  let shape = Nx.shape s in
  let nd = Array.length shape in
  let cols = shape.(nd - 1) and rows = shape.(nd - 2) in
  let planes = if rows = 0 || cols = 0 then 0 else Nx.numel s / (rows * cols) in
  let out = Nx.empty (Nx.dtype s) shape in
  if planes > 0 then begin
    let src = array1_of (Nx.contiguous s) in
    let dst = array1_of out in
    match Nx.dtype s with
    | Nx.Float64 ->
        median_cols_f64 src dst ~planes ~rows ~cols ~k
    | Nx.Float32 ->
        median_cols_f32 src dst ~planes ~rows ~cols ~k
    | _ ->
        assert false
  end ;
  out

(* {1 Masks}

   [tiny dtype] is the smallest positive normal of the element width: the floor
   below which a pair of enhanced values carries no decidable energy. *)

let tiny : type a. (float, a) Nx.dtype -> float = function
  | Nx.Float64 ->
      2.2250738585072014e-308
  | Nx.Float32 ->
      1.1754943508222875e-38
  | _ ->
      assert false

(* [powered x p] is [x ^ p]. Unit and square exponents are the identity and one
   multiply — the exact values a general power function need not return. *)
let powered x p =
  if p = 1. then x else if p = 2. then Nx.mul x x else Nx.pow_s x p

(* [softmask x xref ~power ~split_zeros] is the share of [x] against the
   reference [xref], both non-negative and of one shape.

   At finite [power] the pair is first rescaled by its pointwise maximum [z],
   which keeps [x^p] and [xref^p] representable for any [p] and leaves the
   quotient unchanged; the share is then [x^p / (x^p + xref^p)]. Where [z] falls
   below the smallest positive normal the quotient is undefined and the mask
   takes [0.5] when the decision is a partition ([split_zeros]) and [0.]
   otherwise.

   At infinite [power] the mask is the strict comparison [x > xref] as [0.] and
   [1.]: the limit of the finite masks away from equality, and [0.] on both
   sides of it. *)
let softmask x xref ~power ~split_zeros =
  let dtype = Nx.dtype x in
  if not (Float.is_finite power) then
    Nx.where (Nx.greater x xref) (Nx.ones_like x) (Nx.zeros_like x)
  else begin
    let z = Nx.maximum x xref in
    let bad = Nx.less z (Nx.full_like z (tiny dtype)) in
    let z = Nx.where bad (Nx.ones_like z) z in
    let m = powered (Nx.div x z) power in
    let r = powered (Nx.div xref z) power in
    Nx.where bad
      (Nx.full_like z (if split_zeros then 0.5 else 0.))
      (Nx.div m (Nx.add m r))
  end

(* [masks kernel_size power margin s] is the harmonic and percussive mask pair
   of the magnitude spectrogram [s]. *)
let masks ~kernel_size:(k_h, k_p) ~power ~margin:(m_h, m_p) s =
  let harm = median_time s ~k:k_h in
  let perc = median_freq s ~k:k_p in
  let split_zeros = m_h = 1. && m_p = 1. in
  ( softmask harm (Nx.mul_s perc m_h) ~power ~split_zeros
  , softmask perc (Nx.mul_s harm m_p) ~power ~split_zeros )

(* {1 Validation} *)

let check_dtype : type a. string -> (float, a) Nx.dtype -> unit =
 fun fn dt ->
  match dt with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | dt ->
      invalid_arg
        (Stdlib.Format.asprintf
           "%s: cannot separate %a spectra (the median kernel carries float32 \
            and float64)"
           fn Nx.pp_dtype dt )

let check_rank fn s =
  if Nx.ndim s < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot separate a rank-%d tensor (a spectrogram carries a bin \
          axis and a frame axis)"
         fn (Nx.ndim s) )

let check_kernel fn (k_h, k_p) =
  if k_h < 1 || k_p < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot median-filter with a kernel of (%d, %d) (both kernel \
          sizes must be at least 1)"
         fn k_h k_p )

let check_power fn power =
  if Float.is_nan power || power <= 0. then
    invalid_arg
      (Printf.sprintf
         "%s: cannot raise the mask to the power %g (power must be strictly \
          positive, or infinite for a hard mask)"
         fn power )

let check_margin fn (m_h, m_p) =
  if not (Float.is_finite m_h && Float.is_finite m_p && m_h >= 1. && m_p >= 1.)
  then
    invalid_arg
      (Printf.sprintf
         "%s: cannot bias the decision by a margin of (%g, %g) (both margins \
          must be finite and at least 1)"
         fn m_h m_p )

let validate fn ~kernel_size ~power ~margin s =
  check_kernel fn kernel_size ;
  check_power fn power ;
  check_margin fn margin ;
  check_rank fn s ;
  check_dtype fn (Nx.dtype s)

(* {1 Spectrogram domain} *)

let default_kernel = (31, 31)

let default_margin = (1., 1.)

let hpss_masks ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) s =
  validate "hpss_masks" ~kernel_size ~power ~margin s ;
  masks ~kernel_size ~power ~margin s

let hpss_of_spectrogram ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) s =
  validate "hpss_of_spectrogram" ~kernel_size ~power ~margin s ;
  let mask_h, mask_p = masks ~kernel_size ~power ~margin s in
  (Nx.mul s mask_h, Nx.mul s mask_p)

(* {1 Complex domain}

   [component_witness cdtype] is the real storage whose width matches the
   components of [cdtype]: what the magnitude and the two phase planes carry. *)

type packed_rdtype = Rdtype : (float, 'a) Nx.dtype -> packed_rdtype

let component_witness : type c. (Complex.t, c) Nx.dtype -> packed_rdtype =
  function
  | Nx.Complex64 ->
      Rdtype Nx.float32
  | Nx.Complex128 ->
      Rdtype Nx.float64

let hpss_of_stft ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) z =
  let cdtype = Nx.dtype z in
  let (Rdtype rdtype) = component_witness cdtype in
  let mag = Nx.magnitude rdtype z in
  validate "hpss_of_stft" ~kernel_size ~power ~margin mag ;
  (* The unit phase, component by component: a zero magnitude is offset to one
     so the quotient is defined, and its real part carries the offset back, so
     the phase there is [1 + 0i]. Complex division is avoided — its intermediate
     products underflow on denormal components. *)
  let zeros_to_ones =
    Nx.where
      (Nx.equal mag (Nx.zeros_like mag))
      (Nx.ones_like mag) (Nx.zeros_like mag)
  in
  let denominator = Nx.add mag zeros_to_ones in
  let phase_re = Nx.add (Nx.div (Nx.real rdtype z) denominator) zeros_to_ones in
  let phase_im = Nx.div (Nx.imag rdtype z) denominator in
  let mask_h, mask_p = masks ~kernel_size ~power ~margin mag in
  let apply mask =
    let t = Nx.mul mag mask in
    Nx.complex cdtype ~re:(Nx.mul t phase_re) ~im:(Nx.mul t phase_im)
  in
  (apply mask_h, apply mask_p)

(* {1 Signal domain}

   [spectrum_witness dtype] is the complex storage whose component width matches
   [dtype]: what the signal-domain faces analyse into. *)

type packed_cdtype = Cdtype : (Complex.t, 'c) Nx.dtype -> packed_cdtype

let spectrum_witness : type b. (float, b) Nx.dtype -> packed_cdtype = function
  | Nx.Float64 ->
      Cdtype Nx.complex128
  | Nx.Float32 | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Cdtype Nx.complex64

(* [separate fn c ~kernel_size ~power ~margin x] is the analysis-separation-
   synthesis round trip: the components of the STFT of [x] on the geometry [c],
   each inverted back to [x]'s own length and dtype. *)
let separate fn c ~kernel_size ~power ~margin x =
  let dtype = Nx.dtype x in
  check_kernel fn kernel_size ;
  check_power fn power ;
  check_margin fn margin ;
  if Nx.ndim x < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot separate a rank-zero tensor (the time axis must exist)" fn ) ;
  check_dtype fn dtype ;
  let (Cdtype cdtype) = spectrum_witness dtype in
  let length = Nx.dim (Nx.ndim x - 1) x in
  let z_h, z_p =
    hpss_of_stft ~kernel_size ~power ~margin (Stft.transform cdtype c x)
  in
  (Stft.invert dtype c ~length z_h, Stft.invert dtype c ~length z_p)

let hpss c ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) x =
  separate "hpss" c ~kernel_size ~power ~margin x

let harmonic c ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) x =
  fst (separate "harmonic" c ~kernel_size ~power ~margin x)

let percussive c ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) x =
  snd (separate "percussive" c ~kernel_size ~power ~margin x)
