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

type t =
  | Hann
  | Hamming
  | Blackman
  | Blackman_harris
  | Nuttall
  | Bartlett
  | Kaiser of float
  | Gaussian of float
  | Tukey of float
  | Flat_top
  | Rectangular

let equal a b =
  match (a, b) with
  | Hann, Hann
  | Hamming, Hamming
  | Blackman, Blackman
  | Blackman_harris, Blackman_harris
  | Nuttall, Nuttall
  | Bartlett, Bartlett
  | Flat_top, Flat_top
  | Rectangular, Rectangular ->
      true
  | Kaiser a, Kaiser b | Gaussian a, Gaussian b | Tukey a, Tukey b ->
      Float.equal a b
  | _ ->
      false

let pp fmt = function
  | Hann ->
      Format.pp_print_string fmt "hann"
  | Hamming ->
      Format.pp_print_string fmt "hamming"
  | Blackman ->
      Format.pp_print_string fmt "blackman"
  | Blackman_harris ->
      Format.pp_print_string fmt "blackman_harris"
  | Nuttall ->
      Format.pp_print_string fmt "nuttall"
  | Bartlett ->
      Format.pp_print_string fmt "bartlett"
  | Kaiser beta ->
      Format.fprintf fmt "kaiser(%g)" beta
  | Gaussian std ->
      Format.fprintf fmt "gaussian(%g)" std
  | Tukey taper ->
      Format.fprintf fmt "tukey(%g)" taper
  | Flat_top ->
      Format.pp_print_string fmt "flat_top"
  | Rectangular ->
      Format.pp_print_string fmt "rectangular"

(* [validate op w] raises [Invalid_argument] if the shape parameter of [w] is
   outside its domain. [op] names the calling entry point in the message. *)
let validate op = function
  | Kaiser beta when not (Float.is_finite beta && beta >= 0.) ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot use a kaiser window with beta %g (beta must be finite \
            and non-negative)"
           op beta )
  | Gaussian std when not (Float.is_finite std && std > 0.) ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot use a gaussian window with standard deviation %g \
            (standard deviation must be finite and positive)"
           op std )
  | Tukey taper when not (taper >= 0. && taper <= 1.) ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot use a tukey window with taper %g (taper must lie in [0, \
            1])"
           op taper )
  | _ ->
      ()

(* [bessel_i0 x] is the modified Bessel function of the first kind and order
   zero, evaluated by its power series I0(x) = sum_{k>=0} ((x/2)^k / k!)^2.
   Every term is positive, so there is no cancellation, and successive terms
   satisfy t_k = t_(k-1) * q / k^2 with q = x^2 / 4. Once k exceeds sqrt(q) the
   tail is dominated by a geometric series of ratio q / k^2 < 1, so stopping
   when a term falls below 1e-17 of the running sum bounds the truncation error
   below one unit in the last place of a float64 result. *)
let bessel_i0 x =
  let q = 0.25 *. x *. x in
  let rec go k term sum =
    let term = term *. q /. Float.of_int (k * k) in
    let sum = sum +. term in
    if term <= 1e-17 *. sum then sum else go (k + 1) term sum
  in
  if q = 0. then 1. else go 1 1. 1.

(* {1 Sample generation}

   Every window is generated in float64 into a flat buffer of exactly the
   requested length, one fill function per family. A fill describes the
   [m]-point symmetric window and is told how many of its samples the buffer
   holds, which is what separates the two forms: the symmetric window of length
   [n] is [m = n] into [n] slots, and the periodic window of length [n] is [m =
   n + 1] into [n] slots, its last sample falling off the end.

   Each fill walks the first half of the window and stores every value at both
   [i] and [m - 1 - i]. The families are all even about the midpoint and their
   arguments negate exactly under that reflection, so the mirrored half is
   bit-identical to a direct evaluation. *)

type buffer = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

let buffer len : buffer =
  Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout len

(* [put buf len i v] writes [v] at index [i] of [buf], ignoring the indices at
   or past [len] that a periodic fill runs off the end. *)
let put (buf : buffer) len i v =
  if i < len then Bigarray.Array1.unsafe_set buf i v

(* [cosine_fill buf len coefficients m] writes the [m]-point generalized cosine
   window sum_k a_k * cos(k * theta_i), with theta_i evenly spaced on [-pi, pi]
   (the scipy general_cosine formulation). Writing theta_i as (2i - (m - 1)) *
   pi / (m - 1) makes the reflection i -> m - 1 - i negate an integer factor,
   hence theta exactly. The harmonics follow from the Chebyshev identity cos(k *
   theta) = T_k(cos theta) and its recurrence T_(k+1) = 2 * cos(theta) * T_k -
   T_(k-1), so a single cosine per sample carries the whole sum. Requires [m >=
   2] and at least two coefficients. *)
let cosine_fill buf len coefficients m =
  let step = Float.pi /. Float.of_int (m - 1) in
  let order = Array.length coefficients in
  let a0 = Array.unsafe_get coefficients 0
  and a1 = Array.unsafe_get coefficients 1 in
  for i = 0 to (m - 1) / 2 do
    let c = Float.cos (Float.of_int ((2 * i) - (m - 1)) *. step) in
    let acc = ref (a0 +. (a1 *. c)) in
    let previous = ref 1. and current = ref c in
    for k = 2 to order - 1 do
      let t = (2. *. c *. !current) -. !previous in
      acc := !acc +. (Array.unsafe_get coefficients k *. t) ;
      previous := !current ;
      current := t
    done ;
    put buf len i !acc ;
    put buf len (m - 1 - i) !acc
  done

(* [bartlett_fill buf len m] writes the [m]-point triangular window, rising as
   2i / (m - 1) across the first half. Requires [m >= 2]. *)
let bartlett_fill buf len m =
  let last = Float.of_int (m - 1) in
  for i = 0 to (m - 1) / 2 do
    let v = 2. *. Float.of_int i /. last in
    put buf len i v ;
    put buf len (m - 1 - i) v
  done

(* [gaussian_fill buf len std m] writes the [m]-point Gaussian window exp(-(i -
   (m - 1) / 2)^2 / (2 * std^2)). Requires [m >= 2] and [std > 0]. *)
let gaussian_fill buf len std m =
  let half = Float.of_int (m - 1) /. 2. in
  let scale = -1. /. (2. *. std *. std) in
  for i = 0 to (m - 1) / 2 do
    let x = Float.of_int i -. half in
    let v = Float.exp (x *. x *. scale) in
    put buf len i v ;
    put buf len (m - 1 - i) v
  done

(* [tukey_fill buf len taper m] writes the [m]-point tapered-cosine window: a
   raised-cosine flank over the first [width + 1] samples with [width = floor
   (taper * (m - 1) / 2)], its mirror at the other end, and ones in between.
   Requires [m >= 2] and a [taper] in ]0, 1[, which keeps [2 * width < m - 1]
   and so keeps the two flanks and the flat middle disjoint. *)
let tukey_fill buf len taper m =
  let last = Float.of_int (m - 1) in
  let width = Float.to_int (Float.floor (taper *. last /. 2.)) in
  let step = 2. /. taper /. last in
  for i = 0 to width do
    let v =
      0.5 *. (1. +. Float.cos (Float.pi *. (-1. +. (step *. Float.of_int i))))
    in
    put buf len i v ;
    put buf len (m - 1 - i) v
  done ;
  for i = width + 1 to m - width - 2 do
    put buf len i 1.
  done

(* [inv_factorial_squared.(k)] is 1 / (k!)^2, the coefficient of q^k in I0(2 *
   sqrt q) = sum_(k >= 0) q^k / (k!)^2 — the power series of [bessel_i0] read as
   a polynomial in q = x^2 / 4. *)
let inv_factorial_squared =
  let t = Array.make 96 1. in
  for k = 1 to Array.length t - 1 do
    t.(k) <- t.(k - 1) /. Float.of_int (k * k)
  done ;
  t

(* The same coefficients split by parity: the even and odd halves are two
   independent Horner chains in q^2 that recombine as [even + q * odd]. *)
let even_coefficients = Array.init 48 (fun j -> inv_factorial_squared.(2 * j))

let odd_coefficients =
  Array.init 48 (fun j -> inv_factorial_squared.((2 * j) + 1))

(* The highest degree the split coefficient tables carry. *)
let max_degree = 90

(* [series_degree qmax] is the degree past which the terms of I0(2 * sqrt q)
   stay below 1e-19 for every q in [0, qmax]: the terms are positive and
   increasing in q, so bounding them at [qmax] bounds them throughout, and past
   the peak they decay faster than a geometric series. A result above
   [max_degree] means the series does not converge inside the tables. *)
let series_degree qmax =
  let rec go k term =
    if term <= 1e-19 || k > max_degree then k
    else go (k + 1) (term *. qmax /. Float.of_int ((k + 1) * (k + 1)))
  in
  go 1 qmax

(* [i0_series ~even ~odd q] is I0(2 * sqrt q) summed over the degrees up to [2 *
   even] and [2 * odd + 1]. Every term is positive, so the two Horner chains
   carry no cancellation. *)
let i0_series ~even ~odd q =
  let z = q *. q in
  let pe = ref (Array.unsafe_get even_coefficients even) in
  for j = even - 1 downto 0 do
    pe := (!pe *. z) +. Array.unsafe_get even_coefficients j
  done ;
  let po = ref (Array.unsafe_get odd_coefficients odd) in
  for j = odd - 1 downto 0 do
    po := (!po *. z) +. Array.unsafe_get odd_coefficients j
  done ;
  !pe +. (q *. !po)

(* [kaiser_fill buf len beta m] writes the [m]-point Kaiser window I0(beta *
   sqrt (1 - r^2)) / I0(beta), r = (i - alpha) / alpha with alpha = (m - 1) / 2.
   In terms of q = (beta^2 / 4) * (1 - r^2) the numerator is I0(2 * sqrt q): the
   argument is linear in r^2, so no square root is needed per sample. The series
   is summed by [i0_series] when its degree fits the coefficient tables, and by
   [bessel_i0] on its own argument otherwise. Requires [m >= 2] and [beta >=
   0]. *)
let kaiser_fill buf len beta m =
  let alpha = Float.of_int (m - 1) /. 2. in
  let qmax = 0.25 *. beta *. beta in
  let degree = series_degree qmax in
  let half = (m - 1) / 2 in
  if degree > max_degree then
    let denominator = bessel_i0 beta in
    for i = 0 to half do
      let r = (Float.of_int i -. alpha) /. alpha in
      let v = bessel_i0 (beta *. Float.sqrt (1. -. (r *. r))) /. denominator in
      put buf len i v ;
      put buf len (m - 1 - i) v
    done
  else
    let even = degree / 2 and odd = (degree - 1) / 2 in
    let scale = 1. /. i0_series ~even ~odd qmax in
    for i = 0 to half do
      let r = (Float.of_int i -. alpha) /. alpha in
      let v = i0_series ~even ~odd (qmax *. (1. -. (r *. r))) *. scale in
      put buf len i v ;
      put buf len (m - 1 - i) v
    done

(* [fill buf len w m] writes the [m]-point symmetric window [w] into the [len]
   slots of [buf], following the scipy signal windows formula by formula.
   Requires [m >= 2], a validated [w], and a [buf] of exactly [len] elements
   with [len] equal to [m] or to [m - 1]. *)
let rec fill buf len w m =
  match w with
  | Rectangular ->
      Bigarray.Array1.fill buf 1.
  | Hann ->
      cosine_fill buf len [|0.5; 0.5|] m
  | Hamming ->
      cosine_fill buf len [|0.54; 0.46|] m
  | Blackman ->
      cosine_fill buf len [|0.42; 0.5; 0.08|] m
  | Blackman_harris ->
      cosine_fill buf len [|0.35875; 0.48829; 0.14128; 0.01168|] m
  | Nuttall ->
      cosine_fill buf len [|0.3635819; 0.4891775; 0.1365995; 0.0106411|] m
  | Flat_top ->
      cosine_fill buf len
        [|0.21557895; 0.41663158; 0.277263158; 0.083578947; 0.006947368|]
        m
  | Bartlett ->
      bartlett_fill buf len m
  | Kaiser beta ->
      kaiser_fill buf len beta m
  | Gaussian std ->
      gaussian_fill buf len std m
  | Tukey taper ->
      if taper <= 0. then Bigarray.Array1.fill buf 1.
      else if taper >= 1. then fill buf len Hann m
      else tukey_fill buf len taper m

(* [fill_window buf w ~periodic n] writes the [n]-point window [w] into [buf].
   Requires [n >= 1], a validated [w], and a [buf] of exactly [n] elements. A
   one-point window is [1.] in both forms (the guard applies before the periodic
   extension, as in scipy); for [n >= 2] the periodic window is the symmetric
   window of length [n + 1] with the last sample dropped. *)
let fill_window buf w ~periodic n =
  if n = 1 then Bigarray.Array1.unsafe_set buf 0 1.
  else fill buf n w (if periodic then n + 1 else n)

(* {1 Instantiation}

   Float64 windows are generated straight into the storage of the tensor being
   returned. The narrower float dtypes are generated in float64 and rounded once
   on their way out, so every dtype carries the same values under a single
   rounding. Only the Bigarray-backed kinds can be reached as a flat array; the
   extended float kinds go through the buffer setter element by element. *)

let make : type b.
    (float, b) Nx.dtype -> ?periodic:bool -> t -> int -> (float, b) Nx.t =
 fun dtype ?(periodic = true) w n ->
  if n < 1 then
    invalid_arg
      (Printf.sprintf
         "make: cannot make a %d-point window (length must be at least 1)" n ) ;
  validate "make" w ;
  match dtype with
  | Nx.Float64 ->
      let tensor = Nx.empty Nx.Float64 [|n|] in
      fill_window (Nx_buffer.to_bigarray1 (Nx.to_buffer tensor)) w ~periodic n ;
      tensor
  | Nx.Float32 ->
      let source = buffer n in
      fill_window source w ~periodic n ;
      let tensor = Nx.empty Nx.Float32 [|n|] in
      let destination = Nx_buffer.to_bigarray1 (Nx.to_buffer tensor) in
      for i = 0 to n - 1 do
        Bigarray.Array1.unsafe_set destination i
          (Bigarray.Array1.unsafe_get source i)
      done ;
      tensor
  | dtype ->
      let source = buffer n in
      fill_window source w ~periodic n ;
      let tensor = Nx.empty dtype [|n|] in
      let destination = Nx.to_buffer tensor in
      for i = 0 to n - 1 do
        Nx_buffer.unsafe_set destination i (Bigarray.Array1.unsafe_get source i)
      done ;
      tensor

(* Relative tolerance of the numerical constant-overlap-add check. *)
let cola_tolerance = 1e-10

let cola w ~length ~hop =
  if length < 1 then
    invalid_arg
      (Printf.sprintf
         "cola: cannot check overlap-add of a %d-point window (length must be \
          at least 1)"
         length ) ;
  if hop < 1 || hop > length then
    invalid_arg
      (Printf.sprintf
         "cola: cannot check overlap-add at hop %d (hop must lie in [1, %d])"
         hop length ) ;
  validate "cola" w ;
  let window = buffer length in
  fill_window window w ~periodic:true length ;
  let sums = Array.make hop 0. in
  for i = 0 to length - 1 do
    let v = Bigarray.Array1.unsafe_get window i in
    sums.(i mod hop) <- sums.(i mod hop) +. v
  done ;
  let mean = Array.fold_left ( +. ) 0. sums /. Float.of_int hop in
  mean > 0.
  && Array.for_all
       (fun s -> Float.abs (s -. mean) <= cola_tolerance *. mean)
       sums
