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

(* [cosine_sum coefficients m] is the symmetric [m]-point generalized cosine
   window sum_k a_k * cos(k * fac_i) with fac_i evenly spaced on [-pi, pi], the
   scipy general_cosine formulation. Requires [m >= 2]. *)
let cosine_sum coefficients m =
  let step = 2. *. Float.pi /. Float.of_int (m - 1) in
  Array.init m (fun i ->
      let fac = -.Float.pi +. (step *. Float.of_int i) in
      let acc = ref 0. in
      Array.iteri
        (fun k a -> acc := !acc +. (a *. Float.cos (Float.of_int k *. fac)))
        coefficients ;
      !acc )

(* [symmetric w m] is the symmetric [m]-point window [w] as a float64 array,
   following the scipy signal windows formula by formula. Requires [m >= 2]; the
   [m <= 1] guards live in [values]. *)
let rec symmetric w m =
  let last = Float.of_int (m - 1) in
  match w with
  | Rectangular ->
      Array.make m 1.
  | Hann ->
      cosine_sum [|0.5; 0.5|] m
  | Hamming ->
      cosine_sum [|0.54; 0.46|] m
  | Blackman ->
      cosine_sum [|0.42; 0.5; 0.08|] m
  | Blackman_harris ->
      cosine_sum [|0.35875; 0.48829; 0.14128; 0.01168|] m
  | Nuttall ->
      cosine_sum [|0.3635819; 0.4891775; 0.1365995; 0.0106411|] m
  | Flat_top ->
      cosine_sum
        [|0.21557895; 0.41663158; 0.277263158; 0.083578947; 0.006947368|]
        m
  | Bartlett ->
      Array.init m (fun i ->
          let n = Float.of_int i in
          if n <= last /. 2. then 2. *. n /. last else 2. -. (2. *. n /. last) )
  | Kaiser beta ->
      let alpha = last /. 2. in
      let denominator = bessel_i0 beta in
      Array.init m (fun i ->
          let r = (Float.of_int i -. alpha) /. alpha in
          bessel_i0 (beta *. Float.sqrt (1. -. (r *. r))) /. denominator )
  | Gaussian std ->
      let half = last /. 2. in
      let sig2 = 2. *. std *. std in
      Array.init m (fun i ->
          let n = Float.of_int i -. half in
          Float.exp (-.(n *. n) /. sig2) )
  | Tukey taper ->
      if taper <= 0. then Array.make m 1.
      else if taper >= 1. then symmetric Hann m
      else
        let width = Float.to_int (Float.floor (taper *. last /. 2.)) in
        Array.init m (fun i ->
            let n = Float.of_int i in
            if i <= width then
              0.5
              *. ( 1.
                 +. Float.cos (Float.pi *. (-1. +. (2. *. n /. taper /. last)))
                 )
            else if i < m - width - 1 then 1.
            else
              0.5
              *. ( 1.
                 +. Float.cos
                      ( Float.pi
                      *. ((-2. /. taper) +. 1. +. (2. *. n /. taper /. last)) )
                 ) )

(* [values w ~periodic n] is the window as a float64 array. Requires [n >= 1]
   and a validated [w]. A one-point window is [[|1.|]] in both forms (the guard
   applies before the periodic extension, as in scipy); for [n >= 2] the
   periodic window is the symmetric window of length [n + 1] with the last
   sample dropped. *)
let values w ~periodic n =
  if n = 1 then [|1.|]
  else if periodic then Array.sub (symmetric w (n + 1)) 0 n
  else symmetric w n

let make dtype ?(periodic = true) w n =
  if n < 1 then
    invalid_arg
      (Printf.sprintf
         "make: cannot make a %d-point window (length must be at least 1)" n ) ;
  validate "make" w ;
  Nx.create dtype [|n|] (values w ~periodic n)

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
  let window = values w ~periodic:true length in
  let sums = Array.make hop 0. in
  Array.iteri (fun i v -> sums.(i mod hop) <- sums.(i mod hop) +. v) window ;
  let mean = Array.fold_left ( +. ) 0. sums /. Float.of_int hop in
  mean > 0.
  && Array.for_all
       (fun s -> Float.abs (s -. mean) <= cola_tolerance *. mean)
       sums
