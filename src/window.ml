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

type window = [`Hanning | `Hamming | `Blackman | `Boxcar]

let cosine_sum ?(fftbins = false) (device : 'dev Rune.device)
    (dtype : (float, 'b) Rune.dtype) (a : float array) m =
  if m < 0 then invalid_arg "Window length M must be a non-negative integer"
  else if m = 0 then Rune.empty device dtype [|0|]
  else if m = 1 then Rune.ones device dtype [|1|]
  else
    let sym = not fftbins in
    let fac =
      if sym then Rune.linspace device dtype (-.Float.pi) Float.pi m
      else
        let full_range =
          Rune.linspace device dtype (-.Float.pi) Float.pi (m + 1)
        in
        Rune.slice [Rs (0, m, 1)] full_range
    in
    let w =
      if Array.length a > 0 then Rune.full device dtype [|m|] a.(0)
      else Rune.zeros device dtype [|m|]
    in
    let w =
      Array.fold_left
        (fun acc (k, coeff) ->
          if k = 0 || coeff = 0.0 then acc
          else
            let cos_args = Rune.mul_s fac (float_of_int k) in
            let cos_terms = Rune.cos cos_args in
            let term = Rune.mul_s cos_terms coeff in
            Rune.add acc term )
        w
        (Array.mapi (fun i x -> (i, x)) a)
    in
    w

let hanning ?(fftbins = false) (device : 'dev Rune.device)
    (dtype : (float, 'b) Rune.dtype) m =
  cosine_sum ~fftbins device dtype [|0.5; 1. -. 0.5|] m

let hamming ?(fftbins = false) (device : 'dev Rune.device)
    (dtype : (float, 'b) Rune.dtype) m =
  cosine_sum ~fftbins device dtype [|0.54; 1. -. 0.54|] m

let blackman ?(fftbins = false) (device : 'dev Rune.device)
    (dtype : (float, 'b) Rune.dtype) m =
  cosine_sum ~fftbins device dtype [|0.42; 0.5; 0.08|] m

let boxcar (device : 'dev Rune.device) (dtype : ('a, 'b) Rune.dtype) (size : int)
    : ('a, 'b, 'dev) Rune.t =
  if size < 0 then failwith "Window length M must be non-negative"
  else if size = 0 then Rune.empty device dtype [|0|]
  else Rune.ones device dtype [|size|]

let get (typ : window) (device : 'dev Rune.device)
    (dtype : (float, 'b) Rune.dtype) :
    ?fftbins:bool -> int -> (float, 'b, 'dev) Rune.t =
 fun ?fftbins size ->
  match typ with
  | `Hanning ->
      hanning ?fftbins device dtype size
  | `Hamming ->
      hamming ?fftbins device dtype size
  | `Blackman ->
      blackman ?fftbins device dtype size
  | `Boxcar ->
      boxcar device dtype size
