(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023                                                       *)
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

open Types

type window = [`Hanning | `Hamming | `Blackman | `Boxcar]

let kind_of_precision : type a b. (a, b) precision -> (float, a) Bigarray.kind =
 fun prec -> match prec with B32 -> Bigarray.Float32 | B64 -> Bigarray.Float64

let cosine_sum ?(fftbins = false) (prec : ('a, 'b) precision) (a : float array)
    m =
  let kd = kind_of_precision prec in
  if m < 0 then invalid_arg "Window length M must be a non-negative integer"
  else if m = 0 then Audio.G.empty kd [|0|]
  else if m = 1 then Audio.G.ones kd [|1|]
  else
    let sym = not fftbins in
    let m_extended, needs_trunc =
      if not sym then (m + 1, true) else (m, false)
    in
    let fac = Audio.G.linspace kd (-.Owl_const.pi) Owl_const.pi m_extended in
    let w = Audio.G.zeros kd [|m_extended|] in
    Array.iteri
      (fun k coeff_val ->
        if coeff_val <> 0.0 then
          let term =
            if k = 0 then Audio.G.create kd [|m_extended|] coeff_val
            else
              let k_float = float_of_int k in
              let cos_args = Audio.G.mul_scalar fac k_float in
              let cos_terms = Audio.G.cos cos_args in
              Audio.G.mul_scalar cos_terms coeff_val
          in
          Audio.G.add_ ~out:w w term )
      a ;
    if needs_trunc then Audio.G.get_slice [[0; m - 1]] w else w

let hanning ?(fftbins = false) (prec : ('a, 'b) precision) m =
  cosine_sum ~fftbins prec [|0.5; 1. -. 0.5|] m

let hamming ?(fftbins = false) (prec : ('a, 'b) precision) m =
  cosine_sum ~fftbins prec [|0.54; 1. -. 0.54|] m

let blackman ?(fftbins = false) (prec : ('a, 'b) precision) m =
  cosine_sum ~fftbins prec [|0.42; 0.5; 0.08|] m

let boxcar ?(fftbins = false) (prec : ('a, 'b) precision) (size : int) :
    (float, 'a) Audio.G.t =
  let kd = kind_of_precision prec in
  if size < 0 then failwith "Window length M must be non-negative"
  else if size = 0 then Audio.G.empty kd [|0|]
  else Audio.G.ones kd [|size|]
[@@warning "-27"]

let get (typ : window) (prec : ('a, 'b) precision) :
    ?fftbins:bool -> int -> (float, 'a) Audio.G.t =
 fun ?fftbins size ->
  match typ with
  | `Hanning ->
      hanning ?fftbins prec size
  | `Hamming ->
      hamming ?fftbins prec size
  | `Blackman ->
      blackman ?fftbins prec size
  | `Boxcar ->
      boxcar ?fftbins prec size
