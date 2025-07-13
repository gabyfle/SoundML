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

let cosine_sum ?(fftbins = false) (dtype : ('a, 'b) Nx.dtype) (a : float array)
    m =
  if m < 0 then invalid_arg "Window length M must be a non-negative integer"
  else if m = 0 then Nx.empty dtype [|0|]
  else if m = 1 then Nx.ones dtype [|1|]
  else
    let sym = not fftbins in
    let fac =
      if sym then
        Nx.linspace dtype (-.Float.pi) Float.pi m
      else
        let full_range = Nx.linspace dtype (-.Float.pi) Float.pi (m + 1) in
        Nx.slice [R [0; m]] full_range
    in
    let w =
      if Array.length a > 0 then Nx.full dtype [|m|] a.(0)
      else Nx.zeros dtype [|m|]
    in
    let w =
      Array.fold_left
        (fun acc (k, coeff) ->
          if k = 0 || coeff = 0.0 then acc
          else
            let cos_args = Nx.mul_s fac (float_of_int k) in
            let cos_terms = Nx.cos cos_args in
            let term = Nx.mul_s cos_terms coeff in
            Nx.add acc term )
        w
        (Array.mapi (fun i x -> (i, x)) a)
    in
    w

let hanning ?(fftbins = false) (dtype : ('a, 'b) Nx.dtype) m =
  cosine_sum ~fftbins dtype [|0.5; 1. -. 0.5|] m

let hamming ?(fftbins = false) (dtype : ('a, 'b) Nx.dtype) m =
  cosine_sum ~fftbins dtype [|0.54; 1. -. 0.54|] m

let blackman ?(fftbins = false) (dtype : ('a, 'b) Nx.dtype) m =
  cosine_sum ~fftbins dtype [|0.42; 0.5; 0.08|] m

let boxcar (dtype : ('a, 'b) Nx.dtype) (size : int) : ('a, 'b) Nx.t =
  if size < 0 then failwith "Window length M must be non-negative"
  else if size = 0 then Nx.empty dtype [|0|]
  else Nx.ones dtype [|size|]

let get (typ : window) (dtype : ('a, 'b) Nx.dtype) :
    ?fftbins:bool -> int -> ('a, 'b) Nx.t =
 fun ?fftbins size ->
  match typ with
  | `Hanning ->
      hanning ?fftbins dtype size
  | `Hamming ->
      hamming ?fftbins dtype size
  | `Blackman ->
      blackman ?fftbins dtype size
  | `Boxcar ->
      boxcar dtype size
