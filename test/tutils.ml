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

module Check = struct
  open Soundml

  let shape (x : ('a, 'b) Audio.G.t) (y : ('a, 'b) Audio.G.t) =
    let shape_x = Audio.G.shape x in
    let shape_y = Audio.G.shape y in
    if Array.length shape_x <> Array.length shape_y then false
    else Array.for_all2 (fun x y -> x = y) shape_x shape_y

  let rallclose ?(rtol = 1e-05) ?(atol = 1e-10) (x : ('a, 'b) Audio.G.t)
      (y : ('a, 'b) Audio.G.t) : bool =
    if not (shape x y) then false
    else if Audio.G.numel x = 0 && Audio.G.numel y = 0 then true
    else
      let abs_diff = Audio.G.abs (Audio.G.sub x y) in
      let tolerance = Audio.G.(add_scalar (mul_scalar (abs y) rtol) atol) in
      let comparison_mask = Audio.G.elt_less_equal abs_diff tolerance in
      Audio.G.min' comparison_mask >= 1.0

  let callclose : type a.
         ?rtol:float
      -> ?atol:float
      -> (Complex.t, a) Audio.G.t
      -> (Complex.t, a) Audio.G.t
      -> bool =
   fun ?(rtol = 1e-05) ?(atol = 1e-08) (x : (Complex.t, a) Audio.G.t)
       (y : (Complex.t, a) Audio.G.t) ->
    if not (shape x y) then false
    else if Audio.G.numel x = 0 && Audio.G.numel y = 0 then true
    else
      let x, y =
        match Audio.G.kind x with
        | Bigarray.Complex32 ->
            (Audio.G.cast_c2z x, Audio.G.cast_c2z y)
        | Bigarray.Complex64 ->
            (x, y)
        | _ ->
            .
      in
      let diff = Audio.G.sub x y in
      let abs_diff = Audio.G.abs2_z2d diff in
      let abs_y = Audio.G.abs2_z2d y in
      let tolerance = Audio.G.(add_scalar (mul_scalar abs_y rtol) atol) in
      let comparison_mask = Audio.G.elt_less_equal abs_diff tolerance in
      Audio.G.min' comparison_mask >= 1.0
end

(* This snippet has been gathered from the exact same code but for Matrix in
   Owl. See:
   https://github.com/tachukao/owl/blob/046f703a6890a5ed5ecf4a8c5750d4e392e4ec54/src/owl/dense/owl_dense_matrix_generic.ml#L606-L609
   Unfortunately, for the moment this is not yet available for Ndarrays. *)
let load_npy (path : string) (kind : ('a, 'b) Bigarray.kind) :
    ('a, 'b) Audio.G.t =
  let npy : ('a, 'b) Audio.G.t =
    match Npy.read_copy path |> Npy.to_bigarray Bigarray.c_layout kind with
    | Some x ->
        x
    | None ->
        failwith Printf.(sprintf "%s: incorrect format" path)
  in
  npy
