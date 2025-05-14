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
   fun ?(rtol = 1e-05) ?(atol = 1e-05) (x : (Complex.t, a) Audio.G.t)
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

let allclose : type a b.
       (a, b) Bigarray.kind
    -> ?rtol:float
    -> ?atol:float
    -> (a, b) Owl_dense_ndarray.Generic.t
    -> (a, b) Owl_dense_ndarray.Generic.t
    -> bool =
 fun kd ->
  match kd with
  | Bigarray.Complex32 ->
      Check.callclose
  | Bigarray.Complex64 ->
      Check.callclose
  | Bigarray.Float32 ->
      Check.rallclose
  | Bigarray.Float64 ->
      Check.rallclose
  | _ ->
      failwith "Unsupported datatype."

let dense_testable : type a b.
    (a, b) Bigarray.kind -> (a, b) Audio.G.t Alcotest.testable =
 fun (_ : (a, b) Bigarray.kind) ->
  let kd_to_string (type a b) (kd : (a, b) Bigarray.kind) =
    match kd with
    | Bigarray.Float32 ->
        "Float32"
    | Bigarray.Float64 ->
        "Float64"
    | Bigarray.Complex32 ->
        "Complex32"
    | Bigarray.Complex64 ->
        "Complex64"
    | _ ->
        failwith "Unsupported kind"
  in
  let pp_kind fmt k =
    let str_k = kd_to_string k in
    Format.fprintf fmt "%s" str_k
  in
  let to_string (type a b) (kd : (a, b) Bigarray.kind) (v : a) =
    match kd with
    | Bigarray.Float32 ->
        Printf.sprintf "%f" v
    | Bigarray.Float64 ->
        Printf.sprintf "%f" v
    | Bigarray.Complex32 ->
        Printf.sprintf "%f + %fi" v.re v.im
    | Bigarray.Complex64 ->
        Printf.sprintf "%f + %fi" v.re v.im
    | _ ->
        failwith "Unsupported kind"
  in
  let pp fmt arr =
    let kd = Audio.G.kind arr in
    let dims = Audio.G.shape arr in
    let first_few_max = 10 in
    let first_few = ref [] in
    let total_elements = Array.fold_left ( * ) 1 dims in
    let flattened = Audio.G.flatten arr in
    if total_elements > 0 && Array.length dims == 1 then
      for i = 0 to first_few_max - 1 do
        first_few := Audio.G.get flattened [|i|] :: !first_few
      done ;
    Format.fprintf fmt
      "Audio.G.t <kind: %a, shape: [%s], data (first %d): [%s]>" pp_kind
      (Audio.G.kind arr)
      (String.concat "; " (Array.to_list (Array.map string_of_int dims)))
      first_few_max
      (String.concat "; " (List.map (to_string kd) (List.rev !first_few)))
  in
  let equal a b =
    let kd = Audio.G.kind a in
    allclose kd a b
  in
  Alcotest.testable pp equal

let float32_g_testable = dense_testable Bigarray.Float32

let float64_g_testable = dense_testable Bigarray.Float64

let complex32_g_testable = dense_testable Bigarray.Complex32

let complex64_g_testable = dense_testable Bigarray.Complex64

let get_dense_testable (type a b) (kd : (a, b) Bigarray.kind) :
    (a, b) Audio.G.t Alcotest.testable =
  match kd with
  | Bigarray.Float32 ->
      float32_g_testable
  | Bigarray.Float64 ->
      float64_g_testable
  | Bigarray.Complex32 ->
      complex32_g_testable
  | Bigarray.Complex64 ->
      complex64_g_testable
  | _ ->
      failwith "Unsupported kind"

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
