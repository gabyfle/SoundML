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
  let shape (x : ('a, 'b, 'dev) Rune.t) (y : ('a, 'b2, 'dev) Rune.t) =
    let shape_x = Rune.shape x in
    let shape_y = Rune.shape y in
    if Array.length shape_x <> Array.length shape_y then false
    else Array.for_all2 (fun x y -> x = y) shape_x shape_y

  let rallclose ?(rtol = 1e-5) ?(atol = 1e-10) (x : (float, 'b, 'dev) Rune.t)
      (y : (float, 'b2, 'dev) Rune.t) : bool =
    if not (shape x y) then false
    else if Rune.size x = 0 && Rune.size y = 0 then true
    else
      (* For now, assume tensors are on compatible devices or handle device
         mismatch gracefully *)
      let abs_diff = Rune.abs (Rune.sub x y) in
      let tolerance =
        Rune.add
          (Rune.mul_s (Rune.abs y) rtol)
          (Rune.scalar (Rune.device y) (Rune.dtype y) atol)
      in
      let comparison_mask = Rune.less_equal abs_diff tolerance in
      let all_close = Rune.all comparison_mask in
      Rune.item [] all_close = 1

  let callclose : type a dev.
         ?rtol:float
      -> ?atol:float
      -> (Complex.t, a, dev) Rune.t
      -> (Complex.t, a, dev) Rune.t
      -> bool =
   fun ?(rtol = 1e-05) ?(atol = 1e-05) x y ->
    if not (shape x y) then false
    else if Rune.size x = 0 && Rune.size y = 0 then true
    else
      (* For complex tensors, convert to float64 for comparison *)
      let x_float = Rune.cast Rune.float64 x in
      let y_float = Rune.cast Rune.float64 y in
      let abs_diff = Rune.abs (Rune.sub x_float y_float) in
      let abs_y = Rune.abs y_float in
      let tolerance =
        Rune.add (Rune.mul_s abs_y rtol)
          (Rune.scalar (Rune.device abs_y) (Rune.dtype abs_y) atol)
      in
      let comparison_mask = Rune.less_equal abs_diff tolerance in
      let all_close = Rune.all comparison_mask in
      Rune.item [] all_close = 1
end

let allclose_aux : type a b dev.
       (a, b) Rune.dtype
    -> ?rtol:float
    -> ?atol:float
    -> (a, b, dev) Rune.t
    -> (a, b, dev) Rune.t
    -> bool =
 fun kd ?(rtol = 1e-5) ?(atol = 1e-8) x y ->
  match kd with
  | Rune.Complex32 ->
      Check.callclose ~rtol ~atol x y
  | Rune.Complex64 ->
      Check.callclose ~rtol ~atol x y
  | Rune.Float16 ->
      Check.rallclose ~rtol ~atol x y
  | Rune.Float32 ->
      Check.rallclose ~rtol ~atol x y
  | Rune.Float64 ->
      Check.rallclose ~rtol ~atol x y
  | _ ->
      failwith "Unsupported datatype."

let allclose : type a b dev.
       ?rtol:float
    -> ?atol:float
    -> (a, b, dev) Rune.t
    -> (a, b, dev) Rune.t
    -> bool =
 fun ?rtol ?atol x y -> allclose_aux (Rune.dtype x) ?rtol ?atol x y

let tensor_testable : type a b dev.
       (a, b) Rune.dtype
    -> rtol:float
    -> atol:float
    -> (a, b, dev) Rune.t Alcotest.testable =
 fun _ ~rtol ~atol ->
  let equal a b = allclose ~rtol ~atol a b in
  Alcotest.testable Rune.pp equal

let device = Rune.c
