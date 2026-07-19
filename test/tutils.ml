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
  let shape (x : ('a, 'b) Nx.t) (y : ('a, 'c) Nx.t) =
    let shape_x = Nx.shape x in
    let shape_y = Nx.shape y in
    if Array.length shape_x <> Array.length shape_y then false
    else Array.for_all2 (fun x y -> x = y) shape_x shape_y

  let rallclose ?(rtol = 1e-5) ?(atol = 1e-10) (x : (float, 'b) Nx.t)
      (y : (float, 'b) Nx.t) =
    if not (shape x y) then false
    else if Nx.numel x = 0 then true
    else
      let abs_diff = Nx.abs (Nx.sub x y) in
      let tolerance = Nx.add_s (Nx.mul_s (Nx.abs y) rtol) atol in
      Nx.item [] (Nx.all (Nx.less_equal abs_diff tolerance))

  let callclose : type a.
         ?rtol:float
      -> ?atol:float
      -> (Complex.t, a) Nx.t
      -> (Complex.t, a) Nx.t
      -> bool =
   fun ?(rtol = 1e-05) ?(atol = 1e-05) x y ->
    if not (shape x y) then false
    else if Nx.numel x = 0 then true
    else
      Array.for_all2
        (fun actual expected ->
          Complex.norm (Complex.sub actual expected)
          <= atol +. (rtol *. Complex.norm expected) )
        (Nx.to_array x) (Nx.to_array y)
end

let allclose_aux : type a b.
       (a, b) Nx.dtype
    -> ?rtol:float
    -> ?atol:float
    -> (a, b) Nx.t
    -> (a, b) Nx.t
    -> bool =
 fun kd ?(rtol = 1e-5) ?(atol = 1e-8) x y ->
  match kd with
  | Nx.Complex64 ->
      Check.callclose ~rtol ~atol x y
  | Nx.Complex128 ->
      Check.callclose ~rtol ~atol x y
  | Nx.Float16 ->
      Check.rallclose ~rtol ~atol x y
  | Nx.Float32 ->
      Check.rallclose ~rtol ~atol x y
  | Nx.Float64 ->
      Check.rallclose ~rtol ~atol x y
  | Nx.BFloat16 ->
      Check.rallclose ~rtol ~atol x y
  | Nx.Float8_e4m3 ->
      Check.rallclose ~rtol ~atol x y
  | Nx.Float8_e5m2 ->
      Check.rallclose ~rtol ~atol x y
  | _ ->
      failwith "Unsupported datatype."

let allclose : type a b.
    ?rtol:float -> ?atol:float -> (a, b) Nx.t -> (a, b) Nx.t -> bool =
 fun ?rtol ?atol x y -> allclose_aux (Nx.dtype x) ?rtol ?atol x y

let tensor_testable : type a b.
    rtol:float -> atol:float -> (a, b) Nx.t Alcotest.testable =
 fun ~rtol ~atol ->
  let equal a b = allclose ~rtol ~atol a b in
  Alcotest.testable Nx.pp equal
