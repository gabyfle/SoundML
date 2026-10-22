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

open Bigarray

type f32 = (float, float32_elt, c_layout) Array1.t

type f64 = (float, float64_elt, c_layout) Array1.t

let numel shape = Array.fold_left ( * ) 1 shape

(* A fresh buffer is the tensor's alone until [of_buffer] shares it, so writing
   it first is writing memory nothing else reads. *)
let make dtype kind bytes shape fill =
  let n = numel shape in
  let buffer = Rig.Buffer.create Rig.host (bytes * n) in
  if n > 0 then Rig.Buffer.write buffer kind fill ;
  Nx.of_buffer dtype shape buffer

let make32 shape fill = make Nx.float32 float32 4 shape fill

let make64 shape fill = make Nx.float64 float64 8 shape fill

let make_c128 shape fill = make Nx.complex128 float64 16 shape fill

let make64_pair shape fill =
  let n = numel shape in
  let first = Rig.Buffer.create Rig.host (8 * n)
  and second = Rig.Buffer.create Rig.host (8 * n) in
  if n > 0 then
    Rig.Buffer.write first float64 (fun a ->
        Rig.Buffer.write second float64 (fun b -> fill a b) ) ;
  (Nx.of_buffer Nx.float64 shape first, Nx.of_buffer Nx.float64 shape second)

let empty32 = Array1.create float32 c_layout 0

let empty64 = Array1.create float64 c_layout 0

let read32 x f =
  if Nx.numel x = 0 then f empty32
  else Rig.Buffer.read (Nx.to_buffer x) float32 f

let read64 x f =
  if Nx.numel x = 0 then f empty64
  else Rig.Buffer.read (Nx.to_buffer x) float64 f

let read_parts32 z f =
  if Nx.numel z = 0 then f empty32
  else Rig.Buffer.read (Nx.to_buffer z) float32 f

let read_parts64 z f =
  if Nx.numel z = 0 then f empty64
  else Rig.Buffer.read (Nx.to_buffer z) float64 f
