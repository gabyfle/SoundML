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

type params = {a: float array; b: float array}

type t =
  { b: (float, Bigarray.float32_elt) Audio.G.t
  ; a: (float, Bigarray.float32_elt) Audio.G.t
  ; state: (float, Bigarray.float32_elt) Audio.G.t }

let reset t = Audio.G.fill t.state 0. ; t

let create ({a; b} : params) =
  let a = Audio.G.of_array Bigarray.Float32 a [|Array.length a|] in
  let b = Audio.G.of_array Bigarray.Float32 b [|Array.length b|] in
  let size = max (Audio.G.numel a) (Audio.G.numel b) in
  let a = Audio.G.(a /$ get a [|0|]) in
  (*let b = Audio.G.(b /$ get b [|0|]) in*)
  let state = Audio.G.create Bigarray.Float32 [|size|] 0. in
  {b; a; state}

let process_sample t (x : float) =
  let n = Audio.G.numel t.state in
  let y =
    if n > 0 then (Audio.G.get t.b [|0|] *. x) +. Audio.G.get t.state [|0|]
    else 0.
  in
  let nb = Audio.G.numel t.b in
  let na = Audio.G.numel t.a in
  for i = 0 to Audio.G.numel t.state - 1 do
    let b = if i + 1 < nb then Audio.G.get t.b [|i + 1|] *. x else 0. in
    let a =
      if i + 1 < na then Float.neg (Audio.G.get t.a [|i + 1|]) *. y else 0.
    in
    if i < n - 1 then
      Audio.G.set t.state [|i|] (Audio.G.get t.state [|i + 1|] +. b +. a)
    else Audio.G.set t.state [|i|] (b +. a)
  done ;
  y
