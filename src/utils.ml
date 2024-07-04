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

open Owl

let fftfreq (n : int) (d : float) =
  let nslice = ((n - 1) / 2) + 1 in
  let fhalf =
    Audio.G.linspace Bigarray.Float32 0. (float_of_int nslice) nslice
  in
  let shalf =
    Audio.G.linspace Bigarray.Float32 (-.float_of_int nslice) (-1.) nslice
  in
  let v = Audio.G.concatenate ~axis:0 [|fhalf; shalf|] in
  Arr.(1. /. (d *. float_of_int n) $* v)
