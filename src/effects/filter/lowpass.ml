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

open Iir

type t = Iir.t

type params = {cutoff: float; sample_rate: int}

let create ({cutoff; sample_rate} : params) =
  let fs = float_of_int sample_rate in
  let fc = cutoff in
  let r = Float.tan (Float.pi *. fc /. fs) in
  let c = (r -. 1.) /. (r +. 1.) in
  let a = [|1.0; c|] in
  let b = [|(1.0 +. c) /. 2.0; (1.0 +. c) /. 2.0|] in
  Iir.create {a; b}

let reset = reset

let process_sample = process_sample
