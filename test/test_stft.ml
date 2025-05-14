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

open Soundml
open Vutils

module StftTestable = struct
  type t = Complex.t

  type p = Bigarray.complex32_elt

  type pf = Bigarray.float32_elt

  type pc = Bigarray.complex32_elt

  type ('a, 'b) precision = ('a, 'b) Types.precision

  let precision = Types.B32

  let kd = Bigarray.Complex32

  let typ = "stft"

  let generate (precision : (pf, pc) precision)
      (_cases : string * string * Parameters.t)
      (audio : (float, 'c) Owl_dense_ndarray.Generic.t) =
    let stft = Transform.stft precision audio in
    let _kd = kd in
    stft
end

module Tests = Tests_cases (StftTestable)

let () =
  let name = "Vectors: STFT Comparison" in
  let data = Testdata.get StftTestable.typ Vutils.data in
  let tests = Tests.create_tests data in
  Tests.run name tests
