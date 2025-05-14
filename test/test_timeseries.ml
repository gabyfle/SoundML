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

module Timeseries = struct
  type t = Float.t

  type p = Bigarray.float64_elt

  type pf = Bigarray.float64_elt

  type pc = Bigarray.complex64_elt

  type ('a, 'b) precision = ('a, 'b) Types.precision

  let precision = Types.B64

  let kd = Bigarray.Float64

  let typ = "timeseries"

  let generate (_ : (pf, pc) precision) (_ : string * string * Parameters.t)
      (audio : (float, 'c) Owl_dense_ndarray.Generic.t) =
    audio
end

module Tests = Tests_cases (Timeseries)

let () =
  let name = "Vectors: Timeseries Comparison" in
  let data = Testdata.get Timeseries.typ Vutils.data in
  let tests = Tests.create_tests data in
  Tests.run name tests
