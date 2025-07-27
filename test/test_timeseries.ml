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

open Vutils

module Timeseries = struct
  type a = float

  type b = Nx.float64_elt

  let dtype = Nx.Float64

  let typ = "timeseries"

  let generate (_ : (a, b) Nx.dtype) (_ : string * string * Parameters.t)
      (audio : (float, Bigarray.float64_elt) Nx.t) =
    Nx.cast dtype audio
end

module Tests = Tests_cases (Timeseries)

let () =
  let name = "Vectors: Timeseries Comparison" in
  let data = Testdata.get Timeseries.typ Vutils.data in
  let tests = Tests.create_tests data 1e-5 1e-8 in
  Tests.run name tests
