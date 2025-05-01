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

let tests = [("timeseries", (module Test_timeseries.Tests : Testable))]

let () =
  let types = List.fold_left (fun acc (x, _) -> x :: acc) [] tests in
  let data = Testdata.create test_vectors_dir test_audio_dir types in
  let tests =
    let aux acc (typ, md) =
      let module Tests = (val md : Testable) in
      (typ_to_readable typ, Tests.create_test_set (Testdata.get typ data))
      :: acc
    in
    List.fold_left aux [] tests
  in
  Alcotest.run "SoundML: Vectors Comparison" tests
