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
open Tutils

let read_audio (path : string) (sample_rate : int) (mono : bool) :
    (float, Bigarray.float32_elt) Audio.G.t =
  let audio = Io.read ~sample_rate ~mono Bigarray.Float32 path in
  Audio.data audio

module Tests : Testable = struct
  let typ = "timeseries"

  let create_test_set (data : (string * string * Parameters.t) list) =
    let create_tests (basename : string) (case : string * string * Parameters.t)
        =
      let vector_path, audio_path, params = case in
      let sr = Parameters.get_int "sr" params in
      let mono = Parameters.get_bool "mono" params in
      let audio = read_audio audio_path sr mono in
      let kind = Audio.G.kind audio in
      let vector = load_npy vector_path kind in
      let test_allclose_name = typ ^ "_allclose_" ^ basename in
      let test_rallclose () =
        Alcotest.(check bool)
          test_allclose_name true
          (Check.rallclose audio vector)
      in
      let test_shape_name = typ ^ "_shape_" ^ basename in
      let test_shape () =
        Alcotest.(check bool) test_shape_name true (Check.shape audio vector)
      in
      (test_shape, test_rallclose)
    in
    let aux acc x =
      let f, _, _ = x in
      let basename = Filename.basename f |> Filename.remove_extension in
      let tshape, tallclose = create_tests basename x in
      let test_name = Printf.sprintf "test_%s" basename in
      (test_name ^ "_shape", `Slow, tshape)
      :: (test_name ^ "_allclose", `Slow, tallclose)
      :: acc
    in
    List.fold_left aux [] data
end
