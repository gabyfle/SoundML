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
open Vutils

let string_to_resample_typ = function
  | "soxr_vhq" ->
      Io.SOXR_VHQ
  | "soxr_hq" ->
      Io.SOXR_HQ
  | "soxr_mq" ->
      Io.SOXR_MQ
  | "soxr_lq" ->
      Io.SOXR_LQ
  | _ ->
      Io.NONE

let read_audio (path : string) (res_typ : Io.resampling_t) (sample_rate : int)
    (mono : bool) : (float, Bigarray.float64_elt) Audio.G.t =
  let audio = Io.read ~res_typ ~sample_rate ~mono Bigarray.Float64 path in
  Audio.data audio

module Tests : Testable = struct
  let typ = "timeseries"

  let create_test_set (data : (string * string * Parameters.t) list) =
    let create_tests (basename : string) (case : string * string * Parameters.t)
        =
      let vector_path, audio_path, params = case in
      let sr = Option.value ~default:22050 @@ Parameters.get_int "sr" params in
      let mono =
        Option.value ~default:true @@ Parameters.get_bool "mono" params
      in
      let resampler =
        string_to_resample_typ
          ( Option.value ~default:"None"
          @@ Parameters.get_string "res_type" params )
      in
      let audio = read_audio audio_path resampler sr mono in
      let vector = load_npy vector_path Bigarray.Float64 in
      let test_allclose_name = typ ^ "_allclose_" ^ basename in
      let test_rallclose () =
        Alcotest.(check bool)
          test_allclose_name true
          (Check.rallclose ~atol:1e-7 audio vector)
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
      let basename = Option.get @@ Testdata.get_test_filename basename in
      let tshape, tallclose = create_tests basename x in
      ("SHAPE:    " ^ basename, `Slow, tshape)
      :: ("ALLCLOSE: " ^ basename, `Slow, tallclose)
      :: acc
    in
    List.fold_left aux [] data
end
