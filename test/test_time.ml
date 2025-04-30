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

open Soundml.Effects.Time

let config_testable : Config.t Alcotest.testable =
  Alcotest.testable
    (Fmt.of_to_string (fun fmt ->
         Format.sprintf "Effects.Time.Config int:%d" (Config.to_int fmt) ) )
    (fun a b -> Config.to_int a = Config.to_int b)

let test_default_int () =
  let expected = 0x00000000 in
  let actual = Config.to_int Config.default in
  Alcotest.check Alcotest.int "Default config integer value" expected actual

let test_percussive_int () =
  let expected = 0x00102000 in
  let actual = Config.to_int Config.percussive in
  Alcotest.check Alcotest.int "Percussive config integer" expected actual ;
  let expected = {Config.default with window= Short; phase= Independent} in
  Alcotest.check config_testable "Percussive config record" expected
    Config.percussive

let test_single_options () =
  Alcotest.check Alcotest.int "EngineFiner" 0x20000000
    (Config.to_int (Config.with_engine Finer Config.default)) ;
  Alcotest.check Alcotest.int "TransientsMixed" 0x00000100
    (Config.to_int (Config.with_transients Mixed Config.default)) ;
  Alcotest.check Alcotest.int "TransientsSmooth" 0x00000200
    (Config.to_int (Config.with_transients Smooth Config.default)) ;
  Alcotest.check Alcotest.int "DetectorPercussive" 0x00000400
    (Config.to_int (Config.with_detector Percussive Config.default)) ;
  Alcotest.check Alcotest.int "DetectorSoft" 0x00000800
    (Config.to_int (Config.with_detector Soft Config.default)) ;
  Alcotest.check Alcotest.int "PhaseIndependent" 0x00002000
    (Config.to_int (Config.with_phase Independent Config.default)) ;
  Alcotest.check Alcotest.int "ThreadingNever" 0x00010000
    (Config.to_int (Config.with_threading Never Config.default)) ;
  Alcotest.check Alcotest.int "ThreadingAlways" 0x00020000
    (Config.to_int (Config.with_threading Always Config.default)) ;
  Alcotest.check Alcotest.int "WindowShort" 0x00100000
    (Config.to_int (Config.with_window Short Config.default)) ;
  Alcotest.check Alcotest.int "WindowLong" 0x00200000
    (Config.to_int (Config.with_window Long Config.default)) ;
  Alcotest.check Alcotest.int "SmoothingOn" 0x00800000
    (Config.to_int (Config.with_smoothing On Config.default)) ;
  Alcotest.check Alcotest.int "FormantPreserved" 0x01000000
    (Config.to_int (Config.with_formant Preserved Config.default)) ;
  Alcotest.check Alcotest.int "PitchHighQuality" 0x02000000
    (Config.to_int (Config.with_pitch HighQuality Config.default)) ;
  Alcotest.check Alcotest.int "PitchHighConsistency" 0x04000000
    (Config.to_int (Config.with_pitch HighConsistency Config.default)) ;
  Alcotest.check Alcotest.int "ChannelsTogether" 0x10000000
    (Config.to_int (Config.with_channels Together Config.default))

let test_combinations () =
  let cfg = Config.default |> Config.with_engine Finer in
  let expected = 0x20000000 in
  Alcotest.check Alcotest.int "Combo: RealTime | Finer" expected
    (Config.to_int cfg) ;
  let cfg =
    Config.default |> Config.with_window Short
    |> Config.with_threading Never
    |> Config.with_formant Preserved
  in
  let expected = 0x01110000 in
  Alcotest.check Alcotest.int "Combo: Short | Never | Preserved" expected
    (Config.to_int cfg) ;
  let cfg =
    Config.
      { engine= Finer
      ; (* 0x20000000 *)
        transients= Smooth
      ; (* 0x00000200 *)
        detector= Soft
      ; (* 0x00000800 *)
        phase= Independent
      ; (* 0x00002000 *)
        threading= Always
      ; (* 0x00020000 *)
        window= Long
      ; (* 0x00200000 *)
        smoothing= On
      ; (* 0x00800000 *)
        formant= Preserved
      ; (* 0x01000000 *)
        pitch= HighConsistency
      ; (* 0x04000000 *)
        channels= Together (* 0x10000000 *) }
  in
  let expected = 0x35A22A00 in
  Alcotest.check Alcotest.int "Combo: All non-default" expected
    (Config.to_int cfg)

let test_modifiers () =
  let base = Config.default in
  let modified_engine = Config.with_engine Finer base in
  Alcotest.check config_testable "with_engine changes only engine"
    {base with engine= Finer} modified_engine ;
  let modified_window_phase =
    base |> Config.with_window Short |> Config.with_phase Independent
  in
  Alcotest.check config_testable "with_window then with_phase"
    {base with window= Short; phase= Independent}
    modified_window_phase ;
  Alcotest.check config_testable "Manual percussive matches preset"
    Config.percussive modified_window_phase

let () =
  Alcotest.run "Effects.Time: Config"
    [ ( "Presets"
      , [ Alcotest.test_case "Default integer value" `Quick test_default_int
        ; Alcotest.test_case "Percussive integer value" `Quick
            test_percussive_int ] )
    ; ( "Single Options"
      , [ Alcotest.test_case "Integer values for single flags" `Quick
            test_single_options ] )
    ; ( "Combinations"
      , [ Alcotest.test_case "Integer values for combined flags" `Quick
            test_combinations ] )
    ; ( "Modifiers"
      , [ Alcotest.test_case "Modifiers create correct configs" `Quick
            test_modifiers ] ) ]
