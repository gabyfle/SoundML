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

let sf_format_wav = 0x010000

let sf_format_aiff = 0x020000

let sf_format_raw = 0x040000

let sf_format_w64 = 0x0B0000

let sf_format_flac = 0x170000

let sf_format_ogg = 0x200000

let sf_format_mpeg = 0x230000

let sf_format_pcm_s8 = 0x0001

let sf_format_pcm_16 = 0x0002

let sf_format_pcm_24 = 0x0003

let sf_format_pcm_32 = 0x0004

let sf_format_float = 0x0006

let sf_format_alaw = 0x0011

let sf_format_vorbis = 0x0060

let sf_format_mpeg_layer_iii = 0x0082

let sf_endian_file = 0x00000000

let sf_endian_little = 0x10000000

let sf_endian_big = 0x20000000

let sf_endian_cpu = 0x30000000

let aformat_testable : Aformat.t Alcotest.testable =
  Alcotest.testable
    (Fmt.of_to_string (fun fmt ->
         Format.sprintf "Aformat int:%d" (Aformat.to_int fmt) ) )
    (fun a b -> Aformat.to_int a = Aformat.to_int b)

let afmt_result_testable : (Aformat.t, string) result Alcotest.testable =
  Alcotest.result aformat_testable Alcotest.string

let check_create_to_int ?(subtype : Aformat.subtype option)
    ?(endian = Aformat.FILE) (ftype : Aformat.ftype) (expected_int : int)
    (test_name : string) () =
  let result = Aformat.create ?subtype ~endian ftype in
  match result with
  | Ok fmt ->
      Alcotest.check Alcotest.int test_name expected_int (Aformat.to_int fmt)
  | Error msg ->
      Alcotest.failf "%s: Aformat.create failed unexpectedly: %s" test_name msg

let test_create_simple_wav () =
  check_create_to_int ~subtype:Aformat.PCM_16 Aformat.WAV
    (sf_format_wav lor sf_format_pcm_16 lor sf_endian_file)
    "create wav pcm16 default endian" ()

let test_create_aiff_float_big () =
  check_create_to_int ~subtype:Aformat.FLOAT ~endian:Aformat.BIG Aformat.AIFF
    (sf_format_aiff lor sf_format_float lor sf_endian_big)
    "create aiff float big endian" ()

let test_create_flac_default () =
  check_create_to_int ~subtype:Aformat.PCM_24 Aformat.FLAC
    (sf_format_flac lor sf_format_pcm_24 lor sf_endian_file)
    "create flac pcm24 default endian" ()

let test_create_ogg_vorbis () =
  check_create_to_int ~subtype:Aformat.VORBIS Aformat.OGG
    (sf_format_ogg lor sf_format_vorbis lor sf_endian_file)
    "create ogg vorbis default endian" ()

let test_create_mp3 () =
  check_create_to_int ~subtype:Aformat.MPEG_LAYER_III Aformat.MP3
    (sf_format_mpeg lor sf_format_mpeg_layer_iii lor sf_endian_file)
    "create mp3 (mpeg layer 3) default endian" ()

let test_create_raw_requires_subtype () =
  let result = Aformat.create Aformat.RAW in
  match result with
  | Ok _ ->
      Alcotest.fail "Creating RAW format without subtype should fail"
  | Error _ ->
      ()

let test_create_raw_ok () =
  check_create_to_int ~subtype:Aformat.PCM_S8 ~endian:Aformat.LITTLE Aformat.RAW
    (sf_format_raw lor sf_format_pcm_s8 lor sf_endian_little)
    "create raw pcm_s8 little endian" ()

let test_create_endian_cpu () =
  check_create_to_int ~subtype:Aformat.PCM_32 ~endian:Aformat.CPU Aformat.W64
    (sf_format_w64 lor sf_format_pcm_32 lor sf_endian_cpu)
    "create w64 pcm32 cpu endian" ()

let check_of_int_roundtrip (code : int) (test_name : string) () =
  let result = Aformat.of_int code in
  match result with
  | Ok fmt ->
      Alcotest.check Alcotest.int (test_name ^ " roundtrip") code
        (Aformat.to_int fmt)
  | Error msg ->
      Alcotest.failf "%s: Aformat.of_int failed unexpectedly for code %d: %s"
        test_name code msg

let check_of_int_error (code : int) (test_name : string) () =
  let result = Aformat.of_int code in
  match result with
  | Ok fmt ->
      Alcotest.failf
        "%s: Aformat.of_int succeeded unexpectedly for code %d, got %d"
        test_name code (Aformat.to_int fmt)
  | Error _ ->
      ()

let test_of_int_wav_pcm16 () =
  check_of_int_roundtrip
    (sf_format_wav lor sf_format_pcm_16 lor sf_endian_file)
    "of_int wav pcm16 default endian" ()

let test_of_int_aiff_float_big () =
  check_of_int_roundtrip
    (sf_format_aiff lor sf_format_float lor sf_endian_big)
    "of_int aiff float big endian" ()

let test_of_int_flac_pcm24 () =
  check_of_int_roundtrip
    (sf_format_flac lor sf_format_pcm_24 lor sf_endian_file)
    "of_int flac pcm24 default endian" ()

let test_of_int_ogg_vorbis () =
  check_of_int_roundtrip
    (sf_format_ogg lor sf_format_vorbis lor sf_endian_file)
    "of_int ogg vorbis default endian" ()

let test_of_int_mp3 () =
  check_of_int_roundtrip
    (sf_format_mpeg lor sf_format_mpeg_layer_iii lor sf_endian_file)
    "of_int mp3 default endian" ()

let test_of_int_raw_pcm_s8_little () =
  check_of_int_roundtrip
    (sf_format_raw lor sf_format_pcm_s8 lor sf_endian_little)
    "of_int raw pcm_s8 little endian" ()

let test_of_int_w64_pcm32_cpu () =
  check_of_int_roundtrip
    (sf_format_w64 lor sf_format_pcm_32 lor sf_endian_cpu)
    "of_int w64 pcm32 cpu endian" ()

let test_of_int_invalid_major () =
  check_of_int_error
    (0xDEAD0000 lor sf_format_pcm_16)
    "of_int invalid major format" ()

let test_of_int_invalid_minor () =
  check_of_int_error (sf_format_wav lor 0xBEEF) "of_int invalid minor format" ()

let test_of_int_invalid_endian () =
  check_of_int_error
    (sf_format_wav lor sf_format_pcm_16 lor 0x40000000)
    "of_int invalid endian" ()

let test_of_int_zero () = check_of_int_error 0 "of_int zero" ()

let check_of_ext ?(sub : Aformat.subtype option) ?(endian = Aformat.FILE)
    (ext : string) (expected_result : (Aformat.t, string) result)
    (test_name : string) () =
  let actual_result = Aformat.of_ext ?sub ~endian ext in
  Alcotest.check afmt_result_testable test_name expected_result actual_result

let test_of_ext_wav () =
  let expected_int = sf_format_wav lor sf_format_pcm_16 lor sf_endian_file in
  let expected_res = Aformat.of_int expected_int in
  check_of_ext ~sub:Aformat.PCM_16 ".wav" expected_res "of_ext .wav pcm16" ()

let test_of_ext_aiff_big () =
  let expected_int = sf_format_aiff lor sf_format_alaw lor sf_endian_big in
  let expected_res = Aformat.of_int expected_int in
  check_of_ext ~sub:Aformat.ALAW ~endian:Aformat.BIG ".aiff" expected_res
    "of_ext .aiff alaw big" ()

let test_of_ext_flac () =
  let expected_int = sf_format_flac lor sf_format_pcm_24 lor sf_endian_file in
  let expected_res = Aformat.of_int expected_int in
  check_of_ext ~sub:Aformat.PCM_24 ".flac" expected_res "of_ext .flac pcm24" ()

let test_of_ext_ogg () =
  let expected_int = sf_format_ogg lor sf_format_vorbis lor sf_endian_file in
  let expected_res = Aformat.of_int expected_int in
  check_of_ext ~sub:Aformat.VORBIS ".ogg" expected_res "of_ext .ogg vorbis" ()

let test_of_ext_mp3 () =
  let expected_int =
    sf_format_mpeg lor sf_format_mpeg_layer_iii lor sf_endian_file
  in
  let expected_res = Aformat.of_int expected_int in
  check_of_ext ~sub:Aformat.MPEG_LAYER_III ".mp3" expected_res "of_ext .mp3" ()

let test_of_ext_case_insensitive () =
  let expected_int = sf_format_wav lor sf_format_pcm_16 lor sf_endian_file in
  let expected_res = Aformat.of_int expected_int in
  check_of_ext ~sub:Aformat.PCM_16 ".WAV" expected_res "of_ext .WAV (uppercase)"
    () ;
  check_of_ext ~sub:Aformat.PCM_16 ".wAv" expected_res
    "of_ext .wAv (mixed case)" ()

let test_of_ext_unknown () =
  check_of_ext ".gabyfle"
    (Error "Couldn't find any format matching extension: .gabyfle")
    "of_ext unknown extension" ()

let test_of_ext_no_dot () =
  check_of_ext "wav" (Error "Couldn't find any format matching extension: wav")
    "of_ext no leading dot" ()

let test_of_ext_empty () =
  check_of_ext "" (Error "Couldn't find any format matching extension: ")
    "of_ext empty string" ()

let test_of_ext_dot_only () =
  check_of_ext "." (Error "Couldn't find any format matching extension: .")
    "of_ext dot only" ()

let create_to_int_suite =
  [ Alcotest.test_case "create: WAV PCM16 Default Endian" `Quick
      test_create_simple_wav
  ; Alcotest.test_case "create: AIFF FLOAT Big Endian" `Quick
      test_create_aiff_float_big
  ; Alcotest.test_case "create: FLAC PCM24 Default Endian" `Quick
      test_create_flac_default
  ; Alcotest.test_case "create: OGG VORBIS Default Endian" `Quick
      test_create_ogg_vorbis
  ; Alcotest.test_case "create: MP3 MPEG_LAYER_III Default Endian" `Quick
      test_create_mp3
  ; Alcotest.test_case "create: RAW requires subtype" `Quick
      test_create_raw_requires_subtype
  ; Alcotest.test_case "create: RAW PCM_S8 Little Endian" `Quick
      test_create_raw_ok
  ; Alcotest.test_case "create: W64 PCM32 CPU Endian" `Quick
      test_create_endian_cpu ]

let of_int_suite =
  [ Alcotest.test_case "of_int: WAV PCM16 Default Endian" `Quick
      test_of_int_wav_pcm16
  ; Alcotest.test_case "of_int: AIFF FLOAT Big Endian" `Quick
      test_of_int_aiff_float_big
  ; Alcotest.test_case "of_int: FLAC PCM24 Default Endian" `Quick
      test_of_int_flac_pcm24
  ; Alcotest.test_case "of_int: OGG VORBIS Default Endian" `Quick
      test_of_int_ogg_vorbis
  ; Alcotest.test_case "of_int: MP3 Default Endian" `Quick test_of_int_mp3
  ; Alcotest.test_case "of_int: RAW PCM_S8 Little Endian" `Quick
      test_of_int_raw_pcm_s8_little
  ; Alcotest.test_case "of_int: W64 PCM32 CPU Endian" `Quick
      test_of_int_w64_pcm32_cpu
  ; Alcotest.test_case "of_int: Invalid Major Format" `Quick
      test_of_int_invalid_major
  ; Alcotest.test_case "of_int: Invalid Minor Format" `Quick
      test_of_int_invalid_minor
  ; Alcotest.test_case "of_int: Invalid Endian" `Quick
      test_of_int_invalid_endian
  ; Alcotest.test_case "of_int: Zero" `Quick test_of_int_zero ]

let of_ext_suite =
  [ Alcotest.test_case "of_ext: .wav PCM16" `Quick test_of_ext_wav
  ; Alcotest.test_case "of_ext .aiff ALAW Big Endian" `Quick
      test_of_ext_aiff_big
  ; Alcotest.test_case "of_ext: .flac PCM24" `Quick test_of_ext_flac
  ; Alcotest.test_case "of_ext: .ogg VORBIS" `Quick test_of_ext_ogg
  ; Alcotest.test_case "of_ext: .mp3 MPEG_LAYER_III" `Quick test_of_ext_mp3
  ; Alcotest.test_case "of_ext: Case Insensitive (.WAV)" `Quick
      test_of_ext_case_insensitive
  ; Alcotest.test_case "of_ext: Unknown (.gabyfle)" `Quick test_of_ext_unknown
  ; Alcotest.test_case "of_ext: No Leading Dot (wav)" `Quick test_of_ext_no_dot
  ; Alcotest.test_case "of_ext: Empty String" `Quick test_of_ext_empty
  ; Alcotest.test_case "of_ext: Dot Only (.)" `Quick test_of_ext_dot_only ]

let () =
  Alcotest.run "Audio Format Module"
    [ ("Create", create_to_int_suite)
    ; ("From Int", of_int_suite)
    ; ("From extension", of_ext_suite) ]
