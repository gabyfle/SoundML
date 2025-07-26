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

open Bigarray
open Soundml
open Tutils

let temp_dir_name = ref ""

let setup_test_dir () =
  let dir = Filename.temp_dir "soundml_test_" "" in
  temp_dir_name := dir ;
  if not (Sys.file_exists dir && Sys.is_directory dir) then Unix.mkdir dir 0o755

let delete_test_dir () =
  let rec rm_rf path =
    if Sys.is_directory path then (
      let files = Sys.readdir path in
      Array.iter (fun f -> rm_rf (Filename.concat path f)) files ;
      Unix.rmdir path )
    else Sys.remove path
  in
  if !temp_dir_name <> "" && Sys.file_exists !temp_dir_name then
    rm_rf !temp_dir_name ;
  temp_dir_name := ""

let temp_file ?(ext = ".wav") name = Filename.concat !temp_dir_name (name ^ ext)

let file_exists name =
  try
    Unix.access name [Unix.F_OK] ;
    true
  with
  | Unix.Unix_error (Unix.ENOENT, _, _) ->
      false
  | _ ->
      true

let create_test_audio channels samples sample_rate format =
  let shape = if channels > 1 then [|channels; samples|] else [|samples|] in
  let data = Nx.zeros Nx.float32 shape in
  let freq = 23000. in
  for channel = 0 to channels - 1 do
    for i = 0 to samples - 1 do
      let idx = if channels > 1 then [channel; i] else [i] in
      Nx.set_item idx
        (sin (2. *. Float.pi *. (freq *. Float.of_int channel)))
        data
    done
  done ;
  let meta = Audio.Metadata.create channels samples sample_rate format in
  let audio_data = Audio.create meta data in
  (audio_data, sample_rate)

let create_empty_audio channels sample_rate format =
  let shape = if channels > 1 then [|channels; 0|] else [|0|] in
  let data = Nx.zeros Nx.float32 shape in
  let meta = Audio.Metadata.create channels 0 sample_rate format in
  let audio_data = Audio.create meta data in
  (audio_data, sample_rate)

let audio_testable =
  let pp fmt (a : float32_elt Audio.t) =
    Format.fprintf fmt "{ channels=%d; samples/channel=%d; }" (Audio.channels a)
      (if Audio.channels a > 0 then Audio.samples a else 0)
  in
  let equal a b =
    Check.rallclose ~rtol:1e-05 ~atol:1e-08 (Audio.data a) (Audio.data b)
  in
  Alcotest.testable pp equal

let check_write_read name
    ?(format : Aformat.t = Aformat.{ftype= WAV; sub= PCM_16; endian= FILE})
    channels samples target_sr ext =
  let test_name =
    Printf.sprintf "%s_%dch_%dsamples_%dHz%s" name channels samples target_sr
      ext
  in
  Alcotest.test_case test_name `Quick (fun () ->
      let filename = temp_file ~ext test_name in
      let audio, sr = create_test_audio channels samples target_sr format in
      Io.write ~format filename (Audio.data audio) sr ;
      Alcotest.check Alcotest.bool "Output file exists after write"
        (file_exists filename) true ;
      let read_audio =
        try
          Io.read ~mono:(channels = 1) ~sample_rate:target_sr Nx.Float32
            filename
        with ex ->
          Alcotest.failf "Failed to read back file %s: %s" filename
            (Printexc.to_string ex)
      in
      Alcotest.check Alcotest.int "Channels match after write" channels
        (Audio.channels read_audio) ;
      Alcotest.check Alcotest.int "Sample rate match after write" target_sr
        (Audio.sr read_audio) ;
      Alcotest.check Alcotest.int "Frames match after write" samples
        (Audio.samples read_audio) ;
      Alcotest.check
        (Alcotest.testable Aformat.pp Stdlib.( = ))
        "Format match after write" format (Audio.format read_audio) ;
      Alcotest.check audio_testable "Data unchanged after write" audio
        read_audio )

let check_write_empty name
    ?(format : Aformat.t = Aformat.{ftype= WAV; sub= PCM_16; endian= FILE})
    channels target_sr ext =
  let test_name =
    Printf.sprintf "%s_%dch_empty_%dHz%s" name channels target_sr ext
  in
  Alcotest.test_case test_name `Quick (fun () ->
      let filename = temp_file ~ext test_name in
      let audio, sr = create_empty_audio channels target_sr format in
      Alcotest.check
        (Alcotest.neg Alcotest.reject)
        "Write empty audio don't raise"
        (fun () -> ())
        (fun () -> Io.write ~format filename (Audio.data audio) sr) )

let tests =
  let wav = Result.get_ok (Aformat.create Aformat.WAV) in
  let flac = Result.get_ok (Aformat.create Aformat.FLAC) in
  let ogg = Result.get_ok (Aformat.create Aformat.OGG) in
  [ check_write_read "write_f32_mono_wav_deduced" 1 1024 44100 ".wav"
  ; check_write_read "write_f32_stereo_wav_deduced" 2 1024 44100 ".wav"
  ; check_write_read "write_f32_stereo_flac_deduced" 2 512 22050 ".flac"
  ; check_write_read "write_f32_mono_ogg_deduced" 1 2048 48000 ".ogg"
  ; check_write_read "write_f32_stereo_wav_explicit" ~format:wav 2 1024 44100
      ".wav"
  ; check_write_read "write_f32_stereo_flac_explicit" ~format:flac 2 512 22050
      ".flac"
  ; check_write_read "write_f32_mono_ogg_explicit" ~format:ogg 1 2048 48000
      ".ogg"
  ; check_write_empty "write_f32_mono_empty" 1 44100 ".wav"
  ; check_write_empty "write_f32_stereo_empty" 2 44100 ".wav" ]

let suite = [("Write/Read Roundtrip", tests)]

let () =
  setup_test_dir () ;
  Alcotest.run "SoundML Io.write" suite ;
  delete_test_dir ()
