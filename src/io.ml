(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023                                                       *)
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

open Audio

let ( %> ) f g x = g (f x)

(* decoding frames *)
module FrameToS32Bytes =
  Swresample.Make (Swresample.Frame) (Swresample.S32Bytes)

(* encoding arrays *)
module FloatArrayToFrame =
  Swresample.Make (Swresample.FloatArray) (Swresample.Frame)

let read_audio (filename : string) (format : string) : audio =
  let format =
    match Av.Format.find_input_format format with
    | Some f ->
        f
    | None ->
        failwith ("Could not find format: " ^ format)
  in
  let input = Av.open_input ~format filename in
  let idx, istream, icodec = Av.find_best_audio_stream input in
  let out_sr = Avcodec.Audio.get_sample_rate icodec in
  let channels = Avcodec.Audio.get_channel_layout icodec in
  let options = [`Engine_soxr] in
  let rsp = FrameToS32Bytes.from_codec ~options icodec channels out_sr in
  let data = Dynarray.create () in
  let rec f start =
    match Av.read_input ~audio_frame:[istream] input with
    | `Audio_frame (i, frame) when i = idx ->
        let bytes = FrameToS32Bytes.convert rsp frame in
        let length = Bytes.length bytes in
        for i = 0 to length / 4 do
          let offset = i * 4 in
          if offset + 4 <= length then
            let value = Int32.to_float (Bytes.get_int32_ne bytes offset) in
            Dynarray.add_last data value
        done ;
        f (start + (length / 4))
    | exception Avutil.Error `Eof ->
        ()
    | _ ->
        f start
  in
  f 0 ;
  let data =
    G.of_array Bigarray.Float64 (Dynarray.to_array data) [|Dynarray.length data|]
  in
  Av.get_input istream |> Av.close ;
  Gc.full_major () ;
  Gc.full_major () ;
  create ~name:filename ~data ~sampling:out_sr ~codec:icodec

let write_audio (a : audio) (filename : string) (format : string) : unit =
  let open Avcodec in
  let format =
    match Av.Format.guess_output_format ~short_name:format ~filename () with
    | Some f ->
        f
    | None ->
        failwith ("Could not find format: " ^ format)
  in
  let ocodec = Av.Format.get_audio_codec_id format |> Audio.find_encoder in
  Printf.printf "Codec: %s\n" (Audio.get_name ocodec) ;
  let name =
    match
      Avutil.Sample_format.get_name (Audio.find_best_sample_format ocodec `Dbl)
    with
    | Some n ->
        n
    | None ->
        "unknown"
  in
  Printf.printf "Sample format %s\n" name ;
  (* we first need to gather data about the codec used to decode the file *)
  let icodec = codec a in
  let in_cl = Audio.get_channel_layout icodec in
  let channels = Audio.get_nb_channels icodec in
  let in_sample_rate = Audio.get_sample_rate icodec in
  let in_sample_format = Audio.get_sample_format icodec in
  let out_sample_format = Audio.find_best_sample_format ocodec `Dbl in
  let out_sample_rate = Audio.find_best_sample_rate ocodec 44100 in
  let time_base = {Avutil.num= 1; den= in_sample_rate} in
  let encoder =
    Audio.create_encoder ~channel_layout:in_cl ~channels ~time_base
      ~sample_format:out_sample_format ~sample_rate:out_sample_rate ocodec
  in
  let frame_size =
    if List.mem `Variable_frame_size (capabilities ocodec) then 512
    else Audio.frame_size encoder
  in
  let rsp =
    FloatArrayToFrame.create in_cl ~in_sample_format in_sample_rate in_cl
      ~out_sample_format out_sample_rate
  in
  let out_file = open_out_bin filename in
  let values = data a |> G.to_array in
  let length = Array.length values in
  for i = 0 to length / frame_size do
    let start = i * frame_size in
    let finish = min (start + frame_size) length in
    let slice = Array.sub values start (finish - start) in
    try
      let frame = FloatArrayToFrame.convert rsp slice in
      encode encoder (Packet.to_bytes %> output_bytes out_file) frame
    with _ -> ()
  done ;
  flush_encoder encoder (Packet.to_bytes %> output_bytes out_file) ;
  close_out out_file ;
  Gc.full_major () ;
  Gc.full_major ()
