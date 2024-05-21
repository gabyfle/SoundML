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

open Owl
open Audio

let ( %> ) f g x = g (f x)

(* decoding frames *)
module FrameToS32Bytes =
  Swresample.Make (Swresample.Frame) (Swresample.S32Bytes)

(* encoding arrays *)
module FloatArrayToFrame =
  Swresample.Make (Swresample.FloatArray) (Swresample.Frame)

let read_audio ?(channels = `Mono) (filename : string) (format : string) : audio
    =
  let format =
    match Av.Format.find_input_format format with
    | Some f ->
        f
    | None ->
        failwith ("Could not find format: " ^ format)
  in
  let input = Av.open_input ~format filename in
  let idx, istream, icodec = Av.find_best_audio_stream input in
  let options = [`Engine_soxr] in
  let sampling = 96000 in
  let rsp = FrameToS32Bytes.from_codec ~options icodec channels sampling in
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
  let size = Dynarray.length data in
  let data =
    G.of_array Bigarray.Float64 (Dynarray.to_array data) [|Dynarray.length data|]
  in
  Av.get_input istream |> Av.close ;
  {name= filename; data; sampling; size}

let write_audio ?(sampling = None) (a : audio) (output : string) (fmt : string)
    : unit =
  let sampling = match sampling with Some s -> s | None -> a.sampling in
  let open Avcodec in
  let codec =
    try Audio.find_encoder_by_name fmt
    with _ ->
      Log.error "Could not find codec %s" fmt ;
      exit 1
  in
  let out_sample_format = Audio.find_best_sample_format codec `Dbl in
  let rsp =
    FloatArrayToFrame.create `Mono sampling `Stereo ~out_sample_format sampling
  in
  let time_base = {Avutil.num= 1; den= sampling} in
  let encoder =
    Audio.create_encoder ~channel_layout:`Mono ~channels:1 ~time_base
      ~sample_format:out_sample_format ~sample_rate:sampling codec
  in
  let frame_size =
    if List.mem `Variable_frame_size (capabilities codec) then 512
    else Audio.frame_size encoder
  in
  let out_file = open_out_bin output in
  let values = a.data |> G.to_array in
  let length = Array.length values in
  for i = 0 to length / frame_size do
    let start = i * frame_size in
    let finish = min (start + frame_size) length in
    let slice = Array.sub values start (finish - start - 1) in
    let frame = FloatArrayToFrame.convert rsp slice in
    encode encoder (Packet.to_bytes %> output_bytes out_file) frame
  done ;
  flush_encoder encoder (Packet.to_bytes %> output_bytes out_file) ;
  close_out out_file ;
  Gc.full_major () ;
  Gc.full_major ()
