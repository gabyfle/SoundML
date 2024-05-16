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

type audio = {buffer: Buffer.t; channels: Avutil.Channel_layout.t}

module FrameToS32Bytes =
  Swresample.Make (Swresample.Frame) (Swresample.S32Bytes)

let read_audio ?(channels = `Mono) (filename : string) (format : string) : audio
    =
  let buffer = Buffer.create 0 in
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
  let rsp = FrameToS32Bytes.from_codec ~options icodec channels 44100 in
  let rec f () =
    match Av.read_input ~audio_frame:[istream] input with
    | `Audio_frame (i, frame) when i = idx ->
        Buffer.add_bytes buffer (FrameToS32Bytes.convert rsp frame) ;
        f ()
    | exception Avutil.Error `Eof ->
        ()
    | _ ->
        f ()
  in
  f () ;
  Av.get_input istream |> Av.close ;
  Gc.full_major () ;
  Gc.full_major () ;
  {buffer; channels}
