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
module FrameToS32Bytes =
  Swresample.Make (Swresample.Frame) (Swresample.S32Bytes)
module G = Dense.Ndarray.Generic

type audio =
  { name: string
  ; data: (Complex.t, Bigarray.complex64_elt) G.t
  ; _channels: Avutil.Channel_layout.t }

let size (a : audio) = Dense.Ndarray.Generic.size_in_bytes a.data

let data (a : audio) = a.data

let audio_to_array (buf : Bytes.t) : (Complex.t, Bigarray.complex64_elt) G.t =
  let sample_size = 4 in
  let num_samples = Bytes.length buf in
  let arr = G.create Bigarray.Float32 [|num_samples|] 0. in
  for i = 0 to num_samples / 4 do
    let offset = i * sample_size in
    if offset + sample_size <= Bytes.length buf then
      let sample = Bytes.sub buf offset sample_size in
      let value = Int32.to_float (Bytes.get_int32_be sample 0) in
      G.set arr [|i|] value
  done ;
  Log.info "Done converting into an array of %d 32-bits floats."
    (G.size_in_bytes arr) ;
  arr |> G.cast_s2z

let read_audio ?(channels = `Mono) (filename : string) (format : string) : audio
    =
  Log.info "Reading audio file %s" filename ;
  let buffer = Bytes.create 0 in
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
  let sampling = 44100 in
  let rsp = FrameToS32Bytes.from_codec ~options icodec channels sampling in
  let rec f acc =
    match Av.read_input ~audio_frame:[istream] input with
    | `Audio_frame (i, frame) when i = idx ->
        let bytes = FrameToS32Bytes.convert rsp frame in
        f (Bytes.cat acc bytes)
    | exception Avutil.Error `Eof ->
        acc
    | _ ->
        f acc
  in
  let buffer = f buffer |> audio_to_array in
  Av.get_input istream |> Av.close ;
  Gc.full_major () ;
  Gc.full_major () ;
  Log.info "Read %d bytes from file %s."
    (Dense.Ndarray.Generic.size_in_bytes buffer)
    filename ;
  {name= filename; data= buffer; _channels= channels}

let fft (a : audio) (_start : int) (_finish : int) :
    (Complex.t, Bigarray.complex64_elt) G.t =
  Log.info "Starting to create the FFT of the audio file %s" a.name ;
  a.data |> Owl.Fft.Generic.fft (*|> G.get_slice [[start; finish - 1]]*)
