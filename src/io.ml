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

module type Writer = sig
  type header

  type t = {header: header; converter: float array -> Bytes.t}

  val header_size : int

  val create : Avutil.Sample_format.t -> int -> int -> t

  val convert : t -> float array -> Bytes.t

  val get_header : t -> int -> Bytes.t
end

(* TODO: Write a "generic" writer for compressed files directly handled by
   ffmpeg.

   This generic writer have a header of size zero since it's handled by ffmpeg
   and should use an encoder from ocaml-ffmpeg for the data. *)

(* These Writer modules are private since the goal of the library isn't to deal
   with audio input and output but rather compute analytics on the audio data *)

module WavWriter : Writer = struct
  (* see https://docs.fileformat.com/audio/wav/ *)
  (* see http://soundfile.sapp.org/doc/WaveFormat/ *)
  type header =
    { num_channels: int
    ; sample_rate: int
    ; byte_rate: int
    ; block_align: int
    ; bits_per_sample: int }

  type t = {header: header; converter: float array -> Bytes.t}

  let header_size = 44

  let create_converter (format : Avutil.Sample_format.t) =
    let func (to_bytes : float -> Bytes.t) (slice : float array) =
      let bytes = ref Bytes.empty in
      for i = 0 to Array.length slice - 1 do
        let data = slice.(i) in
        let conv = to_bytes data in
        bytes := Bytes.cat !bytes conv
      done ;
      !bytes
    in
    match format with
    | `Dbl | `Dblp | `Flt | `Fltp | `None ->
        raise (Invalid_argument "Unsupported format")
    | `S16 | `S16p ->
        let to_bytes (data : float) =
          let data = int_of_float (data *. (Float.pow 2. 15. -. 1.)) in
          let bytes = Bytes.create 2 in
          Bytes.set_int16_ne bytes 0 data ;
          bytes
        in
        (func to_bytes, 16)
    | `S32 | `S32p ->
        let to_bytes (data : float) =
          let data = Int32.of_float (data *. (Float.pow 2. 31. -. 1.)) in
          let bytes = Bytes.create 4 in
          Bytes.set_int32_ne bytes 0 data ;
          bytes
        in
        (func to_bytes, 32)
    | `S64 | `S64p ->
        let to_bytes (data : float) =
          let data = Int64.of_float (data *. (Float.pow 2. 63. -. 1.)) in
          let bytes = Bytes.create 8 in
          Bytes.set_int64_ne bytes 0 data ;
          bytes
        in
        (func to_bytes, 64)
    | `U8 | `U8p ->
        let to_bytes (data : float) =
          let data = int_of_float data in
          let bytes = Bytes.create 1 in
          Bytes.set_int8 bytes 0 data ;
          bytes
        in
        (func to_bytes, 8)

  let create (format : Avutil.Sample_format.t) num_channels sample_rate =
    let converter, bits_per_sample = create_converter format in
    let block_align = num_channels * bits_per_sample / 8 in
    let byte_rate = sample_rate * block_align in
    let header =
      {num_channels; sample_rate; byte_rate; block_align; bits_per_sample}
    in
    {header; converter}

  let convert (t : t) (slice : float array) : Bytes.t = t.converter slice

  let get_header (t : t) (data_size : int) : Bytes.t =
    (* Note: these values are only for PCM *)
    let header = Bytes.create 44 in
    Bytes.blit_string "RIFF" 0 header 0 4 ;
    Bytes.set_int32_ne header 4 (Int32.of_int (data_size - 8)) ;
    Bytes.blit_string "WAVE" 0 header 8 4 ;
    Bytes.blit_string "fmt " 0 header 12 4 ;
    Bytes.set_int32_ne header 16 (Int32.of_int 16) ;
    Bytes.set_int16_ne header 20 1 ;
    Bytes.set_int16_ne header 22 t.header.num_channels ;
    Bytes.set_int32_ne header 24 (Int32.of_int t.header.sample_rate) ;
    Bytes.set_int32_ne header 28 (Int32.of_int t.header.byte_rate) ;
    Bytes.set_int16_ne header 32 t.header.block_align ;
    Bytes.set_int16_ne header 34 t.header.bits_per_sample ;
    Bytes.blit_string "data" 0 header 36 4 ;
    Bytes.set_int32_ne header 40 (Int32.of_int data_size) ;
    header
end

(* TODO: Return a first class module representing a writer *)
let get_writer (format : string) : bool =
  match format with "wav" -> true | _ -> false

let read_metadata (filename : string) (format : string) : Metadata.t =
  let open Avcodec in
  let format =
    match Av.Format.find_input_format format with
    | Some f ->
        f
    | None ->
        raise (Invalid_argument ("Could not find format: " ^ format))
  in
  let input = Av.open_input ~format filename in
  let _, _, icodec = Av.find_best_audio_stream input in
  let sr = Audio.get_sample_rate icodec in
  let channels = Audio.get_nb_channels icodec in
  let bit_rate = Audio.get_bit_rate icodec in
  let sample_width = Audio.get_bit_rate icodec / (channels * sr) in
  Av.close input ;
  Gc.full_major () ;
  Gc.full_major () ;
  Metadata.create ~name:filename channels sample_width sr bit_rate

let read_audio (filename : string) (format : string) : audio =
  let open Avcodec in
  let format =
    match Av.Format.find_input_format format with
    | Some f ->
        f
    | None ->
        raise (Invalid_argument ("Could not find format: " ^ format))
  in
  let input = Av.open_input ~format filename in
  let idx, istream, icodec = Av.find_best_audio_stream input in
  let out_sr = Audio.get_sample_rate icodec in
  let channels = Audio.get_channel_layout icodec in
  let nb_channels = Audio.get_nb_channels icodec in
  let options = [`Engine_soxr] in
  let rsp = FrameToS32Bytes.from_codec ~options icodec channels out_sr in
  let duration = Av.get_duration ~format:`Millisecond istream in
  let bit_rate = Audio.get_bit_rate icodec in
  let sample_width = Audio.get_bit_rate icodec / (nb_channels * out_sr) in
  let nsamples =
    Int64.to_float duration *. float_of_int out_sr *. Float.pow 10. (-3.)
    *. float_of_int nb_channels
  in
  let data = G.create Bigarray.Float64 [|int_of_float (nsamples *. 1.01)|] 0. in
  let rsamples = ref 0 in
  (* number of read samples during the process *)
  let rec f () =
    match Av.read_input ~audio_frame:[istream] input with
    | `Audio_frame (i, frame) when i = idx ->
        let bytes = FrameToS32Bytes.convert rsp frame in
        let length = Bytes.length bytes in
        for i = 0 to length / 4 do
          let offset = i * 4 in
          if offset + 4 <= length then (
            let value = Int32.to_float (Bytes.get_int32_ne bytes offset) in
            G.set data [|!rsamples|] value ;
            rsamples := !rsamples + 1 )
        done ;
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
  let data = G.resize data [|!rsamples|] in
  let meta =
    Metadata.create ~name:filename nb_channels sample_width out_sr bit_rate
  in
  create meta icodec data

let write_audio (a : audio) (filename : string) (format : string) : unit =
  let open Avcodec in
  let format =
    match Av.Format.guess_output_format ~short_name:format ~filename () with
    | Some f ->
        f
    | None ->
        raise (Invalid_argument ("Could not find format: " ^ format))
  in
  let ocodec = Av.Format.get_audio_codec_id format |> Audio.find_encoder in
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
  let compressed, frame_size =
    if frame_size = 0 then (false, 1024) else (true, frame_size)
  in
  let out_file = open_out_bin filename in
  let values = data a in
  let length = G.numel values in
  let rsp =
    FloatArrayToFrame.create in_cl ~in_sample_format in_sample_rate in_cl
      ~out_sample_format out_sample_rate
  in
  (* if we're writing out a PCM format, we reserve the first 44 bytes *)
  if not compressed then output_bytes out_file (Bytes.create 44) ;
  let channels = Metadata.channels (meta a) in
  let raw_writer = WavWriter.create out_sample_format channels in_sample_rate in
  let values = data a in
  let data_size = ref 44 in
  for i = 0 to length / frame_size do
    let start = i * frame_size in
    let finish = min (start + frame_size) length in
    let slice = values |> G.get_slice [[start; finish - 1]] |> G.to_array in
    try
      if compressed then
        let frame = FloatArrayToFrame.convert rsp slice in
        encode encoder (Packet.to_bytes %> output_bytes out_file) frame
      else
        let bytes = WavWriter.convert raw_writer slice in
        output_bytes out_file bytes ;
        data_size := !data_size + Bytes.length bytes
    with
    | Avutil.Error e ->
        Printf.eprintf "Error while encoding data: %s\n"
          (Avutil.string_of_error e) ;
        flush stderr ;
        Gc.full_major () ;
        Gc.full_major () ;
        exit 1
    | _ ->
        Printf.eprintf "An unknown error occured while encoding the file.\n" ;
        flush stderr ;
        Gc.full_major () ;
        Gc.full_major () ;
        exit 1
  done ;
  if not compressed then (
    let header = WavWriter.get_header raw_writer !data_size in
    seek_out out_file 0 ;
    output_bytes out_file header )
  else flush_encoder encoder (Packet.to_bytes %> output_bytes out_file) ;
  close_out out_file ;
  Gc.full_major () ;
  Gc.full_major ()
