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

(* decoding frames *)
module FrameToS32Bytes =
  Swresample.Make (Swresample.Frame) (Swresample.S32Bytes)

(* encoding arrays *)
module FloatArrayToFrame =
  Swresample.Make (Swresample.FloatArray) (Swresample.Frame)

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
  Metadata.create ~name:filename channels sample_width sr bit_rate

let read (filename : string) (format : string) : audio =
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
  let bit_depth = bit_rate / (out_sr * nb_channels) in
  (* number of samples in the audio file *)
  let nsamples =
    Int64.to_float duration *. float_of_int out_sr *. Float.pow 10. (-3.)
    *. float_of_int nb_channels
  in
  (* we're a bit over-evaluating the size of the number of samples to alloc
     enought memory just before starting the reading process *)
  let data = G.create Bigarray.Float32 [|int_of_float (nsamples *. 1.01)|] 0. in
  (* number of read samples during the process *)
  let rsamples = ref 0 in
  (* each recursive call decodes a single frame *)
  let rec decode_frames () : unit =
    match Av.read_input ~audio_frame:[istream] input with
    | `Audio_frame (i, frame) when i = idx ->
        let bytes = FrameToS32Bytes.convert rsp frame in
        let length = Bytes.length bytes in
        for i = 0 to length / 4 do
          let offset = i * 4 in
          if offset + 4 <= length then (
            let value = Int32.to_float (Bytes.get_int32_ne bytes offset) in
            G.set data [|!rsamples|] value ;
            incr rsamples )
        done ;
        decode_frames ()
    | exception Avutil.Error `Eof ->
        ()
    | _ ->
        decode_frames ()
  in
  decode_frames () ;
  Av.get_input istream |> Av.close ;
  Gc.full_major () ;
  let data = G.resize data [|!rsamples|] in
  if bit_depth != 3 then
    G.div_scalar_ ~out:data data (Float.pow 2. (float_of_int bit_depth))
  else
    (* Here, the data has been converted from 24-bits to 32-bits and is from a
       bit depth of 3. *)
    (*G.div_scalar_ ~out:data data (Float.pow 2. 8.) ;*)
    () ;
  let meta =
    Metadata.create ~name:filename nb_channels sample_width out_sr bit_rate
  in
  create meta icodec data

module type Writer = sig
  type t

  val header_size : int

  val create :
       Avutil.Channel_layout.t
    -> int
    -> Avutil.rational
    -> Avutil.Sample_format.t * Avutil.Sample_format.t
    -> int * int
    -> Avcodec.encode Avcodec.Audio.t
    -> t

  val convert : t -> float array -> Bytes.t

  val get_header : t -> Bytes.t

  val frame_size : t -> int

  val flush : t -> bytes
end

(* Generic writer to handle the formats handled by FFMPEG *)
module GenericWriter : Writer = struct
  type t =
    { encoder: Avutil.audio Avcodec.encoder
    ; codec: Avcodec.encode Avcodec.Audio.t
    ; converter: float array -> Bytes.t }

  (* everything is handled by ffmpeg *)
  let header_size = 0

  let create channel_layout channels tb (in_sf, out_sf) (in_sr, out_sr) ocodec =
    let open Avcodec in
    let encoder =
      Audio.create_encoder ~channel_layout ~channels ~time_base:tb
        ~sample_format:out_sf ~sample_rate:out_sr ocodec
    in
    let rsp =
      FloatArrayToFrame.create channel_layout ~in_sample_format:in_sf in_sr
        channel_layout ~out_sample_format:out_sf out_sr
    in
    let converter (slice : float array) : Bytes.t =
      let buf = Buffer.create 0 in
      let frame = FloatArrayToFrame.convert rsp slice in
      let write_buff (packet : 'media Packet.t) : unit =
        let bytes = Packet.to_bytes packet in
        Buffer.add_bytes buf bytes
      in
      encode encoder write_buff frame ;
      Buffer.to_bytes buf
    in
    {encoder; codec= ocodec; converter}

  let convert (t : t) (slice : float array) : Bytes.t = t.converter slice

  (* everything is handled by ffmpeg *)
  let get_header _ = Bytes.empty

  let frame_size (w : t) =
    if List.mem `Variable_frame_size (Avcodec.capabilities w.codec) then 512
    else Avcodec.Audio.frame_size w.encoder

  let flush (t : t) =
    let buf = Buffer.create 0 in
    let write_buff (packet : 'media Avcodec.Packet.t) : unit =
      let bytes = Avcodec.Packet.to_bytes packet in
      Buffer.add_bytes buf bytes
    in
    Avcodec.flush_encoder t.encoder write_buff ;
    Buffer.to_bytes buf
end

(* These Writer modules are private since the goal of the library isn't to deal
   with audio input and output but rather compute analytics on the audio data *)
module WavWriter : Writer = struct
  (* see https://docs.fileformat.com/audio/wav/ *)
  (* see http://soundfile.sapp.org/doc/WaveFormat/ *)
  type header =
    { channels: int
    ; sample_rate: int
    ; byte_rate: int
    ; block_align: int
    ; bits_per_sample: int
    ; mutable data_size: int }

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
          let data = int_of_float data in
          let bytes = Bytes.create 2 in
          Bytes.set_int16_ne bytes 0 data ;
          bytes
        in
        (func to_bytes, 16)
    | `S32 | `S32p ->
        let to_bytes (data : float) =
          let data = Int32.of_float data in
          let bytes = Bytes.create 4 in
          Bytes.set_int32_ne bytes 0 data ;
          bytes
        in
        (func to_bytes, 32)
    | `S64 | `S64p ->
        let to_bytes (data : float) =
          let data = Int64.of_float data in
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

  let create _ channels _ (_, format) (_, sample_rate) _ =
    let converter, bits_per_sample = create_converter format in
    let block_align = channels * bits_per_sample / 8 in
    let byte_rate = sample_rate * block_align in
    let header =
      { channels
      ; sample_rate
      ; byte_rate
      ; block_align
      ; bits_per_sample
      ; data_size= 0 }
    in
    {header; converter}

  let convert (t : t) (slice : float array) : Bytes.t =
    let bytes = t.converter slice in
    t.header.data_size <- t.header.data_size + Bytes.length bytes ;
    bytes

  let get_header (w : t) : Bytes.t =
    (* Note: these values are only for PCM *)
    let header = Bytes.create 44 in
    Bytes.blit_string "RIFF" 0 header 0 4 ;
    Bytes.set_int32_ne header 4 (Int32.of_int (w.header.data_size + 36)) ;
    Bytes.blit_string "WAVE" 0 header 8 4 ;
    Bytes.blit_string "fmt " 0 header 12 4 ;
    Bytes.set_int32_ne header 16 (Int32.of_int 16) ;
    Bytes.set_int16_ne header 20 1 ;
    Bytes.set_int16_ne header 22 w.header.channels ;
    Bytes.set_int32_ne header 24 (Int32.of_int w.header.sample_rate) ;
    Bytes.set_int32_ne header 28 (Int32.of_int w.header.byte_rate) ;
    Bytes.set_int16_ne header 32 w.header.block_align ;
    Bytes.set_int16_ne header 34 w.header.bits_per_sample ;
    Bytes.blit_string "data" 0 header 36 4 ;
    Bytes.set_int32_ne header 40 (Int32.of_int (w.header.data_size + 44)) ;
    header

  let frame_size _ = 512

  let flush _ = Bytes.empty
end

let get_writer (format : string) : (module Writer) =
  match format with
  | "wav" ->
      (module WavWriter : Writer)
  | "aiff" ->
      raise (Invalid_argument "AIFF format is not supported yet.")
  | _ ->
      (module GenericWriter : Writer)

let write (a : audio) (filename : string) (ext : string) : unit =
  let open Avcodec in
  let format =
    match Av.Format.guess_output_format ~short_name:ext ~filename () with
    | Some f ->
        f
    | None ->
        raise (Invalid_argument ("Could not find format: " ^ ext))
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
  let module W = (val get_writer ext) in
  let writer =
    W.create in_cl channels time_base
      (in_sample_format, out_sample_format)
      (in_sample_rate, out_sample_rate)
      ocodec
  in
  let frame_size = W.frame_size writer in
  let out_file = open_out_bin filename in
  let values = data a in
  let length = Array.get (G.shape values) 0 in
  (* we're going first to prepare the header size inside the file *)
  output_bytes out_file (Bytes.create W.header_size) ;
  for i = 0 to length / frame_size do
    let start = i * frame_size in
    if start < length then (
      let finish = min (start + frame_size) (length - 1) in
      let slice = values |> G.get_slice [[start; finish]] |> G.to_array in
      try W.convert writer slice |> output_bytes out_file with
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
          exit 1 )
  done ;
  (* flushing the data *)
  W.flush writer |> output_bytes out_file ;
  (* writing the header *)
  let header = W.get_header writer in
  seek_out out_file 0 ;
  output_bytes out_file header ;
  close_out out_file ;
  Gc.full_major () ;
  Gc.full_major ()
