(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023-2025                                                  *)
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
open Bigarray

exception File_not_found of string

exception Invalid_format of string

exception Resampling_error of string

exception Internal_error of string

let _ =
  Callback.register_exception "soundml.exn.file_not_found"
    (File_not_found "file.wav")

let _ =
  Callback.register_exception "soundml.exn.invalid_format"
    (Invalid_format "invalid format")

let _ =
  Callback.register_exception "soundml.exn.resampling_error"
    (Resampling_error "error")

let _ =
  Callback.register_exception "soundml.exn.internal_error"
    (Internal_error "internal error")

type resampling_t = NONE | SOXR_QQ | SOXR_LQ | SOXR_MQ | SOXR_HQ | SOXR_VHQ

(* nframes * channels * sample_rate * format *)
type metadata = int * int * int * int

external caml_read_audio_file_f32 :
     string
  -> resampling_t
  -> int
  -> (float, Bigarray.float32_elt) Audio.G.t * metadata
  = "caml_read_audio_file_f32"

external caml_read_audio_file_f64 :
     string
  -> resampling_t
  -> int
  -> (float, Bigarray.float64_elt) Audio.G.t * metadata
  = "caml_read_audio_file_f64"

let to_mono (x : (float, 'a) G.t) =
  if G.num_dims x > 1 then G.mean ~axis:1 ~keep_dims:false x else x

let read : type a.
       ?res_typ:resampling_t
    -> ?sample_rate:int
    -> ?mono:bool
    -> (float, a) kind
    -> string
    -> a audio =
 fun ?(res_typ : resampling_t = SOXR_HQ) ?(sample_rate : int = 22050)
     ?(mono : bool = true) typ (filename : string) ->
  let read_func : type a.
         (float, a) kind
      -> string
      -> resampling_t
      -> int
      -> (float, a) G.t * metadata =
   fun typ ->
    match typ with
    | Float32 ->
        caml_read_audio_file_f32
    | Float64 ->
        caml_read_audio_file_f64
    | Float16 ->
        raise
          (Invalid_argument
             "Float16 elements kind aren't supported. The array kind must be \
              either Float32 or Float64." )
  in
  let data, meta = read_func typ filename res_typ sample_rate in
  let dshape = Audio.G.shape data in
  let nsamples = dshape.(0) in
  let data = if mono then to_mono data else data in
  let frames, channels, sample_rate, format = meta in
  let data =
    match (res_typ, frames, nsamples) with
    | NONE, real, pred ->
        if real = pred then data else Audio.G.sub_left data 0 real
    | _ ->
        data
  in
  let channels = if mono then 1 else channels in
  let format =
    match Aformat.of_int format with
    | Ok fmt ->
        fmt
    | Error e ->
        raise (Invalid_format e)
  in
  let meta =
    Metadata.create ~name:filename frames channels sample_rate format
  in
  let data = Audio.G.transpose data in
  Audio.create meta data

external caml_write_audio_file_f32 :
     string
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> int * int * int * int
  -> unit = "caml_write_audio_file_f32"

external caml_write_audio_file_f64 :
     string
  -> (float, Bigarray.float64_elt) Audio.G.t
  -> int * int * int * int
  -> unit = "caml_write_audio_file_f64"

let write : type a.
    ?format:Aformat.t -> string -> (float, a) Audio.G.t -> int -> unit =
 fun ?format (filename : string) (x : (float, a) Audio.G.t) sample_rate ->
  let format =
    if format = None then
      match Aformat.of_ext (Filename.extension filename) with
      | Ok fmt ->
          fmt
      | Error e ->
          raise (Invalid_format e)
    else Option.get format
  in
  let format = Aformat.to_int format in
  let data = Audio.G.transpose x in
  let dshape = Audio.G.shape data in
  let nframes = dshape.(0) in
  let channels = if Array.length dshape > 1 then dshape.(1) else 1 in
  (* we get back our interleaved format *)
  match Audio.G.kind data with
  | Float32 ->
      caml_write_audio_file_f32 filename data
        (nframes, sample_rate, channels, format)
  | Float64 ->
      caml_write_audio_file_f64 filename data
        (nframes, sample_rate, channels, format)
  | _ ->
      raise
        (Invalid_argument
           "Float16 elements kind aren't supported. The array kind must be \
            either Float32 or Float64." )
