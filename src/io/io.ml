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

external caml_read_audio_file_f32 :
     string
  -> resampling_t
  -> int
  -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t * int
  = "caml_read_audio_file_f32"

external caml_read_audio_file_f64 :
     string
  -> resampling_t
  -> int
  -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t * int
  = "caml_read_audio_file_f64"

let to_mono (x : (float, 'a) Nx.t) =
  if Nx.ndim x > 1 then Nx.mean ~axes:[|1|] ~keepdims:false x else x

let read : type a.
       ?res_typ:resampling_t
    -> ?sample_rate:int
    -> ?mono:bool
    -> (float, a) Nx.dtype
    -> string
    -> (float, a) Nx.t * int =
 fun ?(res_typ : resampling_t = SOXR_HQ) ?(sample_rate : int = 22050)
     ?(mono : bool = true) typ (filename : string) ->
  let read_func : type a.
         (float, a) Nx.dtype
      -> string
      -> resampling_t
      -> int
      -> (float, a, Bigarray.c_layout) Bigarray.Genarray.t * int =
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
  let data, sample_rate = read_func typ filename res_typ sample_rate in
  let data = Nx.of_bigarray data in
  let data = if mono then to_mono data else data in
  let data = if Nx.ndim data > 1 then Nx.transpose data else data in
  (data, sample_rate)

external caml_write_audio_file_f32 :
     string
  -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Genarray.t
  -> int * int * int * int
  -> unit = "caml_write_audio_file_f32"

external caml_write_audio_file_f64 :
     string
  -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Genarray.t
  -> int * int * int * int
  -> unit = "caml_write_audio_file_f64"

let write : type a.
    ?format:Aformat.t -> string -> (float, a) Nx.t -> int -> unit =
 fun ?format (filename : string) (x : (float, a) Nx.t) sample_rate ->
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
  let data = Nx.transpose x in
  let dshape = Nx.shape data in
  let nframes = dshape.(0) in
  let channels = if Array.length dshape > 1 then dshape.(1) else 1 in
  (* we get back our interleaved format *)
  match Nx.dtype data with
  | Float32 ->
      caml_write_audio_file_f32 filename (Nx.to_bigarray data)
        (nframes, sample_rate, channels, format)
  | Float64 ->
      caml_write_audio_file_f64 filename (Nx.to_bigarray data)
        (nframes, sample_rate, channels, format)
  | _ ->
      raise
        (Invalid_argument
           "Float16 elements kind aren't supported. The array kind must be \
            either Float32 or Float64." )
