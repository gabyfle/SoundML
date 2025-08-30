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

let to_mono (audio_tensor : (float, 'a, 'dev) Rune.t) =
  try
    if Rune.ndim audio_tensor > 1 then
      let shape = Rune.shape audio_tensor in
      if Array.length shape < 2 then
        raise (Invalid_argument "Invalid tensor shape for mono conversion")
      else Rune.mean ~axes:[|1|] ~keepdims:false audio_tensor
    else audio_tensor
  with
  | Invalid_argument _ as e ->
      raise e
  | exn ->
      raise
        (Internal_error
           ("Unexpected error in mono conversion: " ^ Printexc.to_string exn) )

let read : type a.
       ?res_typ:resampling_t
    -> ?sample_rate:int
    -> ?mono:bool
    -> 'dev Rune.device
    -> (float, a) Rune.dtype
    -> string
    -> (float, a, 'dev) Rune.t * int =
 fun ?(res_typ : resampling_t = SOXR_HQ) ?(sample_rate : int = 22050)
     ?(mono : bool = true) (device : 'dev Rune.device) typ
     (filename : string) ->
  if sample_rate <= 0 then
    raise (Invalid_argument "Sample rate must be positive") ;
  if String.length filename = 0 then
    raise (Invalid_argument "Filename cannot be empty") ;
  let read_func : type a.
         (float, a) Rune.dtype
      -> string
      -> resampling_t
      -> int
      -> (float, a, Bigarray.c_layout) Bigarray.Genarray.t * int =
   fun typ ->
    match typ with
    | Rune.Float32 ->
        caml_read_audio_file_f32
    | Rune.Float64 ->
        caml_read_audio_file_f64
    | _ ->
        raise
          (Invalid_argument
             "Float16 elements kind aren't supported. The array kind must be \
              either Float32 or Float64." )
  in
  try
    let data, actual_sample_rate = read_func typ filename res_typ sample_rate in
    let data_shape = Bigarray.Genarray.dims data in
    if Array.length data_shape = 0 || Array.fold_left ( * ) 1 data_shape = 0
    then raise (Invalid_format "Audio file contains no data") ;
    let data = Rune.of_bigarray device data in
    (* Apply mono conversion if requested *)
    let data = if mono then to_mono data else data in
    (* Transpose if multi-dimensional (channels first to channels last) *)
    let data =
      if Rune.ndim data > 1 then
        try Rune.transpose data
        with exn ->
          raise (Internal_error ("Transpose failed: " ^ Printexc.to_string exn))
      else data
    in
    (data, actual_sample_rate)
  with
  | ( File_not_found _
    | Invalid_format _
    | Resampling_error _
    | Internal_error _
    | Invalid_argument _ ) as e ->
      raise e
  | Failure msg ->
      raise (Internal_error ("Internal error during read: " ^ msg))
  | exn ->
      raise
        (Internal_error
           ("Unexpected error during audio read: " ^ Printexc.to_string exn) )

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
    ?format:Aformat.t -> string -> (float, a, 'dev) Rune.t -> int -> unit =
 fun ?format (filename : string) (x : (float, a, 'dev) Rune.t) sample_rate ->
  if sample_rate <= 0 then
    raise (Invalid_argument "Sample rate must be positive") ;
  if String.length filename = 0 then
    raise (Invalid_argument "Filename cannot be empty") ;
  let write_func : type a.
         (float, a) Rune.dtype
      -> string
      -> (float, a, Bigarray.c_layout) Bigarray.Genarray.t
      -> int * int * int * int
      -> unit =
   fun typ ->
    match typ with
    | Rune.Float32 ->
        caml_write_audio_file_f32
    | Rune.Float64 ->
        caml_write_audio_file_f64
    | _ ->
        raise
          (Invalid_argument
             "Float16 elements kind aren't supported. The array kind must be \
              either Float32 or Float64." )
  in
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
  let data = Rune.transpose x in
  let dshape = Rune.shape data in
  let nframes = dshape.(0) in
  let channels = if Array.length dshape > 1 then dshape.(1) else 1 in
  try
    let dtype = Rune.dtype data in
    write_func dtype filename
      (Rune.unsafe_to_bigarray data)
      (nframes, sample_rate, channels, format)
  with
  | ( File_not_found _
    | Invalid_format _
    | Resampling_error _
    | Internal_error _
    | Invalid_argument _ ) as e ->
      raise e
  | Failure msg ->
      raise (Internal_error ("Internal error during write: " ^ msg))
  | exn ->
      raise
        (Internal_error
           ("Unexpected error during audio write: " ^ Printexc.to_string exn) )
