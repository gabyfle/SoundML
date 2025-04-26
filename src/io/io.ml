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
open Bigarray

type metadata = int * int * int * int

external caml_read_audio_file_f32 :
  string -> int -> int -> (float, Bigarray.float32_elt) Audio.G.t * metadata
  = "caml_read_audio_file_f32"

external caml_read_audio_file_f64 :
     string
  -> int
  -> int
  -> (float, Bigarray.float64_elt) Audio.G.t * (int * int * int * int)
  = "caml_read_audio_file_f64"

let to_mono (x : (float, 'a) G.t) =
  if G.num_dims x > 1 then G.mean ~axis:0 ~keep_dims:false x else x

let read : type a.
       ?buffer_size:int
    -> ?sample_rate:int option
    -> ?mono:bool
    -> (float, a) kind
    -> string
    -> a audio =
 fun ?(buffer_size : int = 1024) ?sample_rate ?(mono : bool = true) typ
     (filename : string) ->
  let read_func : type a.
      (float, a) kind -> string -> int -> int -> (float, a) G.t * metadata =
   fun typ ->
    match typ with
    | Float32 ->
        caml_read_audio_file_f32
    | Float64 ->
        caml_read_audio_file_f64
    | _ ->
        failwith "unsupported"
  in
  let sample_rate =
    match sample_rate with
    | None ->
        22050
    | Some None ->
        -1
    | Some (Some rate) ->
        rate
  in
  let data, meta = read_func typ filename buffer_size sample_rate in
  let data = if mono then to_mono data else data in
  let frames, channels, sample_rate, format = meta in
  let channels = if mono then 1 else channels in
  let meta =
    Metadata.create ~name:filename frames channels sample_rate format
  in
  Audio.create meta data
