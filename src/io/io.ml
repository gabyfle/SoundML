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

(* nframes * channels * sample_rate * padded_frames * format *)
type metadata = int * int * int * int * int

type resampling_t = NONE | SOXR_QQ | SOXR_LQ | SOXR_MQ | SOXR_HQ | SOXR_VHQ

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
    -> ?fix:bool
    -> (float, a) kind
    -> string
    -> a audio =
 fun ?(res_typ : resampling_t = SOXR_VHQ) ?sample_rate ?(mono : bool = true)
     ?(fix : bool = true) typ (filename : string) ->
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
    | _ ->
        failwith "Unsupported datatype."
  in
  let sample_rate =
    match sample_rate with None -> 22050 | Some rate -> rate
  in
  let data, meta = read_func typ filename res_typ sample_rate in
  let dshape = Audio.G.shape data in
  let nsamples = if Array.length dshape > 1 then dshape.(1) else dshape.(0) in
  let data = if mono then to_mono data else data in
  let frames, channels, sample_rate, padded_frames, format = meta in
  let data =
    match (res_typ, frames, nsamples) with
    | NONE, real, pred ->
        if real = pred then data else Audio.G.sub_left data 0 real
    | _, real, pred ->
        if fix then
          if nsamples <> padded_frames then
            Audio.G.sub_left data 0 padded_frames
          else data
        else if real = pred then data
        else Audio.G.sub_left data 0 real
  in
  let channels = if mono then 1 else channels in
  let meta =
    Metadata.create ~name:filename frames channels sample_rate format
  in
  Audio.create meta data
