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

(**
    The {!Io} (in/out) module is the entry point for reading and writing audio
    data from and to the filesystem. *)

open Audio
open Bigarray

(**
    {1 Supported formats}
    
    {!Soundml} supports the following formats for reading and writing audio data.

    {2 Reading}
    
    - WAV
    - MP3
    - FLAC
    - OGG
    - AIFF
    - AU
    - RAW

    {2 Writing}

    - WAV
    - MP3 *)

type resampling_t = NONE | SOXR_QQ | SOXR_LQ | SOXR_MQ | SOXR_HQ | SOXR_VHQ
(*| SRC_SINC_BEST_QUALITY | SRC_SINC_MEDIUM_QUALITY | SRC_SINC_FASTEST |
  SRC_ZERO_ORDER_HOLD | SRC_LINEAR*)

(**
    {1 Reading data} *)

val read :
  'a.
     ?res_typ:resampling_t
  -> ?sample_rate:int
  -> ?mono:bool
  -> ?fix:bool
  -> (float, 'a) kind
  -> string
  -> 'a audio
(**
    [read ?sample_rate ?mono kind filename] reads an audio file and returns an [audio].
    open Soundml

    {3 Parameters}
    - [?sample_rate] is the target sample rate to use when reading the file. Default is 22050 Hz. If [None] is passed, the file's sample rate is used.
    - [?mono] is a boolean that indicates if we want to convert to a mono audio. Default is [true].
    - [kind] is the format of audio data to read. It can be either [Bigarray.Float32] or [Bigarray.Float64].
    - [filename] is the path to the file to read audio from.
    
    {3 Usage}
    Reading audio is straightfoward. Simply specify the path to the file you want to read.
    
    {[
      open Soundml
      (* This will read the file.wav audio into a Float32 bigarray *)
      let audio = Io.read Bigarray.Float32 "path/to/file.wav"
    ]}

    {3 Supported formats}

    SoundML relies on {{:https://libsndfile.github.io/libsndfile/}libsndfile} to read audio files. Full detail on the supported formats are available
    on the official sndfile's website: {{:https://libsndfile.github.io/libsndfile/formats.html}Supported formats}. *)
