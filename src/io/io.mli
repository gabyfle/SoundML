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

(**
    The {!Io} (in/out) module is the entry point for reading and writing audio
    data from and to the filesystem. It supports resampling via the {{:https://github.com/chirlu/soxr}SoXr} library. *)

(** Thrown when a requested file cannot be found on the system. *)
exception File_not_found of string

(** Thrown when the file we're trying to read is encoded in an invalid format, or when the format we're trying to write isn't supported. *)
exception Invalid_format of string

(** Thrown when an error occurred while resampling. *)
exception Resampling_error of string

(** Thrown when an internal error occurred. This is should not happen, so please report it. *)
exception Internal_error of string

(** The resampling method to use. The default is [SOXR_HQ]. *)
type resampling_t =
  | NONE  (** Indicates that no resampling is requested *)
  | SOXR_QQ  (** 'Quick' cubic interpolation. *)
  | SOXR_LQ  (** 'Low' 16-bit with larger rolloff. *)
  | SOXR_MQ  (** 'Medium' 16-bit with medium rolloff. *)
  | SOXR_HQ  (** 'High quality'. *)
  | SOXR_VHQ  (** 'Very high quality'. *)

val read :
  'a.
     ?res_typ:resampling_t
  -> ?sample_rate:int
  -> ?mono:bool
  -> (float, 'a) Nx.dtype
  -> string
  -> 'a Audio.t
(**
    [read ?res_typ ?sample_rate ?fix kind filename] reads an audio file and returns an [audio].

    @return an [audio] type that contains the audio data read from the file. The type of the audio's data is determined by the [kind] parameter.

    {2 Parameters}
    @param ?res_typ is the resampling method to use. The default is [SOXR_HQ]. If [NONE] is used, [?sample_rate] is ignored and no resampling will be done.
    @param ?sample_rate is the target sample rate to use when reading the file. Default is 22050 Hz.
    @param ?mono is a boolean that indicates if we want to convert to a mono audio. Default is [true].
    @param dtype is the format of audio data to read. It can be either [Float32] or [Float64].
    @param filename is the path to the file to read audio from.

    @raise File_not_found If the file does not exist.
    @raise Invalid_format If the file is not a valid audio file.
    @raise Resampling_error If the resampling fails.
    @raise Internal_error If an internal error occurs.
    
    {2 Usage}
    Reading audio is straightfoward. Simply specify the path to the file you want to read.
    
    {[
      open Soundml
      (* This will read the file.wav audio into a Float32 bigarray, resampled using SOXR_HQ at 22050Hz. *)
      let audio = Io.read Bigarray.Float32 "path/to/file.wav"
    ]}

    {2 Supported formats}

    SoundML relies on {{:https://libsndfile.github.io/libsndfile/}libsndfile} to read audio files. Full detail on the supported formats are available
    on the official sndfile's website: {{:https://libsndfile.github.io/libsndfile/formats.html}Supported formats} and in the {!Audio.Aformat} module. *)

val write : 'a. ?format:Aformat.t -> string -> (float, 'a) Nx.t -> int -> unit
(**
    [write ?format filename data sample_reat] writes an audio file to the filesystem.

    {2 Parameters}
    @param ?format is the format to use when writing the file. If not specified, the format is determined by the file extension by {!Aformat.of_ext}.
    @param filename is the path to the file to write audio to.
    @param data is the audio data to write. It can be either a [Bigarray.Float32] or [Bigarray.Float64].
    @param sample_rate is the sample rate of the audio data.


    @raise Invalid_format If the file is not a valid audio file.
    @raise Internal_error If an internal error occurs.


    {2 Usage}
    Writing audio is as straightfoward as reading it. Simply specify the path to the file you want to write.
    
    {[
      open Soundml
      open Audio
      let audio = Io.read Bigarray.Float32 "path/to/file.mp3" in
      Io.write "path/to/file.wav" (data audio) 22050 (* we'll automatically detect that you want to write to the WAV format *)
    ]}

    {2 Supported formats}

    SoundML relies on {{:https://libsndfile.github.io/libsndfile/}libsndfile} to read audio files. Full detail on the supported formats are available
    on the official sndfile's website: {{:https://libsndfile.github.io/libsndfile/formats.html}Supported formats} and in the {!Audio.Aformat} module. *)
