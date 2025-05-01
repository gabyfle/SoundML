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
   The {!Aformat} (audio format) module is an abstraction over the different supported audio format from libsndfile. *)

type ftype =
  | WAV
  | AIFF
  | AU
  | RAW
  | PAF
  | SVX
  | NIST
  | VOC
  | IRCAM
  | W64
  | MAT4
  | MAT5
  | PVF
  | XI
  | HTK
  | SDS
  | AVR
  | WAVEX
  | SD2
  | FLAC
  | CAF
  | WVE
  | OGG
  | MPC2K
  | RF64
  | MP3

type subtype =
  | PCM_S8
  | PCM_16
  | PCM_24
  | PCM_32
  | PCM_U8
  | FLOAT
  | DOUBLE
  | ULAW
  | ALAW
  | IMA_ADPCM
  | MS_ADPCM
  | GSM610
  | VOX_ADPCM
  | NMS_ADPCM_16
  | NMS_ADPCM_24
  | NMS_ADPCM_32
  | G721_32
  | G723_24
  | G723_40
  | DVW_12
  | DVW_16
  | DVW_24
  | DVW_N
  | DPCM_8
  | DPCM_16
  | VORBIS
  | OPUS
  | ALAC_16
  | ALAC_20
  | ALAC_24
  | ALAC_32
  | MPEG_LAYER_I
  | MPEG_LAYER_II
  | MPEG_LAYER_III

type endianness = FILE | LITTLE | BIG | CPU

(** The type for an audio format specification. *)
type t = {ftype: ftype; sub: subtype; endian: endianness}

val create :
  ?subtype:subtype -> ?endian:endianness -> ftype -> (t, string) result
(**
    [create ?subtype ?endian ftype] creates a new audio format representation based on the given format specifications.s

    {2 Parameters}
    @param subtype is the subtype of the audio file. If not specified, it'll be set to a default value according to the file type.
    @param endian is the endianness of the audio file. If not specified, it'll be set to [FILE], which is the default file endianness.
    @param ftype is the file type of the audio file.

    {2 Returns}

    @return A result type, where [Ok t] is the created format and [Error msg] is an error message indicating why it failed.

    {2 Usage}

    Creating a new audio format is as simple as calling the [create] function with the desired parameters.
    
    For the [RAW] file type, the subtype is required. Not specifying one will result in an error.

    {[ 
      open Soundml.Io
      (* This will create a new WAV audio format with PCM_16 subtype and little endian. *)
      let fmt = Afmt.create ~subtype:Io.Afmt.PCM_16 ~endian:Io.Afmt.LITTLE Io.Aformat.WAV in
    ]} *)

val to_int : t -> int
(**
    [to_int fmt] converts the audio format to an integer representation compatible with libsndfile.
    
    {2 Parameters}
    @param fmt the format that we need to convert to an integer value.
    
    {2 Returns}
    @return The integer representation of the audio format. *)

val of_int : int -> (t, string) result
(**
    [of_int code] converts the integer representation of the audio format to a {!Aformat.t} type.
    
    {2 Parameters}
    @param code is the integer representation of the audio format we're trying to convert.
    
    @return A result type, where [Ok t] is the created format and [Error msg] is an error message indicating why it failed. *)

val of_ext : ?sub:subtype -> ?endian:endianness -> string -> (t, string) result
(**
    [of_ext ?sub ?endian ext] tries to convert the given file extension to an audio format type.

    This function assumes that the extension is given with its leading dot (e.g. [".wav"]) and is thus compatible with the [Filename] module.
    
    {2 Parameters}
    @param sub is the subtype of the audio file. If not specified, it'll be set to a default value according to the file type.
    @param endian is the endianness of the audio file. If not specified, it'll be set to [FILE], which is the default file endianness.
    @param ext is the file extension we're trying to convert.
    
    {2 Returns}
    @return A result type, where [Ok t] is the created format and [Error msg] is an error message indicating why it failed. *)

val pp : Format.formatter -> t -> unit
(**
    [pp fmt] pretty prints the audio format to the given formatter.
    
    {2 Parameters}
    @param fmt is the formatter to use for printing the audio format. *)
