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

(**
    {1 Reading data} *)

val read_metadata : string -> string -> Metadata.t
(**
    [read_metadata filename format] reads the metadata of an audio file and returns [Metadata.t] type.
    
    Example usage:
    
    {[
    let () =
        let src = Io.read_metadata file.wav wav in
        let open Audio in
        Printf.fprintf "Sample rate: %d\n" (Metadata.sample_rate src);
        (* ... *)
    ]} *)

val read_audio : string -> string -> audio
(**
    [read_audio filename format] reads an audio file returns a representation of the file.
    
    Example usage:
    
    {[
    let () =
        let src = Io.read_audio file.wav wav in
        (* ... *)
    ]}

    you can as well choose to have a stereo representation of the file

    {[
    let () =
        let src = Io.read_audio `Stereo file.wav wav in
        (* ... *)
    ]} *)

(**
    {1 Writing data} *)

val write_audio : audio -> string -> string -> unit
(**
    [write_audio audio filename format] writes an audio file from the given audio data element.
    
    Example usage:
    
    {[
    (* Converting an MP3 file into a WAV file *)
    let () =
        let src = Io.read_audio "file.mp3" "mp3" in
        Io.write_audio src "file.wav" "wav"
    ]} *)
