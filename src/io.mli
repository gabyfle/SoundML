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

val read_audio :
  ?channels:Avutil.Channel_layout.t -> ?sr:int -> string -> string -> audio
(**
    [read_audio ?channels ?sr filename format] reads an audio file returns a representation of the file.

    By default, [channels] is set to `Mono and [sr] is set to 44100.
    
    Example usage:
    
    {[
    let () =
        let src = read_audio file.wav wav in
        (* ... *)
    ]}

    you can as well choose to have a stereo representation of the file

    {[
    let () =
        let src = read_audio `Stereo file.wav wav in
        (* ... *)
    ]} *)

val write_audio : ?sr:int option -> audio -> string -> string -> unit
(**
    [write_audio ?sr audio filename format] writes an audio file from the given audio data element.
    
    Example usage:
    
    {[
    let () =
        let src = read_audio "file.wav" "wav" in
        write_audio src "file.wav" "aac"
    ]} *)
