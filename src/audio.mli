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

open Owl
module G = Dense.Ndarray.Generic

(**
    High level representation of an audio file data, used to store data when reading audio files *)
type audio

val create :
  name:string -> data:(float, Bigarray.float64_elt) G.t -> sampling:int -> audio
(**
    [create ~name ~data ~sampling] creates a new audio data element with the given name, data and sampling rate *)

val name : audio -> string
(**
    [name audio] returns the name (as it was read on the filesystem) of the given audio data element *)

val data : audio -> (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
(**
    [data audio] returns the data of the given audio data element *)

val size : audio -> int
(**
    [size audio] returns the size (in bytes) of the given audio data element *)

val sampling : audio -> int
(**
    [sampling audio] returns the sampling rate of the given audio data element *)
