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
    The {!Audio} module defines the needed types around the representation of 
    an audio file and more precisely an audio data. Most of the things here are
    used internally, but can still be usefull if you want, for example, to create
    your audio data directly from OCaml (instead of reading it from a file). *)

open Owl

(**
    Alias of the generic [Ndarray] datastructure from [Owl]. This is used
    internally to make the computations around the audio data *)
module G = Dense.Ndarray.Generic

(**
    {1 Audio Metadata}

    This module contains the metadata of an audio file, which is used to store
    information about the audio file when reading it from the filesystem.
    
    Note: {!Metadata} in {!Soundml} isn't the same thing as the metadata attached
    to audio files. In {!Soundml}, we refer to {!Metadata} all the data describing
    the audio file in itself (sample rate, number of channels, etc...). If you are
    interested in dealing with author name, label and other metadata, we recommend
    using the {{:https://github.com/savonet/ocaml-mm} ocaml-mm} library instead. *)

module Metadata : sig
  type t

  val create : ?name:string -> int -> int -> int -> int -> t
  (**
      [create ?name channels sample_width sample_rate bit_rate] creates a new metadata with the given parameters *)

  val name : t -> string
  (**
      [name meta] returns the name of the file represented by the metadata *)

  val channels : t -> int
  (**
      [channels meta] returns the number of channels of the audio file *)

  val sample_width : t -> int
  (**
      [sample_width meta] returns the sample width of the audio file *)

  val sample_rate : t -> int
  (**
      [sample_rate meta] returns the sample rate of the audio file *)

  val bit_rate : t -> int
  (**
      [bit_rate meta] returns the bit rate of the audio file *)
end

(**
    {1 Audio manipulation}

    Most of these functions are used internally, and you'll probably just use the {!Audio.normalize}
    function to normalize the audio data before writing it back to a file. *)

(**
    High level representation of an audio file data, used to store data when reading audio files. *)
type audio

val create :
     Metadata.t
  -> Avutil.audio Avcodec.params
  -> (float, Bigarray.float64_elt) G.t
  -> audio
(**
    [create metadata icodec data] creates a new audio with the given name and metadata *)

val meta : audio -> Metadata.t
(**
    [meta audio] returns the metadata attached to the given audio element *)

val rawsize : audio -> int
(**
    [rawsize audio] returns the raw size of the given audio element *)

val length : audio -> int
(**
    [length audio] returns the length (in milliseconds) of the given audio element *)

val data : audio -> (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
(**
    [data audio] returns the data of the given audio element *)

val set_data :
  audio -> (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t -> audio
(**
    [set_data audio data] sets the data of the given audio element *)

val codec : audio -> Avutil.audio Avcodec.params
(**
    [codec audio] returns the codec of the given audio element *)

val normalize : ?factor:float -> audio -> unit
(**
    [normalize ?factor audio] normalizes the data of the given audio data element by
    the maximum value of an int32.

    Use this function when you did not normalized you audio and you need to
    either plot the data or write it back to an audio file.
    
    If you forgot to normalize the data, you might get some values that goes
    beyond 1.0 or under -1.0, which will surely make the audio sound distorted.

    We recommend to always normalize your audio once you first read it from a source
    file.

    The operation is performed in place (impure function).

    Example:

    {[
        let audio = Audio.read "audio.wav" in
        (* you can perform any operation here *)
        (* ... *)
        Audio.normalize audio; (* normalizing before writing *)
        Audio.write audio "audio.wav"
    ]} *)
