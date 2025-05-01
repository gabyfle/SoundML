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
    an audio file and more precisely an audio data. *)

open Owl

(**
    Alias of the generic [Ndarray] datastructure from [Owl]. This is used
    internally to make the computations around the audio data *)
module G = Dense.Ndarray.Generic

(** @canonical Audio.Aformat
    Abstraction over the different supported audio format from libsndfile. *)
module Aformat = Aformat

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

  val create : ?name:string -> int -> int -> int -> Aformat.t -> t
  (**
      [create ?name channels frames sample_rate format] creates a new metadata
      with the given name, number of channels, number of frames, sample rate
      and format. The name is optional and defaults to [""] *)

  val name : t -> string
  (**
      [name meta] returns the name of the file represented by the metadata *)

  val frames : t -> int

  val channels : t -> int
  (**
      [channels meta] returns the number of channels of the audio file *)

  val sample_rate : t -> int
  (**
      [sample_rate meta] returns the sample rate of the audio file *)

  val format : t -> Aformat.t
  (**
        [format meta] returns the format of the audio file *)
end

(**
    {1 Audio manipulation}

    Most of these functions are used internally, and you'll probably just use the {!Audio.normalize}
    function to normalize the audio data before writing it back to a file. *)

(**
    High level representation of an audio file data, used to store data when reading audio files. *)
type 'a audio

val create : Metadata.t -> (float, 'a) G.t -> 'a audio
(**
    [create metadata data] creates a new audio with the given name and metadata *)

val meta : 'a audio -> Metadata.t
(**
    [meta audio] returns the metadata attached to the given audio *)

val rawsize : 'a audio -> int
(**
    [rawsize audio] returns the raw size of the given audio *)

val length : 'a audio -> int
(**
    [length audio] returns the length (in milliseconds) of the given audio *)

val data : 'a audio -> (float, 'a) Owl.Dense.Ndarray.Generic.t
(**
    [data audio] returns the data of the given audio *)

val sr : 'a audio -> int
(**
    [sr audio] returns the sample rate of the given audio *)

val channels : 'a audio -> int
(**
    [channels audio] returns the number of channels of the given audio *)

val samples : 'a audio -> int
(**
    [samples audio] returns the number of samples per channel in the given audio *)

val format : 'a audio -> Aformat.t
(**
    [format audio] returns the format of the given audio *)

val set_data : 'a audio -> (float, 'a) Owl.Dense.Ndarray.Generic.t -> 'a audio
(**
    [set_data audio data] sets the data of the given audio *)

val get : int -> 'a audio -> float
(**
    [get x audio] returns the sample located at the position [x] in milliseconds.

    The position [x] must be between 0 and the length of the audio.
    
    Example:

    {[
        let audio = Audio.read "audio.wav" in
        let sample = Audio.get 1000 audio in (* get the sample at 1 second *)
    ]} *)

val get_slice : int * int -> 'a audio -> 'a audio
(**
    [get_slice (start, stop) audio] returns a slice of the audio from the position [start] to [stop].

    The position [start] and [stop] must be between 0 and the length of the audio.

    This function works like Owl's slicing. Giving negative values to [start] and [stop] will slice the audio
    from the end of the audio .

    Example:

    {[
        let audio = Audio.read "audio.wav" in
        let slice = Audio.get_slice audio 1000 2000 in (* get the slice from 1 to 2 seconds *)
    ]} *)

val normalize : ?factor:float -> 'a audio -> unit
(**
    [normalize ?factor audio] normalizes the data of the given audio data by
    the [?factor] parameter, by default equal to $2^31 - 1$.

    Use this function when you need to normalize the audio data by a certain factor.
    
    Warning: if you normalize the data and end up getting values that goes
    beyond 1.0 or under -1.0, it will surely make the audio sound distorted.

    The operation is performed in place (impure function).

    Example:

    {[
        let audio = Audio.read "audio.wav" in
        (* you can perform any operation here *)
        (* ... *)
        let factor = (* ... *) in
        Audio.normalize ?factor audio; (* normalizing before writing *)
        Audio.write audio "audio.wav"
    ]} *)

val reverse : 'a audio -> 'a audio
(**
    [reverse audio] reverses the audio data.
    This function does not operate in place: a new audio is created with the reversed data.

    Example:

    {[
        let audio = Audio.read "audio.wav" in
        let audio = Audio.reverse audio in
        Audio.write audio "reversed.wav"
    ]} *)

(**
    {2 Operators on audio data}

    Following the Owl's conventions, few operators are available to deal with
    audio data. You can use them to make the code more concise and more readable.
    They are just syntaxic sugar on functions over the {!Audio.audio} type. *)

val ( .%{} ) : 'a audio -> int -> float
(** Operator of {!Audio.get} *)

val ( .${} ) : 'a audio -> int * int -> 'a audio
(** Operator of {!Audio.get_slice} *)

val ( $/ ) : 'a audio -> float -> unit
(** Operator of {!Audio.normalize} *)

val ( /$ ) : float -> 'a audio -> unit
(** Operator of {!Audio.normalize} *)
