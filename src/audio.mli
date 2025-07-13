(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2025                                                       *)
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

(** This module is used to define the basic types used in
    SoundML. It defines the main modules that are used
    throughout the library. *)

module Metadata : sig
  type t =
    { name: string
    ; frames: int
    ; channels: int
    ; sample_rate: int
    ; format: Aformat.t }

  val create : ?name:string -> int -> int -> int -> Aformat.t -> t

  val name : t -> string

  val frames : t -> int

  val channels : t -> int

  val sample_rate : t -> int

  val format : t -> Aformat.t
end

(** The [Audio] module is the main module of the library. It
    contains all the functions to manipulate audio signals. *)
type 'a t = {meta: Metadata.t; data: (float, 'a) Nx.t}

val create : Metadata.t -> (float, 'a) Nx.t -> 'a t
(** Creates a new audio object from a given ndarray. *)

val get : int -> 'a t -> float

val sr : 'a t -> int
(** Returns the sample rate of the audio object. *)

val format : 'a t -> Aformat.t
(** Returns the format of the audio object. *)

val channels : 'a t -> int
(** Returns the number of channels of the audio object. *)

val duration : 'a t -> float
(** Returns the duration of the audio object in seconds. *)

val samples : 'a t -> int
(** Returns the number of samples of the audio object. *)

val set_data : (float, 'a) Nx.t -> 'a t -> 'a t
(** Sets the data of the audio object. *)
