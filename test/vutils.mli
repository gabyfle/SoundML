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

val test_audio_dir : string
(** The directory where the test audio files are located. *)

val test_vectors_dir : string
(** The directory where the test vectors are located. *)

val typ_to_readable : string -> string
(** Converts a test type to a readable format for Alcotest. *)

(** A map from strings to values. *)
module StrMap : Map.S with type key = string

(** A module for handling parameters. *)
module Parameters : sig
  type t

  val create : string -> t

  val get_string : string -> t -> string option

  val get_int : string -> t -> int option

  val get_float : string -> t -> float option

  val get_bool : string -> t -> bool option
end

module Testdata : sig
  type t = (string * string * Parameters.t) list StrMap.t

  val get_test_type : string -> string option
  (** Returns the test type for a given test name *)

  val get_test_filename : string -> string option
  (** Returns the test filename for a given test type *)

  val create : string -> string -> string list -> t
  (** Creates a test set given a directory of vectors files, a directory of audio files
      and a list of test types *)

  val get : string -> t -> (string * string * Parameters.t) list
  (** Returns the test set for a given test type *)
end

module type Testable = sig
  val typ : string

  val create_test_set :
       (string * string * Parameters.t) list
    -> (string * [> `Slow] * (unit -> unit)) list
end
