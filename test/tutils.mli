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

  val get_string : string -> t -> string

  val get_int : string -> t -> int

  val get_float : string -> t -> float

  val get_bool : string -> t -> bool
end

module Testdata : sig
  type t = (string * string * Parameters.t) list StrMap.t

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

(** Module providing usefull checking functions for the tests *)
module Check : sig
  val eps : float
  (** The epsilon value used for floating point comparisons. *)

  val rallclose :
       (float, 'b) Owl_dense_ndarray.Generic.t
    -> (float, 'b) Owl_dense_ndarray.Generic.t
    -> bool
  (** Real-valued all-close function *)

  val callclose :
       (Complex.t, 'b) Owl_dense_ndarray.Generic.t
    -> (Complex.t, 'b) Owl_dense_ndarray.Generic.t
    -> bool
  (** Complex-valued all-close function *)

  val shape :
       ('a, 'b) Owl_dense_ndarray.Generic.t
    -> ('a, 'b) Owl_dense_ndarray.Generic.t
    -> bool
  (** Check the shape of two ndarrays are equal *)
end

val load_npy :
  string -> ('a, 'b) Bigarray.kind -> ('a, 'b) Owl_dense_ndarray.Generic.t
(** Load a numpy file and return the ndarray. 
    @see https://github.com/tachukao/owl/blob/046f703a6890a5ed5ecf4a8c5750d4e392e4ec54/src/owl/dense/owl_dense_matrix_generic.ml#L606-L609 *)
