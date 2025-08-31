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

(** Module providing usefull checking functions for the tests
*)
module Check : sig
  val rallclose :
       ?rtol:float
    -> ?atol:float
    -> (float, 'b, 'dev) Rune.t
    -> (float, 'b, 'dev) Rune.t
    -> bool
  (** Real-valued all-close function *)

  val callclose :
    'a 'dev.
       ?rtol:float
    -> ?atol:float
    -> (Complex.t, 'a, 'dev) Rune.t
    -> (Complex.t, 'a, 'dev) Rune.t
    -> bool
  (** Complex-valued all-close function *)

  val shape : ('a, 'b, 'dev) Rune.t -> ('a, 'b, 'dev) Rune.t -> bool
  (** Check the shape of two ndarrays are equal *)
end

val allclose :
  'a 'b 'dev.
     ?rtol:float
  -> ?atol:float
  -> ('a, 'b, 'dev) Rune.t
  -> ('a, 'b, 'dev) Rune.t
  -> bool
(** Checks if two Rune tensors are allclose. This is equivalent to
    NumPy's allclose function. *)

val tensor_testable :
  'a 'b 'dev.
     ('a, 'b) Rune.dtype
  -> rtol:float
  -> atol:float
  -> ('a, 'b, 'dev) Rune.t Alcotest.testable
(** An Alcotest.testable for Rune tensors. *)

val device : [`c] Rune.device
(** Device used to run the tests *)
