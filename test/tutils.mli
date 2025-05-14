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

(** Module providing usefull checking functions for the tests *)
module Check : sig
  val rallclose :
       ?rtol:float
    -> ?atol:float
    -> (float, 'b) Owl_dense_ndarray.Generic.t
    -> (float, 'b) Owl_dense_ndarray.Generic.t
    -> bool
  (** Real-valued all-close function *)

  val callclose :
    'a.
       ?rtol:float
    -> ?atol:float
    -> (Complex.t, 'a) Owl_dense_ndarray.Generic.t
    -> (Complex.t, 'a) Owl_dense_ndarray.Generic.t
    -> bool
  (** Complex-valued all-close function *)

  val shape :
       ('a, 'b) Owl_dense_ndarray.Generic.t
    -> ('a, 'b) Owl_dense_ndarray.Generic.t
    -> bool
  (** Check the shape of two ndarrays are equal *)
end

val allclose :
  'a 'b.
     ('a, 'b) Bigarray.kind
  -> ?rtol:float
  -> ?atol:float
  -> ('a, 'b) Owl_dense_ndarray.Generic.t
  -> ('a, 'b) Owl_dense_ndarray.Generic.t
  -> bool
(** Checks if two Ndarrays are allclose. This is equivalent to NumPy's allclose function. *)

val get_dense_testable :
     ('a, 'b) Bigarray.kind
  -> ('a, 'b) Owl_dense_ndarray.Generic.t Alcotest.testable
(** Function that returns a correctly-typed testable based on the passed kind for Dense.Ndarray. *)

val load_npy :
  string -> ('a, 'b) Bigarray.kind -> ('a, 'b) Owl_dense_ndarray.Generic.t
(** Load a numpy file and return the ndarray. 
    @see https://github.com/tachukao/owl/blob/046f703a6890a5ed5ecf4a8c5750d4e392e4ec54/src/owl/dense/owl_dense_matrix_generic.ml#L606-L609 *)
