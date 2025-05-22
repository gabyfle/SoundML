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

module type S = sig
  type t

  type params

  val reset : t -> t

  val create : params -> t

  val process_sample : t -> float -> float
end

module Make : functor (S : S) -> sig
  type t = S.t

  type params = S.params

  val reset : S.t -> S.t

  val create : S.params -> S.t

  val process_sample : S.t -> float -> float

  val process :
       S.t
    -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
    -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
end

module IIR : sig
  module Generic : sig
    type t = Iir.t

    type params = Iir.params

    val reset : t -> t

    val create : params -> t

    val process_sample : t -> float -> float

    val process :
         t
      -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
      -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
  end

  module HighPass : sig
    type t = Highpass.t

    type params = Highpass.params

    val reset : t -> t

    val create : params -> t

    val process_sample : t -> float -> float

    val process :
         t
      -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
      -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
  end

  module LowPass : sig
    type t = Lowpass.t

    type params = Lowpass.params

    val reset : t -> t

    val create : params -> t

    val process_sample : t -> float -> float

    val process :
         t
      -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
      -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
  end
end
