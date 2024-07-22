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
    Various utility functions that are used inside the library and that can be usefull
    as well outside of it. *)

val fftfreq :
  int -> float -> (float, Bigarray.float32_elt) Owl_dense_ndarray_generic.t
(**
    Implementation of the Numpy's fftfreq function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html}numpy.fft.fftfreq} for more information. *)

val roll :
     (float, Bigarray.float32_elt) Owl_dense_ndarray_generic.t
  -> int
  -> (float, Bigarray.float32_elt) Owl_dense_ndarray_generic.t
(**
    Implementation of the Numpy's roll function for 1D arrays.
    This function is used to shift elements of an array inside the library (as we deal only with 1D arrays) and is exposed
    as it can be sometimes usefull.

    This function returns a copy of the given ndarray.

    See {{:https://numpy.org/doc/stable/reference/generated/numpy.roll.html}numpy.roll} for more information. *)
