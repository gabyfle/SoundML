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
   Utility conversion module. *)
module Convert : sig
  val mel_to_hz :
       ?htk:bool
    -> (float, 'a) Owl_dense_ndarray_generic.t
    -> (float, 'a) Owl_dense_ndarray_generic.t
  (** Converts mel-scale values to Hz. *)

  val hz_to_mel :
       ?htk:bool
    -> (float, 'a) Owl_dense_ndarray_generic.t
    -> (float, 'a) Owl_dense_ndarray_generic.t
  (** Reverse function of {!mel_to_hz}. *)
end

(**
    Various utility functions that are used inside the library and that can be usefull
    as well outside of it. *)

val fftfreq :
  int -> float -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(**
    Implementation of the Numpy's fftfreq function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html}numpy.fft.fftfreq} for more information. *)

val rfftfreq :
  int -> float -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(**
    Implementation of the Numpy's rfftfreq function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html}numpy.fft.rfftfreq} for more information. *)

val melfreq :
     ?nmels:int
  -> ?fmin:float
  -> ?fmax:float
  -> ?htk:bool
  -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(**
  Implementation of librosa's mel_frequencies. Compute an [Owl.Dense.Ndarray] of acoustic frequencies tuned to the mel scale.
  See: {{:https://librosa.org/doc/main/generated/librosa.mel_frequencies.html}librosa.mel_frequencies} for more information. *)

val roll :
     ('a, 'b) Owl.Dense.Ndarray.Generic.t
  -> int
  -> ('a, 'b) Owl.Dense.Ndarray.Generic.t
(**
    Implementation of the Numpy's roll function on the 0th axis of the given ndarray.
    This function is used to shift elements of an array inside the library and is exposed
    as it can be sometimes usefull.

    This function returns a copy of the given ndarray.

    See {{:https://numpy.org/doc/stable/reference/generated/numpy.roll.html}numpy.roll} for more information. *)

val cov : ?b:('a, 'b) Audio.G.t -> a:('a, 'b) Audio.G.t -> ('a, 'b) Audio.G.t
(**
    (re)Implementation of the matrix covariance function from Owl.
    
    Note: this is temporary and done only because Owl doesn't export any
    cov function for the [Ndarray] module on which [Audio.G] is based. This function is
    likely to be deleted when Owl library will export such a cov function for n-dimensional arrays. *)

val unwrap :
     ?discont:float option
  -> ?axis:int
  -> ?period:float
  -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
  -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(**
    Implementation of the Numpy's unwrap function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html}numpy.unwrap} for more information. *)

val outer :
     (   ('a, 'b) Owl.Dense.Ndarray.Generic.t
      -> ('a, 'b) Owl.Dense.Ndarray.Generic.t
      -> ('a, 'b) Owl.Dense.Ndarray.Generic.t )
  -> ('a, 'b) Owl.Dense.Ndarray.Generic.t
  -> ('a, 'b) Owl.Dense.Ndarray.Generic.t
  -> ('a, 'b) Owl.Dense.Ndarray.Generic.t
(**
  Generalized outer product of any given operator that supports broadcasting (basically all the common Owl's Ndarray operators.) *)
