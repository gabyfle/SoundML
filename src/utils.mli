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
  val mel_to_hz : ?htk:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
  (** Converts mel-scale values to Hz. *)

  val hz_to_mel : ?htk:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
  (** Reverse function of {!mel_to_hz}. *)

  type reference =
    | RefFloat of float
    | RefFunction of ((float, Bigarray.float32_elt) Nx.t -> float)

  val power_to_db :
       ?amin:float
    -> ?top_db:float option
    -> reference
    -> (float, Bigarray.float32_elt) Nx.t
    -> (float, Bigarray.float32_elt) Nx.t

  val db_to_power :
       ?amin:float
    -> reference
    -> (float, Bigarray.float32_elt) Nx.t
    -> (float, Bigarray.float32_elt) Nx.t
end

val pad_center : ('a, 'b) Nx.t -> int -> 'a -> ('a, 'b) Nx.t
(**
    Pads a ndarray such that *)

val frame : ('a, 'b) Nx.t -> int -> int -> int -> ('a, 'b) Nx.t

val fftfreq : int -> float -> (float, Bigarray.float32_elt) Nx.t
(**
    Implementation of the Numpy's fftfreq function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html}numpy.fft.fftfreq} for more information. *)

val rfftfreq : (float, 'b) Nx.dtype -> int -> float -> (float, 'b) Nx.t
(**
    Implementation of the Numpy's rfftfreq function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html}numpy.fft.rfftfreq} for more information. *)

val melfreq :
     ?nmels:int
  -> ?fmin:float
  -> ?fmax:float
  -> ?htk:bool
  -> (float, 'b) Nx.dtype
  -> (float, 'b) Nx.t
(**
  Implementation of librosa's mel_frequencies. Compute an [Nx.t] of acoustic frequencies tuned to the mel scale.
  See: {{:https://librosa.org/doc/main/generated/librosa.mel_frequencies.html}librosa.mel_frequencies} for more information. *)

val unwrap :
     ?discont:float option
  -> ?axis:int
  -> ?period:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(**
    Implementation of the Numpy's unwrap function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html}numpy.unwrap} for more information. *)

val outer :
     (('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
  -> ('a, 'b) Nx.t
  -> ('a, 'b) Nx.t
  -> ('a, 'b) Nx.t
(**
  Generalized outer product of any given operator that supports broadcasting (basically all the common Nx operators.) *)
