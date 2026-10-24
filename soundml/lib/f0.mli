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

(** Fundamental frequency estimation: YIN and probabilistic YIN.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0 ([librosa.yin],
    [librosa.pyin]); parity is enforced against committed golden vectors in
    the test suite. *)

(** The boundary extension of centered frames. See [Soundml.yin]. *)
type pad = [`Constant of float | `Edge | `Reflect]

(** The result of {!pyin}. See [Soundml.pyin]. *)
type 'a pyin =
  { f0: (float, 'a) Nx.t
  ; voiced: (bool, Nx.bool_elt) Nx.t
  ; voiced_probability: (float, 'a) Nx.t }

val yin :
     ?frame_length:int
  -> ?hop:int
  -> ?trough_threshold:float
  -> ?alignment:[`Centered | `Left]
  -> ?pad:pad
  -> fmin:float
  -> fmax:float
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [yin ~fmin ~fmax ~sample_rate x] is the YIN estimate of the fundamental
    frequency of every frame of [x], shaped [[...; frames]]. See
    [Soundml.yin]. *)

val stage :
     ?frame_length:int
  -> ?hop:int
  -> ?trough_threshold:float
  -> ?alignment:[`Centered | `Left]
  -> ?pad:pad
  -> fmin:float
  -> fmax:float
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage ~fmin ~fmax ~sample_rate ()] is {!yin} as a pipeline stage. See
    [Soundml.yin_stage]. *)

val pyin :
     ?frame_length:int
  -> ?hop:int
  -> ?n_thresholds:int
  -> ?beta_parameters:float * float
  -> ?boltzmann_parameter:float
  -> ?resolution:float
  -> ?max_transition_rate:float
  -> ?switch_prob:float
  -> ?no_trough_prob:float
  -> ?fill:[`Value of float | `Decoded]
  -> ?alignment:[`Centered | `Left]
  -> ?pad:pad
  -> ?transition_min_prob:float
  -> fmin:float
  -> fmax:float
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> 'a pyin
(** [pyin ~fmin ~fmax ~sample_rate x] is the probabilistic YIN estimate of
    the fundamental frequency of [x] with its voicing. See [Soundml.pyin]. *)
