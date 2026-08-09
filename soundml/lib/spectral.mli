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

(** Spectral-shape features over magnitude spectrograms.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 0.11
    ([librosa.feature.spectral_centroid] and friends); parity is enforced
    against committed golden vectors in the test suite. *)

val centroid :
     ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [centroid ~sample_rate s] is the spectral centroid of the magnitude
    spectrogram [s], shaped [[...; 1; frames]]. See
    [Soundml.spectral_centroid]. *)

val bandwidth :
     ?p:float
  -> ?freqs:(float, 'a) Nx.t
  -> ?centroid:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [bandwidth ~sample_rate s] is the [p]-th-order spectral bandwidth of the
    magnitude spectrogram [s], shaped [[...; 1; frames]]. See
    [Soundml.spectral_bandwidth]. *)

val rolloff :
     ?roll_percent:float
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [rolloff ~sample_rate s] is the roll-off frequency of each frame of the
    magnitude spectrogram [s], shaped [[...; 1; frames]]. See
    [Soundml.spectral_rolloff]. *)

val flatness :
  ?amin:float -> ?power:float -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [flatness s] is the spectral flatness of the magnitude spectrogram [s],
    shaped [[...; 1; frames]]. See [Soundml.spectral_flatness]. *)

val centroid_stage :
     ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [centroid_stage ~sample_rate ()] is {!centroid} as a memoryless
    {!Pipeline} stage. See [Soundml.spectral_centroid_stage]. *)

val bandwidth_stage :
     ?p:float
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [bandwidth_stage ~sample_rate ()] is {!bandwidth} as a memoryless
    {!Pipeline} stage. See [Soundml.spectral_bandwidth_stage]. *)

val rolloff_stage :
     ?roll_percent:float
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [rolloff_stage ~sample_rate ()] is {!rolloff} as a memoryless {!Pipeline}
    stage. See [Soundml.spectral_rolloff_stage]. *)

val flatness_stage :
     ?amin:float
  -> ?power:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [flatness_stage ()] is {!flatness} as a memoryless {!Pipeline} stage. See
    [Soundml.spectral_flatness_stage]. *)
