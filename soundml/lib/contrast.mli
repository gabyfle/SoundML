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

(** Spectral contrast over octave-spaced bands.

    The implementation module behind [Soundml.spectral_contrast] and
    [Soundml.spectral_contrast_stage]; the semantics are documented there,
    once, on the public face. The band plan — bin ranges and quantile sizes —
    is validated and precomputed when either face is built; the per-frame
    computation is batched over whole spectrogram tensors. *)

val spectral_contrast :
     Stft.Config.t
  -> ?n_bands:int
  -> ?f_min:float
  -> ?quantile:float
  -> ?linear:bool
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [spectral_contrast c ~sample_rate x] is the spectral contrast of the audio
    [x], shaped [[...; n_bands + 1; frames]]. See [Soundml.spectral_contrast]. *)

val spectral_contrast_of_spectrogram :
     ?n_bands:int
  -> ?f_min:float
  -> ?quantile:float
  -> ?linear:bool
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [spectral_contrast_of_spectrogram ~sample_rate s] is the spectral contrast
    of the magnitude spectrogram [s], shaped [[...; n_bands + 1; frames]]. See
    [Soundml.spectral_contrast_of_spectrogram]. *)

val stage :
     Stft.Config.t
  -> ?n_bands:int
  -> ?f_min:float
  -> ?quantile:float
  -> ?linear:bool
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage c ~sample_rate ()] is the contrast computation as a memoryless
    {!Pipeline} stage over magnitude-spectrum chunks. See
    [Soundml.spectral_contrast_stage]. *)
