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

(** Feature inversion: cepstra and mel spectrograms back to mel spectrograms,
    linear magnitude spectrograms and audio.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0
    ([librosa.feature.inverse]) with the named deviations documented there;
    parity is enforced against committed golden vectors in the test suite,
    and the non-negative least squares behind {!mel_to_stft} against the
    properties its documentation states. *)

val mfcc_to_mel :
     Mel.Config.t
  -> ?lifter:float
  -> ?reference:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mfcc_to_mel mel c] is the mel power spectrogram the cepstrum [c]
    encodes. See [Soundml.mfcc_to_mel]. *)

val mfcc_to_mel_stage :
     Mel.Config.t
  -> ?n_mfcc:int
  -> ?lifter:float
  -> ?reference:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mfcc_to_mel_stage mel ()] is {!mfcc_to_mel} as a pipeline stage. See
    [Soundml.mfcc_to_mel_stage]. *)

val mel_to_stft :
  Mel.Config.t -> ?power:float -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [mel_to_stft mel m] is the magnitude spectrogram whose mel projection
    best fits [m]. See [Soundml.mel_to_stft]. *)

val mel_to_stft_stage :
     Mel.Config.t
  -> ?power:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mel_to_stft_stage mel ()] is {!mel_to_stft} as a pipeline stage. See
    [Soundml.mel_to_stft_stage]. *)

val mel_to_audio :
     Stft.Config.t
  -> Mel.Config.t
  -> ?power:float
  -> ?n_iter:int
  -> ?length:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mel_to_audio stft mel m] is a signal whose mel spectrogram approximates
    [m]. See [Soundml.mel_to_audio]. *)

val mfcc_to_audio :
     Stft.Config.t
  -> Mel.Config.t
  -> ?lifter:float
  -> ?reference:float
  -> ?power:float
  -> ?n_iter:int
  -> ?length:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mfcc_to_audio stft mel c] is a signal whose cepstrum approximates [c].
    See [Soundml.mfcc_to_audio]. *)
