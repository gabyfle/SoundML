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

(** Autocorrelation, linear prediction and spectral polynomial fits.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0
    ([librosa.autocorrelate], [librosa.lpc] and
    [librosa.feature.poly_features]); parity is enforced against committed
    golden vectors in the test suite. *)

val autocorrelate : ?max_size:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [autocorrelate x] is the autocorrelation of every row of the last axis of
    [x]. See [Soundml.autocorrelate]. *)

val lpc : order:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [lpc ~order x] is the Burg prediction-error filter of every row of the
    last axis of [x]. See [Soundml.lpc]. *)

val poly_features :
     Stft.Config.t
  -> ?order:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [poly_features c ~sample_rate x] is the per-frame polynomial fit of the
    magnitude spectrogram of the audio [x]. See [Soundml.poly_features]. *)

val poly_features_of_spectrogram :
     ?order:int
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [poly_features_of_spectrogram ~sample_rate s] is the per-frame polynomial
    fit of the spectrogram [s]. See [Soundml.poly_features_of_spectrogram]. *)

val stage :
     Stft.Config.t
  -> ?order:int
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage c ~sample_rate ()] is the per-frame fit as a memoryless
    {!Pipeline} stage over magnitude-spectrum chunks. See
    [Soundml.poly_features_stage]. *)
