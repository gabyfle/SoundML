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

(** Harmonic/percussive source separation by median filtering.

    The implementation module behind [Soundml.hpss] and its spectrogram-,
    mask- and complex-domain faces; the semantics are documented there, once,
    on the public face. Median filtering the magnitude spectrogram along time
    enhances the horizontal structure of sustained partials and filtering it
    along frequency enhances the vertical structure of transients
    (Fitzgerald, {e Harmonic/percussive separation using median filtering},
    DAFx 2010); the two enhanced spectrograms drive a pair of masks whose
    exponent and margins are those of Driedger, Müller and Disch
    ({e Extending harmonic-percussive separation of audio signals}, ISMIR
    2014).

    {2 Median-filter semantics}

    Both filters are one-dimensional. The window of index [i] on a line of
    [n] values is [\[i - k / 2, i + k - 1 - k / 2\]] — left-biased for even
    [k] — and the filtered value is rank [k / 2] of that window sorted
    ascending, the upper middle for even [k] and never the average of two.
    Indices outside the line reflect half-sample-symmetrically with period
    [2 n]: [-1] reads [0], [n] reads [n - 1], and the pattern repeats, so a
    kernel may exceed the line length by any amount. These semantics hold for
    every [k] and every line length, boundaries included.

    {2 Parity}

    Semantics follow librosa 0.11 ([librosa.decompose.hpss],
    [librosa.util.softmask] and the [librosa.effects] signal-domain
    wrappers), whose median filter is [scipy.ndimage.median_filter] under
    [mode="reflect"]; parity is enforced against committed golden vectors in
    the test suite. Three deviations are named and permanent:

    - hard masks ([power = infinity]) are carried as the floats [0.] and
      [1.] in the dtype of the input, where the reference returns a boolean
      array;
    - at an exponent other than [1] and [2] the mask carries up to one unit
      in the last place against the reference, which reaches those two
      exponents through fast paths this implementation reproduces exactly
      and the others through a general power function;
    - where a kernel exceeds [2 n + 1] on a line of [n] values, the
      reflection above continues to hold here, while
      [scipy.ndimage.median_filter] departs from the boundary mode it
      documents; the golden vectors stay inside [k <= 2 n + 1]. *)

val hpss_of_spectrogram :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [hpss_of_spectrogram s] is the harmonic and percussive components of the
    magnitude or power spectrogram [s]. See [Soundml.hpss_of_spectrogram]. *)

val hpss_of_stft :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (Complex.t, 'c) Nx.t
  -> (Complex.t, 'c) Nx.t * (Complex.t, 'c) Nx.t
(** [hpss_of_stft z] is the harmonic and percussive components of the complex
    spectrum [z], phase preserved. See [Soundml.hpss_of_stft]. *)

val hpss_masks :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [hpss_masks s] is the harmonic and percussive mask pair of the magnitude
    or power spectrogram [s]. See [Soundml.hpss_masks]. *)

val hpss :
     Stft.Config.t
  -> ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [hpss c x] is the harmonic and percussive parts of the audio [x] on the
    analysis geometry [c]. See [Soundml.hpss]. *)

val harmonic :
     Stft.Config.t
  -> ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [harmonic c x] is the harmonic part of the audio [x]. See
    [Soundml.harmonic]. *)

val percussive :
     Stft.Config.t
  -> ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [percussive c x] is the percussive part of the audio [x]. See
    [Soundml.percussive]. *)
