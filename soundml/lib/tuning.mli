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

(** Pitch tracking and tuning estimation over magnitude spectrograms.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0 ([librosa.piptrack],
    [librosa.pitch_tuning] and [librosa.estimate_tuning]); parity is enforced
    against committed golden vectors in the test suite. *)

(** The magnitude a bin must exceed to be detected. See
    [Soundml.piptrack_reference]. *)
type reference = [`Maximum | `Value of float]

val piptrack :
     ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:reference
  -> ?fft_size:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [piptrack ~sample_rate s] is the pitches and magnitudes of the peaks of
    the magnitude spectrogram [s]. See [Soundml.piptrack]. *)

val stage :
     ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:reference
  -> ?fft_size:int
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t * (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage ~sample_rate ()] is {!piptrack} as a memoryless pipeline stage.
    See [Soundml.piptrack_stage]. *)

val pitch_tuning :
  ?resolution:float -> ?bins_per_octave:int -> (float, 'a) Nx.t -> float
(** [pitch_tuning f] is the tuning of the frequencies [f], in fractions of a
    bin. See [Soundml.pitch_tuning]. *)

val estimate_tuning :
     Stft.Config.t
  -> ?resolution:float
  -> ?bins_per_octave:int
  -> ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:reference
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> float
(** [estimate_tuning c ~sample_rate x] is the tuning of the audio [x]. See
    [Soundml.estimate_tuning]. *)

val estimate_tuning_of_spectrogram :
     ?resolution:float
  -> ?bins_per_octave:int
  -> ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:reference
  -> ?fft_size:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> float
(** [estimate_tuning_of_spectrogram ~sample_rate s] is the tuning of the
    magnitude spectrogram [s]. See [Soundml.estimate_tuning_of_spectrogram]. *)
