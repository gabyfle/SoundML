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

(** Tempo and beat: tempograms of an onset envelope, tempo estimation, beat
    tracking and the predominant local pulse.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0
    ([librosa.feature.tempogram], [fourier_tempogram] and [tempo],
    [librosa.beat.beat_track] and [plp], [librosa.tempo_frequencies] and
    [fourier_tempo_frequencies]); parity is enforced against committed golden
    vectors in the test suite. *)

type norm = Chroma.norm

type aggregate = [`Mean | `Median | `None]

type kernels = [`Reference | `Unit]

val tempo_frequencies :
  (float, 'a) Nx.dtype -> ?hop:int -> sample_rate:int -> int -> (float, 'a) Nx.t
(** [tempo_frequencies dtype ~sample_rate n] is the tempo of each lag bin of
    an [n]-bin tempogram. See [Soundml.tempo_frequencies]. *)

val fourier_tempo_frequencies :
  (float, 'a) Nx.dtype -> ?hop:int -> sample_rate:int -> int -> (float, 'a) Nx.t
(** [fourier_tempo_frequencies dtype ~sample_rate win] is the tempo of each
    bin of a [win]-frame Fourier tempogram. See
    [Soundml.fourier_tempo_frequencies]. *)

val tempogram :
     ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> ?norm:norm
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [tempogram x] is the autocorrelation tempogram of the onset envelope [x].
    See [Soundml.tempogram]. *)

val tempogram_stage :
     ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> ?norm:norm
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [tempogram_stage ()] is {!tempogram} as a pipeline stage. See
    [Soundml.tempogram_stage]. *)

val fourier_tempogram :
     (Complex.t, 'c) Nx.dtype
  -> ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> (float, 'a) Nx.t
  -> (Complex.t, 'c) Nx.t
(** [fourier_tempogram cdtype x] is the Fourier tempogram of the onset
    envelope [x]. See [Soundml.fourier_tempogram]. *)

val fourier_tempogram_stage :
     (Complex.t, 'c) Nx.dtype
  -> ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> unit
  -> ((float, 'a) Nx.t, (Complex.t, 'c) Nx.t, 'k) Pipeline.t
(** [fourier_tempogram_stage cdtype ()] is {!fourier_tempogram} as a pipeline
    stage. See [Soundml.fourier_tempogram_stage]. *)

val tempo :
     ?hop:int
  -> ?start_bpm:float
  -> ?std_bpm:float
  -> ?ac_size:float
  -> ?tempo_max:float
  -> ?aggregate:aggregate
  -> ?prior:(float -> float)
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [tempo ~sample_rate x] is the tempo of the onset envelope [x]. See
    [Soundml.tempo]. *)

val beat_track :
     ?hop:int
  -> ?start_bpm:float
  -> ?tightness:float
  -> ?trim:bool
  -> ?bpm:float
  -> ?prior:(float -> float)
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> float * int array
(** [beat_track ~sample_rate x] is the tempo and the beat frames of the onset
    envelope [x]. See [Soundml.beat_track]. *)

val plp :
     ?hop:int
  -> ?win_length:int
  -> ?tempo_min:float
  -> ?tempo_max:float
  -> ?prior:(float -> float)
  -> ?kernels:kernels
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [plp ~sample_rate x] is the predominant local pulse of the onset envelope
    [x]. See [Soundml.plp]. *)
