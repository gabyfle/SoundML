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

(** Synthetic signals and mu-law companding: tones, chirps, click tracks, and
    the mu-law compressor and expander.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0 ([librosa.tone],
    [librosa.chirp] over [scipy.signal.chirp], [librosa.clicks],
    [librosa.mu_compress] and [librosa.mu_expand]); parity is enforced
    against committed golden vectors in the test suite. *)

type positions =
  [`Frames of int array | `Samples of int array | `Times of float array]

val tone :
     (float, 'a) Nx.dtype
  -> ?length:int
  -> ?duration:float
  -> ?phase:float
  -> sample_rate:int
  -> float
  -> (float, 'a) Nx.t
(** [tone dtype ~sample_rate ~length f] is the pure tone at [f] Hz. See
    [Soundml.tone]. *)

val chirp :
     (float, 'a) Nx.dtype
  -> ?length:int
  -> ?duration:float
  -> ?linear:bool
  -> ?phase:float
  -> sample_rate:int
  -> fmin:float
  -> fmax:float
  -> unit
  -> (float, 'a) Nx.t
(** [chirp dtype ~sample_rate ~fmin ~fmax ~length ()] is the sine sweep from
    [fmin] to [fmax]. See [Soundml.chirp]. *)

val clicks :
     (float, 'a) Nx.dtype
  -> ?hop:int
  -> ?click:(float, 'a) Nx.t
  -> ?click_frequency:float
  -> ?click_duration:float
  -> ?length:int
  -> sample_rate:int
  -> positions
  -> (float, 'a) Nx.t
(** [clicks dtype ~sample_rate positions] is the click track sounding a click
    at each position. See [Soundml.clicks]. *)

val mu_compress :
  ?mu:float -> ?quantize:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [mu_compress x] is the mu-law compression of [x]. See
    [Soundml.mu_compress]. *)

val mu_compress_stage :
     ?mu:float
  -> ?quantize:bool
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mu_compress_stage ()] is {!mu_compress} as a pipeline stage. See
    [Soundml.mu_compress_stage]. *)

val mu_expand :
  ?mu:float -> ?quantize:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [mu_expand y] is the mu-law expansion of [y]. See [Soundml.mu_expand]. *)

val mu_expand_stage :
     ?mu:float
  -> ?quantize:bool
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mu_expand_stage ()] is {!mu_expand} as a pipeline stage. See
    [Soundml.mu_expand_stage]. *)
