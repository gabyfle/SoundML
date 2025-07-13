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

(** This module provides functions to create filterbanks. A
    filterbank is a set of filters that are applied to a
    signal. *)

type norm = Slaney | PNorm of float

val mel :
     ?fmax:float option
  -> ?htk:bool
  -> ?norm:norm option
  -> (float, 'b) Nx.dtype
  -> int
  -> int
  -> int
  -> float
  -> (float, 'b) Nx.t
(** Creates a Mel filterbank. A Mel filterbank is a set of
    filters that are spaced according to the Mel scale.

    @param n_mels The number of Mel filters to generate.
    @param f_min The minimum frequency of the filterbank.
    @param f_max The maximum frequency of the filterbank.
    @param n_fft The number of FFT components.
    @param sample_rate The sample rate of the signal.
    @return A Mel filterbank of shape (n_mels, n_fft). *)
