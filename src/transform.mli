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

(** This module provides functions to transform audio signals.
*)

val stft :
     ?n_fft:int
  -> ?hop_length:int
  -> ?win_length:int
  -> ?window:(float, 'a) Nx.t
  -> ?center:bool
  -> (float, 'a) Nx.t
  -> (Complex.t, 'b) Nx.t
(** Computes the Short-Time Fourier Transform (STFT) of a
    signal.

    @param n_fft The number of FFT components.
    @param hop_length
      The number of samples between consecutive frames.
    @param win_length The length of the window.
    @param window The window function to use.
    @param center Whether to pad the signal on both sides.
    @param pad_mode The padding mode to use.
    @param signal The input signal.
    @return The STFT of the signal. *)
