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

(** This module provides functions to transform audio signals. *)

val stft :
     ?n_fft:int
  -> ?hop_size:int
  -> ?win_length:int
  -> ?window:Window.window
  -> ?center:bool
  -> (float, Bigarray.float64_elt) Nx.t
  -> (Complex.t, Bigarray.complex64_elt) Nx.t
(** Generic STFT function that accepts any float input type and returns complex64.
    
    This is the most convenient function to use as it handles all float types
    (float16, float32, float64) and always returns complex64 for consistency.

    @param n_fft The number of FFT components (default: 2048).
    @param hop_size
      The number of samples between consecutive frames (default: 512).
    @param win_length The length of the window (default: 2048).
    @param window The window function to use (default: Hanning window).
    @param center Whether to pad the signal on both sides (default: true).
    @param signal The input signal.
    @return The STFT of the signal with shape [n_fft/2 + 1, n_frames]. *)
