(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023                                                       *)
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

val fft :
  ?norm:bool -> Audio.audio -> (Complex.t, Bigarray.complex64_elt) Audio.G.t
(**
    [fft ?norm audio] computes an FFT on the slice [start; finish] of the given audio data.
    By default, [norm] is set to [true]. This will normalise the audio data before computing the FFT.
    
    Examples:

    {[
        let () =
            let src = read_audio file.wav wav in
            let fft = fft src in
            (* ... *)
    ]} *)

val ifft :
     (Complex.t, Bigarray.complex64_elt) Audio.G.t
  -> (float, Bigarray.float64_elt) Audio.G.t
(**
    [ifft fft] computes the inverse FFT of the given FFT data.
    
    Example:

    {[
        let () =
            let src = read_audio file.wav wav in
            let fft = fft src in
            let ifft = ifft fft in
            (* ... *)
    ]} *)

val fftfreq : Audio.audio -> (float, Bigarray.float64_elt) Audio.G.t
(**
    [fftfreq audio] return the FT sample frequencies.
    
    Inspired from: np.fft.fft.
    @see <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html> Numpy Documentation *)
