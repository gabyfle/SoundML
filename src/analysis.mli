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

val fft : Audio.audio -> (Complex.t, Bigarray.complex64_elt) Audio.G.t
(**
    [fft audio] computes an FFT on the the given audio data.
    
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
    
    Implementation and inspiration from: {{:https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html} np.fft.fftfreq}.
    @see <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html> Numpy Documentation *)

val spectrogram :
     ?window:(int -> Owl_dense_ndarray.D.arr)
  -> ?nfft:int
  -> ?window_size:int option
  -> Audio.audio
  -> int
  -> (Complex.t, Bigarray.complex64_elt) Audio.G.t
(**
    [spectrogram ?window ?nfft ?window_size audio n] computes the spectrogram of the given audio data.

    [?window] is the window function to apply to the audio data. The default window function is the hamming function
    from {!Owl.Signal.hamming}.
    [?nfft] is the number of points to use for the FFT. Default is [2048].
    [?window_size] is the size of the window to apply to the audio data. Default is [None].
    [audio] is the audio data.
    [n] is the number of points to use for the FFT.
    
    Examples:

    {[
        let () =
            let src = read_audio file.wav wav in
            let spec = spectrogram src 1024 in
            (* ... *)
    ]} *)
