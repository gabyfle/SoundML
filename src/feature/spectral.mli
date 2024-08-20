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

(**
    The {!Feature.Spectral} module focus on extracting spectral-related features
    of an audio data. *)

(**
    {1 The Fast Fourier Transform (FFT)}

    SoundML allow you to compute the FFT of an audio data in an efficient and compliant way.
    The FFTs functions are simple wrappers around the Owl library FFT functions, that are themselves
    wrappers around the FFTPack library. *)

val fft : Audio.audio -> (Complex.t, Bigarray.complex32_elt) Audio.G.t
(**
    [fft audio] computes an FFT on the the given audio data.
    
    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let fft = fft src in
            (* ... *)
    ]} *)

val ifft :
     (Complex.t, Bigarray.complex32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Audio.G.t
(**
    [ifft fft] computes the inverse FFT of the given FFT data.
    
    Example:

    {[
        let () =
            let src = read file.wav wav in
            let fft = fft src in
            let ifft = ifft fft in
            (* ... *)
    ]} *)

(**
    {1 Spectral representations}

    Spectrograms are one of the most-used representation of time-frequency audio data.
    SoundML allow you to compute the spectrogram of an {!Audio.audio}. Algorithms are based
    on the work done by the authors and maintainers of the matplotlib.mlab Python module. *)

type mode =
  | PSD
  | Angle
  | Phase
  | Magnitude
  | Complex
  | Default  (** Mode used to compute a spectrogram *)

type side = OneSided | TwoSided  (** Side used to compute a spectrogram *)

(**
   Detrend functions module *)
module Detrend : sig
  val none : 'a -> 'a
  (**
    Identity function, no detrend *)
end

val specgram :
     ?nfft:int
  -> ?fs:int
  -> ?noverlap:int
  -> ?detrend:
       (   (float, Bigarray.float32_elt) Audio.G.t
        -> (float, Bigarray.float32_elt) Audio.G.t )
  -> Audio.audio
  -> (Complex.t, Bigarray.complex32_elt) Audio.G.t
     * (float, Bigarray.float32_elt) Owl_dense_ndarray_generic.t
(**
    [spectrogram ?nfft ?fs ?noverlap audio] computes the spectrogram of the given audio data.

    [?window] is the window function to apply to the audio data. The default window function is the hamming function
    from [Owl.Signal].
    [?nfft] is the number of points to use for the FFT. Default is [2048].
    [?window_size] is the size of the window to apply to the audio data. Default is [None].
    [audio] is the audio data.
    [n] is the number of points to use for the FFT.

    {i Note:} The spectrogram implementation is based on the work from the authors and maintainers of the matplotlib library,
    especially the matplotlib.mlab module. All the credits go to them.
    
    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let spec = specgram src in
            (* ... *)
    ]} *)

val rms :
     ?window:int
  -> ?step:int
  -> Audio.audio
  -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
(**
    [rms ~window ~step audio] computes the Root Mean Square (RMS) of the given audio data for each frame.

    [?window] is the window size to use for the RMS computation. Default is [2048].
    [?step] is the step size to use for the RMS computation. Default is [1024].

    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let rms = rms src in
            (* ... *)
    ]} *)
