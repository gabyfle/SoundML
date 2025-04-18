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
    {1 Spectral}

    Spectrograms are one of the most-used representation of time-frequency audio data.
    SoundML allow you to compute the spectrogram of an {!Audio.audio}. Algorithms are based
    on the work done by the authors and maintainers of the matplotlib.mlab Python module. *)

(** Mode used to compute a spectrogram *)
type mode =
  | PSD  (** Power Spectral Density *)
  | Angle  (** Phase angle *)
  | Phase  (** Alias for Angle *)
  | Magnitude  (** Magnitude spectrum *)
  | Complex  (** Complex spectrum *)
  | Default  (** Default to PSD *)

type side = OneSided | TwoSided  (** Side used to compute a spectrogram *)

(**
    Window functions module *)
module Window : sig
  (** Window type *)
  type t =
    | Hann  (** Hanning window *)
    | Hamming  (** Hamming window *)
    | Blackman  (** Blackman window *)
    | Rectangle  (** Rectangle window *)
    | Custom of
        (int -> (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t)
        (** Custom user-defined window *)

  val get_window : t -> int -> Owl.Dense.Ndarray.D.arr
  (**
    [get_window window n] returns the window type [window] function of size [n] *)

  val default : t
  (**
    Default window: Hann *)
end

(**
   Detrend functions module *)
module Detrend : sig
  val none : 'a -> 'a
  (**
    Identity function, no detrend *)

  val constant :
       (Complex.t, Bigarray.complex32_elt) Audio.G.t
    -> (Complex.t, Bigarray.complex32_elt) Audio.G.t
  (**
    Constant detrend function *)

  val linear :
       (Complex.t, Bigarray.complex32_elt) Audio.G.t
    -> (Complex.t, Bigarray.complex32_elt) Audio.G.t
  (**
    Linear detrend function *)
end

module Filterbank : sig
  type norm = Slaney | PNorm of float

  val mel :
       ?fmax:float option
    -> ?htk:bool
    -> ?norm:norm option
    -> sample_rate:int
    -> nfft:int
    -> nmels:int
    -> fmin:float
    -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
end

module Config : sig
  type t =
    { nfft: int
    ; window: Window.t
    ; fs: int
    ; noverlap: int
    ; side: side
    ; mode: mode
    ; pad_to: int option
    ; scale_by_freq: bool option }

  val default : t
end

val specgram :
     ?config:Config.t
  -> ?detrend:
       (   (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
        -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t )
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(**
    [spectrogram ?nfft ?fs ?noverlap audio] computes the spectrogram of the given audio data.

    [?window] is the window function to apply to the audio data. The default window function is the hamming function
    from [Owl.Signal].
    [?nfft] is the number of points to use for the FFT. Default is [2048].
    [?window_size] is the size of the window to apply to the audio data. Default is [None].
    [audio] is the audio data.

    {i Note:} The spectrogram implementation is based on the work from the authors and maintainers of the matplotlib library,
    especially the matplotlib.mlab module. All the credits go to them.
    
    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let spec = specgram src in
            (* ... *)
    ]} *)

val complex_specgram :
     ?config:Config.t
  -> ?detrend:
       (   (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
        -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t )
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (Complex.t, Bigarray.complex32_elt) Owl.Dense.Ndarray.Generic.t

(** 
    [complex_specgram ?nfft ?fs ?noverlap audio] computes the complex spectrogram of the given audio data.

    [?window] is the window function to apply to the audio data. The default window function is the hamming function
    from [Owl.Signal].
    [?nfft] is the number of points to use for the FFT. Default is [2048].
    [?window_size] is the size of the window to apply to the audio data. Default is [None].
    [audio] is the audio data.

    {i Note:} The spectrogram implementation is based on the work from the authors and maintainers of the matplotlib library,
    especially the matplotlib.mlab module. All the credits go to them.
    
    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let spec = complex_specgram src in
            (* ... *)
    ]} *)

val magnitude_specgram :
     ?config:Config.t
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(**
    [magnitude_specgram ?nfft ?fs ?noverlap audio] computes the magnitude spectrogram of the given audio data.

    [?window] is the window function to apply to the audio data. The default window function is the hamming function
    from [Owl.Signal].
    [?nfft] is the number of points to use for the FFT. Default is [2048].
    [?window_size] is the size of the window to apply to the audio data. Default is [None].
    [audio] is the audio data.

    {i Note:} The spectrogram implementation is based on the work from the authors and maintainers of the matplotlib library,
    especially the matplotlib.mlab module. All the credits go to them.
    
    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let spec = magnitude_specgram src in
            (* ... *)
    ]} *)

val phase_specgram :
     ?config:Config.t
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(**
    [phase_specgram ?nfft ?fs ?noverlap audio] computes the phase spectrogram of the given audio data.

    [?window] is the window function to apply to the audio data. The default window function is the hamming function
    from [Owl.Signal].
    [?nfft] is the number of points to use for the FFT. Default is [2048].
    [?window_size] is the size of the window to apply to the audio data. Default is [None].
    [audio] is the audio data.

    {i Note:} The spectrogram implementation is based on the work from the authors and maintainers of the matplotlib library,
    especially the matplotlib.mlab module. All the credits go to them.
    
    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let spec = phase_specgram src in
            (* ... *)
    ]} *)

val mel_specgram :
     ?config:Config.t
  -> ?nmels:int
  -> ?fmin:float
  -> ?fmax:float option
  -> ?htk:bool
  -> ?norm:Filterbank.norm option
  -> ?sample_rate:int
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Owl.Dense.Ndarray.Generic.t
(** 
  [mel_specgram] *)

val mfcc :
     ?config:Config.t
  -> ?nmfcc:int
  -> ?nmels:int
  -> ?fmin:float
  -> ?fmax:float option
  -> ?htk:bool
  -> ?norm:Filterbank.norm
  -> ?dct_type:Owl_fft_generic.ttrig_transform
  -> ?lifter:int
  -> ?sample_rate:int
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Audio.G.t

val rms :
     ?window:int
  -> ?step:int
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Audio.G.t
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

val zero_crossings :
     ?threshold:float
  -> ?zero_pos:bool
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Audio.G.t
(**
    [zero_crossings ~threshold ~zero_pos x] computes the zero-crossings of the given audio data for each frame.

    [?threshold] is the threshold to use for the zero-crossing computation. Default is [1e-10].
    [?zero_pos] is a boolean to determine if zero-crossings should be considered positive. Default is [true].

    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let zc = zero_crossings src in
            (* ... *)
    ]} *)

val zero_crossing_rate :
     ?window:int
  -> ?hop_length:int
  -> ?threshold:float
  -> ?zero_pos:bool
  -> (float, Bigarray.float32_elt) Audio.G.t
  -> (float, Bigarray.float32_elt) Audio.G.t
(**
    [zero_crossing_rate ~window ~step audio] computes the zero-crossing rate of the given audio data for each frame.

    [?window] is the window size to use for the zero-crossing computation. Default is [256].
    [?noverlap] is the step size to use for the zero-crossing computation. Default is [128].

    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let zcr = zero_crossing src in
            (* ... *)
    ]} *)
