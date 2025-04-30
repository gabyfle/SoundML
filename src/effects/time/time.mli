(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023-2025                                                  *)
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

(** {1 Time-stretching and pitch-shifting} *)

(** This module provides functions for time-stretching and pitch-shifting audio signals. *)

(** {2 Configuration} *)

(** This module is based on the {{:https://breakfastquay.com/rubberband/}RubberBand library}.
    It expose some utilities to configure the Rubberband stretcher.
    The documentation of the {!Config} module as well as the options types has been directly taken
    from the Rubberband's documentation. For more in-depth informations, please visit {{:https://breakfastquay.com/rubberband/}the official Rubberband website}. *)

(** Engine to use for the stretching *)
type engine =
  | Faster  (** Use the Rubber Band Library R2 (Faster) engine. *)
  | Finer  (** Use the Rubber Band Library R3 (Finer) engine. *)

(** Option to control the component frequency phase-reset mechanism in the R2 engine. These options have no effect when using the R3 engine. *)
type transients =
  | Crisp
      (** Reset component phases at the peak of each transient (the start of a significant note or percussive event).*)
  | Mixed
      (** Reset component phases at the peak of each transient, outside a frequency range typical of musical fundamental frequencies.*)
  | Smooth  (** Do not reset component phases at any point. *)

(** Option to control the type of transient detector used in the R2 engine. These options have no effect when using the R3 engine. *)
type detector =
  | Compound
      (** Use a general-purpose transient detector which is likely to be good for most situations. *)
  | Percussive  (** Detect percussive transients. *)
  | Soft
      (** Use an onset detector with less of a bias toward percussive transients. *)

(** Option to control the adjustment of component frequency phases in the R2 engine from one analysis window to the next during non-transient segments. These options have no effect when using the R3 engine. *)
type phase =
  | Laminar
      (** Adjust phases when stretching in such a way as to try to retain the continuity of phase relationships between adjacent frequency bins whose phases are behaving in similar ways. *)
  | Independent
      (** Adjust the phase in each frequency bin independently from its neighbours. *)

(** Option to control the threading model of the stretcher. *)
type threading =
  | Auto  (** Permit the stretcher to determine its own threading model. *)
  | Never  (** Never use more than one thread. *)
  | Always
      (** Use multiple threads in any situation where [Auto] would do so, except omit the check for multiple CPUs and instead assume it to be true. *)

(** Option to control the window size for FFT processing. *)
type window =
  | Standard  (** Use the default window size. *)
  | Short  (** Use a shorter window. *)
  | Long  (** Use a longer window. *)

(** Option to control the use of window-presum FFT and time-domain smoothing in the R2 engine. These options have no effect when using the R3 engine. *)
type smoothing =
  | Off  (** Do not use time-domain smoothing. *)
  | On  (** Use time-domain smoothing. *)

(** Option to control the handling of formant shape (spectral envelope) when pitch-shifting. *)
type formant =
  | Shifted  (** Apply no special formant processing. *)
  | Preserved  (** Preserve the spectral envelope of the unshifted signal. *)

(** Option to control the method used for pitch shifting. *)
type pitch =
  | HighSpeed  (** Favour CPU cost over sound quality. *)
  | HighQuality  (** Favour sound quality over CPU cost. *)
  | HighConsistency
      (** Use a method that supports dynamic pitch changes without discontinuities, including when crossing the 1.0 pitch scale. *)

(** Option to control the method used for processing two-channel stereo audio. *)
type channels =
  | Apart
      (** Channels are handled for maximum individual fidelity, at the expense of synchronisation. *)
  | Together
      (** Channels are handled for higher synchronisation at some expense of individual fidelity. *)

module Config : sig
  type t =
    { engine: engine
    ; transients: transients
    ; detector: detector
    ; phase: phase
    ; threading: threading
    ; window: window
    ; smoothing: smoothing
    ; formant: formant
    ; pitch: pitch
    ; channels: channels }

  val default : t

  val percussive : t

  val with_engine : engine -> t -> t

  val with_transients : transients -> t -> t

  val with_detector : detector -> t -> t

  val with_phase : phase -> t -> t

  val with_threading : threading -> t -> t

  val with_window : window -> t -> t

  val with_smoothing : smoothing -> t -> t

  val with_formant : formant -> t -> t

  val with_pitch : pitch -> t -> t

  val with_channels : channels -> t -> t

  val to_int : t -> int
end

(** {2 Functions} *)

val time_stretch :
  'a.
     ?config:Config.t
  -> (float, 'a) Owl_dense_ndarray.Generic.t
  -> int
  -> float
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(**
    [time_stretch ?config x sample_rate ratio] time-strech the input signal [x] with [ratio].

    {2 Parameters}
    @param ?config is the configuration to use. Default is {!Config.default}.
    @param x is the input signal to time-stretch. It can be either a [Bigarray.Float32] or [Bigarray.Float64].
    @param sample_rate is the sample rate of the input signal.
    @param ratio is the time-stretching ratio. It can't be a negative value. A value of [1.0] means no time-stretching, a value of [2.0] means double speed and a value of [0.5] means half speed.

    {2 Usage}
    After loading your audio with the {!Io} module, you can time-stretch it like this.
    
    {[
      open Soundml
      open Effects
      let audio = Io.read Bigarray.Float32 "path/to/file.mp3" in
      let sample_rate = Audio.sample_rate audio in
      let data = Audio.data audio in
      let stretched = Time.time_stretch data sample_rate 2.0 in
      Io.write "path/to/file_stretched.mp3" data sample_rate
    ]} *)

val pitch_shift :
  'a.
     ?config:Config.t
  -> ?bins_per_octave:int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
  -> int
  -> int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(**
    [pitch_shift ?config ?bins_per_octave x sample_rate steps] shift the pitch of the input signal [x] by [steps] semitones.

    {2 Parameters}
    @param ?config is the configuration to use. Default is {!Config.default}.
    @param ?bins_per_octave is the number of bins per octave. Default is [12].
    @param x is the input signal to time-stretch. It can be either a [Bigarray.Float32] or [Bigarray.Float64].
    @param sample_rate is the sample rate of the input signal.
    @param steps is the number of semitones to shift the pitch. A value of [0] means no pitch-shifting, a value of [1] means one semitone up and a value of [-1] means one semitone down.

    {2 Usage}
    After loading your audio with the {!Io} module, you can pitch-shift it like this.
    
    {[
      open Soundml
      open Effects
      let audio = Io.read Bigarray.Float32 "path/to/file.mp3" in
      let sample_rate = Audio.sample_rate audio in
      let data = Audio.data audio in
      let stretched = Time.pitch_shift data sample_rate -6 in
      Io.write "path/to/file_pitched.mp3" data sample_rate
    ]} *)
