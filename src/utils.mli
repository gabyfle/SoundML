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

(** Audio processing utility functions.
    
    This module provides essential utility functions for audio signal processing,
    including signal framing, padding, frequency conversions, and mathematical
    operations commonly used in audio analysis pipelines. *)

(** {2 Signal Framing and Windowing} *)

val frame :
     ?axis:int
  -> ('a, 'b, 'dev) Rune.t
  -> frame_length:int
  -> hop_length:int
  -> ('a, 'b, 'dev) Rune.t
(** [frame signal ~frame_length ~hop_length] slices a signal into overlapping frames.

   This function creates overlapping windows of the input signal, which is essential
   for Short-Time Fourier Transform (STFT) and other windowed analysis techniques.
   The implementation uses efficient stride manipulation to avoid copying data.

   @param axis The axis along which to frame (default: -1, last axis)
   @param signal Input signal to frame
   @param frame_length Length of each frame in samples (must be > 0)
   @param hop_length Number of samples to advance between frames (must be >= 1)
   @return Framed view of the input signal with one additional dimension

   @raise Stdlib.Invalid_argument if frame_length <= 0
   @raise Stdlib.Invalid_argument if hop_length < 1
   @raise Stdlib.Invalid_argument if signal is too short for even one frame
   @raise Stdlib.Invalid_argument if axis is out of bounds

   {3 Examples}

   Basic signal framing:
   {[
     let signal = Rune.arange Rune.float32 0.0 10.0 1.0 in
     let frames = Utils.frame signal ~frame_length:3 ~hop_length:2 in
     (* frames has shape [3; 4] with overlapping windows *)
   ]}

   Frame a stereo signal:
   {[
     let stereo = Rune.ones Rune.float32 [|2; 1000|] in
     let frames = Utils.frame stereo ~frame_length:512 ~hop_length:256 in
     (* frames has shape [2; 512; 3] - frames along last axis *)
   ]}

   Frame along first axis:
   {[
     let signal = Rune.arange Rune.float32 0.0 100.0 1.0 in
     let frames = Utils.frame ~axis:0 signal ~frame_length:10 ~hop_length:5 in
     (* frames along first dimension *)
   ]}

   {3 Performance Notes}

   - Creates a view of the original data, not a copy (O(1) memory)
   - The returned tensor shares memory with the input
   - Modifications to frames affect the original signal
   - Most efficient when frame_length and hop_length are powers of 2 *)

val pad_center :
  ('a, 'b, 'dev) Rune.t -> size:int -> pad_value:'a -> ('a, 'b, 'dev) Rune.t
(** [pad_center signal ~size ~pad_value] pads a signal to center it.

   Centers the signal by padding it symmetrically on both sides. This is commonly
   used in STFT preprocessing to ensure the first and last frames are properly
   centered on the signal boundaries.

   @param device Rune device to use for computation (default: same as input)
   @param signal Input signal to pad
   @param size Total desired size after padding
   @param pad_value Value to use for padding
   @return Padded signal of the specified size

   @raise Stdlib.Invalid_argument if size < signal length
   @raise Stdlib.Invalid_argument if signal is not 1D

   {3 Example}

   {[
     let signal = Rune.ones Rune.float32 [|100|] in
     let padded = Utils.pad_center signal ~size:200 ~pad_value:0.0 in
     (* padded has 50 zeros, then 100 ones, then 50 zeros *)
   ]} *)

(** {2 Frequency and Scale Conversions} *)

module Convert : sig
  (** Frequency and amplitude scale conversion functions. *)

  val mel_to_hz :
    ?htk:bool -> (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  (** [mel_to_hz mel_values] converts mel-scale values to Hz.
      
      @param htk Use HTK mel scale formula (default: false, uses Slaney formula)
      @param mel_values Mel-scale frequency values
      @return Corresponding Hz frequency values
      
      {3 Example}
      
      {[
        let mel_freqs = Rune.linspace Rune.float32 0.0 2595.0 100 in
        let hz_freqs = Utils.Convert.mel_to_hz mel_freqs in
        (* Convert 100 mel frequencies to Hz *)
      ]} *)

  val hz_to_mel :
    ?htk:bool -> (float, 'a, 'dev) Rune.t -> (float, 'a, 'dev) Rune.t
  (** [hz_to_mel hz_values] converts Hz frequency values to mel scale.
      
      @param htk Use HTK mel scale formula (default: false, uses Slaney formula)
      @param hz_values Hz frequency values
      @return Corresponding mel-scale frequency values
      
      This is the inverse function of {!mel_to_hz}. *)

  (** Reference value specification for dB conversions. *)
  type ('a, 'dev) reference =
    | RefFloat of float  (** Use a constant reference value. *)
    | RefFunction of ((float, 'a, 'dev) Rune.t -> float)
        (** Compute reference from the input tensor (e.g., maximum value). *)

  val power_to_db :
       ?amin:float
    -> ?top_db:float
    -> ('a, 'dev) reference
    -> (float, 'a, 'dev) Rune.t
    -> (float, 'a, 'dev) Rune.t
  (** [power_to_db reference power_spectrum] converts power values to decibels.
      
      @param amin Minimum amplitude threshold (default: 1e-10)
      @param top_db Maximum dB value relative to reference (default: 80.0)
      @param reference Reference value for dB calculation
      @param power_spectrum Power spectrum values
      @return dB values: 10 * log10(power / reference)
      
      @raise Stdlib.Invalid_argument if amin <= 0.0
      
      {3 Example}
      
      {[
        let power_spec = (* compute power spectrum *) in
        let db_spec = Utils.Convert.power_to_db 
          (RefFunction Rune.max) 
          power_spec in
        (* Convert to dB relative to maximum value *)
      ]} *)

  val db_to_power :
       ?amin:float
    -> ('a, 'dev) reference
    -> (float, 'a, 'dev) Rune.t
    -> (float, 'a, 'dev) Rune.t
  (** [decibels_to_power reference db_values] converts decibel values to power.
      
      This is the inverse function of {!power_to_db}.
      
      @param amin Minimum amplitude threshold (default: 1e-10)
      @param reference Reference value used in original dB conversion
      @param db_values Decibel values
      @return Power values: reference * 10^(db/10) *)
end

(** {2 Frequency Generation} *)

val melfreqs :
     ?n_mels:int
  -> ?f_min:float
  -> ?f_max:float
  -> ?htk:bool
  -> 'dev Rune.device
  -> (float, 'b) Rune.dtype
  -> (float, 'b, 'dev) Rune.t
(** [melfreqs device dtype] generates mel-scale frequency values.

   Computes a tensor of acoustic frequencies tuned to the mel scale, which
   approximates human auditory perception. This is used internally by
   mel filterbank functions but can also be useful for custom applications.

   @param n_mels Number of mel frequencies to generate (default: 128)
   @param f_min Minimum frequency in Hz (default: 0.0)
   @param f_max Maximum frequency in Hz (default: 11025.0)
   @param htk Use HTK mel scale formula (default: false)
   @param device Rune device on which the Tensor should be created
   @param dtype Data type for the output tensor
   @return Tensor of mel-scale frequencies

   @raise Invalid_argument if n_mels <= 0
   @raise Invalid_argument if f_min < 0.0 or f_min >= f_max

   {3 Example}

   {[
     let mel_freqs = Utils.melfreqs 
       Rune.float32 
       ~n_mels:80 
       ~f_min:80.0 
       ~f_max:7600.0 in
     (* 80 mel-scale frequencies from 80 Hz to 7600 Hz *)
   ]}

   See: {{:https://librosa.org/doc/main/generated/librosa.mel_frequencies.html}librosa.mel_frequencies} *)

(** {2 Mathematical Utilities} *)

val unwrap :
     ?discontinuity:float
  -> ?axis:int
  -> ?period:float
  -> (float, 'a, 'dev) Rune.t
  -> (float, 'a, 'dev) Rune.t
(** [unwrap phase_values] unwraps phase values by removing discontinuities.

   Corrects phase values by adding multiples of 2π to remove artificial
   discontinuities caused by the periodic nature of phase. This is essential
   for phase-based audio analysis and synthesis.

   @param discontinuity Threshold for detecting discontinuities (default: π)
   @param axis Axis along which to unwrap (default: -1)
   @param period Period of the phase values (default: 2π)
   @param phase_values Phase values to unwrap
   @return Unwrapped phase values

   @raise Invalid_argument if discontinuity <= 0.0
   @raise Invalid_argument if period <= 0.0
   @raise Invalid_argument if axis is out of bounds

   {3 Example}

   {[
     let wrapped_phase = (* phase from STFT *) in
     let unwrapped = Utils.unwrap wrapped_phase in
     (* Remove 2π discontinuities for smooth phase *)
   ]}

   See: {{:https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html}numpy.unwrap} *)

val outer :
     (('a, 'b, 'dev) Rune.t -> ('a, 'b, 'dev) Rune.t -> ('a, 'b, 'dev) Rune.t)
  -> ('a, 'b, 'dev) Rune.t
  -> ('a, 'b, 'dev) Rune.t
  -> ('a, 'b, 'dev) Rune.t
(** [outer_product operation x y] computes generalized outer product.

   Applies a binary operation between all pairs of elements from two tensors,
   creating a matrix where result[i,j] = operation(x[i], y[j]). This is useful
   for creating distance matrices, correlation matrices, and other pairwise operations.

   @param operation Binary operation to apply (e.g., Rune.add, Rune.mul)
   @param x First input tensor (1D)
   @param y Second input tensor (1D)
   @return 2D tensor with shape [length(x); length(y)]

   @raise Invalid_argument if x or y are not 1D

   {3 Example}

   {[
     let x = Rune.arange Rune.float32 1.0 4.0 1.0 in
     let y = Rune.arange Rune.float32 1.0 3.0 1.0 in
     let product = Utils.outer_product Rune.mul x y in
     (* Multiplication table: [[1,2], [2,4], [3,6]] *)
   ]} *)
