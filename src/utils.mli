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

val pad_center : ('a, 'b) Nx.t -> size:int -> pad_value:'a -> ('a, 'b) Nx.t
(** [pad_center signal ~size ~pad_value] pads a signal to center it.

   Centers the signal by padding it symmetrically on both sides. This is commonly
   used in STFT preprocessing to ensure the first and last frames are properly
   centered on the signal boundaries.

   @param signal Input signal to pad
   @param size Total desired size after padding
   @param pad_value Value to use for padding
   @return Padded signal of the specified size

   @raise Stdlib.Invalid_argument if size < signal length
   @raise Stdlib.Invalid_argument if signal is not 1D

   {3 Example}

   {[
     let signal = Nx.ones Nx.float32 [|100|] in
     let padded = Utils.pad_center signal ~size:200 ~pad_value:0.0 in
     (* padded has 50 zeros, then 100 ones, then 50 zeros *)
   ]} *)

(** {2 Frequency and Scale Conversions} *)

module Convert : sig
  (** Frequency and amplitude scale conversion functions. *)

  val mel_to_hz : ?htk:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
  (** [mel_to_hz mel_values] converts mel-scale values to Hz.
      
      @param htk Use HTK mel scale formula (default: false, uses Slaney formula)
      @param mel_values Mel-scale frequency values
      @return Corresponding Hz frequency values
      
      {3 Example}
      
      {[
        let mel_freqs = Nx.linspace Nx.float32 0.0 2595.0 100 in
        let hz_freqs = Utils.Convert.mel_to_hz mel_freqs in
        (* Convert 100 mel frequencies to Hz *)
      ]} *)

  val hz_to_mel : ?htk:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
  (** [hz_to_mel hz_values] converts Hz frequency values to mel scale.
      
      @param htk Use HTK mel scale formula (default: false, uses Slaney formula)
      @param hz_values Hz frequency values
      @return Corresponding mel-scale frequency values
      
      This is the inverse function of {!mel_to_hz}. *)

  (** Reference value specification for dB conversions. *)
  type 'a reference =
    | RefFloat of float  (** Use a constant reference value. *)
    | RefFunction of ((float, 'a) Nx.t -> float)
        (** Compute reference from the input tensor (e.g., maximum value). *)

  val power_to_db :
       ?amin:float
    -> ?top_db:float
    -> 'a reference
    -> (float, 'a) Nx.t
    -> (float, 'a) Nx.t
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
          (RefFunction (fun x -> Nx.item [] (Nx.max x)))
          power_spec in
        (* Convert to dB relative to maximum value *)
      ]} *)

  val db_to_power :
    ?amin:float -> 'a reference -> (float, 'a) Nx.t -> (float, 'a) Nx.t
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
  -> (float, 'b) Nx.dtype
  -> (float, 'b) Nx.t
(** [melfreqs dtype] generates mel-scale frequency values.

   Computes a tensor of acoustic frequencies tuned to the mel scale, which
   approximates human auditory perception. This is used internally by
   mel filterbank functions but can also be useful for custom applications.

   @param n_mels Number of mel frequencies to generate (default: 128)
   @param f_min Minimum frequency in Hz (default: 0.0)
   @param f_max Maximum frequency in Hz (default: 11025.0)
   @param htk Use HTK mel scale formula (default: false)
   @param dtype Data type for the output tensor
   @return Tensor of mel-scale frequencies

   @raise Invalid_argument if n_mels <= 0
   @raise Invalid_argument if f_min < 0.0 or f_min >= f_max

   {3 Example}

   {[
     let mel_freqs = Utils.melfreqs 
       ~n_mels:80 
       ~f_min:80.0 
       ~f_max:7600.0
       Nx.float32 in
     (* 80 mel-scale frequencies from 80 Hz to 7600 Hz *)
   ]}

   See: {{:https://librosa.org/doc/main/generated/librosa.mel_frequencies.html}librosa.mel_frequencies} *)

(** {2 Mathematical Utilities} *)

val outer :
     (('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
  -> ('a, 'b) Nx.t
  -> ('a, 'b) Nx.t
  -> ('a, 'b) Nx.t
(** [outer_product operation x y] computes generalized outer product.

   Applies a binary operation between all pairs of elements from two tensors,
   creating a matrix where result[i,j] = operation(x[i], y[j]). This is useful
   for creating distance matrices, correlation matrices, and other pairwise operations.

   @param operation Binary operation to apply (e.g., Nx.add, Nx.mul)
   @param x First input tensor (1D)
   @param y Second input tensor (1D)
   @return 2D tensor with shape [length(x); length(y)]

   @raise Invalid_argument if x or y are not 1D

   {3 Example}

   {[
     let x = Nx.arange_f Nx.float32 1.0 4.0 1.0 in
     let y = Nx.arange_f Nx.float32 1.0 3.0 1.0 in
     let product = Utils.outer Nx.mul x y in
     (* Multiplication table: [[1,2], [2,4], [3,6]] *)
   ]} *)
