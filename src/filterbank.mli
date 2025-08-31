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

(** Mel-scale filterbank operations for audio processing.
    
    This module provides functions to create mel-scale filterbanks commonly
    used in audio feature extraction and speech processing. Mel filterbanks
    are essential components in many audio analysis pipelines, particularly
    for spectral feature extraction and machine learning applications. *)

(** Normalization methods for mel filterbanks. *)
type normalization =
  | Slaney
      (** Slaney normalization (default in librosa). Each filter is normalized
      to have unit area under the curve. *)
  | Power_norm of float
      (** L^p normalization with given power. For example, [Power_norm 2.0]
      applies L2 normalization to each filter. *)

val mel_filterbank :
     ?f_max:float
  -> ?htk:bool
  -> ?norm:normalization
  -> sample_rate:int
  -> n_fft:int
  -> n_mels:int
  -> f_min:float
  -> 'dev Rune.device
  -> (float, 'b) Rune.dtype
  -> (float, 'b, 'dev) Rune.t
(** [mel_filterbank dtype ~sample_rate ~n_fft ~n_mels ~f_min] creates a mel-scale filterbank.
    
    A mel filterbank is a collection of triangular filters spaced according to the
    mel scale, which approximates human auditory perception. Each filter has a
    triangular shape in the mel-frequency domain and is used to extract spectral
    features from audio signals.
    
    @param f_max Maximum frequency in Hz (default: sample_rate/2.0)
    @param htk Use HTK mel scale formula instead of Slaney formula (default: false)
    @param norm Normalization method to apply to filters (default: none)
    @param sample_rate Audio sample rate in Hz (must be positive)
    @param n_fft FFT size, determines frequency resolution (must be positive)
    @param n_mels Number of mel bands to generate (must be positive)
    @param f_min Minimum frequency in Hz (must be non-negative and < f_max)
    @param device Rune device to use for computation (default: automatic selection)
    @param dtype Data type for the filterbank (e.g., Rune.float32, Rune.float64)
    @return Mel filterbank matrix of shape [n_mels; n_fft/2 + 1]
    
    @raise Invalid_argument if sample_rate <= 0
    @raise Invalid_argument if n_fft <= 0  
    @raise Invalid_argument if n_mels <= 0
    @raise Invalid_argument if f_min < 0.0
    @raise Invalid_argument if f_min >= f_max (when f_max is specified)
    
    {3 Examples}
    
    Basic usage with automatic device selection:
    {[
      let filterbank = Filterbank.mel_filterbank 
        Rune.float32 
        ~sample_rate:22050 
        ~n_fft:1024 
        ~n_mels:128 
        ~f_min:0.0 in
      (* filterbank has shape [128; 513] *)
    ]}
    
    With explicit device and normalization:
    {[
      let device = Rune.metal () in
      let filterbank = Filterbank.mel_filterbank 
        ~norm:Slaney
        ~f_max:8000.0 
        ~sample_rate:16000 
        ~n_fft:512 
        ~n_mels:80 
        ~f_min:80.0
        device
        Rune.float64
      in
      (* filterbank has shape [80; 257] with Slaney normalization *)
    ]}
    
    Using HTK mel scale:
    {[
      let filterbank = Filterbank.mel_filterbank 
        ~htk:true
        ~sample_rate:44100 
        ~n_fft:2048 
        ~n_mels:40 
        ~f_min:20.0
        Rune.float32
        Rune.ocaml
      in
      (* filterbank uses HTK mel scale formula *)
    ]}
    
    {3 Mathematical Background}
    
    The mel scale is a perceptual scale of pitches judged by listeners to be
    equal in distance from one another. The mel filterbank creates triangular
    filters that are linearly spaced in the mel domain but logarithmically
    spaced in the frequency domain.
    
    The conversion between Hz and mel depends on the formula used:
    - Slaney formula (default): mel = 2595 * log10(1 + f/700)
    - HTK formula: mel = 2595 * log10(1 + f/700) with different constants
    
    {3 Performance Notes}
    
    - Use Metal device for best performance on Apple Silicon
    - Use C device for good performance on other platforms  
    - Float32 is usually sufficient precision and faster than Float64
    - Larger n_fft values provide better frequency resolution but are slower *)
