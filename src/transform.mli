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

(** Time-frequency transforms for audio analysis.
    
    This module provides Short-Time Fourier Transform (STFT) and related
    operations for converting between time and frequency domain representations.
    These transforms are fundamental tools in audio signal processing, enabling
    spectral analysis, feature extraction, and time-frequency visualization. *)

val stft :
     ?window:Window.window
  -> ?win_length:int
  -> ?center:bool
  -> n_fft:int
  -> hop_length:int
  -> (float, 'a, 'dev) Rune.t
  -> (Complex.t, Bigarray_ext.complex64_elt, 'dev) Rune.t
(** [stft ~n_fft ~hop_length x] computes the Short-Time Fourier Transform.
    
    The STFT transforms a time-domain signal into a time-frequency representation
    by applying the Fourier transform to overlapping windowed segments of the signal.
    This allows analysis of how the frequency content of a signal changes over time.
    
    @param window Window function specification (default: Hann window)
    @param win_length Length of the window in samples (default: n_fft)
    @param center Whether to center the signal by padding (default: true)
    @param n_fft FFT size, determines frequency resolution (must be positive)
    @param hop_length Number of samples between successive frames (must be positive)
    @param x Input audio signal (1D tensor)
    @return Complex STFT matrix of shape [n_fft/2 + 1; n_frames]
    
    @raise Invalid_argument if n_fft <= 0
    @raise Invalid_argument if hop_length <= 0
    @raise Invalid_argument if win_length > n_fft (when specified)
    @raise Invalid_argument if signal is not 1D
    
    {3 Examples}
    
    Basic STFT with default parameters:
    {[
      let signal = (* load audio signal *) in
      let stft_result = Transform.stft 
        signal 
        ~n_fft:1024 
        ~hop_length:512 in
      (* stft_result has shape [513; n_frames] *)
    ]}
    
    {3 Mathematical Background}
    
    The STFT is defined as:
    X(m,k) = Σ x(n) * w(n-m*H) * exp(-j*2π*k*n/N)
    
    Where:
    - x(n) is the input signal
    - w(n) is the window function
    - m is the frame index
    - k is the frequency bin index  
    - H is the hop length
    - N is the FFT size
    
    {3 Parameter Guidelines}
    
    - n_fft: Larger values give better frequency resolution but worse time resolution
    - hop_length: Smaller values give better time resolution but more computation
    - Common ratio: hop_length = n_fft/4 (75% overlap) or n_fft/2 (50% overlap)
    - win_length: Usually equal to n_fft, can be smaller for zero-padding
    
    {3 Performance Notes}
    
    - Use Metal device for best performance on Apple Silicon
    - Powers of 2 for n_fft are most efficient (e.g., 512, 1024, 2048)
    - Larger hop_length reduces computation but may miss transient events
    - Consider using float32 input for better performance *)
