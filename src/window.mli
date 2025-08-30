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

(** Window functions for audio signal processing.
    
    This module provides commonly used window functions for spectral analysis,
    particularly for Short-Time Fourier Transform (STFT) and other windowed
    operations. Window functions reduce spectral leakage by tapering the signal
    at frame boundaries. *)

(** Window function types supported by SoundML. *)
type window =
  [ `Hanning
    (** Hann window (raised cosine) - good general-purpose window with moderate
      spectral leakage and good frequency resolution. Recommended for most applications. *)
  | `Hamming
    (** Hamming window - similar to Hann but with different coefficients,
      slightly better sidelobe suppression. *)
  | `Blackman
    (** Blackman window - excellent sidelobe suppression but wider main lobe,
      good for applications requiring low spectral leakage. *)
  | `Boxcar
    (** Rectangular window (no tapering) - best frequency resolution but
      high spectral leakage. Use only when necessary. *)
  ]

val get :
     window
  -> 'dev Rune.device
  -> (float, 'b) Rune.dtype
  -> ?fftbins:bool
  -> int
  -> (float, 'b, 'dev) Rune.t
(** [get window dtype n] generates a window of size [n] using the specified window function.

    Creates a window function suitable for spectral analysis applications.
    The window is normalized and ready to use with STFT or other windowed operations.

    @param window Window function type to generate
    @param device Device on which the resulting Tensor should be created
    @param dtype Data type for the window values (e.g., Rune.float32, Rune.float64)
    @param fftbins Whether to use FFT-compatible window length (default: true)
    @param n Window size in samples (must be > 0)
    @return Window function tensor of length n
    
    @raise Invalid_argument if n <= 0
    
    {3 Examples}
    
    Basic window generation:
    {[
      let hann_window = Window.get `Hanning Rune.float32 1024 in
      (* 1024-sample Hann window *)
    ]}
    
    Window for STFT:
    {[
      let window = Window.get `Hamming Rune.float64 512 in
      (* Use with Transform.stft ~window:(Custom_window window) *)
    ]}
    
    {3 Window Selection Guidelines}
    
    - Hann: Best general-purpose window, good balance of properties
    - Hamming: Similar to Hann, slightly better sidelobe suppression  
    - Blackman: Use when you need very low spectral leakage
    - Boxcar: Use only when you need maximum frequency resolution *)

val cosine_sum :
     ?fftbins:bool
  -> 'dev Rune.device
  -> (float, 'b) Rune.dtype
  -> float array
  -> int
  -> (float, 'b, 'dev) Rune.t
(** [cosine_sum dtype coeffs n] generates a generalized cosine-sum window.

    Creates a window function as a weighted sum of cosine terms. This is the
    general form that includes Hann, Hamming, Blackman, and other cosine-based
    windows as special cases. Useful for creating custom window functions.

    @param fftbins Whether to use FFT-compatible window (default: true)
    @param device Device on which the resulting Tensor should be created
    @param dtype Data type for window values (e.g., Rune.float32, Rune.float64)
    @param coeffs Coefficients for the cosine terms (must have length >= 1)
    @param n Window size in samples (must be > 0)
    @return Cosine-sum window tensor of length n

    @raise Invalid_argument if n <= 0
    @raise Invalid_argument if coeffs array is empty

    {3 Examples}

    Create a Hann window using cosine_sum:
    {[
      let hann_coeffs = [|0.5; -0.5|] in
      let hann_window = Window.cosine_sum Rune.float32 hann_coeffs 1024 in
      (* Equivalent to Window.hanning Rune.float32 1024 *)
    ]}

    Create a custom window:
    {[
      let custom_coeffs = [|0.4; -0.4; 0.2|] in
      let custom_window = Window.cosine_sum Rune.float32 custom_coeffs 512 in
      (* Custom 3-term cosine window *)
    ]}

    Mathematical definition: w(n) = Σ(k=0 to K-1) coeffs[k] * cos(2π*k*n/(N-1))

    See: {{:https://en.wikipedia.org/wiki/Window_function#Cosine-sum_windows}Wikipedia: Cosine-sum windows}
    See: {{:https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_cosine.html}SciPy general_cosine} *)

val hanning :
     ?fftbins:bool
  -> 'dev Rune.device
  -> (float, 'b) Rune.dtype
  -> int
  -> (float, 'b, 'dev) Rune.t
(** [hanning dtype n] generates a Hann window of size [n].

    The Hann window (also called Hanning window) is a raised cosine window
    that provides good general-purpose characteristics for spectral analysis.
    It offers a good balance between frequency resolution and spectral leakage.

    @param fftbins Whether to use FFT-compatible window (default: true)
    @param device Device on which the resulting Tensor should be created
    @param dtype Data type for window values (e.g., Rune.float32, Rune.float64)
    @param n Window size in samples (must be > 0)
    @return Hann window tensor of length n
    
    @raise Invalid_argument if n <= 0
    
    {3 Example}
    
    {[
      let window = Window.hanning Rune.float32 1024 in
      (* Use with STFT: Transform.stft ~window:(Custom_window window) signal *)
    ]}
    
    Mathematical definition: w(n) = 0.5 * (1 - cos(2π*n/(N-1))) *)

val hamming :
     ?fftbins:bool
  -> 'dev Rune.device
  -> (float, 'b) Rune.dtype
  -> int
  -> (float, 'b, 'dev) Rune.t
(** [hamming dtype n] generates a Hamming window of size [n].

    The Hamming window is similar to the Hann window but uses different
    coefficients that provide slightly better sidelobe suppression at
    the cost of a slightly wider main lobe.

    @param fftbins Whether to use FFT-compatible window (default: true)
    @param device Device on which the resulting Tensor should be created
    @param dtype Data type for window values (e.g., Rune.float32, Rune.float64)
    @param n Window size in samples (must be > 0)
    @return Hamming window tensor of length n
    
    @raise Invalid_argument if n <= 0
    
    Mathematical definition: w(n) = 0.54 - 0.46 * cos(2π*n/(N-1)) *)

val blackman :
     ?fftbins:bool
  -> 'dev Rune.device
  -> (float, 'b) Rune.dtype
  -> int
  -> (float, 'b, 'dev) Rune.t
(** [blackman dtype n] generates a Blackman window of size [n].

    The Blackman window provides excellent sidelobe suppression (very low
    spectral leakage) but has a wider main lobe than Hann or Hamming windows.
    Use when you need minimal spectral leakage.

    @param fftbins Whether to use FFT-compatible window (default: true)
    @param device Device on which the resulting Tensor should be created
    @param dtype Data type for window values (e.g., Rune.float32, Rune.float64)
    @param n Window size in samples (must be > 0)
    @return Blackman window tensor of length n
    
    @raise Invalid_argument if n <= 0
    
    Mathematical definition: w(n) = 0.42 - 0.5*cos(2π*n/(N-1)) + 0.08*cos(4π*n/(N-1)) *)

val boxcar :
  'dev Rune.device -> ('a, 'b) Rune.dtype -> int -> ('a, 'b, 'dev) Rune.t
(** [boxcar device dtype n] generates a rectangular (boxcar) window of size [n].

    The boxcar window is simply a rectangular window (all ones) that provides
    the best frequency resolution but the worst spectral leakage. Use only
    when you specifically need maximum frequency resolution.

    @param device Device on which the resulting Tensor should be created
    @param dtype Data type for window values
    @param n Window size in samples (must be > 0)
    @return Rectangular window tensor of length n (all ones)
    
    @raise Invalid_argument if n <= 0
    
    Mathematical definition: w(n) = 1 for all n *)
