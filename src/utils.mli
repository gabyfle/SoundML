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
   Utility conversion module. *)
module Convert : sig
  val mel_to_hz : ?htk:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
  (** Converts mel-scale values to Hz. *)

  val hz_to_mel : ?htk:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
  (** Reverse function of {!mel_to_hz}. *)

  type reference =
    | RefFloat of float
    | RefFunction of ((float, Bigarray.float32_elt) Nx.t -> float)

  val power_to_db :
       ?amin:float
    -> ?top_db:float option
    -> reference
    -> (float, Bigarray.float32_elt) Nx.t
    -> (float, Bigarray.float32_elt) Nx.t

  val db_to_power :
       ?amin:float
    -> reference
    -> (float, Bigarray.float32_elt) Nx.t
    -> (float, Bigarray.float32_elt) Nx.t
end

val pad_center : ('a, 'b) Nx.t -> int -> 'a -> ('a, 'b) Nx.t
(**
    Pads a ndarray such that *)

val melfreq :
     ?nmels:int
  -> ?fmin:float
  -> ?fmax:float
  -> ?htk:bool
  -> (float, 'b) Nx.dtype
  -> (float, 'b) Nx.t
(**
  Implementation of librosa's mel_frequencies. Compute an [Nx.t] of acoustic frequencies tuned to the mel scale.
  See: {{:https://librosa.org/doc/main/generated/librosa.mel_frequencies.html}librosa.mel_frequencies} for more information. *)

val unwrap :
     ?discont:float option
  -> ?axis:int
  -> ?period:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(**
    Implementation of the Numpy's unwrap function.
    See {{:https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html}numpy.unwrap} for more information. *)

val outer :
     (('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t)
  -> ('a, 'b) Nx.t
  -> ('a, 'b) Nx.t
  -> ('a, 'b) Nx.t
(**
  Generalized outer product of any given operator that supports broadcasting (basically all the common Nx operators.) *)

val frame :
     ?axis:int
  -> ('a, 'b) Nx.t
  -> frame_length:int
  -> hop_length:int
  -> ('a, 'b) Nx.t
(**
   [frame ?axis x ~frame_length ~hop_length] slices a data array into overlapping frames.

   This implementation uses low-level stride manipulation to avoid making a copy of the data.
   The resulting frame representation is a new view of the same input data, providing maximum
   performance for audio processing applications.

   The framing operation increments by 1 the number of dimensions, adding a new "frame axis"
   either before the framing axis (if [axis < 0]) or after the framing axis (if [axis >= 0]).

   @param axis The axis along which to frame. Default is [-1] (last axis).
   @param x Array to frame
   @param frame_length Length of each frame (must be > 0)
   @param hop_length Number of steps to advance between frames (must be >= 1)

   @return A framed view of [x]. For example, with [axis=-1] (framing on the last dimension):
           [x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]]
           
           If [axis=0] (framing on the first dimension), then:
           [x_frames[j] = x[j * hop_length : j * hop_length + frame_length]]

   @raise Invalid_argument if [x.shape[axis] < frame_length] (not enough data to fill one frame)
   @raise Invalid_argument if [hop_length < 1] (frames cannot advance)
   @raise Invalid_argument if [frame_length <= 0] (invalid frame length)
   @raise Invalid_argument if [axis] is out of bounds

   {3 Examples}

   Extract overlapping frames from a 1D signal:
   {[
     let signal = Nx.arange Nx.float32 0 10 1 in
     let frames = frame signal ~frame_length:3 ~hop_length:2 in
     (* frames will have shape [3; 4] containing:
        [[0, 2, 4, 6],
         [1, 3, 5, 7], 
         [2, 4, 6, 8]] *)
   ]}

   Frame along the first axis instead:
   {[
     let signal = Nx.arange Nx.float32 0 10 1 in
     let frames = frame ~axis:0 signal ~frame_length:3 ~hop_length:2 in
     (* frames will have shape [4; 3] containing:
        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6],
         [6, 7, 8]] *)
   ]}

   Frame a stereo signal (2D input):
   {[
     let stereo = Nx.ones Nx.float32 [|2; 1000|] in
     let frames = frame stereo ~frame_length:512 ~hop_length:256 in
     (* frames will have shape [2; 512; 3] *)
   ]}

   This function is particularly useful for:
   - Short-Time Fourier Transform (STFT) preprocessing
   - Creating overlapping windows for spectral analysis
   - Preparing audio data for neural network training
   - Any sliding window operation on multi-dimensional arrays

   {3 Performance Notes}

   - This function creates a view of the original data, not a copy
   - Memory usage is O(1) regardless of frame size or hop length
   - The returned tensor shares memory with the input tensor
   - Modifications to the framed view will affect the original data

   See also: {{:https://librosa.org/doc/main/generated/librosa.util.frame.html}librosa.util.frame}
   for the equivalent Python implementation.
*)
