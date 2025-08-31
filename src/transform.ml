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

let stft ?(window = `Hanning) ?(win_length = 2048) ?(center = true) ~n_fft
    ~hop_length (x : (float, 'a, 'dev) Rune.t) =
  (* Input validation *)
  if n_fft <= 0 then invalid_arg "n_fft must be positive" ;
  if hop_length <= 0 then invalid_arg "hop_size must be positive" ;
  if win_length <= 0 then invalid_arg "win_length must be positive" ;
  if win_length > n_fft then
    invalid_arg "win_length cannot be larger than n_fft" ;
  let device = Rune.device x in
  let fft_window =
    Window.get window device (Rune.dtype x) ~fftbins:true win_length
  in
  let fft_window =
    if win_length < n_fft then
      Utils.pad_center fft_window ~size:n_fft ~pad_value:0.0
    else fft_window
  in
  let x_padded =
    if center then (
      let x_shape = Rune.shape x in
      let pad_width = n_fft / 2 in
      let padding = Array.make (Array.length x_shape) (0, 0) in
      padding.(Array.length x_shape - 1) <- (pad_width, pad_width) ;
      Rune.pad padding 0.0 x )
    else x
  in
  let padded_shape = Rune.shape x_padded in
  let signal_length = padded_shape.(Array.length padded_shape - 1) in
  if n_fft > signal_length then
    invalid_arg
      (Printf.sprintf "n_fft=%d is too large for input signal of length=%d"
         n_fft signal_length ) ;
  let y_frames =
    Utils.frame ~axis:(-1) ~frame_length:n_fft ~hop_length x_padded
  in
  let frames_shape = Rune.shape y_frames in
  let frames_ndim = Array.length frames_shape in
  let window_shape = Array.make frames_ndim 1 in
  window_shape.(frames_ndim - 2) <- n_fft ;
  let fft_window_reshaped = Rune.reshape window_shape fft_window in
  let windowed_frames = Rune.mul y_frames fft_window_reshaped in
  let stft_result = Rune.rfft ~axis:(-2) windowed_frames in
  stft_result
