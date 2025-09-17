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
  if hop_length <= 0 then invalid_arg "hop_length must be positive" ;
  if win_length <= 0 then invalid_arg "win_length must be positive" ;
  if win_length > n_fft then
    invalid_arg "win_length cannot be larger than n_fft" ;
  let device = Rune.device x in
  let dtype = Rune.dtype x in
  (* 1. Create window and pad it if necessary to n_fft *)
  let fft_window = Window.get window device dtype ~fftbins:true win_length in
  let fft_window =
    if win_length < n_fft then
      let pad_width = (n_fft - win_length) / 2 in
      Rune.pad [|(pad_width, n_fft - win_length - pad_width)|] 0.0 fft_window
    else fft_window
  in
  (* 2. Pad the input signal for centering *)
  let x_padded =
    if center then (
      let pad_width = n_fft / 2 in
      let rank = Rune.ndim x in
      let padding = Array.make rank (0, 0) in
      padding.(rank - 1) <- (pad_width, pad_width) ;
      Rune.pad padding 0.0 x )
    else x
  in
  let padded_shape = Rune.shape x_padded in
  let signal_length = padded_shape.(Array.length padded_shape - 1) in
  if n_fft > signal_length then
    invalid_arg
      (Printf.sprintf "n_fft=%d is too large for input signal of length=%d"
         n_fft signal_length ) ;
  (* 3. Calculate number of frames *)
  let num_frames = 1 + ((signal_length - n_fft) / hop_length) in
  let frame_indices = Rune.arange device Rune.int32 0 num_frames 1 in
  (* 4. Define the function to be vmapped *)
  let process_single_frame frame_index =
    let start = Rune.mul_s frame_index (Int32.of_int hop_length) in
    let start_reshaped = Rune.unsqueeze ~axes:[|1|] start in
    let offsets = Rune.arange device Rune.int32 0 n_fft 1 in
    let indices = Rune.add start_reshaped offsets in
    (* Dynamic slice using take. Assuming take flattens the taken dimensions. *)
    let frame_flat = Rune.take ~axis:(-1) indices x_padded in
    (* Reshape back to (..., num_frames, n_fft) using -1 for num_frames *)
    let x_rank = Rune.ndim x_padded in
    let other_dims = Array.sub (Rune.shape x_padded) 0 (x_rank - 1) in
    let target_shape = Array.append other_dims [|-1; n_fft|] in
    let frame = Rune.reshape target_shape frame_flat in
    (* Apply window. *)
    let windowed_frame = Rune.mul frame fft_window in
    (* FFT on the last axis *)
    Rune.rfft ~axis:(-1) windowed_frame
  in
  (* 5. vmap the function over the frame indices *)
  let stft_matrix = Rune.vmap process_single_frame frame_indices in
  (* 6. Transpose to (..., freq, time) by moving the first axis (frames) to the
     end *)
  let rank = Rune.ndim stft_matrix in
  let axes = Array.init rank (fun i -> if i = rank - 1 then 0 else i + 1) in
  let stft_transposed = Rune.transpose ~axes stft_matrix in
  stft_transposed
