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

let stft ?(window = `Hanning) ?win_length ?(center = true) ~n_fft ~hop_length
    (x : (float, 'a) Nx.t) =
  let win_length = Option.value win_length ~default:n_fft in
  (* Input validation *)
  if n_fft <= 0 then invalid_arg "n_fft must be positive" ;
  if hop_length <= 0 then invalid_arg "hop_length must be positive" ;
  if win_length <= 0 then invalid_arg "win_length must be positive" ;
  if win_length > n_fft then
    invalid_arg "win_length cannot be larger than n_fft" ;
  let dtype = Nx.dtype x in
  (* 1. Create window and pad it if necessary to n_fft *)
  let fft_window = Window.get window dtype ~fftbins:true win_length in
  let fft_window =
    if win_length < n_fft then
      let pad_width = (n_fft - win_length) / 2 in
      Nx.pad [|(pad_width, n_fft - win_length - pad_width)|] 0.0 fft_window
    else fft_window
  in
  (* 2. Pad the input signal for centering *)
  let x_padded =
    if center then (
      let pad_width = n_fft / 2 in
      let rank = Nx.ndim x in
      let padding = Array.make rank (0, 0) in
      padding.(rank - 1) <- (pad_width, pad_width) ;
      Nx.pad padding 0.0 x )
    else x
  in
  let padded_shape = Nx.shape x_padded in
  let signal_length = padded_shape.(Array.length padded_shape - 1) in
  if n_fft > signal_length then
    invalid_arg
      (Printf.sprintf "n_fft=%d is too large for input signal of length=%d"
         n_fft signal_length ) ;
  let frames =
    Nx.extract_patches ~kernel_size:[|n_fft|] ~stride:[|hop_length|]
      ~dilation:[|1|]
      ~padding:[|(0, 0)|]
      x_padded
  in
  let fft_window = Nx.reshape [|n_fft; 1|] fft_window in
  Nx.rfft ~axis:(-2) (Nx.mul frames fft_window)
