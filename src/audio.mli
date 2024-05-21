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
    High level representation of an audio file data, used to store data when reading audio files *)
type audio

val data :
  audio -> (Complex.t, Bigarray.complex32_elt) Owl.Dense.Ndarray.Generic.t
(**
    [data audio] returns the data of the given audio data element *)

val size : audio -> int
(**
    [size audio] returns the size (in bytes) of the given audio data element *)

val sampling : audio -> int
(**
    [sampling audio] returns the sampling rate of the given audio data element *)

val read_audio : ?channels:Avutil.Channel_layout.t -> string -> string -> audio
(**
    [read_audio ~channel filename format] reads an audio file returns a representation of the file.
    
    Example usage:
    
    {[
    let () =
        let src = read_audio file.wav wav in
        (* ... *)
    ]}

    you can as well choose to have a stereo representation of the file

    {[
    let () =
        let src = read_audio `Stereo file.wav wav in
        (* ... *)
    ]} *)

val fft :
     audio
  -> int
  -> int
  -> (Complex.t, Bigarray.complex32_elt) Owl.Dense.Ndarray.Generic.t
(**
    [fft audio start finish] computes an FFT on the slice [start; finish] of the given audio data.
    
    Examples:

    {[
        let () =
            let src = read_audio file.wav wav in
            let fft = fft src 0 1024 in
            (* ... *)
    ]} *)

val fftfreq : audio -> (float, Bigarray.float64_elt) Owl.Dense.Ndarray.Generic.t
(**
    [fftfreq audio] return the FT sample frequencies.
    
    Inspired from: np.fft.fft.
    @see <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html> Numpy Documentation *)
