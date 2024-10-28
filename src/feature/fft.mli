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
    The {!Feature.Fft} module expose Fast Fourier Transform utility functions. *)

(**
    {1 The Fast Fourier Transform (FFT)}

    SoundML allow you to compute the FFT of an audio data in an efficient and compliant way.
    The FFTs functions are simple wrappers around the Owl library FFT functions, that are themselves
    wrappers around the FFTPack library. *)

(**
    Type of a Fourier Transform *)
type t = (Complex.t, Bigarray.complex32_elt) Audio.G.t

val forward : Audio.audio -> t
(**
    [forward audio] computes an FFT on the the given audio data.
    
    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let fft = forward src in
            (* ... *)
    ]} *)

val inverse : t -> (float, Bigarray.float32_elt) Audio.G.t
(**
    [inverse fft] computes the inverse FFT of the given FFT data.
    
    Example:

    {[
        let () =
            let src = read file.wav wav in
            let fft = forward src in
            let ifft = inverse fft in
            (* ... *)
    ]} *)
