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

open Owl

let _magnitudes = List.map (fun c -> Complex.norm c)

let () =
  let audio = Audio.read_audio "sin_1k.wav" "wav" in
  let fft = Audio.fft audio 0 4098 in
  Log.warn "FFT: %d" (Audio.size audio) ;
  let fft =
    fft
    |> Dense.Ndarray.Generic.get_slice [[0; Audio.size audio / 2]]
    |> Dense.Ndarray.Generic.to_array |> Array.to_list
  in
  Log.info "FFT: %d" (List.length fft) ;
  let x =
    Audio.fftfreq audio
    |> Dense.Ndarray.Generic.get_slice [[0; Audio.size audio / 2]]
    |> Dense.Ndarray.Generic.to_array |> Array.to_list
  in
  Log.info "X: %d" (List.length x)
