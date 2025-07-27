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

open Soundml
open Vutils

let string_to_window = function
  | "hann" ->
      `Hanning
  | "hamming" ->
      `Hamming
  | "blackman" ->
      `Blackman
  | "boxcar" ->
      `Boxcar
  | _ ->
      failwith "Unknown window type"

module StftTestable = struct
  type a = Complex.t

  type b = Nx.complex64_elt

  let dtype = Nx.Complex64

  let typ = "stft"

  let generate (_ : (a, b) Nx.dtype) (case : string * string * Parameters.t)
      (audio : (float, Bigarray.float64_elt) Nx.t) =
    let _, _, params = case in
    let n_fft =
      Option.value ~default:2048 @@ Parameters.get_int "n_fft" params
    in
    let hop_size =
      Option.value ~default:512 @@ Parameters.get_int "hop_size" params
    in
    let win_length =
      Option.value ~default:2048 @@ Parameters.get_int "window_length" params
    in
    let window =
      string_to_window
        (Option.value ~default:"hann" @@ Parameters.get_string "window" params)
    in
    let center =
      Option.value ~default:false @@ Parameters.get_bool "center" params
    in
    let stft =
      Transform.stft ~n_fft ~hop_size ~win_length ~window ~center audio
    in
    stft
end

module Tests = Tests_cases (StftTestable)

let () =
  let name = "Vectors: STFT Comparison" in
  let data = Testdata.get StftTestable.typ Vutils.data in
  let tests = Tests.create_tests data 1e-5 1e-8 in
  Tests.run name tests
