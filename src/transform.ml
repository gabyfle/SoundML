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

open Bigarray
open Types

module Config = struct
  type t =
    { n_fft: int
    ; hop_size: int
    ; win_length: int
    ; window: Window.window
    ; center: bool }

  let default =
    { n_fft= 2048
    ; hop_size= 512
    ; win_length= 2048
    ; window= `Hanning
    ; center= true }
end

module G = Owl.Dense.Ndarray.Generic

let to_complex (x : float) : Complex.t = Complex.{re= x; im= 0.}

let stft : type a b.
    ?config:Config.t -> (a, b) precision -> (float, a) G.t -> (Complex.t, b) G.t
    =
 fun ?(config : Config.t = Config.default) p (x : (float, a) G.t) ->
  let kd : (Complex.t, b) kind =
    match p with B32 -> Complex32 | B64 -> Complex64
  in
  let window = (Window.get config.window p ~fftbins:true) config.win_length in
  let out_shape =
    [| (config.n_fft / 2) + 1
     ; ((G.numel x - config.win_length) / config.hop_size) + 1 |]
  in
  let spectrum = Audio.G.create kd out_shape Complex.zero in
  let ym = Audio.G.zeros kd [|config.n_fft; 1|] in
  for m = 0 to out_shape.(1) - 1 do
    Audio.G.fill ym Complex.zero ;
    for p = 0 to config.win_length - 1 do
      Audio.G.(
        ym.%{p; 0} <-
          to_complex
          @@ Float.mul
               (get x [|Int.(add p (mul m config.hop_size))|])
               window.%{p} )
    done ;
    let ym_fft = Owl.Fft.Generic.fft ~axis:0 ym in
    let spectrum_slice =
      Audio.G.get_slice [[0; out_shape.(0) - 1]; [0]] ym_fft
    in
    Audio.G.set_slice_ ~out:spectrum [[]; [m]] spectrum spectrum_slice
  done ;
  spectrum
