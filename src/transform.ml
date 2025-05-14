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
  type t = {n_fft: int; hop_size: int; win_length: int; center: bool}

  let default = {n_fft= 2048; hop_size= 512; win_length= 2048; center= true}
end

module G = Owl.Dense.Ndarray.Generic

let stft : type a b.
    ?config:Config.t -> (a, b) precision -> (float, a) G.t -> (Complex.t, b) G.t
    =
 fun ?(config : Config.t = Config.default) p (x : (float, a) G.t) ->
  let kd : (Complex.t, b) kind =
    match p with B32 -> Complex32 | B64 -> Complex64
  in
  let signal_length = Float.of_int @@ Audio.G.numel x in
  let m =
    Float.to_int
      (Float.ceil
         ( signal_length
         -. (Float.of_int config.win_length /. Float.of_int config.hop_size) ) )
    + 1
  in
  let spectrum = Audio.G.create kd [|m; config.n_fft|] Complex.zero in
  spectrum
