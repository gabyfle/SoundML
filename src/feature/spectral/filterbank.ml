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

let mel ?(fmax : float option = None) ?(htk : bool = false) ~(sample_rate : int)
    ~(nfft : int) ~(nmels : int) ~fmin =
  let fmax =
    match fmax with Some fmax -> fmax | None -> float_of_int sample_rate /. 2.
  in
  let weights = Audio.G.zeros Bigarray.Float32 [|nmels; 1 + (nfft / 2)|] in
  let fftfreqs = Utils.fftfreq nfft (1. /. float_of_int sample_rate) in
  let mel_freqs = Utils.Convert.mel_freqs ~nmels:(nmels + 2) ~fmin ~fmax ~htk in
  let fdiff = Audio.G.diff mel_freqs in
  (* TODO: add outer product substract *)
  ()
[@@warning "-unerasable-optional-argument"] [@@warning "-unused-var"]
