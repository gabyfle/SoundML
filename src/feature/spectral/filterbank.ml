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

type norm = Slaney | PNorm of float

let mel ?(fmax : float option = None) ?(htk : bool = false)
    ?(norm : norm = Slaney) ~(sample_rate : int) ~(nfft : int) ~(nmels : int)
    ~fmin =
  let fmax =
    match fmax with Some fmax -> fmax | None -> float_of_int sample_rate /. 2.
  in
  let fftfreqs = Utils.rfftfreq nfft (1. /. float_of_int sample_rate) in
  let mel_freqs = Utils.melfreq ~nmels:(nmels + 2) ~fmin ~fmax ~htk in
  let fdiff = Audio.G.diff mel_freqs in
  let ramps = Utils.outer Audio.G.sub mel_freqs fftfreqs in
  let open Audio.G in
  let lower =
    neg ramps.${[0; Int.sub nmels 1]}
    / reshape fdiff.${[0; Int.sub nmels 1]} [|nmels; 1|]
  in
  let upper =
    ramps.${[2; Int.add nmels 1]} / reshape fdiff.${[1; nmels]} [|nmels; 1|]
  in
  (* Intersect slopes *)
  let weights =
    max2 (zeros Bigarray.Float32 (shape lower)) (min2 lower upper)
  in
  let weights =
    match norm with
    | Slaney ->
        let enorm =
          2.0
          $/ sub
               mel_freqs.${[2; Int.add nmels 1]}
               mel_freqs.${[0; Int.sub nmels 1]}
        in
        let enorm = reshape enorm [|nmels; 1|] in
        weights * enorm
    | PNorm p ->
        Audio.G.vecnorm ~p ~axis:(-1) weights
  in
  weights
[@@warning "-unerasable-optional-argument"]
