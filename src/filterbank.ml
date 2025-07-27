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

type norm = Slaney | PNorm of float

let mel ?(fmax : float option = None) ?(htk : bool = false)
    ?(norm : norm option = None) (dtype : ('a, 'b) Nx.dtype) (sample_rate : int)
    (nfft : int) (nmels : int) (fmin : float) =
  if nmels = 0 then Nx.empty dtype [|0; (nfft / 2) + 1|]
  else
    let fmax =
      match fmax with
      | Some fmax ->
          fmax
      | None ->
          float_of_int sample_rate /. 2.
    in
    let fftfreqs = Nx.rfftfreq ~d:(1. /. float_of_int sample_rate) nfft in
    let mel_freqs = Utils.melfreq dtype ~nmels:(nmels + 2) ~fmin ~fmax ~htk in
    let fdiff =
      let n = Nx.size mel_freqs in
      Nx.sub (Nx.slice [R [1; n]] mel_freqs) (Nx.slice [R [0; n - 1]] mel_freqs)
    in
    let ramps = Utils.outer Nx.sub mel_freqs fftfreqs in
    let lower =
      Nx.div
        (Nx.neg (Nx.slice [R [0; nmels]] ramps))
        (Nx.reshape [|nmels; 1|] (Nx.slice [R [0; nmels]] fdiff))
    in
    let upper =
      Nx.div
        (Nx.slice [R [2; nmels + 2]] ramps)
        (Nx.reshape [|nmels; 1|] (Nx.slice [R [1; nmels + 1]] fdiff))
    in
    (* Intersect slopes *)
    let weights = Nx.maximum (Nx.zeros_like lower) (Nx.minimum lower upper) in
    let weights =
      match norm with
      | Some Slaney ->
          let enorm =
            Nx.rdiv_s 2.0
              (Nx.sub
                 (Nx.slice [R [2; nmels + 2]] mel_freqs)
                 (Nx.slice [R [0; nmels]] mel_freqs) )
          in
          let enorm = Nx.reshape [|nmels; 1|] enorm in
          Nx.mul weights enorm
      | Some (PNorm p) ->
          let norm =
            Nx.pow_s
              (Nx.sum ~axes:[|-1|] ~keepdims:true (Nx.pow_s (Nx.abs weights) p))
              (1. /. p)
          in
          Nx.div weights (Nx.add_s norm 1e-8)
      | None ->
          weights
    in
    weights
