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

(** Normalization methods for mel filterbanks *)
type normalization = Slaney | Power_norm of float

let mel_filterbank ?f_max ?(htk : bool = false) ?norm ~sample_rate ~n_fft
    ~n_mels ~f_min (dtype : (float, 'b) Nx.dtype) =
  (* Input validation *)
  if sample_rate <= 0 then invalid_arg "sample_rate must be positive" ;
  if n_fft <= 0 then invalid_arg "n_fft must be positive" ;
  if n_mels <= 0 then invalid_arg "n_mels must be positive" ;
  if f_min < 0.0 then invalid_arg "f_min must be non-negative" ;
  if n_mels = 0 then Nx.empty dtype [|0; (n_fft / 2) + 1|]
  else
    let f_max =
      match f_max with
      | Some fmax ->
          if fmax <= f_min then invalid_arg "f_max must be greater than f_min" ;
          fmax
      | None ->
          float_of_int sample_rate /. 2.
    in
    (* Check that f_min < sample_rate/2 *)
    assert (f_min < float_of_int sample_rate /. 2.0) ;
    (* Create fftfreqs manually to ensure correct dtype *)
    let d = 1. /. float_of_int sample_rate in
    let n_freqs = (n_fft / 2) + 1 in
    let freq_step = 1.0 /. (float_of_int n_fft *. d) in
    let fftfreqs =
      let indices = Array.init n_freqs (fun i -> float_of_int i *. freq_step) in
      Nx.create dtype [|n_freqs|] indices
    in
    let mel_freqs =
      Utils.melfreqs ~n_mels:(n_mels + 2) ~f_min ~f_max ~htk dtype
    in
    let fdiff =
      let n = Nx.numel mel_freqs in
      Nx.sub (Nx.slice [R (1, n)] mel_freqs) (Nx.slice [R (0, n - 1)] mel_freqs)
    in
    let ramps = Utils.outer Nx.sub mel_freqs fftfreqs in
    let lower =
      Nx.div
        (Nx.neg (Nx.slice [R (0, n_mels)] ramps))
        (Nx.reshape [|n_mels; 1|] (Nx.slice [R (0, n_mels)] fdiff))
    in
    let upper =
      Nx.div
        (Nx.slice [R (2, n_mels + 2)] ramps)
        (Nx.reshape [|n_mels; 1|] (Nx.slice [R (1, n_mels + 1)] fdiff))
    in
    (* Intersect slopes *)
    let weights = Nx.maximum (Nx.zeros_like lower) (Nx.minimum lower upper) in
    let weights =
      match norm with
      | Some Slaney ->
          let enorm =
            Nx.div (Nx.scalar dtype 2.0)
              (Nx.sub
                 (Nx.slice [R (2, n_mels + 2)] mel_freqs)
                 (Nx.slice [R (0, n_mels)] mel_freqs) )
          in
          let enorm = Nx.reshape [|n_mels; 1|] enorm in
          Nx.mul weights enorm
      | Some (Power_norm p) ->
          let norm =
            Nx.pow_s
              (Nx.sum ~axes:[-1] ~keepdims:true (Nx.pow_s (Nx.abs weights) p))
              (1. /. p)
          in
          Nx.div weights (Nx.add_s norm 1e-8)
      | None ->
          weights
    in
    weights
