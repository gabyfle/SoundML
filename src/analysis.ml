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
open Audio

let fft (a : audio) : (Complex.t, Bigarray.complex64_elt) G.t =
  Owl.Fft.D.rfft (data a)

let ifft (ft : (Complex.t, Bigarray.complex64_elt) G.t) :
    (float, Bigarray.float64_elt) G.t =
  Owl.Fft.D.irfft ft

let fftfreq (a : audio) =
  let meta = meta a in
  let size = rawsize a in
  let sampling = Metadata.sample_rate meta in
  let n = float_of_int size in
  let d = 1. /. float_of_int sampling in
  let nslice = ((int_of_float n - 1) / 2) + 1 in
  let fhalf = Arr.linspace 0. (float_of_int nslice) nslice in
  let shalf = Arr.linspace (-.float_of_int nslice) (-1.) nslice in
  let v = Arr.concatenate ~axis:0 [|fhalf; shalf|] in
  Arr.(1. /. (d *. n) $* v)

let spectrogram ?(window = Owl.Signal.hamming) ?(nfft = 2048)
    ?(window_size = None) (a : audio) (noverlap : int) =
  let window_size = match window_size with Some x -> x | None -> nfft in
  let window = window window_size in
  let hop_size = window_size - noverlap in
  let nframes = ((rawsize a - window_size) / hop_size) + 1 in
  let spectrogram = G.zeros Bigarray.Complex64 [|(nfft / 2) + 1; nframes|] in
  for i = 0 to nframes - 1 do
    let start = i * hop_size in
    let stop = start + window_size in
    let frame = G.get_slice [[start; stop - 1]] (data a) in
    G.(frame *= window) ;
    let ft = Owl.Fft.D.rfft frame in
    let ft = G.reshape ft [|(G.shape ft).(0); 1|] in
    G.set_slice [[]; [i]] spectrogram ft
  done ;
  spectrogram
