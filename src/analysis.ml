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

let fft ?(norm = true) (a : audio) : (Complex.t, Bigarray.complex64_elt) G.t =
  let nsplit = Domain.recommended_domain_count () in
  let norm = if norm then Fun.id else normalise in
  let data = data (norm a) in
  let shape = (G.shape data).(0) in
  let data = data |> G.cast_d2z in
  (*we're playing with 1-D arrays*)
  let n = shape / nsplit in
  let slices =
    Array.init nsplit (fun i ->
        let start = i * n in
        let finish = min (start + n) shape in
        G.get_slice [[start; finish - 1]] data )
  in
  let fft_slice slice =
    let fft = Owl.Fft.D.fft slice in
    fft
  in
  let data = G.empty Bigarray.Complex64 [|shape|] in
  let domains = Dynarray.create () in
  (* we start the computing of each slice inside nslit domains *)
  for i = 0 to nsplit - 1 do
    let di = Domain.spawn (fun _ -> fft_slice slices.(i)) in
    Dynarray.add_last domains di
  done ;
  for i = 0 to nsplit - 1 do
    let di = Dynarray.get domains i in
    let slice = Domain.join di in
    G.set_slice [[i * n; ((i + 1) * n) - 1]] data slice
  done ;
  data

let fftfreq (a : audio) =
  let size = size a in
  let sampling = sampling a in
  let n = float_of_int size in
  let d = 1. /. float_of_int sampling in
  let nslice = ((int_of_float n - 1) / 2) + 1 in
  let fhalf = Arr.linspace 0. (float_of_int nslice) nslice in
  let shalf = Arr.linspace (-.float_of_int nslice) (-1.) nslice in
  let v = Arr.concatenate ~axis:0 [|fhalf; shalf|] in
  Arr.(1. /. (d *. n) $* v)
