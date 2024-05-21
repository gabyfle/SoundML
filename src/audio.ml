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
module FrameToS32Bytes =
  Swresample.Make (Swresample.Frame) (Swresample.S32Bytes)
module G = Dense.Ndarray.Generic

type audio =
  { name: string
  ; data: (Complex.t, Bigarray.complex32_elt) G.t
  ; sampling: int
  ; size: int }

let size (a : audio) = a.size

let data (a : audio) = a.data

let sampling (a : audio) = a.sampling

let normalise (data : (float, Bigarray.float32_elt) G.t) :
    (float, Bigarray.float32_elt) G.t =
  let c = 2147483648 in
  G.(1. /. float_of_int c $* data)

let read_audio ?(channels = `Mono) (filename : string) (format : string) : audio
    =
  Log.info "Reading audio file %s" filename ;
  let format =
    match Av.Format.find_input_format format with
    | Some f ->
        f
    | None ->
        failwith ("Could not find format: " ^ format)
  in
  let input = Av.open_input ~format filename in
  let idx, istream, icodec = Av.find_best_audio_stream input in
  let options = [`Engine_soxr] in
  let sampling = 22050 in
  let rsp = FrameToS32Bytes.from_codec ~options icodec channels sampling in
  let data = Dynarray.create () in
  let rec f start =
    match Av.read_input ~audio_frame:[istream] input with
    | `Audio_frame (i, frame) when i = idx ->
        let bytes = FrameToS32Bytes.convert rsp frame in
        let length = Bytes.length bytes in
        for i = 0 to length / 4 do
          let offset = i * 4 in
          if offset + 4 <= length then
            let value = Int32.to_float (Bytes.get_int32_ne bytes offset) in
            Dynarray.add_last data value
        done ;
        f (start + (length / 4))
    | exception Avutil.Error `Eof ->
        ()
    | _ ->
        f start
  in
  f 0 ;
  let size = Dynarray.length data in
  let data =
    G.of_array Bigarray.Float32 (Dynarray.to_array data) [|Dynarray.length data|]
  in
  let data = normalise data |> G.cast_s2c in
  Av.get_input istream |> Av.close ;
  {name= filename; data; sampling; size}

let fft (a : audio) (_start : int) (_finish : int) :
    (Complex.t, Bigarray.complex32_elt) G.t =
  Log.info "Starting to create the FFT of the audio file %s" a.name ;
  let stime = Unix.time () in
  let nsplit = Domain.recommended_domain_count () in
  let shape = (G.shape a.data).(0) in
  (*we're playing with 1-D arrays*)
  let n = shape / nsplit in
  let slices =
    Array.init nsplit (fun i ->
        let start = i * n in
        let finish = min (start + n) shape in
        G.get_slice [[start; finish - 1]] a.data )
  in
  let fft_slice slice =
    let fft = Owl.Fft.S.fft slice in
    fft
  in
  let data = G.empty Bigarray.Complex32 [|shape|] in
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
  Log.info "Finished creating the FFT of the audio file %s. Time: %f" a.name
    (Unix.time () -. stime) ;
  data

let fftfreq (a : audio) =
  let n = float_of_int a.size in
  let d = 1. /. float_of_int a.sampling in
  let nslice = ((int_of_float n - 1) / 2) + 1 in
  let fhalf = Arr.linspace 0. (float_of_int nslice) nslice in
  let shalf = Arr.linspace (-.float_of_int nslice) (-1.) nslice in
  let v = Arr.concatenate ~axis:0 [|fhalf; shalf|] in
  Arr.(1. /. (d *. n) $* v)
