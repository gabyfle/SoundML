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
module FrameToS16Bytes =
  Swresample.Make (Swresample.Frame) (Swresample.S16Bytes)
module G = Dense.Ndarray.Generic

type audio =
  { name: string
  ; data: (Complex.t, Bigarray.complex64_elt) G.t
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
  let rsp = FrameToS16Bytes.from_codec ~options icodec channels sampling in
  let data = Dynarray.create () in
  let rec f start =
    match Av.read_input ~audio_frame:[istream] input with
    | `Audio_frame (i, frame) when i = idx ->
        let bytes = FrameToS16Bytes.convert rsp frame in
        let length = Bytes.length bytes in
        for i = 0 to length / 2 do
          let offset = i * 2 in
          if offset + 2 <= length then
            let value = Int.to_float (Bytes.get_int16_ne bytes offset) in
            Dynarray.add_last data value
        done ;
        f (start + (length / 2))
    | exception Avutil.Error `Eof ->
        start
    | _ ->
        f start
  in
  let total = f 0 in
  Log.info "Size of data: %d" (Dynarray.length data) ;
  let size = Dynarray.length data in
  Log.info "Read %d samples" total ;
  let data =
    G.of_array Bigarray.Float32 (Dynarray.to_array data) [|Dynarray.length data|]
  in
  let data = normalise data |> G.cast_s2z in
  Av.get_input istream |> Av.close ;
  Gc.full_major () ;
  Gc.full_major () ;
  {name= filename; data; sampling; size}

let fft (a : audio) (_start : int) (_finish : int) :
    (Complex.t, Bigarray.complex64_elt) G.t =
  Log.info "Starting to create the FFT of the audio file %s" a.name ;
  Log.info "Recommended domains: %d" (Domain.recommended_domain_count ()) ;
  (* TODO: Spawn n domains and compute the FFT on each of the domain before
     combining the results *)
  ignore (Domain.spawn (fun _ -> print_endline "I ran in parallel")) ;
  a.data |> Owl.Fft.D.fft (*|> G.get_slice [[start; finish - 1]]*)

let fftfreq (a : audio) =
  let n = float_of_int a.size in
  let d = 1. /. float_of_int a.sampling in
  let nslice = ((int_of_float n - 1) / 2) + 1 in
  let fhalf = Arr.linspace 0. (float_of_int nslice) nslice in
  let shalf = Arr.linspace (-.float_of_int nslice) (-1.) nslice in
  let v = Arr.concatenate ~axis:0 [|fhalf; shalf|] in
  Arr.(1. /. (d *. n) $* v)
