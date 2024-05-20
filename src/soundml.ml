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

let magnitudes = List.map (fun c -> Complex.norm c)

let _phases = List.map (fun c -> Complex.arg c)

let () =
  let audio = Audio.read_audio "test/sin_2k.wav" "wav" in
  let fft =
    Audio.fft audio 0 4096 |> Dense.Ndarray.Generic.to_array |> Array.to_list
  in
  let x =
    Arr.linspace 1. 0. (List.length fft) |> Arr.to_array |> Array.to_list
  in
  Printf.printf "Length of x: %d\n" (List.length x) ;
  Printf.printf "Length of fft: %d\n" (List.length fft) ;
  let mags = magnitudes fft in
  Printf.printf "Max element of fft_mags: %f\n"
    (List.fold_left (fun acc v -> if v >= acc then v else acc) 0. mags) ;
  let open Oplot.Plt in
  let axis = axis 0. 0. in
  let get_points a b : plot_object =
    let mmap x y : Oplot.Points.Point2.t = {x; y} in
    let res = List.map2 mmap a b in
    Lines [res]
  in
  let y1 = get_points x mags in
  display [Color red; y1; Color black; axis]
