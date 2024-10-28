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
open Bigarray

let fftfreq (n : int) (d : float) =
  let nslice = ((n - 1) / 2) + 1 in
  let fhalf =
    Audio.G.linspace Bigarray.Float32 0. (float_of_int nslice) nslice
  in
  let shalf =
    Audio.G.linspace Bigarray.Float32 (-.float_of_int nslice) (-1.) nslice
  in
  let v = Audio.G.concatenate ~axis:0 [|fhalf; shalf|] in
  Arr.(1. /. (d *. float_of_int n) $* v)

let roll (x : ('a, 'b) Audio.G.t) (shift : int) =
  let n = Array.get (Owl.Dense.Ndarray.Generic.shape x) 0 in
  let shift = if n = 0 then 0 else shift mod n in
  if shift = 0 then x
  else
    let shift = if shift < 0 then shift + n else shift in
    let result = Audio.G.copy x in
    Audio.G.set_slice_ ~out:result
      [[shift; n - 1]]
      result
      (Audio.G.get_slice [[0; n - shift - 1]; []] x) ;
    Audio.G.set_slice_ ~out:result
      [[0; shift - 1]]
      result
      (Audio.G.get_slice [[n - shift; n - 1]; []] x) ;
    result

let _float_typ_elt : type a b. (a, b) kind -> float -> a = function
  | Float32 ->
      fun a -> a
  | Float64 ->
      fun a -> a
  | Complex32 ->
      fun a -> Complex.{re= a; im= 0.}
  | Complex64 ->
      fun a -> Complex.{re= a; im= 0.}
  | Int8_signed ->
      int_of_float
  | Int8_unsigned ->
      int_of_float
  | Int16_signed ->
      int_of_float
  | Int16_unsigned ->
      int_of_float
  | Int32 ->
      fun a -> int_of_float a |> Int32.of_int
  | Int64 ->
      fun a -> int_of_float a |> Int64.of_int
  | _ ->
      failwith "_float_typ_elt: unsupported operation"

let cov ?(b : ('a, 'b) Audio.G.t option) ~(a : ('a, 'b) Audio.G.t) =
  let a =
    match b with
    | Some b ->
        let na = Audio.G.numel a in
        let nb = Audio.G.numel b in
        assert (na = nb) ;
        let a = Audio.G.reshape a [|na; 1|] in
        let b = Audio.G.reshape b [|nb; 1|] in
        Audio.G.concat_horizontal a b
    | None ->
        a
  in
  let mu = Audio.G.mean ~axis:0 ~keep_dims:true a in
  let a = Audio.G.sub a mu in
  let a' = Audio.G.transpose a in
  let c = Audio.G.dot a' a in
  let n =
    Audio.G.row_num a - 1
    |> Stdlib.max 1 |> float_of_int
    |> _float_typ_elt (Genarray.kind a)
  in
  Audio.G.div_scalar c n
[@@warning "-unerasable-optional-argument"]

let get_phase (z : Complex.t) = Complex.{re= atan2 z.im z.re; im= 0.}

let unwrap ?(discont = None) ?(axis = -1) ?(period = 2. *. Owl.Const.pi)
    (p : (Complex.t, Bigarray.complex32_elt) Owl.Dense.Ndarray.Generic.t) =
  let phase_arr = Audio.G.map get_phase p in
  let nd = Audio.G.num_dims p in
  let axis = if axis < 0 then nd + axis else axis in
  let shp = Audio.G.shape p in
  let slice_size = shp.(axis) - 1 in
  let rest_slice = Array.make nd (R []) in
  rest_slice.(axis) <- R [1; -1; 1] ;
  let init_slice = Array.make nd (R []) in
  init_slice.(axis) <- R [0; slice_size; 1] ;
  let p_rest = Audio.G.get_fancy_ext rest_slice phase_arr in
  let p_init = Audio.G.get_fancy_ext init_slice phase_arr in
  let dd = Audio.G.sub p_rest p_init in
  let discont =
    match discont with None -> period /. 2. | Some d -> max d (period /. 2.)
  in
  let interval_high = period /. 2. in
  let interval_low = -.interval_high in
  let dd_shifted = Audio.G.sub_scalar dd (Complex.polar 1. interval_low) in
  let ddmod =
    Audio.G.map
      (fun x -> Complex.polar 1. (mod_float (get_phase x).re period))
      dd_shifted
  in
  let ddmod = Audio.G.add_scalar ddmod (Complex.polar 1. interval_low) in
  let handle_boundary x =
    let phase = get_phase x in
    if abs_float phase.re = interval_high && phase.re < 0. then
      Complex.polar 1. interval_high
    else if abs_float phase.re = interval_high && phase.re > 0. then
      Complex.polar 1. (-.interval_high)
    else x
  in
  let ddmod = Audio.G.map handle_boundary ddmod in
  let ph_correct = Audio.G.sub ddmod dd in
  let ph_correct =
    Audio.G.map
      (fun x -> if abs_float (get_phase x).re < discont then Complex.zero else x)
      ph_correct
  in
  let cumsum = Audio.G.cumsum ~axis ph_correct in
  let result = Audio.G.copy p in
  let final_slice = Array.make nd (R []) in
  final_slice.(axis) <- R [1; -1; 1] ;
  let corrected_phase = Audio.G.add p_rest cumsum in
  let corrected_complex =
    Audio.G.map (fun z -> Complex.polar 1. Complex.(z.re)) corrected_phase
  in
  let _ = Audio.G.set_fancy_ext final_slice result corrected_complex in
  result
