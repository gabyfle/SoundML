(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023-2025                                                  *)
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

module Convert = struct
  let mel_to_hz ?(htk : bool = false) mels =
    let open Audio.G in
    if htk then 700. $* (10. $** (mels /$ 2595.) -$ 1.)
    else
      let f_min = 0.0 in
      let f_sp = 200.0 /. 3. in
      let min_log_hz = 1000. in
      let min_log_mel = (min_log_hz -. f_min) /. f_sp in
      let logstep = Maths.log 6.4 /. 27.0 in
      let linear_mask = mels <=.$ min_log_mel in
      let log_mask = mels >=.$ min_log_mel in
      let linear_result = f_min $+ (f_sp $* mels) in
      let log_result = min_log_hz $* exp (logstep $* mels -$ min_log_mel) in
      (linear_mask * linear_result) + (log_mask * log_result)

  let hz_to_mel ?(htk : bool = false) freqs =
    let open Audio.G in
    if htk then 2595. $* log10 (1. $+ freqs /$ 700.)
    else
      let f_min = 0.0 in
      let f_sp = 200.0 /. 3. in
      let min_log_hz = 1000. in
      let min_log_mel = (min_log_hz -. f_min) /. f_sp in
      let logstep = Maths.log 6.4 /. 27.0 in
      let linear_mask = freqs <=.$ min_log_hz in
      let log_mask = freqs >=.$ min_log_hz in
      let linear_result = (freqs -$ f_min) /$ f_sp in
      let log_result = min_log_mel $+ log (freqs /$ min_log_hz) /$ logstep in
      (linear_mask * linear_result)
      (* we need to filter out possible nans here since log can result in
         nans *)
      + map (fun x -> if Float.is_nan x then 0.0 else x) (log_mask * log_result)

  type reference =
    | RefFloat of float
    | RefFunction of ((float, float32_elt) Audio.G.t -> float)

  let power_to_db ?(amin = 1e-10) ?(top_db : float option = Some 80.)
      (ref : reference) (s : (float, float32_elt) Audio.G.t) =
    assert (amin > 0.) ;
    let ref = match ref with RefFloat x -> x | RefFunction f -> f s in
    let amin = Audio.G.(init (kind s) (shape s) (fun _ -> amin)) in
    let ref = Audio.G.(init (kind s) (shape s) (fun _ -> ref)) in
    let log_spec = Audio.G.(10.0 $* log10 (max2 amin s)) in
    Audio.G.(log_spec -= (10.0 $* log10 (max2 amin ref))) ;
    let res =
      match top_db with
      | None ->
          log_spec
      | Some top_db ->
          assert (top_db >= 0.0) ;
          let max_spec = Audio.G.max' log_spec in
          let max_spec = max_spec -. top_db in
          let max_spec =
            Audio.G.(init (kind log_spec) (shape log_spec) (fun _ -> max_spec))
          in
          Audio.G.max2 log_spec max_spec
    in
    res

  let db_to_power ?(amin = 1e-10) (ref : reference)
      (s : ('a, Bigarray.float32_elt) Audio.G.t) =
    assert (amin > 0.) ;
    let ref = match ref with RefFloat x -> x | RefFunction f -> f s in
    let amin = Audio.G.(init (kind s) (shape s) (fun _ -> amin)) in
    let ref = Audio.G.(init (kind s) (shape s) (fun _ -> ref)) in
    let spec = Audio.G.(10.0 $* s /$ 10.0) in
    Audio.G.(spec += (10.0 $* log10 (max2 amin ref))) ;
    Audio.G.(exp10 spec)
end

let pad_center (data : ('a, 'b) Audio.G.t) (target_size : int) (value : 'a) :
    ('a, 'b) Audio.G.t =
  (*if Audio.G.numel data = 0 then data else*)
  let size = Audio.G.shape data |> fun s -> s.(0) in
  if size = target_size then data
  else if size > target_size then
    raise
      (Invalid_argument
         "An error occured while trying to pad: current_size > target_size" )
  else
    let pad_total = target_size - size in
    let pad_left = pad_total / 2 in
    let pad_right = pad_total - pad_left in
    let padding = [[pad_left; pad_right]] in
    Audio.G.pad ~v:value padding data

let frame (x : ('a, 'b) Audio.G.t) (frame_size : int) (hop_size : int)
    (axis : int) : ('a, 'b) Audio.G.t =
  if frame_size <= 0 then invalid_arg "frame_size must be positive" ;
  if hop_size <= 0 then invalid_arg "hop_size must be positive" ;
  let shape_in = Audio.G.shape x in
  let num_dims_in = Array.length shape_in in
  let len_axis = shape_in.(axis) in
  let num_frames =
    if len_axis < frame_size then 0
    else ((len_axis - frame_size) / hop_size) + 1
  in
  let shape_out_list =
    let prefix_dims = Array.to_list (Array.sub shape_in 0 axis) in
    let suffix_dims =
      if axis + 1 < num_dims_in then
        Array.to_list (Array.sub shape_in (axis + 1) (num_dims_in - (axis + 1)))
      else []
    in
    prefix_dims @ [num_frames; frame_size] @ suffix_dims
  in
  let shape_out = Array.of_list shape_out_list in
  if num_frames <= 0 then Audio.G.empty (Audio.G.kind x) shape_out
  else
    let get_value_for_output_indices (out_indices : int array) : 'a =
      let in_indices = Array.make num_dims_in 0 in
      for d = 0 to axis - 1 do
        in_indices.(d) <- out_indices.(d)
      done ;
      let frame_idx = out_indices.(axis) in
      let offset_in_frame = out_indices.(axis + 1) in
      in_indices.(axis) <- (frame_idx * hop_size) + offset_in_frame ;
      for d = axis + 1 to num_dims_in - 1 do
        in_indices.(d) <- out_indices.(d + 1)
      done ;
      Audio.G.get x in_indices
    in
    Audio.G.init_nd (Audio.G.kind x) shape_out get_value_for_output_indices

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

let rfftfreq (n : int) (d : float) =
  let nslice = n / 2 in
  let res =
    Audio.G.linspace Bigarray.Float32 0. (float_of_int nslice) (nslice + 1)
  in
  Arr.(1. /. (d *. float_of_int n) $* res)

let melfreq ?(nmels : int = 128) ?(fmin : float = 0.) ?(fmax : float = 11025.)
    ?(htk : bool = false) =
  let open Audio.G in
  let bounds =
    of_array Bigarray.Float32 [|fmin; fmax|] [|2|] |> Convert.hz_to_mel ~htk
  in
  let mel_f =
    linspace Bigarray.Float32 (get bounds [|0|]) (get bounds [|1|]) nmels
  in
  Convert.mel_to_hz mel_f ~htk
[@@warning "-unerasable-optional-argument"]

let roll (x : ('a, 'b) Audio.G.t) (shift : int) =
  let n = Array.get (Audio.G.shape x) 0 in
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

let unwrap ?(discont = None) ?(axis = -1) ?(period = 2. *. Owl.Const.pi)
    (p : (float, 'a) Owl.Dense.Ndarray.Generic.t) =
  let nd = Audio.G.num_dims p in
  let dd = Audio.G.diff ~axis p in
  let discont = match discont with Some d -> d | None -> period /. 2. in
  let slices = Array.init nd (fun _ -> R [0; -1]) in
  slices.(axis) <- R [1; -1] ;
  let boundary_ambiguous, interval_high =
    if Float.is_integer period then (mod_float period 2. = 0., period /. 2.)
    else (true, period /. 2.)
  in
  let open Audio.G in
  let interval_low = -.interval_high in
  let ddmod = (dd -$ interval_low) %$ period in
  let mask = ddmod <.$ 0. in
  ddmod += (mask *$ period) ;
  ddmod +$= interval_low ;
  if boundary_ambiguous then (
    let mask = (ddmod =.$ interval_low) * (dd >.$ 0.) in
    ddmod *= (1. $- mask) ;
    ddmod += (mask *$ interval_high) ) ;
  let ph_correct = ddmod - dd in
  let mask = abs dd >.$ discont in
  let ph_correct = ph_correct * mask in
  let up = copy p in
  set_fancy_ext slices up (get_fancy_ext slices p + cumsum ~axis ph_correct) ;
  up

let outer (op : ('a, 'b) Audio.G.t -> ('a, 'b) Audio.G.t -> ('a, 'b) Audio.G.t)
    (x : ('a, 'b) Audio.G.t) (y : ('a, 'b) Audio.G.t) =
  let nx = (Audio.G.shape x).(0) in
  let ny = (Audio.G.shape y).(0) in
  let x = Audio.G.reshape x [|nx; 1|] in
  let y = Audio.G.reshape y [|1; ny|] in
  op x y
