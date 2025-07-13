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

open Nx

module Convert = struct
  let mel_to_hz ?(htk = false) mels =
    let dtype = Nx.dtype mels in
    if htk then
      let term = Nx.div_s mels 2595. in
      let term = Nx.pow (Nx.scalar dtype 10.) term in
      let term = Nx.sub_s term 1. in
      Nx.mul_s term 700.
    else
      let f_min = 0.0 in
      let f_sp = 200.0 /. 3. in
      let min_log_hz = 1000. in
      let min_log_mel = (min_log_hz -. f_min) /. f_sp in
      let logstep = Float.log 6.4 /. 27.0 in
      let linear_mask = Nx.less_equal mels (Nx.scalar dtype min_log_mel) in
      let linear_result = Nx.(add_s (mul_s mels f_sp) f_min) in
      let log_result =
        Nx.(mul_s (exp (mul_s (sub_s mels min_log_mel) logstep)) min_log_hz)
      in
      Nx.where linear_mask linear_result log_result

  let hz_to_mel ?(htk = false) freqs =
    let dtype = Nx.dtype freqs in
    if htk then
      let term = Nx.div_s freqs 700. in
      let term = Nx.add_s term 1. in
      let term = Nx.log term in
      let term = Nx.div_s term (Float.log 10.) in
      Nx.mul_s term 2595.
    else
      let f_min = 0.0 in
      let f_sp = 200.0 /. 3. in
      let min_log_hz = 1000. in
      let min_log_mel = (min_log_hz -. f_min) /. f_sp in
      let logstep = Float.log 6.4 /. 27.0 in
      let linear_mask = Nx.less_equal freqs (Nx.scalar dtype min_log_hz) in
      let linear_result = Nx.(div_s (sub_s freqs f_min) f_sp) in
      let log_result =
        Nx.(add_s (div_s (log (div_s freqs min_log_hz)) logstep) min_log_mel)
      in
      let res = Nx.where linear_mask linear_result log_result in
      Nx.map_item (fun x -> if Float.is_nan x then 0.0 else x) res

  type reference =
    | RefFloat of float
    | RefFunction of ((float, float32_elt) t -> float)

  let power_to_db ?(amin = 1e-10) ?(top_db : float option = Some 80.) ref
      (s : (float, float32_elt) t) =
    assert (amin > 0.) ;
    let ref_value = match ref with RefFloat x -> x | RefFunction f -> f s in
    let log_spec =
      Nx.mul_s (Nx.div_s (Nx.log (Nx.maximum_s s amin)) (Float.log 10.)) 10.
    in
    let log_spec =
      Nx.sub log_spec
        (Nx.mul_s
           (Nx.div_s
              (Nx.log
                 (Nx.maximum
                    (Nx.scalar (Nx.dtype s) amin)
                    (Nx.scalar (Nx.dtype s) ref_value) ) )
              (Float.log 10.) )
           10. )
    in
    match top_db with
    | None ->
        log_spec
    | Some top_db ->
        assert (top_db >= 0.0) ;
        let max_val = Nx.max log_spec |> Nx.to_array in
        Nx.maximum_s log_spec (max_val.(0) -. top_db)

  let db_to_power ?(amin = 1e-10) (ref : reference) (s : ('a, float32_elt) t) =
    assert (amin > 0.) ;
    let ref_value = match ref with RefFloat x -> x | RefFunction f -> f s in
    let amin = Nx.scalar (Nx.dtype s) amin in
    let ref_value = Nx.scalar (Nx.dtype s) ref_value in
    let spec = Nx.div_s (Nx.mul_s s 10.) 10. in
    let log_ref =
      Nx.div_s (Nx.log (Nx.maximum amin ref_value)) (Float.log 10.)
    in
    let log_ref = Nx.mul_s log_ref 10. in
    let spec = Nx.add spec log_ref in
    Nx.pow (Nx.scalar (Nx.dtype s) 10.) spec
end

let pad_center (data : ('a, 'b) t) (target_size : int) (value : 'a) : ('a, 'b) t
    =
  let size = (Nx.shape data).(0) in
  if size = target_size then data
  else if size > target_size then
    raise
      (Invalid_argument
         "An error occured while trying to pad: current_size > target_size" )
  else if size = 0 then
    Nx.full (Nx.dtype data) [|target_size|] value
  else
    let pad_total = target_size - size in
    let pad_left = pad_total / 2 in
    let pad_right = pad_total - pad_left in
    Nx.pad [|(pad_left, pad_right)|] value data

let frame (x : ('a, 'b) t) (frame_size : int) (hop_size : int) (axis : int) :
    ('a, 'b) t =
  if frame_size <= 0 then invalid_arg "frame_size must be positive"
  else if hop_size <= 0 then invalid_arg "hop_size must be positive"
  else
    let shape_in = Nx.shape x in
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
          Array.to_list
            (Array.sub shape_in (axis + 1) (num_dims_in - (axis + 1)))
        else []
      in
      prefix_dims @ [num_frames; frame_size] @ suffix_dims
    in
    let shape_out = Array.of_list shape_out_list in
    if num_frames <= 0 then Nx.empty (Nx.dtype x) shape_out
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
        Nx.get_item (Array.to_list in_indices) x
      in
      Nx.init (Nx.dtype x) shape_out get_value_for_output_indices

let fftfreq (n : int) (d : float) =
  let nslice = ((n - 1) / 2) + 1 in
  let fhalf = Nx.arange Float32 0 nslice 1 in
  let shalf = Nx.arange Float32 (-n / 2) 0 1 in
  let v = Nx.concatenate ~axis:0 [fhalf; shalf] in
  Nx.mul (Nx.scalar Float32 (1. /. (d *. float_of_int n))) v

let rfftfreq (kd : ('a, 'b) Nx.dtype) (n : int) (d : float) =
  let nslice = n / 2 in
  let res = Nx.arange kd 0 (nslice + 1) 1 in
  let factor = 1. /. (d *. float_of_int n) in
  let factor_nx = Nx.scalar kd factor in
  Nx.mul factor_nx res

let melfreq ?(nmels = 128) ?(fmin = 0.) ?(fmax = 11025.) ?(htk = false)
    (kd : ('a, 'b) Nx.dtype) =
  let bounds = Nx.create kd [|2|] [|fmin; fmax|] |> Convert.hz_to_mel ~htk in
  let mel_f =
    Nx.linspace kd (Nx.get_item [0] bounds) (Nx.get_item [1] bounds) nmels
  in
  Convert.mel_to_hz mel_f ~htk

let unwrap ?(discont = None) ?(axis = -1) ?(period = 2. *. Float.pi)
    (p : (float, 'a) t) =
  let ndim = Nx.ndim p in
  let axis = if axis < 0 then ndim + axis else axis in
  let diff (p : (float, 'a) t) =
    let p_swapped = Nx.swapaxes axis (-1) p in
    let ndim = Nx.ndim p_swapped in
    let shape = Nx.shape p_swapped in
    let n = shape.(ndim - 1) in
    if n <= 1 then (
      let new_shape = Array.copy (Nx.shape p) in
      new_shape.(axis) <- 0 ;
      Nx.empty (Nx.dtype p) new_shape )
    else
      let starts1 = Array.make ndim 0 in
      starts1.(ndim - 1) <- 1 ;
      let stops1 = shape in
      let p1 =
        Nx.slice_ranges (Array.to_list starts1) (Array.to_list stops1) p_swapped
      in
      let starts2 = Array.make ndim 0 in
      let stops2 = Array.copy shape in
      stops2.(ndim - 1) <- n - 1 ;
      let p2 =
        Nx.slice_ranges (Array.to_list starts2) (Array.to_list stops2) p_swapped
      in
      let d = Nx.sub p1 p2 in
      Nx.swapaxes axis (-1) d
  in
  let cumsum (x : (float, 'a) t) =
    let x_moved = Nx.moveaxis axis 0 x in
    let shape = Nx.shape x_moved in
    let n = shape.(0) in
    if n = 0 then x
    else
      let result = Nx.copy x_moved in
      for i = 1 to n - 1 do
        let current_slice = Nx.slice [I i] result in
        let prev_slice = Nx.slice [I (i - 1)] result in
        let new_slice = Nx.add current_slice prev_slice in
        Nx.set_slice [I i] result new_slice
      done ;
      Nx.moveaxis 0 axis result
  in
  let d = diff p in
  let d =
    let pad_shape = Array.copy (Nx.shape p) in
    pad_shape.(axis) <- 1 ;
    let padding = Nx.zeros (Nx.dtype p) pad_shape in
    Nx.concatenate ~axis [padding; d]
  in
  let discont = match discont with Some d -> d | None -> period /. 2. in
  let d_mod =
    Nx.sub_s (Nx.mod_s (Nx.add_s d (period /. 2.)) period) (period /. 2.)
  in
  let php = period /. 2. in
  let cond1 = Nx.equal d_mod (Nx.scalar (Nx.dtype p) (-.php)) in
  let cond2 = Nx.greater d (Nx.scalar (Nx.dtype p) 0.) in
  let cond = Nx.logical_and cond1 cond2 in
  let php_scalar = Nx.full_like d_mod php in
  let d_mod = Nx.where cond php_scalar d_mod in
  let ph_correct = Nx.sub d_mod d in
  let cond_abs = Nx.less (Nx.abs d) (Nx.scalar (Nx.dtype p) discont) in
  let p_correct = Nx.where cond_abs (Nx.zeros_like ph_correct) ph_correct in
  let up = cumsum p_correct in
  Nx.add p up

let outer (op : ('a, 'b) t -> ('a, 'b) t -> ('a, 'b) t) (x : ('a, 'b) t)
    (y : ('a, 'b) t) =
  let nx = (Nx.shape x).(0) in
  let ny = (Nx.shape y).(0) in
  let x = Nx.reshape [|nx; 1|] x in
  let y = Nx.reshape [|1; ny|] y in
  op x y
