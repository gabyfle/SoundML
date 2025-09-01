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

module Convert = struct
  let mel_to_hz ?(htk = false) mels =
    let dtype = Rune.dtype mels in
    if htk then
      let term = Rune.div_s mels 2595. in
      let term = Rune.pow (Rune.scalar (Rune.device mels) dtype 10.) term in
      let term = Rune.sub_s term 1. in
      Rune.mul_s term 700.
    else
      let f_min = 0.0 in
      let f_sp = 200.0 /. 3. in
      let min_log_hz = 1000. in
      let min_log_mel = (min_log_hz -. f_min) /. f_sp in
      let logstep = Float.log 6.4 /. 27.0 in
      let linear_mask =
        Rune.less_equal mels (Rune.scalar (Rune.device mels) dtype min_log_mel)
      in
      let linear_result = Rune.(add_s (mul_s mels f_sp) f_min) in
      let log_result =
        Rune.(mul_s (exp (mul_s (sub_s mels min_log_mel) logstep)) min_log_hz)
      in
      Rune.where linear_mask linear_result log_result

  let hz_to_mel ?(htk = false) freqs =
    let dtype = Rune.dtype freqs in
    if htk then
      let term = Rune.div_s freqs 700. in
      let term = Rune.add_s term 1. in
      let term = Rune.log term in
      let term = Rune.div_s term (Float.log 10.) in
      Rune.mul_s term 2595.
    else
      let f_min = 0.0 in
      let f_sp = 200.0 /. 3. in
      let min_log_hz = 1000. in
      let min_log_mel = (min_log_hz -. f_min) /. f_sp in
      let logstep = Float.log 6.4 /. 27.0 in
      let linear_mask =
        Rune.less_equal freqs (Rune.scalar (Rune.device freqs) dtype min_log_hz)
      in
      let linear_result = Rune.(div_s (sub_s freqs f_min) f_sp) in
      let log_result =
        Rune.(add_s (div_s (log (div_s freqs min_log_hz)) logstep) min_log_mel)
      in
      let res = Rune.where linear_mask linear_result log_result in
      (* Handle NaN values by replacing them with 0.0 *)
      let nan_mask = Rune.isnan res in
      let zero_tensor = Rune.scalar (Rune.device res) (Rune.dtype res) 0.0 in
      Rune.where nan_mask zero_tensor res

  type ('a, 'dev) reference =
    | RefFloat of float
    | RefFunction of ((float, 'a, 'dev) Rune.t -> float)

  let power_to_db ?(amin = 1e-10) ?(top_db = 80.) ref
      (s : (float, 'a, 'dev) Rune.t) =
    assert (amin > 0.) ;
    let ref_value = match ref with RefFloat x -> x | RefFunction f -> f s in
    let log_spec = Rune.mul_s (Rune.div_s (Rune.log s) (Float.log 10.)) 10. in
    let log_spec =
      Rune.(
        sub log_spec
          (mul_s
             (div_s
                (log
                   (maximum
                      (scalar (device s) (dtype s) amin)
                      (scalar (device s) (dtype s) ref_value) ) )
                (Float.log 10.) )
             10. ) )
    in
    assert (top_db >= 0.0) ;
    let finite_values =
      Rune.where (Rune.isfinite log_spec) log_spec
        (Rune.scalar (Rune.device log_spec) (Rune.dtype log_spec) (-1e8))
    in
    let max_val = Rune.max finite_values in
    let max_scalar = Rune.item [] max_val in
    let threshold =
      Rune.scalar (Rune.device log_spec) (Rune.dtype log_spec)
        (max_scalar -. top_db)
    in
    Rune.maximum log_spec threshold

  let db_to_power ?(amin = 1e-10) (ref : ('a, 'dev) reference)
      (s : (float, 'a, 'dev) Rune.t) =
    assert (amin > 0.) ;
    let ref_value = match ref with RefFloat x -> x | RefFunction f -> f s in
    let amin_t = Rune.scalar (Rune.device s) (Rune.dtype s) amin in
    let ref_value_t = Rune.scalar (Rune.device s) (Rune.dtype s) ref_value in
    let log_ref =
      Rune.mul_s
        (Rune.div_s
           (Rune.log (Rune.maximum amin_t ref_value_t))
           (Float.log 10.) )
        10.
    in
    let spec = Rune.add s log_ref in
    let spec = Rune.div_s spec 10. in
    Rune.pow (Rune.scalar (Rune.device s) (Rune.dtype s) 10.) spec
end

let melfreqs ?(n_mels = 128) ?(f_min = 0.) ?(f_max = 11025.) ?(htk = false)
    (device : 'dev Rune.device) (kd : (float, 'b) Rune.dtype) =
  (* Input validation *)
  if n_mels <= 0 then invalid_arg "n_mels must be positive" ;
  if f_min < 0.0 then invalid_arg "f_min must be non-negative" ;
  if f_min >= f_max then invalid_arg "f_min must be less than f_max" ;
  let bounds =
    Rune.create device kd [|2|] [|f_min; f_max|] |> Convert.hz_to_mel ~htk
  in
  let mel_f =
    Rune.linspace device kd ~endpoint:true (Rune.item [0] bounds)
      (Rune.item [1] bounds) n_mels
  in
  Convert.mel_to_hz mel_f ~htk

let outer op x y =
  (* Generalized outer product using reshape and broadcasting *)
  let x_shape = Rune.shape x in
  let y_shape = Rune.shape y in
  let x_ndim = Array.length x_shape in
  let y_ndim = Array.length y_shape in
  (* Reshape x to add singleton dimensions for y's dimensions *)
  (* x: [x_shape] -> [x_shape, 1, 1, ..., 1] (y_ndim ones) *)
  let x_new_shape = Array.concat [x_shape; Array.make y_ndim 1] in
  let x_reshaped = Rune.reshape x_new_shape x in
  (* Reshape y to add singleton dimensions for x's dimensions *)
  (* y: [y_shape] -> [1, 1, ..., 1, y_shape] (x_ndim ones) *)
  let y_new_shape = Array.concat [Array.make x_ndim 1; y_shape] in
  let y_reshaped = Rune.reshape y_new_shape y in
  (* Apply the operation - broadcasting will handle the rest *)
  op x_reshaped y_reshaped

let frame ?(axis = -1) (x : ('a, 'b, 'dev) Rune.t) ~frame_length ~hop_length :
    ('a, 'b, 'dev) Rune.t =
  if frame_length <= 0 then
    raise (Invalid_argument "frame_length must be positive") ;
  if hop_length < 1 then
    raise (Invalid_argument (Printf.sprintf "Invalid hop_length: %d" hop_length)) ;
  let device = Rune.device x in
  let x = Rune.to_nx x in
  let ndim = Nx.ndim x in
  let shape = Nx.shape x in
  let axis_resolved = if axis < 0 then ndim + axis else axis in
  if axis_resolved < 0 || axis_resolved >= ndim then
    raise (Invalid_argument "axis out of bounds") ;
  if shape.(axis_resolved) < frame_length then
    raise
      (Invalid_argument
         (Printf.sprintf "Input is too short (n=%d) for frame_length=%d"
            shape.(axis_resolved) frame_length ) ) ;
  (* Ensure we have a contiguous array for as_strided to work correctly *)
  let x_contiguous = if Nx.is_c_contiguous x then x else Nx.copy x in
  (* Calculate the number of frames *)
  let n_frames = ((shape.(axis_resolved) - frame_length) / hop_length) + 1 in
  let out_shape =
    let shape_arr = Array.make (ndim + 1) 0 in
    for i = 0 to axis_resolved - 1 do
      shape_arr.(i) <- shape.(i)
    done ;
    if axis < 0 then (
      (* Negative axes: frame_length, then n_frames *)
      shape_arr.(axis_resolved) <- frame_length ;
      shape_arr.(axis_resolved + 1) <- n_frames )
    else (
      (* Positive axes: n_frames, then frame_length *)
      shape_arr.(axis_resolved) <- n_frames ;
      shape_arr.(axis_resolved + 1) <- frame_length ) ;
    for i = axis_resolved + 1 to ndim - 1 do
      shape_arr.(i + 1) <- shape.(i)
    done ;
    shape_arr
  in
  let x_strides = Nx.strides x_contiguous in
  let itemsize = Nx.itemsize x_contiguous in
  let out_strides = Array.make (Array.length out_shape) 0 in
  if axis < 0 then (
    (* Copy strides for dimensions before the framed axis *)
    for i = 0 to axis_resolved - 1 do
      out_strides.(i) <- x_strides.(i) / itemsize
    done ;
    (* Frame dimension stride: 1 element along the framed axis *)
    out_strides.(axis_resolved) <- x_strides.(axis_resolved) / itemsize ;
    (* Hop dimension stride: hop_length elements along the framed axis *)
    out_strides.(axis_resolved + 1) <-
      x_strides.(axis_resolved) / itemsize * hop_length ;
    (* Copy remaining strides (if any) *)
    for i = axis_resolved + 1 to ndim - 1 do
      out_strides.(i + 1) <- x_strides.(i) / itemsize
    done )
  else (
    (* Positive axes: [n_frames, frame_length] order *)
    for i = 0 to axis_resolved - 1 do
      out_strides.(i) <- x_strides.(i) / itemsize
    done ;
    (* Hop dimension stride: hop_length elements along the framed axis *)
    out_strides.(axis_resolved) <-
      x_strides.(axis_resolved) / itemsize * hop_length ;
    (* Frame dimension stride: 1 element along the framed axis *)
    out_strides.(axis_resolved + 1) <- x_strides.(axis_resolved) / itemsize ;
    (* Copy remaining strides *)
    for i = axis_resolved + 1 to ndim - 1 do
      out_strides.(i + 1) <- x_strides.(i) / itemsize
    done ) ;
  let strided = Nx.as_strided out_shape out_strides ~offset:0 x_contiguous in
  Rune.of_nx device strided

let pad_center signal ~size ~pad_value =
  let signal_shape = Rune.shape signal in
  let signal_length = signal_shape.(0) in
  if size < signal_length then invalid_arg "size must be >= signal length" ;
  if size = signal_length then signal
  else
    let pad_total = size - signal_length in
    let pad_left = pad_total / 2 in
    let pad_right = pad_total - pad_left in
    let padding = [|(pad_left, pad_right)|] in
    Rune.pad padding pad_value signal
