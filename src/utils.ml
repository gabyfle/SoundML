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
    let max_scalar = Rune.unsafe_get [] max_val in
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
  let bounds_array = Rune.unsafe_to_array bounds in
  let mel_f =
    Rune.linspace device kd ~endpoint:true bounds_array.(0) bounds_array.(1)
      n_mels
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

let frame ?(axis = -1) signal ~frame_length ~hop_length =
  (* Input validation *)
  if frame_length <= 0 then invalid_arg "frame_length must be positive" ;
  if hop_length < 1 then invalid_arg "hop_length must be positive" ;
  let shape = Rune.shape signal in
  let ndim = Rune.ndim signal in
  let axis_resolved = if axis < 0 then ndim + axis else axis in
  (* Validate axis bounds *)
  if axis_resolved < 0 || axis_resolved >= ndim then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for %dD tensor" axis ndim) ;
  let signal_length = shape.(axis_resolved) in
  if signal_length < frame_length then
    invalid_arg
      (Printf.sprintf "Input too short (n=%d) for frame_length=%d" signal_length
         frame_length ) ;
  (* Calculate number of frames *)
  let n_frames = ((signal_length - frame_length) / hop_length) + 1 in
  if n_frames <= 0 then
    invalid_arg
      (Printf.sprintf
         "No frames can be extracted with frame_length=%d, hop_length=%d from \
          signal of length %d"
         frame_length hop_length signal_length ) ;
  (* Extract all frames using efficient slicing operations *)
  let frame_list = ref [] in
  for i = 0 to n_frames - 1 do
    let start_idx = i * hop_length in
    let slice_starts = Array.make ndim 0 in
    let slice_stops = Array.copy shape in
    slice_starts.(axis_resolved) <- start_idx ;
    slice_stops.(axis_resolved) <- start_idx + frame_length ;
    let frame =
      Rune.slice_ranges
        (Array.to_list slice_starts)
        (Array.to_list slice_stops)
        signal
    in
    frame_list := frame :: !frame_list
  done ;
  (* Stack all frames together *)
  let frames = Rune.stack ~axis:0 (List.rev !frame_list) in
  (* Adjust output shape to match expected frame layout *)
  (* The stack operation puts frames in the first dimension [n_frames, ...] 
     We need to move the frame dimension to the correct position based on axis *)
  if axis < 0 then
    (* For negative axis, the frame dimension should come before the framing axis *)
    (* Move axis 0 (frames) to position axis_resolved *)
    Rune.moveaxis 0 axis_resolved frames
  else
    (* For positive axis, the frame dimension should come after the framing axis *)
    (* Move axis 0 (frames) to position axis_resolved + 1 *)
    Rune.moveaxis 0 (axis_resolved + 1) frames

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

let unwrap ?(discontinuity = Float.pi) ?(axis = -1) ?(period = 2. *. Float.pi)
    phase_values =
  (* Input validation *)
  let shape = Rune.shape phase_values in
  let ndim = Rune.ndim phase_values in
  let axis_resolved = if axis < 0 then ndim + axis else axis in
  if axis_resolved < 0 || axis_resolved >= ndim then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for %dD tensor" axis ndim) ;
  let axis_size = shape.(axis_resolved) in
  if axis_size <= 1 then phase_values
    (* No unwrapping needed for single element *)
  else
    let device = Rune.device phase_values in
    let dtype = Rune.dtype phase_values in
    (* Default discontinuity threshold is π *)
    let discont_threshold = discontinuity in
    (* Compute differences along the specified axis *)
    let slice_starts_prev = Array.make ndim 0 in
    let slice_stops_prev = Array.copy shape in
    slice_stops_prev.(axis_resolved) <- axis_size - 1 ;
    let slice_starts_curr = Array.make ndim 0 in
    let slice_stops_curr = Array.copy shape in
    slice_starts_curr.(axis_resolved) <- 1 ;
    let p_prev =
      Rune.slice_ranges
        (Array.to_list slice_starts_prev)
        (Array.to_list slice_stops_prev)
        phase_values
    in
    let p_curr =
      Rune.slice_ranges
        (Array.to_list slice_starts_curr)
        (Array.to_list slice_stops_curr)
        phase_values
    in
    (* Calculate differences *)
    let diff = Rune.sub p_curr p_prev in
    (* Find discontinuities: |diff| > discont_threshold *)
    let abs_diff = Rune.abs diff in
    let discont_mask =
      Rune.greater abs_diff (Rune.scalar device dtype discont_threshold)
    in
    (* Calculate correction: round(diff / period) * period *)
    let diff_normalized = Rune.div_s diff period in
    let correction_multiplier = Rune.round diff_normalized in
    let correction = Rune.mul_s correction_multiplier period in
    (* Apply correction only where discontinuities exist *)
    let correction_masked =
      Rune.where discont_mask correction (Rune.scalar device dtype 0.0)
    in
    (* Build cumulative sum manually by collecting slices *)
    let correction_shape = Rune.shape correction_masked in
    let axis_len = correction_shape.(axis_resolved) in
    let cumsum_slices = ref [] in
    let running_sum =
      ref
        (Rune.zeros device dtype
           (Array.mapi
              (fun i s -> if i = axis_resolved then 1 else s)
              correction_shape ) )
    in
    for i = 0 to axis_len - 1 do
      let slice_starts = Array.make ndim 0 in
      let slice_stops = Array.copy correction_shape in
      slice_starts.(axis_resolved) <- i ;
      slice_stops.(axis_resolved) <- i + 1 ;
      let current_slice =
        Rune.slice_ranges
          (Array.to_list slice_starts)
          (Array.to_list slice_stops)
          correction_masked
      in
      running_sum := Rune.add !running_sum current_slice ;
      cumsum_slices := !running_sum :: !cumsum_slices
    done ;
    let cumsum_correction =
      Rune.concatenate ~axis:axis_resolved (List.rev !cumsum_slices)
    in
    (* Pad cumsum_correction to match original shape by prepending zeros *)
    let zero_slice_shape = Array.copy shape in
    zero_slice_shape.(axis_resolved) <- 1 ;
    let zero_slice = Rune.zeros device dtype zero_slice_shape in
    let cumsum_padded =
      Rune.concatenate ~axis:axis_resolved [zero_slice; cumsum_correction]
    in
    (* Apply corrections to original array *)
    Rune.sub phase_values cumsum_padded
