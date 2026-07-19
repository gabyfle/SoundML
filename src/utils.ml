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
      (* Handle NaN values by replacing them with 0.0 *)
      let nan_mask = Nx.isnan res in
      let zero_tensor = Nx.scalar (Nx.dtype res) 0.0 in
      Nx.where nan_mask zero_tensor res

  type 'a reference =
    | RefFloat of float
    | RefFunction of ((float, 'a) Nx.t -> float)

  let power_to_db ?(amin = 1e-10) ?(top_db = 80.) ref (s : (float, 'a) Nx.t) =
    assert (amin > 0.) ;
    let ref_value = match ref with RefFloat x -> x | RefFunction f -> f s in
    let log_spec = Nx.mul_s (Nx.div_s (Nx.log s) (Float.log 10.)) 10. in
    let log_spec =
      Nx.(
        sub log_spec
          (mul_s
             (div_s
                (log
                   (maximum
                      (scalar (dtype s) amin)
                      (scalar (dtype s) ref_value) ) )
                (Float.log 10.) )
             10. ) )
    in
    assert (top_db >= 0.0) ;
    let finite_values =
      Nx.where (Nx.isfinite log_spec) log_spec
        (Nx.scalar (Nx.dtype log_spec) (-1e8))
    in
    let max_val = Nx.max finite_values in
    let max_scalar = Nx.item [] max_val in
    let threshold = Nx.scalar (Nx.dtype log_spec) (max_scalar -. top_db) in
    Nx.maximum log_spec threshold

  let db_to_power ?(amin = 1e-10) (ref : 'a reference) (s : (float, 'a) Nx.t) =
    assert (amin > 0.) ;
    let ref_value = match ref with RefFloat x -> x | RefFunction f -> f s in
    let amin_t = Nx.scalar (Nx.dtype s) amin in
    let ref_value_t = Nx.scalar (Nx.dtype s) ref_value in
    let log_ref =
      Nx.mul_s
        (Nx.div_s (Nx.log (Nx.maximum amin_t ref_value_t)) (Float.log 10.))
        10.
    in
    let spec = Nx.add s log_ref in
    let spec = Nx.div_s spec 10. in
    Nx.pow (Nx.scalar (Nx.dtype s) 10.) spec
end

let melfreqs ?(n_mels = 128) ?(f_min = 0.) ?(f_max = 11025.) ?(htk = false)
    (kd : (float, 'b) Nx.dtype) =
  (* Input validation *)
  if n_mels <= 0 then invalid_arg "n_mels must be positive" ;
  if f_min < 0.0 then invalid_arg "f_min must be non-negative" ;
  if f_min >= f_max then invalid_arg "f_min must be less than f_max" ;
  let bounds = Nx.create kd [|2|] [|f_min; f_max|] |> Convert.hz_to_mel ~htk in
  let mel_f =
    Nx.linspace kd ~endpoint:true (Nx.item [0] bounds) (Nx.item [1] bounds)
      n_mels
  in
  Convert.mel_to_hz mel_f ~htk

let outer op x y =
  (* Generalized outer product using reshape and broadcasting *)
  let x_shape = Nx.shape x in
  let y_shape = Nx.shape y in
  let x_ndim = Array.length x_shape in
  let y_ndim = Array.length y_shape in
  (* Reshape x to add singleton dimensions for y's dimensions *)
  (* x: [x_shape] -> [x_shape, 1, 1, ..., 1] (y_ndim ones) *)
  let x_new_shape = Array.concat [x_shape; Array.make y_ndim 1] in
  let x_reshaped = Nx.reshape x_new_shape x in
  (* Reshape y to add singleton dimensions for x's dimensions *)
  (* y: [y_shape] -> [1, 1, ..., 1, y_shape] (x_ndim ones) *)
  let y_new_shape = Array.concat [Array.make x_ndim 1; y_shape] in
  let y_reshaped = Nx.reshape y_new_shape y in
  (* Apply the operation - broadcasting will handle the rest *)
  op x_reshaped y_reshaped

let pad_center signal ~size ~pad_value =
  let signal_shape = Nx.shape signal in
  let signal_length = signal_shape.(0) in
  if size < signal_length then invalid_arg "size must be >= signal length" ;
  if size = signal_length then signal
  else
    let pad_total = size - signal_length in
    let pad_left = pad_total / 2 in
    let pad_right = pad_total - pad_left in
    let padding = [|(pad_left, pad_right)|] in
    Nx.pad padding pad_value signal
