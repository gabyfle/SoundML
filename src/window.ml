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

open Bigarray

let of_owl : type a.
       (int -> Owl.Dense.Ndarray.D.arr)
    -> (float, a) kind
    -> int
    -> (float, a) Audio.G.t =
 fun owl_window kd size ->
  if size <= 0 then raise (Invalid_argument "Window size must be positive.") ;
  match kd with
  | Float64 ->
      owl_window size
  | Float32 ->
      Audio.G.cast_d2s @@ owl_window size
  | Float16 ->
      raise
        (Invalid_argument
           "Float16 elements kind aren't supported for window functions. \
            Please use Float32 or Float64." )

let hanning (kd : (float, 'a) kind) (size : int) : (float, 'a) Audio.G.t =
  of_owl Owl.Signal.hann kd size

let hamming (kd : (float, 'a) kind) (size : int) : (float, 'a) Audio.G.t =
  of_owl Owl.Signal.hamming kd size

let blackman (kd : (float, 'a) kind) (size : int) : (float, 'a) Audio.G.t =
  of_owl Owl.Signal.blackman kd size

let rectangular : type a. (float, a) kind -> int -> (float, a) Audio.G.t =
 fun (kd : (float, a) kind) (size : int) ->
  match kd with
  | Float32 ->
      Audio.G.ones Bigarray.Float32 [|size|]
  | Float64 ->
      Audio.G.ones Bigarray.Float64 [|size|]
  | Float16 ->
      raise
        (Invalid_argument
           "Float16 elements kind aren't supported. The array kind must be \
            either Float32 or Float64." )
