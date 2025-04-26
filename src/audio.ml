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

(* generic multi-dimensionnal array *)
module G = Dense.Ndarray.Generic

module Metadata = struct
  type t =
    {name: string; frames: int; channels: int; sample_rate: int; format: int}

  let create ?(name : string = "Unknown") frames channels sample_rate format =
    {name; frames; channels; sample_rate; format}

  let name (m : t) = m.name

  let frames (m : t) = m.frames

  let channels (m : t) = m.channels

  let sample_rate (m : t) = m.sample_rate

  let format (m : t) = m.format
end

type 'a audio = {meta: Metadata.t; data: (float, 'a) G.t}

let create (meta : Metadata.t) data = {meta; data}

let meta (a : 'a audio) = a.meta

let rawsize (a : 'a audio) = G.numel a.data

let length (a : 'a audio) : int =
  let meta = meta a in
  let channels = float_of_int (Metadata.channels meta) in
  let sr = float_of_int (Metadata.sample_rate meta) in
  let size = float_of_int (rawsize a) /. channels in
  Int.of_float (size /. sr *. 1000.)

let data (a : 'a audio) = a.data

let set_data (a : 'a audio) (d : (float, 'a) G.t) = {a with data= d}

let sample_pos (a : 'a audio) (x : int) =
  Int.of_float
    ( float_of_int x /. 1000.
    *. float_of_int (Metadata.sample_rate (meta a))
    *. float_of_int (Metadata.channels (meta a)) )

let get_slice (slice : int * int) (a : 'a audio) : 'a audio =
  let x, y = slice in
  let x, y =
    match (sample_pos a x, sample_pos a y) with
    | x, y when x < 0 ->
        (rawsize a + x, y)
    | x, y when y < 0 ->
        (x, rawsize a + y)
    | x, y when x < 0 && y < 0 ->
        (rawsize a + x, rawsize a + y)
    | x, y ->
        (x, y)
  in
  let x, y = if x < y then (x, y) else (y, x) in
  if x < 0 || y < 0 then
    raise
      (Invalid_argument "Audio.get_slice: slice out of bounds, negative values")
  else if x >= rawsize a || y >= rawsize a then
    raise
      (Invalid_argument
         "Audio.get_slice: slice out of bounds, values greater than rawsize" )
  else
    let data = G.get_slice [[x; y]] a.data in
    {a with data}

let get (x : int) (a : 'a audio) : float =
  let slice = get_slice (x, x) a |> data in
  G.get slice [|0|]

let normalize ?(factor : float = 2147483647.) (a : 'a audio) : unit =
  G.scalar_mul_ (1. /. factor) a.data

let reverse (x : 'a audio) : 'a audio =
  let data = G.reverse x.data in
  {x with data}

let ( .${} ) x s = get_slice s x

let ( .%{} ) i x = get x i

let ( $/ ) x f = normalize ~factor:f x

let ( /$ ) f x = normalize ~factor:f x
