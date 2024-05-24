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
    { name: string
    ; channels: int
    ; sample_width: int
    ; sample_rate: int
    ; bit_rate: int }

  let create ?(name : string = "Unknown") channels sample_width sample_rate
      bit_rate =
    {name; channels; sample_width; sample_rate; bit_rate}

  let name (m : t) = m.name

  let channels (m : t) = m.channels

  let sample_width (m : t) = m.sample_width

  let sample_rate (m : t) = m.sample_rate

  let bit_rate (m : t) = m.bit_rate
end

type audio =
  { meta: Metadata.t
  ; icodec: Avutil.audio Avcodec.params
  ; data: (float, Bigarray.float64_elt) G.t }

let create (meta : Metadata.t) icodec data = {meta; icodec; data}

let meta (a : audio) = a.meta

let rawsize (a : audio) = G.numel a.data

let length (a : audio) : int =
  let meta = meta a in
  let channels = float_of_int (Metadata.channels meta) in
  let sr = float_of_int (Metadata.sample_rate meta) in
  let size = float_of_int (rawsize a) /. channels in
  Int.of_float (size /. sr *. 1000.)

let data (a : audio) = a.data

let set_data (a : audio) (d : (float, Bigarray.float64_elt) G.t) =
  {a with data= d}

let codec (a : audio) = a.icodec

let normalize ?(factor : float = 2147483648.) (a : audio) : unit =
  G.scalar_mul_ (1. /. factor) a.data
