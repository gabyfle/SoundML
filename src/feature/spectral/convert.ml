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

type reference =
  | RefFloat of float
  | RefFunction of ((float, Bigarray.float32_elt) Audio.G.t -> float)

let power_to_db ?(amin = 1e-10) ?(top_db : float option = Some 80.)
    (ref : reference) (s : ('a, Bigarray.float32_elt) Audio.G.t) =
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
