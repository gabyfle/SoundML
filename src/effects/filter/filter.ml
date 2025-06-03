(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2025                                                       *)
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

module type S = sig
  type t

  type params

  val reset : t -> t

  val create : params -> t

  val process_sample : t -> float -> float
end

module Make (S : S) = struct
  type t = S.t

  type params = S.params

  let reset = S.reset

  let create = S.create

  let process_sample = S.process_sample

  let process (t : t) (x : (Float.t, 'a) Audio.G.t) =
    let kd = Audio.G.kind x in
    let n = Audio.G.numel x in
    let y = Audio.G.create kd [|n|] 0. in
    for i = 0 to n - 1 do
      Audio.G.set y [|i|] (process_sample t (Audio.G.get x [|i|]))
    done ;
    y
end

module IIR = struct
  module Generic = Make (Iir)
  module HighPass = Make (Highpass)
  module LowPass = Make (Lowpass)
end

module FIR = struct
  module Generic = Make (Fir)
end
