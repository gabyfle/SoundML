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

(** Delta features: the local derivative of a feature sequence along its
    frame axis.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0
    ([librosa.feature.delta], over [scipy.signal.savgol_filter]); parity is
    enforced against committed golden vectors in the test suite. *)

type mode = [`Interp | `Nearest]

val delta :
  ?width:int -> ?order:int -> ?mode:mode -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [delta x] is the derivative of [x] along its last axis. See
    [Soundml.delta]. *)

val stage :
     ?width:int
  -> ?order:int
  -> ?mode:mode
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage ()] is {!delta} as a pipeline stage. See [Soundml.delta_stage]. *)
