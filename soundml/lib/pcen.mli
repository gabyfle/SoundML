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

(** Per-channel energy normalization of a spectrogram.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0 ([librosa.pcen]);
    parity is enforced against committed golden vectors in the test suite. *)

val pcen :
     ?hop:int
  -> ?gain:float
  -> ?bias:float
  -> ?power:float
  -> ?time_constant:float
  -> ?eps:float
  -> ?b:float
  -> ?max_size:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [pcen ~sample_rate s] is the per-channel energy normalization of [s]. See
    [Soundml.pcen]. *)

val stage :
     ?hop:int
  -> ?gain:float
  -> ?bias:float
  -> ?power:float
  -> ?time_constant:float
  -> ?eps:float
  -> ?b:float
  -> ?max_size:int
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage ~sample_rate ()] is {!pcen} as a pipeline stage. See
    [Soundml.pcen_stage]. *)
