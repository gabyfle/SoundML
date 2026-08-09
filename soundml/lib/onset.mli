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

(** Spectral-flux onset strength.

    The implementation module behind [Soundml.onset_strength] and
    [Soundml.onset_strength_stage]; the semantics are documented there, once,
    on the public face. The lagged positive difference is a small Mealy
    kernel — [lag] frames of carried state — and the offline function is its
    one-chunk instance, so the two cannot disagree. *)

val onset_strength :
     Stft.Config.t
  -> Mel.Config.t
  -> ?lag:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [onset_strength stft mel x] is the onset strength envelope of the audio
    [x], shaped [[...; frames]]. See [Soundml.onset_strength]. *)

val stage :
  ?lag:int -> unit -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage ()] is the lagged positive spectral flux as a causal {!Pipeline}
    stage over log-spectral chunks. See [Soundml.onset_strength_stage]. *)
