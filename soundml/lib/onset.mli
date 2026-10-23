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

(** Spectral-flux onset strength and onset detection.

    The implementation module behind [Soundml.onset_strength],
    [Soundml.onset_strength_stage], [Soundml.peak_pick],
    [Soundml.peak_pick_stage], [Soundml.onset_detect],
    [Soundml.onset_detect_of_envelope] and [Soundml.onset_backtrack]; the
    semantics are documented there, once, on the public face. The lagged
    positive difference is a small Mealy kernel — [lag] frames of carried
    state — and the offline function is its one-chunk instance, so the two
    cannot disagree. So is the greedy peak picker, whose wait rule carries
    one integer of state; its dynamic-programming rules, the detector's
    normalisation and the backtracking are offline. *)

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

(** How {!peak_pick} chooses among the candidate peaks. See
    [Soundml.peak_selection]. *)
type selection = [`Greedy | `Dp_count | `Dp_value]

val peak_pick :
     ?selection:selection
  -> pre_max:int
  -> post_max:int
  -> pre_avg:int
  -> post_avg:int
  -> delta:float
  -> wait:int
  -> (float, 'a) Nx.t
  -> int array
(** [peak_pick ~pre_max ~post_max ~pre_avg ~post_avg ~delta ~wait x] is the
    frames of the one signal [x] that the peak picker selects, ascending.
    See [Soundml.peak_pick]. *)

val peak_pick_stage :
     pre_max:int
  -> post_max:int
  -> pre_avg:int
  -> post_avg:int
  -> delta:float
  -> wait:int
  -> ((float, 'a) Nx.t, int array, 'k) Pipeline.t
(** [peak_pick_stage ~pre_max ~post_max ~pre_avg ~post_avg ~delta ~wait] is
    the greedy {!peak_pick} as a causal pipeline stage emitting the frame
    indices of the peaks each chunk decides. See [Soundml.peak_pick_stage]. *)

val onset_detect :
     Stft.Config.t
  -> Mel.Config.t
  -> ?normalize:bool
  -> ?backtrack:[`Envelope | `Energy of (float, 'e) Nx.t]
  -> ?selection:selection
  -> ?pre_max:int
  -> ?post_max:int
  -> ?pre_avg:int
  -> ?post_avg:int
  -> ?delta:float
  -> ?wait:int
  -> (float, 'a) Nx.t
  -> int array
(** [onset_detect stft mel x] is the onset frames of the audio [x],
    ascending. See [Soundml.onset_detect]. *)

val onset_detect_of_envelope :
     ?hop:int
  -> ?normalize:bool
  -> ?backtrack:[`Envelope | `Energy of (float, 'e) Nx.t]
  -> ?selection:selection
  -> ?pre_max:int
  -> ?post_max:int
  -> ?pre_avg:int
  -> ?post_avg:int
  -> ?delta:float
  -> ?wait:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> int array
(** [onset_detect_of_envelope ~sample_rate e] is the onset frames of the onset
    strength envelope [e], ascending. See [Soundml.onset_detect_of_envelope]. *)

val onset_backtrack : int array -> (float, 'a) Nx.t -> int array
(** [onset_backtrack events energy] is each of [events] moved back to the
    latest minimum of [energy] at or before it. See
    [Soundml.onset_backtrack]. *)
