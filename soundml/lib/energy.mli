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

(** Frame-rate energy features over framed audio.

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0
    ([librosa.feature.rms], [librosa.feature.zero_crossing_rate]); parity is
    enforced against committed golden vectors in the test suite. *)

val rms : ?frame_length:int -> ?hop:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [rms x] is the root-mean-square energy of each frame of the audio [x],
    shaped [[...; 1; frames]]. See [Soundml.rms]. *)

val rms_of_spectrogram :
  ?frame_length:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [rms_of_spectrogram s] is the frame energy recovered from a magnitude
    spectrogram. See [Soundml.rms_of_spectrogram]. *)

val rms_stage :
     ?frame_length:int
  -> ?hop:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [rms_stage ()] is {!rms} as a pipeline stage. See [Soundml.rms_stage]. *)

(** The streaming carry behind {!rms_stage}, for library stages that judge
    frame energies as they arrive ([Effects.split_stage]). Its frames are
    those of {!rms}, bit for bit, on every partitioning of the signal. *)
module Rms_kernel : sig
  (** The type for the carry of one stream. Mutable; single-owner. *)
  type 'a t

  val create : frame_length:int -> hop:int -> 'a t
  (** [create ~frame_length ~hop] is a fresh carry on that geometry, which
      the caller has validated. *)

  val step : 'a t -> (float, 'a) Nx.t -> (float, 'a) Nx.t option
  (** [step k chunk] consumes the next [m] samples of the stream, [chunk]
      shaped [[...; m]] in the leading shape of the chunks before it, and is
      the frames they complete, shaped [[...; 1; frames]]. The chunk is
      borrowed.

      Raises [Invalid_argument] after {!flush}, or if a leading axis of
      [chunk] has size zero. *)

  val flush : 'a t -> (float, 'a) Nx.t option
  (** [flush k] ends the stream and is the frames its right border
      completes, in one tensor. A second [flush] is [None]. *)

  val reset : 'a t -> unit
  (** [reset k] restores the freshly created carry. *)

  val latency : frame_length:int -> int
  (** [latency ~frame_length] is the lookahead {!rms_stage} declares,
      [frame_length / 2] samples. *)

  val rate : hop:int -> Pipeline.Rate.t
  (** [rate ~hop] is the frame rate {!rms_stage} declares, [1/hop]. *)

  val out_format :
    frame_length:int -> hop:int -> Pipeline.Format.t -> Pipeline.Format.t
  (** [out_format ~frame_length ~hop f] is the output format {!rms_stage}
      derives from [f]: items per second scaled by {!rate}, and the most
      frames one step can complete as the per-chunk bound. *)
end

val zero_crossing_rate :
     ?frame_length:int
  -> ?hop:int
  -> ?threshold:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [zero_crossing_rate x] is the fraction of sign changes in each frame of
    the audio [x], shaped [[...; 1; frames]]. See
    [Soundml.zero_crossing_rate]. *)

val zero_crossing_rate_stage :
     ?frame_length:int
  -> ?hop:int
  -> ?threshold:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [zero_crossing_rate_stage ()] is {!zero_crossing_rate} as a pipeline
    stage. See [Soundml.zero_crossing_rate_stage]. *)
