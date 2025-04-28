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

type engine = Faster | Finer

type quality = HighSpeed | HighQuality | HighConsistency

type formant = Shifted | Preserved

type window = Standard | Short | Long

type smoothing = Off | On

type threading = Auto | Always | Never

type phase = Laminar | Independent

type detector = Compound | Percussive | Soft

type transients = Crisp | Smooth | Mixed

type channels = Apart | Together

type process = RealTime | Offline

module Config : sig
  type t =
    { engine: engine
    ; quality: quality
    ; formant: formant
    ; window: window
    ; smoothing: smoothing
    ; threading: threading
    ; phase: phase
    ; detector: detector
    ; transients: transients
    ; channels: channels
    ; process: process }

  val default : t

  val percussive : t

  val set_engine : engine -> t -> t

  val set_quality : quality -> t -> t

  val set_formant : formant -> t -> t

  val set_window : window -> t -> t

  val set_smoothing : smoothing -> t -> t

  val set_threading : threading -> t -> t

  val set_phase : phase -> t -> t

  val set_detector : detector -> t -> t

  val set_transients : transients -> t -> t

  val set_channels : channels -> t -> t

  val set_process : process -> t -> t

  val to_int : t -> int
end

val time_stretch :
     ?config:Config.t
  -> Bigarray.float32_elt Audio.audio
  -> float
  -> Bigarray.float32_elt Audio.audio
(**
  [time_stretch ?config audio factor] stretches or compresses the audio by the
  given factor. 
  
  [config] is the configuration of the stretcher. It uses by default the same default configuration as RubberBand.
  [audio] is the audio to stretch.
  [factor] must be a positive number. It's the ratio of stretched to unstretched duration -- not tempo !
  
   *)

val pitch_shift :
     ?config:Config.t
  -> Bigarray.float32_elt Audio.audio
  -> int
  -> Bigarray.float32_elt Audio.audio
(**
  [pitch_shift ?config audio semitones] shifts the pitch of the audio by the
  given number of semitones. 
  
  [config] is the configuration of the stretcher. It uses by default the same default configuration as RubberBand.
  [audio] is the audio to shift.
  [semitones] is the number of semitones to shift the pitch. A negative value indicates a downward shift, a positive value an upward shift.
  
   *)
