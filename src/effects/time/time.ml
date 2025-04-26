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

module OptionBits = struct
  let process_offline = 0x00000000

  let process_realtime = 0x00000001

  let transients_crisp = 0x00000000

  let transients_mixed = 0x00000100

  let transients_smooth = 0x00000200

  let detector_compound = 0x00000000

  let detector_percussive = 0x00000400

  let detector_soft = 0x00000800

  let phase_laminar = 0x00000000

  let phase_independent = 0x00002000

  let threading_auto = 0x00000000

  let threading_never = 0x00010000

  let threading_always = 0x00020000

  let window_standard = 0x00000000

  let window_short = 0x00100000

  let window_long = 0x00200000

  let smoothing_off = 0x00000000

  let smoothing_on = 0x00800000

  let formant_shifted = 0x00000000

  let formant_preserved = 0x01000000

  let pitch_high_speed = 0x00000000

  let pitch_high_quality = 0x02000000

  let pitch_high_consistency = 0x04000000

  let channels_apart = 0x00000000

  let channels_together = 0x10000000

  let engine_faster = 0x00000000

  let engine_finer = 0x20000000
end

module Config = struct
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

  let default =
    { engine= Faster
    ; quality= HighSpeed
    ; formant= Shifted
    ; window= Standard
    ; smoothing= Off
    ; threading= Auto
    ; phase= Laminar
    ; detector= Compound
    ; transients= Crisp
    ; channels= Apart
    ; process= Offline }

  let percussive =
    { engine= Faster
    ; quality= HighSpeed
    ; formant= Shifted
    ; window= Short
    ; smoothing= Off
    ; threading= Auto
    ; phase= Independent
    ; detector= Compound
    ; transients= Crisp
    ; channels= Apart
    ; process= Offline }

  let set_engine engine config = {config with engine}

  let set_quality quality config = {config with quality}

  let set_formant formant config = {config with formant}

  let set_window window config = {config with window}

  let set_smoothing smoothing config = {config with smoothing}

  let set_threading threading config = {config with threading}

  let set_phase phase config = {config with phase}

  let set_detector detector config = {config with detector}

  let set_transients transients config = {config with transients}

  let set_channels channels config = {config with channels}

  let set_process process config = {config with process}

  let to_int config =
    let engine_val =
      match config.engine with
      | Faster ->
          OptionBits.engine_faster
      | Finer ->
          OptionBits.engine_finer
    in
    let quality_val =
      match config.quality with
      | HighSpeed ->
          OptionBits.pitch_high_speed
      | HighQuality ->
          OptionBits.pitch_high_quality
      | HighConsistency ->
          OptionBits.pitch_high_consistency
    in
    let formant_val =
      match config.formant with
      | Shifted ->
          OptionBits.formant_shifted
      | Preserved ->
          OptionBits.formant_preserved
    in
    let window_val =
      match config.window with
      | Standard ->
          OptionBits.window_standard
      | Short ->
          OptionBits.window_short
      | Long ->
          OptionBits.window_long
    in
    let smoothing_val =
      match config.smoothing with
      | Off ->
          OptionBits.smoothing_off
      | On ->
          OptionBits.smoothing_on
    in
    let threading_val =
      match config.threading with
      | Auto ->
          OptionBits.threading_auto
      | Always ->
          OptionBits.threading_always
      | Never ->
          OptionBits.threading_never
    in
    let phase_val =
      match config.phase with
      | Laminar ->
          OptionBits.phase_laminar
      | Independent ->
          OptionBits.phase_independent
    in
    let detector_val =
      match config.detector with
      | Compound ->
          OptionBits.detector_compound
      | Percussive ->
          OptionBits.detector_percussive
      | Soft ->
          OptionBits.detector_soft
    in
    let transients_val =
      match config.transients with
      | Crisp ->
          OptionBits.transients_crisp
      | Mixed ->
          OptionBits.transients_mixed
      | Smooth ->
          OptionBits.transients_smooth
    in
    let channels_val =
      match config.channels with
      | Apart ->
          OptionBits.channels_apart
      | Together ->
          OptionBits.channels_together
    in
    let process_val =
      match config.process with
      | Offline ->
          OptionBits.process_offline
      | RealTime ->
          OptionBits.process_realtime
    in
    engine_val lor quality_val lor formant_val lor window_val lor smoothing_val
    lor threading_val lor phase_val lor detector_val lor transients_val
    lor channels_val lor process_val
end

external rubberband_time_stretch :
     (float, Bigarray.float32_elt) Audio.G.t
  -> float
  -> int
  -> int
  -> int
  -> (float, Bigarray.float32_elt) Audio.G.t = "caml_rubberband_time_stretch"

external rubberband_pitch_shift :
     (float, Bigarray.float32_elt) Audio.G.t
  -> int
  -> int
  -> int
  -> int
  -> (float, Bigarray.float32_elt) Audio.G.t = "caml_rubberband_pitch_shift"

let time_stretch ?(config : Config.t = Config.default)
    (x : Bigarray.float32_elt Audio.audio) (rate : float) :
    Bigarray.float32_elt Audio.audio =
  if not (rate > 0.) then failwith "rate must be > 0."
  else
    let data = Audio.data x in
    let meta = Audio.meta x in
    let sr = Audio.Metadata.sample_rate meta in
    let channels = Audio.Metadata.channels meta in
    let config = Config.to_int config in
    let y = rubberband_time_stretch data rate sr channels config in
    Audio.set_data x y

let pitch_shift ?(config : Config.t = Config.default)
    (x : Bigarray.float32_elt Audio.audio) (semitones : int) :
    Bigarray.float32_elt Audio.audio =
  let data = Audio.data x in
  let meta = Audio.meta x in
  let sr = Audio.Metadata.sample_rate meta in
  let channels = Audio.Metadata.channels meta in
  let config = Config.to_int config in
  let y = rubberband_pitch_shift data semitones sr channels config in
  Audio.set_data x y
