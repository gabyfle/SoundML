(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023-2025                                                  *)
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

let engine_to_int = function Faster -> 0x00000000 | Finer -> 0x20000000

type transients = Crisp | Mixed | Smooth

let transients_to_int = function
  | Crisp ->
      0x00000000
  | Mixed ->
      0x00000100
  | Smooth ->
      0x00000200

type detector = Compound | Percussive | Soft

let detector_to_int = function
  | Compound ->
      0x00000000
  | Percussive ->
      0x00000400
  | Soft ->
      0x00000800

type phase = Laminar | Independent

let phase_to_int = function Laminar -> 0x00000000 | Independent -> 0x00002000

type threading = Auto | Never | Always

let threading_to_int = function
  | Auto ->
      0x00000000
  | Never ->
      0x00010000
  | Always ->
      0x00020000

type window = Standard | Short | Long

let window_to_int = function
  | Standard ->
      0x00000000
  | Short ->
      0x00100000
  | Long ->
      0x00200000

type smoothing = Off | On

let smoothing_to_int = function Off -> 0x00000000 | On -> 0x00800000

type formant = Shifted | Preserved

let formant_to_int = function Shifted -> 0x00000000 | Preserved -> 0x01000000

type pitch = HighSpeed | HighQuality | HighConsistency

let pitch_to_int = function
  | HighSpeed ->
      0x00000000
  | HighQuality ->
      0x02000000
  | HighConsistency ->
      0x04000000

type channels = Apart | Together

let channels_to_int = function Apart -> 0x00000000 | Together -> 0x10000000

module Config = struct
  type t =
    { engine: engine
    ; transients: transients
    ; detector: detector
    ; phase: phase
    ; threading: threading
    ; window: window
    ; smoothing: smoothing
    ; formant: formant
    ; pitch: pitch
    ; channels: channels }

  let default : t =
    { engine= Faster
    ; transients= Crisp
    ; detector= Compound
    ; phase= Laminar
    ; threading= Auto
    ; window= Standard
    ; smoothing= Off
    ; formant= Shifted
    ; pitch= HighSpeed
    ; channels= Apart }

  let percussive : t = {default with window= Short; phase= Independent}

  let with_engine engine config = {config with engine}

  let with_transients transients config = {config with transients}

  let with_detector detector config = {config with detector}

  let with_phase phase config = {config with phase}

  let with_threading threading config = {config with threading}

  let with_window window config = {config with window}

  let with_smoothing smoothing config = {config with smoothing}

  let with_formant formant config = {config with formant}

  let with_pitch pitch config = {config with pitch}

  let with_channels channels config = {config with channels}

  let to_int (cfg : t) : int =
    0 lor engine_to_int cfg.engine
    lor transients_to_int cfg.transients
    lor detector_to_int cfg.detector
    lor phase_to_int cfg.phase
    lor threading_to_int cfg.threading
    lor window_to_int cfg.window
    lor smoothing_to_int cfg.smoothing
    lor formant_to_int cfg.formant lor pitch_to_int cfg.pitch
    lor channels_to_int cfg.channels
end

external rubberband_stretch :
     (float, Bigarray.float32_elt) Audio.G.t
  -> int * int * int * int * float * float
  -> (float, Bigarray.float32_elt) Audio.G.t = "caml_rubberband_stretch"

let to_float32 : type b.
       (float, b) Bigarray.kind
    -> (float, b) Audio.G.t
    -> (float, Bigarray.float32_elt) Audio.G.t =
 fun (kd : (float, b) Bigarray.kind) ->
  match kd with
  | Float32 ->
      Fun.id
  | Float64 ->
      Audio.G.cast_d2s
  | Float16 ->
      raise
        (Invalid_argument
           "Float16 elements kind aren't supported. The array kind must be \
            either Float32 or Float64." )

let of_float32 : type b.
       (float, b) Bigarray.kind
    -> (float, Bigarray.float32_elt) Audio.G.t
    -> (float, b) Audio.G.t =
 fun (kd : (float, b) Bigarray.kind) ->
  match kd with
  | Float32 ->
      Fun.id
  | Float64 ->
      Audio.G.cast_s2d
  | Float16 ->
      raise
        (Invalid_argument
           "Float16 elements kind aren't supported. The array kind must be \
            either Float32 or Float64." )

let time_stretch : type a.
       ?config:Config.t
    -> (float, a) Audio.G.t
    -> int
    -> float
    -> (float, a) Audio.G.t =
 fun ?(config : Config.t = Config.default) (x : (float, a) Audio.G.t)
     (sample_rate : int) (ratio : float) : (float, a) Audio.G.t ->
  if not (ratio > 0.) then failwith "rate must be > 0."
  else
    let dshape = Audio.G.shape x in
    let channels = if Array.length dshape > 1 then dshape.(0) else 1 in
    let samples = if Array.length dshape > 1 then dshape.(1) else dshape.(0) in
    let config = Config.to_int config in
    let to_float32 = to_float32 (Audio.G.kind x) in
    let of_float32 = of_float32 (Audio.G.kind x) in
    of_float32
      (rubberband_stretch (to_float32 x)
         (samples, sample_rate, channels, config, ratio, 1.0) )

let pitch_shift : type a.
       ?config:Config.t
    -> ?bins_per_octave:int
    -> (float, a) Audio.G.t
    -> int
    -> int
    -> (float, a) Audio.G.t =
 fun ?(config : Config.t = Config.default) ?(bins_per_octave : int = 12)
     (x : (float, a) Audio.G.t) (sample_rate : int) (steps : int) :
     (float, a) Audio.G.t ->
  let bins_per_octave = Float.of_int bins_per_octave in
  let steps = Float.of_int steps in
  let scale = Float.pow 2.0 (steps /. bins_per_octave) in
  let dshape = Audio.G.shape x in
  let channels = if Array.length dshape > 1 then dshape.(0) else 1 in
  let samples = if Array.length dshape > 1 then dshape.(1) else dshape.(0) in
  let config = Config.to_int config in
  let to_float32 = to_float32 (Audio.G.kind x) in
  let of_float32 = of_float32 (Audio.G.kind x) in
  of_float32
    (rubberband_stretch (to_float32 x)
       (samples, sample_rate, channels, config, 1.0, scale) )
