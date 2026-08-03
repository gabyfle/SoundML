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

(* librosa 0.11 [onset.onset_strength], on the default feature chain: the
   log-power mel spectrogram, the lagged positive difference along the frame
   axis, the mean over the bin axis, and — for centered analysis — the [fft_size
   / (2 * hop)]-frame compensation shift. The difference itself is the Mealy
   kernel below ([lag] frames of carried state); the flat function runs it on
   one chunk. *)

let check_lag fn lag =
  if lag < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot difference frames at a lag of %d (lag must be at least 1)"
         fn lag )

(* {1 Shape helpers} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.shrink
    (Array.init nd (fun i ->
         if i = nd - 1 then (start, stop) else (0, Nx.dim i t) ) )
    t

let concat_last parts = Nx.concatenate ~axis:(-1) parts

(* [collapsed_shape t frames] is the shape of the aggregated envelope: the shape
   of the spectral tensor [t] with its bin axis dropped and [frames] frames.
   Requires rank >= 2. *)
let collapsed_shape t frames =
  let nd = Nx.ndim t in
  let shape = Nx.shape t in
  Array.init (nd - 1) (fun i -> if i = nd - 2 then frames else shape.(i))

(* {1 The lagged-difference kernel}

   State of one flux stream: [carry] holds the last [min lag seen] input frames,
   kernel-owned copies — incoming chunks are borrowed. Output frame [t] is the
   mean over bins of [max 0 (s t - s (t - lag))], and [0] for the first [lag]
   frames of the stream (librosa's zero prefix, its lag compensation pad); every
   input frame yields exactly one output frame, so there is no tail to drain. *)

type 'a state = {lag: int; mutable carry: (float, 'a) Nx.t option}

let state_create lag = {lag; carry= None}

let state_reset s = s.carry <- None

(* [flux dtype cur prev] is the mean over bins of [relu (cur - prev)], in
   double, rounded to [dtype] once at the boundary. *)
let flux dtype cur prev =
  Nx.cast dtype
    (Nx.mean ~axes:[-2]
       (Nx.relu (Nx.sub (Nx.cast Nx.float64 cur) (Nx.cast Nx.float64 prev))) )

(* [state_step fn s chunk] feeds [chunk] ([[...; bins; frames]], borrowed) and
   is the envelope of exactly its frames. The differences run batched over the
   chunk's whole frame axis against the carried context; the returned tensor
   aliases neither [chunk] nor kernel state. *)
let state_step fn s chunk =
  let nd = Nx.ndim chunk in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot aggregate a rank-%d tensor (the onset flux needs [...; \
          bins; frames])"
         fn nd ) ;
  let shape = Nx.shape chunk in
  if Array.exists (fun d -> d = 0) (Array.sub shape 0 (nd - 1)) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot aggregate a chunk with a zero-size axis (bins and \
          channels must be at least 1)"
         fn ) ;
  let m = last_dim chunk in
  let dtype = Nx.dtype chunk in
  if m = 0 then Nx.zeros dtype (collapsed_shape chunk 0)
  else begin
    let ext =
      match s.carry with None -> chunk | Some t -> concat_last [t; chunk]
    in
    let l = last_dim ext in
    let count = Stdlib.min m (Stdlib.max 0 (l - s.lag)) in
    let out =
      if count = 0 then Nx.zeros dtype (collapsed_shape chunk m)
      else
        let values =
          flux dtype
            (shrink_last ext (l - count) l)
            (shrink_last ext (l - count - s.lag) (l - s.lag))
        in
        if count = m then values
        else
          concat_last
            [Nx.zeros dtype (collapsed_shape chunk (m - count)); values]
    in
    let keep = Stdlib.min s.lag l in
    s.carry <- Some (Nx.copy (shrink_last ext (l - keep) l)) ;
    out
  end

(* {1 Pipeline stage} *)

let stage ?(lag = 1) () =
  check_lag "onset_strength_stage" lag ;
  (* The element dtype is only witnessed by incoming chunks; [step] records it
     so [concat []] (reached only through [run], where a step always precedes)
     can build the empty envelope. One stage value is monomorphic in its element
     type, so the shared cell is coherent across prepares. *)
  let witness = ref None in
  Pipeline.kernel ~reset:state_reset
    ~concat:(function
      | [] -> (
        match !witness with
        | Some dtype ->
            Nx.zeros dtype [|0|]
        | None ->
            invalid_arg
              "onset_strength_stage: cannot concatenate zero chunks before any \
               chunk fixed the element dtype" )
      | parts ->
          concat_last parts )
    ~prepare:(fun _format -> state_create lag)
    ~step:(fun s chunk ->
      witness := Some (Nx.dtype chunk) ;
      Some (state_step "onset_strength_stage" s chunk) )
    ()

(* {1 The flat function} *)

let onset_strength stft_config mel_config ?(lag = 1) x =
  let fn = "onset_strength" in
  check_lag fn lag ;
  let stft_size = Stft.Config.fft_size stft_config in
  let mel_size = Mel.Config.fft_size mel_config in
  if stft_size <> mel_size then
    invalid_arg
      (Printf.sprintf
         "%s: cannot project a %d-point STFT through a filterbank built for an \
          FFT of size %d (the two configurations must agree on fft_size)"
         fn stft_size mel_size ) ;
  if Nx.ndim x < 1 then
    invalid_arg
      (fn ^ ": cannot analyse a rank-zero tensor (the time axis must exist)") ;
  let mel = Mel.apply mel_config (Stft.power_spectrum stft_config x) in
  let dtype = Nx.dtype x in
  let m = last_dim mel in
  if Nx.numel mel = 0 then
    (* No frames, or no signals at all: nothing to difference — produce the
       broadcast-consistent empty envelope directly. *)
    Nx.zeros dtype (collapsed_shape mel m)
  else begin
    (* librosa's log-power mel: power_to_db with its defaults — reference 1,
       floor 1e-10 and the 80 dB whole-tensor clamp — in double precision. *)
    let db = Convert.power_to_db ~top_db:80. (Nx.cast Nx.float64 mel) in
    (* The flat flux is the kernel on one chunk. *)
    let z = state_step fn (state_create lag) db in
    (* Centered analysis reads [fft_size / 2] samples ahead of each grid
       position; librosa compensates by shifting the envelope right by [fft_size
       / (2 * hop)] frames, trimmed back to the frame grid. *)
    let shift =
      match Stft.Config.alignment stft_config with
      | `Centered ->
          stft_size / (2 * Stft.Config.hop stft_config)
      | `Left | `Right ->
          0
    in
    let out64 =
      if shift = 0 then z
      else if shift >= m then Nx.zeros Nx.float64 (Nx.shape z)
      else
        concat_last
          [ Nx.zeros Nx.float64 (collapsed_shape mel shift)
          ; shrink_last z 0 (m - shift) ]
    in
    Nx.cast dtype out64
  end
