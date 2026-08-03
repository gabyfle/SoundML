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

(* Every feature is one batched pass over the whole spectrogram tensor:
   reductions run along the bin axis (axis -2) with [keepdims], so the result is
   [[...; 1; frames]] and leading axes broadcast untouched. The interior is
   double precision with librosa 0.11's exact operation order; the one cast to
   the input dtype sits at the boundary. *)

let bins_of s = Nx.dim (Nx.ndim s - 2) s

let frames_of s = Nx.dim (Nx.ndim s - 1) s

let check_rank op s =
  if Nx.ndim s < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-%d tensor (a spectrogram is [...; bins; \
          frames])"
         op (Nx.ndim s) )

let check_sample_rate op sample_rate =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use a sample rate of %d Hz (sample_rate must be at least \
          1)"
         op sample_rate )

let check_freqs_rank op freqs =
  Option.iter
    (fun f ->
      if Nx.ndim f <> 1 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot use a rank-%d freqs tensor (freqs is rank-one, one \
              frequency per bin)"
             op (Nx.ndim f) ) )
    freqs

let check_p op p =
  if not (Float.is_finite p && p > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot raise deviations to the power %g (p must be finite and \
          positive)"
         op p )

let check_roll_percent op roll_percent =
  if not (roll_percent > 0. && roll_percent < 1.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot keep %g of the spectral energy (roll_percent must lie \
          strictly between 0 and 1)"
         op roll_percent )

let check_amin op amin =
  if not (Float.is_finite amin && amin > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot floor the spectrum at %g (amin must be finite and \
          positive)"
         op amin )

let check_power op power =
  if not (Float.is_finite power && power > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot raise magnitudes to the power %g (power must be finite \
          and positive)"
         op power )

(* [check_magnitudes op s] rejects what librosa rejects: the features are only
   defined over non-negative energies. NaN entries fail the same comparison, so
   they are caught here too. *)
let check_magnitudes op s =
  if Nx.numel s > 0 && not (Nx.item [] (Nx.all (Nx.greater_equal_s s 0.))) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a spectrogram with negative or NaN values (a \
          magnitude spectrogram is non-negative)"
         op )

(* [grid op ?freqs ~sample_rate s] is the rank-one double-precision bin
   frequency vector: [freqs] validated and cast, or the FFT grid of the [2 *
   (bins - 1)]-point transform the bin count implies — bin [k] at [k *
   sample_rate / fft_size], with librosa's exact arithmetic ([np.fft.rfftfreq]:
   one reciprocal, one multiply per bin). *)
let grid op ?freqs ~sample_rate s =
  check_sample_rate op sample_rate ;
  check_freqs_rank op freqs ;
  let bins = bins_of s in
  match freqs with
  | Some f ->
      if Nx.dim 0 f <> bins then
        invalid_arg
          (Printf.sprintf
             "%s: cannot pair %d bin frequencies with %d bins (freqs holds one \
              frequency per bin)"
             op (Nx.dim 0 f) bins ) ;
      Nx.cast Nx.float64 f
  | None ->
      if bins < 2 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot derive bin frequencies for a %d-bin spectrogram (the \
              implied FFT size is %d; pass freqs explicitly)"
             op bins
             (2 * (bins - 1)) ) ;
      let fft_size = 2 * (bins - 1) in
      let step =
        1.0 /. (Float.of_int fft_size *. (1.0 /. Float.of_int sample_rate))
      in
      Nx.mul_s (Nx.arange_f Nx.float64 0. (Float.of_int bins) 1.) step

(* [col fq] is the grid as a [[bins; 1]] column, broadcastable against [[...;
   bins; frames]]. *)
let col fq = Nx.reshape [|Nx.dim 0 fq; 1|] fq

let is_empty s = Array.exists (fun d -> d = 0) (Nx.shape s)

(* [empty_feature s] is the all-zero result for the cases that reduce nothing:
   no frames, no bins, or a zero-size leading axis. *)
let empty_feature s =
  let out = Array.copy (Nx.shape s) in
  out.(Array.length out - 2) <- 1 ;
  Nx.zeros (Nx.dtype s) out

(* [normalised s64] is each frame scaled to unit magnitude sum — librosa's
   [util.normalize norm=1 axis=-2]: frames whose sum falls below the smallest
   positive normal double divide by [1] instead (the underflow guard), so an
   all-zero frame stays all-zero. *)
let normalised s64 =
  let length = Nx.sum ~axes:[-2] ~keepdims:true s64 in
  let safe =
    Nx.where
      (Nx.less length (Nx.scalar Nx.float64 Float.min_float))
      (Nx.scalar Nx.float64 1.) length
  in
  Nx.div s64 safe

(* [centroid_core fq s64] is the double-precision centroid, librosa's exact
   operation order: normalise, weight by the bin frequencies, reduce. *)
let centroid_core fq s64 =
  Nx.sum ~axes:[-2] ~keepdims:true (Nx.mul (col fq) (normalised s64))

let centroid ?freqs ~sample_rate s =
  let op = "spectral_centroid" in
  check_rank op s ;
  let fq = grid op ?freqs ~sample_rate s in
  check_magnitudes op s ;
  if is_empty s then empty_feature s
  else Nx.cast (Nx.dtype s) (centroid_core fq (Nx.cast Nx.float64 s))

let bandwidth ?(p = 2.) ?freqs ?centroid ~sample_rate s =
  let op = "spectral_bandwidth" in
  check_rank op s ;
  check_p op p ;
  let fq = grid op ?freqs ~sample_rate s in
  ( match centroid with
  | None ->
      ()
  | Some c ->
      if Nx.ndim c < 2 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot reuse a rank-%d centroid (centroid must be [...; 1; \
              frames])"
             op (Nx.ndim c) ) ;
      if Nx.dim (Nx.ndim c - 2) c <> 1 || frames_of c <> frames_of s then
        invalid_arg
          (Printf.sprintf
             "%s: cannot reuse a centroid with %d rows over %d frames for a \
              %d-frame spectrogram (centroid must be [...; 1; frames], one \
              frequency per frame)"
             op
             (Nx.dim (Nx.ndim c - 2) c)
             (frames_of c) (frames_of s) ) ) ;
  check_magnitudes op s ;
  if is_empty s then empty_feature s
  else
    let s64 = Nx.cast Nx.float64 s in
    let c64 =
      match centroid with
      | Some c ->
          Nx.cast Nx.float64 c
      | None ->
          centroid_core fq s64
    in
    let deviation = Nx.abs (Nx.sub c64 (col fq)) in
    let weighted = Nx.mul (normalised s64) (Nx.pow_s deviation p) in
    Nx.cast (Nx.dtype s)
      (Nx.pow_s (Nx.sum ~axes:[-2] ~keepdims:true weighted) (1. /. p))

let rolloff ?(roll_percent = 0.85) ?freqs ~sample_rate s =
  let op = "spectral_rolloff" in
  check_rank op s ;
  check_roll_percent op roll_percent ;
  let fq = grid op ?freqs ~sample_rate s in
  check_magnitudes op s ;
  if is_empty s then empty_feature s
  else
    let s64 = Nx.cast Nx.float64 s in
    let nd = Nx.ndim s64 in
    let cumulative = Nx.cumsum ~axis:(nd - 2) s64 in
    let bins = bins_of s in
    let total =
      Nx.shrink
        (Array.init nd (fun i ->
             if i = nd - 2 then (bins - 1, bins) else (0, Nx.dim i cumulative) )
        )
        cumulative
    in
    let reached = Nx.greater_equal cumulative (Nx.mul_s total roll_percent) in
    let candidates =
      Nx.where reached (col fq) (Nx.scalar Nx.float64 Float.infinity)
    in
    Nx.cast (Nx.dtype s) (Nx.min ~axes:[-2] ~keepdims:true candidates)

let flatness ?(amin = 1e-10) ?(power = 2.) s =
  let op = "spectral_flatness" in
  check_rank op s ;
  check_amin op amin ;
  check_power op power ;
  check_magnitudes op s ;
  if is_empty s then empty_feature s
  else
    let floored = Nx.maximum_s (Nx.pow_s (Nx.cast Nx.float64 s) power) amin in
    let geometric =
      Nx.exp (Nx.mean ~axes:[-2] ~keepdims:true (Nx.log floored))
    in
    let arithmetic = Nx.mean ~axes:[-2] ~keepdims:true floored in
    Nx.cast (Nx.dtype s) (Nx.div geometric arithmetic)

(* {1 Pipeline stages}

   Each constructor validates every chunk-independent parameter up front, so a
   misbuilt stage fails where it is built; only the checks that need a chunk's
   shape or data — the freqs/bins pairing and the non-negativity of the
   magnitudes — are left to the flat function at the first chunk. *)

let centroid_stage ?freqs ~sample_rate () =
  let op = "spectral_centroid_stage" in
  check_sample_rate op sample_rate ;
  check_freqs_rank op freqs ;
  Pipeline.stateless (fun s -> centroid ?freqs ~sample_rate s)

let bandwidth_stage ?(p = 2.) ?freqs ~sample_rate () =
  let op = "spectral_bandwidth_stage" in
  check_p op p ;
  check_sample_rate op sample_rate ;
  check_freqs_rank op freqs ;
  Pipeline.stateless (fun s -> bandwidth ~p ?freqs ~sample_rate s)

let rolloff_stage ?(roll_percent = 0.85) ?freqs ~sample_rate () =
  let op = "spectral_rolloff_stage" in
  check_roll_percent op roll_percent ;
  check_sample_rate op sample_rate ;
  check_freqs_rank op freqs ;
  Pipeline.stateless (fun s -> rolloff ~roll_percent ?freqs ~sample_rate s)

let flatness_stage ?(amin = 1e-10) ?(power = 2.) () =
  let op = "spectral_flatness_stage" in
  check_amin op amin ;
  check_power op power ;
  Pipeline.stateless (fun s -> flatness ~amin ~power s)
