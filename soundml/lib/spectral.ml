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

(* Every feature is one fused loop over the whole spectrogram ({!Loops}): each
   frame's bins are reduced in ascending order from [0.], so the result is
   [[...; 1; frames]], leading axes are mapped independently, and a frame's
   value is the same bits alone or among others — the stages agree with the
   offline functions on every chunking. The interior is double precision, in the
   operation order the module's parity contract fixes (see the interface); the
   one rounding to the input dtype sits at the boundary. The loops also check
   that the magnitudes are non-negative, in the same pass. *)

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

(* [negative op ()] rejects negative entries: the features are only defined over
   non-negative energies. NaN entries fail the same comparison, so the loops
   report them too. *)
let negative op () =
  invalid_arg
    (Printf.sprintf
       "%s: cannot analyse a spectrogram with negative or NaN values (a \
        magnitude spectrogram is non-negative)"
       op )

(* [grid op ?freqs ~sample_rate s] is the double-precision bin frequencies:
   [freqs] validated and widened, or the FFT grid of the [2 * (bins - 1)]-point
   transform the bin count implies — bin [k] at [k * sample_rate / fft_size],
   evaluated as one reciprocal and one multiply per bin, the reference
   arithmetic the parity contract fixes. *)
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
      Float.Array.map_from_array Fun.id (Nx.to_array (Nx.cast Nx.float64 f))
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
      Float.Array.init bins (fun k -> Float.of_int k *. step)

let is_empty s = Array.exists (fun d -> d = 0) (Nx.shape s)

(* [empty_feature s] is the all-zero result for the cases that reduce nothing:
   no frames, no bins, or a zero-size leading axis. *)
let empty_feature s =
  let out = Array.copy (Nx.shape s) in
  out.(Array.length out - 2) <- 1 ;
  Nx.zeros (Nx.dtype s) out

(* The centroid normalises each frame to unit magnitude sum — frames whose sum
   falls below the smallest positive normal double divide by [1] instead (the
   underflow guard), so an all-zero frame stays all-zero — then weights the
   normalised bins by their frequencies and reduces: the operation order the
   parity contract fixes. *)
let centroid ?freqs ~sample_rate s =
  let op = "spectral_centroid" in
  check_rank op s ;
  let fq = grid op ?freqs ~sample_rate s in
  if is_empty s then empty_feature s
  else Loops.centroid ~fq ~invalid:(negative op) s

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
  if is_empty s then empty_feature s
  else
    let centroid =
      Option.map
        (fun c ->
          let shape = Array.copy (Nx.shape s) in
          shape.(Array.length shape - 2) <- 1 ;
          Nx.broadcast_to shape (Nx.cast Nx.float64 c) )
        centroid
    in
    Loops.bandwidth ~fq ~p ?centroid ~invalid:(negative op) s

let rolloff ?(roll_percent = 0.85) ?freqs ~sample_rate s =
  let op = "spectral_rolloff" in
  check_rank op s ;
  check_roll_percent op roll_percent ;
  let fq = grid op ?freqs ~sample_rate s in
  if is_empty s then empty_feature s
  else Loops.rolloff ~fq ~roll:roll_percent ~invalid:(negative op) s

let flatness ?(amin = 1e-10) ?(power = 2.) s =
  let op = "spectral_flatness" in
  check_rank op s ;
  check_amin op amin ;
  check_power op power ;
  if is_empty s then empty_feature s
  else Loops.flatness ~amin ~power ~invalid:(negative op) s

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
