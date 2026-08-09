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

(* The exact-rational polyphase resampler. The geometry, once and for all:

   [target / sample_rate] reduces to [L / M]. A configuration is a plan of one
   or two polyphase stages, decided at [Config.create] by arithmetic (see the
   planner below); every stage is the same machine. For one stage with factors
   [l / m]: the prototype lowpass runs at the interpolated rate [l *
   stage_rate]; its length is rounded up to [2*K*l + 1] so the linear-phase
   group delay is exactly [K] stage-input samples. Output [i] sits at input time
   [i * m / l]: with [t = i * m], its phase is [t mod l] and it reads the [2K +
   1] input samples [t/l - K .. t/l + K] (delay already compensated). The bank
   stores one row per phase, each row the prototype decimated by [l] and
   reversed so the executor reads input windows forward — laid out twice:
   phase-major ([sbank], the mathematical object the GEMM arrangement is built
   from) and in executor visit order ([svisit], see [visit_bank]);
   [Kernel.prepare] hands the executor whichever layout its instantiated bank
   wants.

   Each stage runs one of three executors, decided at [Config.create]. The C
   executor (resample_stubs.c) computes each output as one dot product over
   [history ++ chunk] with a fixed summation order; the OLS executor (the
   [ols_*] machinery below) runs integer-factor sharp stages as overlap-save FFT
   convolution on a fixed, config-derived block grid; the GEMM executor (the
   [gemm_*] machinery below) runs a phase-rich single stage as one banded matrix
   product per fixed group of block-rows. Whichever it is, offline equals
   streaming bit for bit on every partitioning — the pipeline law is structural,
   not tested-in. A cascade chains two such executors: stage 1 is
   chunk-invariant, so the sample sequence entering stage 2 does not depend on
   the input partitioning, and stage 2 is invariant to how that sequence is
   partitioned — the same argument, applied twice, makes the composite
   chunk-invariant. OCaml orchestrates at chunk granularity only: exact integer
   phase and grid state, no float accumulator anywhere. *)

type spec = {attenuation: float; passband: float}

type quality = [`Fast | `High | `Best | `Custom of spec]

(* {1 Small shape helpers (shared conventions with the other modules)} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let check_rank op t =
  if Nx.ndim t < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot resample a rank-zero tensor (the time axis must exist)" op )

let check_dtype : type a. string -> (float, a) Nx.dtype -> unit =
 fun op dt ->
  match dt with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | dt ->
      invalid_arg
        (Stdlib.Format.asprintf
           "%s: cannot resample %a audio (the executor carries float32 and \
            float64)"
           op Nx.pp_dtype dt )

(* [ceil_pos a b] is [ceil (a / b)] for [b > 0], clamped at zero for [a <=
   0]. *)
let ceil_pos a b = if a <= 0 then 0 else ((a - 1) / b) + 1

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* The flat storage of a contiguous tensor, shared (not copied): the seam
   between config-owned filter tensors and the C executor's arrays. *)
let array1_of t = Nx_buffer.to_bigarray1 (Nx.to_buffer t)

(* {1 Filter design}

   Kaiser-windowed sinc, designed by the standard empirical Kaiser relations
   (the [kaiserord] convention): the transition band of a stage runs between the
   band edges the plan assigns it — in units of the interpolated-rate Nyquist —
   the cutoff sits mid-transition, and attenuation fixes both the window shape
   and the length. Prototypes are designed in float64 once per configuration and
   cast to the kernel dtype at prepare. *)

let kaiser_beta att =
  if att > 50. then 0.1102 *. (att -. 8.7)
  else if att > 21. then
    (0.5842 *. ((att -. 21.) ** 0.4)) +. (0.07886 *. (att -. 21.))
  else 0.

(* [width] is the transition width as a fraction of the rate the filter runs at,
   Nyquist = 1. *)
let kaiser_numtaps att width =
  let n = Float.ceil (((att -. 7.95) /. 2.285 /. (Float.pi *. width)) +. 1.) in
  (* odd length: linear phase with an integral group delay *)
  if Float.rem n 2. = 0. then n +. 1. else n

(* [1 / k^2] for the [bessel_i0] series: the per-term division dominates the
   series cost (the window evaluates [I0] once per tap of a filter that can run
   tens of thousands of taps), so the reciprocals are precomputed once. The
   series at the betas admitted here converges within a few dozen terms; the
   table covers far more, and the fallback division keeps the function total. *)
let i0_inv_sq =
  Array.init 129 (fun k -> if k = 0 then 0. else 1. /. Float.of_int (k * k))

let bessel_i0 x =
  let hx2 = 0.25 *. x *. x in
  let rec go k term sum =
    let term =
      if k < 129 then term *. hx2 *. i0_inv_sq.(k)
      else term *. hx2 /. Float.of_int (k * k)
    in
    let sum' = sum +. term in
    if term <= Float.epsilon *. sum' || k > 1000 then sum'
    else go (k + 1) term sum'
  in
  go 1 1. 1.

(* [design_prototype ~l ~k ~fc ~beta] is the [2*K*L + 1]-tap prototype: a
   windowed sinc with cutoff [fc] (Nyquist units of the interpolated rate),
   normalised so the full filter sums to [L] — unit passband gain after the
   zero-stuffing of upsampling. Both factors are even in [z], so only the right
   half is evaluated and the left half mirrors it — the symmetry is exact, not a
   rounding shortcut. *)
let design_prototype ~l ~k ~fc ~beta =
  let mid = k * l in
  let n = (2 * mid) + 1 in
  let i0_beta = bessel_i0 beta in
  let h = Array.make n 0. in
  for i = mid to n - 1 do
    let z = Float.of_int (i - mid) in
    let s =
      if i = mid then fc else Float.sin (Float.pi *. fc *. z) /. (Float.pi *. z)
    in
    let r = z /. Float.of_int mid in
    let v = s *. (bessel_i0 (beta *. Float.sqrt (1. -. (r *. r))) /. i0_beta) in
    h.(i) <- v ;
    h.(n - 1 - i) <- v
  done ;
  let sum = Array.fold_left ( +. ) 0. h in
  let gain = Float.of_int l /. sum in
  Array.iteri (fun i v -> h.(i) <- v *. gain) h ;
  h

(* [bank_of_prototype ~l ~k h] lays the prototype out phase-major, [L] rows of
   [2K + 1] taps, each row reversed so the executor's dot product walks the
   input window forward: row [p], slot [s] holds [h.(p + (2K - s) * L)] (zero
   where that index passes the end). *)
let bank_of_prototype ~l ~k h =
  let taps = (2 * k) + 1 in
  let n = Array.length h in
  let b = Array.make (l * taps) 0. in
  for p = 0 to l - 1 do
    for s = 0 to taps - 1 do
      let idx = p + ((taps - 1 - s) * l) in
      if idx < n then b.((p * taps) + s) <- h.(idx)
    done
  done ;
  b

(* [visit_bank ~l ~m ~taps bank] re-lays the phase-major bank tensor in the
   executor's visit order: consecutive outputs walk phases [(i * M) mod L], so
   slot [j] holds the phase-[(j * M) mod L] row and the row of absolute output
   [i] is simply slot [i mod L] — the walk reads the bank forward, one wrap per
   [L] outputs, instead of hopping [M] rows per output. The hop was the cost on
   banks past L1: it streams them from L2 with no prefetchable pattern (measured
   57-60% of the load-bound dot ceiling vs 76-94% for resident or sequential
   banks, Apple M4 Pro). Row contents and the per-output summation order are
   untouched, so the relayout cannot move a bit. [Kernel.prepare] picks this
   layout only when the dtype-instantiated bank passes the L1 edge: below it
   both layouts are resident, the walk order is free, and the executor's
   phase-major loop carries no row counter — measured 6% on the short-dot wide
   stages of the 8 -> 48 kHz cascade. Built by blitting rows between the two
   tensors' storage: element-wise construction (an [Nx.create] from a staging
   array, or generic bigarray access) boxes per element — a measured +15% on
   [Config.create]. *)
let visit_bank ~l ~m ~taps bank =
  let v = Nx.zeros Nx.float64 [|l; taps|] in
  let src = array1_of bank and dst = array1_of v in
  for j = 0 to l - 1 do
    Bigarray.Array1.blit
      (Bigarray.Array1.sub src (j * m mod l * taps) taps)
      (Bigarray.Array1.sub dst (j * taps) taps)
  done ;
  v

(* [gemm_bank ~l ~m ~k ~blocks bank] is the [P; blocks*L] matrix [G] of the
   tensor formulation ([apply_gemm]): column [r] (an output within a
   [blocks]-phase-cycle group) is the phase-[(r*M) mod L] row of the polyphase
   bank, shifted down by the integer input advance [floor(r*M/L)] of output [r]
   within the group, and [P = 2K + 1 + floor((blocks*L - 1)*M/L)]. With [blocks
   = 1] this is the natural one-cycle arrangement; larger [blocks] express the
   same stage as the unreduced [blocks*L / blocks*M] resampler (see [gemm_block]
   above). [bank] is the phase-major array of [bank_of_prototype]. *)
let gemm_bank ~l ~m ~k ~blocks bank =
  let taps = (2 * k) + 1 in
  let lp = blocks * l in
  let p_len = taps + ((lp - 1) * m / l) in
  let g = Array.make (p_len * lp) 0. in
  for r = 0 to lp - 1 do
    let d = r * m / l and ph = r * m mod l in
    for s = 0 to taps - 1 do
      g.(((d + s) * lp) + r) <- bank.((ph * taps) + s)
    done
  done ;
  Nx.create Nx.float64 [|p_len; lp|] g

(* {1 Configuration} *)

let bank_budget_bytes = 8 * 1024 * 1024

(* The L1 residency edge (128 KB of data L1 on the reference machine's
   performance cores), shared by the planner's cost model and the bank-layout
   choice at [Kernel.prepare]. *)
let l1_edge_bytes = 128 * 1024

(* {1 OLS plan constants}

   All frozen reference-machine numbers, like [l1_edge_bytes]: plans, latencies
   and output geometry must be identical on every machine, so nothing here is
   ever probed at run time. *)

(* Whether multi-block OLS work may run as one stacked transform ([channels;
   blocks; N] — the blocks of a whole offline signal, or of one large streaming
   chunk, as extra transform lines) instead of a per-block loop. Admitted only
   under the standing byte-identity probe (test/resample/resample_fft_probe.ml):
   nx's FFT transforms lines independently through one plan per (length, sign),
   so batched equals single bit for bit — verified green over both dtypes, every
   plannable length, lines 1-64 [measured 2026-08-04]. If the probe ever fails
   on a backend, flip this to [false]: every face then presents the per-block
   [channels; N] call shape and the partition law holds structurally, at
   batching-throughput cost. The law never rests on this switch — with it on,
   all shapes produce identical bits; with it off, the shape is unique. *)
let ols_batch = true

(* The stacked-transform tile: at most this many transform lines ([channels *
   blocks]) enter one batched rfft/irfft call. Tiling is free — the standing
   probe makes every stacking bit-identical — so the tile exists purely to bound
   the transform transient, which costs on the order of 150 KB of complex128
   temporaries per line: unbounded, the one-step offline stack grows with the
   signal (measured 3.9 GB peak footprint — a ~30x multiple of the payload — and
   1.5x the tiled wall for a 10-minute 44.1 -> 48 kHz float32 apply); tiled, a
   call's transient tops out near 150 MB however long the signal. Wall is flat
   from 256 to 2048 lines on that probe, so 1024 sits well past the point where
   stacking still buys throughput (a 30-second mono conversion runs in one to
   three stacks). *)
let ols_tile_lines = 1024

(* The streaming emission-granularity ceiling: an OLS block may not span more
   than ~130 ms at its stage rate (soxr's own DFT stages buffer 1-4 k samples,
   the same class). A sharp stage whose block rule cannot fit under the ceiling
   is not OLS-eligible and stays on the direct executor. *)
let ols_ceiling_ms = 130

(* [ols_block_n ~rate ~f_div ~k] is the OLS block length for a sharp stage of
   group delay [k] consuming [rate] Hz: the smallest [2^j] (times 3 when a ÷3
   stage needs [3 | N]) with [N >= 10*K] — edge waste [N/(N - 2K) <= 1.25] — or
   [None] past the emission ceiling. *)
let ols_block_n ~rate ~f_div ~k =
  let target = Stdlib.max 64 (10 * k) in
  let base = if f_div mod 3 = 0 then 3 else 1 in
  let n = ref base in
  while !n < target do
    n := 2 * !n
  done ;
  if !n * 1000 <= ols_ceiling_ms * rate then Some !n else None

(* [ols_geom ~rate ~l ~m ~k] is [Some (n, b, delta)] when the stage is
   OLS-eligible: [b = n - 2k] rounded down to a multiple of F for ÷F stages (so
   the block grid stays on the decimated output grid), [delta] the ÷F phase
   alignment of the module comment. *)
let ols_geom ~rate ~l ~m ~k =
  let f_div = if l = 1 then m else 1 in
  match ols_block_n ~rate ~f_div ~k with
  | None ->
      None
  | Some n ->
      let b = (n - (2 * k)) / f_div * f_div in
      let delta = (f_div - (3 * k mod f_div)) mod f_div in
      if b < 1 then None else Some (n, b, delta)

(* The executor cost model's rate constants, in nanoseconds, measured with
   bench/profile/profile_fft.ml on the reference machine (Apple M4 Pro, quiet
   host) at the batched steady state — the condition offline [apply] and >=
   4096-sample streaming chunks actually run. Measured [2026-08-04]: rfft
   complex128 1.7-2.0 ns/element across N in [512, 16384] (batched lines), irfft
   1.9-2.2, complex multiply 0.5-0.7 ns/bin (bandwidth-class through Nx.mul),
   spectrum extension ~0.5 ns/bin plus op dispatch — and cross-checked against
   the measured end-to-end block cost of a shipped plan (the ×2 sharp stage of
   48 -> 44.1 kHz runs ~12-14 µs per 4096-point block, within ~10 % of the model
   at these constants). The direct kernel's dot product runs 34.3 GFLOP/s
   float32 = 0.058 ns/MAC, measured on the shipped geometries like the residency
   factors above. *)
let ols_rfft_ns_per_elt = 1.7

let ols_irfft_ns_per_elt = 1.9

let ols_mul_ns_per_bin = 0.55

let ols_copy_ns_per_bin = 0.5

let ols_block_fixed_ns = 1000.

let dot_ns_per_mac = 0.058

(* [ols_cost ~l ~m ~k (n, b, _)] is the modeled OLS execution cost per stage
   output, in direct-kernel MAC equivalents — the unit [bank_cost] prices direct
   stages in, so the two executors compare in one currency. *)
let ols_cost ~l ~m ~k:_ (n, b, _) =
  let fi = Float.of_int in
  let block_ns =
    if l > 1 then
      (* ×F: forward rfft of N, extension + multiply on the N*F/2 + 1 grid,
         inverse of N*F *)
      (ols_rfft_ns_per_elt *. fi n)
      +. ((ols_copy_ns_per_bin +. ols_mul_ns_per_bin) *. fi (n * l / 2))
      +. (ols_irfft_ns_per_elt *. fi (n * l))
      +. ols_block_fixed_ns
    else
      (* ÷F (F >= 1): multiply on the N/2 + 1 grid, extension + alias fold on
         the N grid, inverse of N/F *)
      (ols_rfft_ns_per_elt *. fi n)
      +. (ols_mul_ns_per_bin *. fi (n / 2))
      +. (ols_copy_ns_per_bin *. fi (if m > 1 then n + (n / m) else 0))
      +. (ols_irfft_ns_per_elt *. fi (n / m))
      +. ols_block_fixed_ns
  in
  let outputs_per_block = fi (b * l) /. fi m in
  block_ns /. outputs_per_block /. dot_ns_per_mac

(* Near-unity ratios (no integer factor to strip; 44.1 <-> 48 kHz) admit one OLS
   shape — oversample past the transition, then a cheap rational descent. Plans
   are dtype-blind, so the shape ships only if it clears both dtype bars against
   the previously shipped single-stage FIR: float32 >= 0.95x and float64 >= 1.5x
   — a plan may not trade one dtype's throughput away for the other's. Measured
   on the reference machine [2026-08-04], 30 s mono clips, min of 10, `High, in
   Msamples-out/s (OLS plan vs single stage): 44.1 -> 48 kHz 118.1 vs 91.6
   float32 (1.29x) and 115.7 vs 46.6 float64 (2.48x); 48 -> 44.1 kHz 120.4 vs
   85.3 float32 (1.41x) and 120.0 vs 43.1 float64 (2.78x). Both pairs clear both
   bars, so the gate ships enabled and the near-unity plans are the ×2-first OLS
   cascades of the pp pins. *)
let near_unity_ols = true

(* {1 The overlap-save (OLS) executor plan}

   A stage the planner tags for FFT execution runs the same designed filter as
   frequency-domain convolution on a fixed block grid instead of dense dot
   products. Only integer-factor sharp stages are eligible — interpolate-by-F
   and decimate-by-F, F in {2, 3, 4} — because those are the shapes whose
   frequency-domain arithmetic collapses (the ×F spectrum is the periodic
   extension of the input spectrum; the ÷F spectrum is the exact alias fold),
   while a rational L/M stage would pay a full-band multiply per phase and lose
   to the direct kernel on the count alone.

   The grid is a function of the configuration, never of chunk arrival: with
   plan constants (N, B, K, delta), block [b] covers absolute stage-input
   positions [b*B, (b+1)*B) and transforms the window [x[b*B - 2K - delta .. b*B
   - 2K - delta + N)] — zeros before the stream start, the shipped left-edge
   convention. Every execution is the same call shape (real [channels; N] ->
   [Nx.rfft complex128] -> multiply/fold against the plan-owned spectrum ->
   [Nx.irfft] at the kernel dtype), the first 2K + delta samples of every
   circular result carry the wrap and are discarded, and the emitted run of
   every block is fixed integer arithmetic in [b] — so the emitted sequence is
   invariant under every input partitioning, and the pipeline law holds by the
   same structural argument as the direct kernel. [delta] is zero except for ÷F
   stages whose K is not aligned to the decimation phase: shifting the window
   start by [(-3K) mod F] puts the kept samples on the decimated grid without
   touching the filter, so the executed impulse response stays exactly the
   designed prototype.

   The whole frequency path runs complex128 — [oh] is the float64 prototype's
   spectrum, computed once per configuration and shared by every kernel and both
   dtypes; only the final inverse transform rounds into the kernel dtype. The
   float32 OLS path therefore carries a float64 interior end to end: more
   accurate than the direct float32 dot kernel, and nothing is ever cached
   narrower than the state's dtype. *)

type ols_plan =
  { on: int (* N: block length in stage-input samples *)
  ; ob: int (* B: block advance in stage-input samples *)
  ; odelta: int (* window offset aligning ÷F decimation phase *)
  ; oh: Nx.complex128_t Lazy.t
        (* the prototype's spectrum on the transform grid: [rfft] of the
           zero-padded prototype — length [N*F/2 + 1] for ×F stages, [N/2 + 1]
           (pre-scaled by 1/F, the alias-fold weight) for ÷F. Forced on the
           first prepared kernel; not domain-safe, like [sgemm] *) }

(* {1 The GEMM executor plan}

   A phase-rich stage is one banded matrix product. Block-row [r] holds the [L]
   outputs of one phase cycle — outputs [r*L .. r*L + L - 1] — and reads the [P
   = 2K + 1 + floor((L-1)*M/L)] consecutive stage inputs starting at [r*M - K]
   (zeros outside the extent), so a run of block-rows gathered as [R; P] times
   the [P; L] arrangement of the bank ([gemm_bank] with [blocks = 1]) is [R*L]
   outputs — already the outputs in order, which is why the product's own layout
   is the emitted run. Every call carries exactly the [R] block-rows
   [gemm_rows_of] gives the stage: call [b] covers stage inputs [b*B - K,
   (b+1)*B + K) with [B = R*M], the streaming state carries that span, and the
   drain completes the calls covering the output total with virtual zeros — the
   final call zero-pads its trailing rows, so the product shape is a plan
   constant. Interior arithmetic is the kernel dtype: a call gathers its
   block-rows straight out of the carry and the input, and the product lands in
   the kernel's own precision, the same arithmetic the dot-product executor
   carries.

   The reduction of a single output is the length-[P] row times the banded
   column, in whatever order the platform's matrix product blocks it. That order
   is a property of the build, not of the call: with the row count and both
   extents fixed by the plan, every output of a given absolute index is produced
   by a call whose shape and content are functions of the plan constants and the
   totals alone, so the emitted sequence is invariant under every input
   partitioning, exactly as for the other two executors. *)

(* The stage inputs one call spans. A call is the stage's emission unit, so its
   size is fixed here in stage-input samples: one call's worth of signal is then
   the same duration whatever the stage's factors, and the product it drives is
   wide enough to carry the gather that feeds it. Frozen plan geometry like
   [l1_edge_bytes], never probed. *)
let gemm_span = 16384

(* The block-rows one call carries — as many phase cycles as [gemm_span] holds,
   at least two so the product never degenerates to a matrix-vector shape. It
   fixes the product's batch extent, hence the emission granularity ([R*M] stage
   inputs) and the per-call gather transient. *)
let gemm_rows_of ~m = Stdlib.max 2 (gemm_span / m)

(* The stages the matrix product takes: at least this many phases — the product
   is [L] columns wide, and a narrow one leaves the machine idle between
   reductions — together with a window no wider than twice the filter, [M <= 2K
   + 1], which bounds the arithmetic a block-row spends against the [2K + 1]
   multiplies an output needs. Everything else keeps the dot products or the
   transforms. *)
let gemm_min_phases = 64

(* One polyphase stage of a plan: factors [sl / sm], group delay [sk] in
   stage-input samples, the designed filter in its three layouts, and the
   executor choice. *)
type stage_plan =
  { sl: int
  ; sm: int
  ; sk: int
  ; sproto: (float, Nx.float64_elt) Nx.t (* [2*sk*sl + 1] *)
  ; sbank: (float, Nx.float64_elt) Nx.t (* [sl; 2*sk + 1], phase-major *)
  ; svisit: (float, Nx.float64_elt) Nx.t
        (* [sbank] rows permuted into executor visit order ([visit_bank]); the
           layout the C executor is handed at [Kernel.prepare] *)
  ; sgemm: (float, Nx.float64_elt) Nx.t Lazy.t
        (* [P; sl], built on the first [apply_gemm] and shared by every later
           call on this config; forcing is not domain-safe, like every other
           piece of state in this module *)
  ; sexec: stage_exec
        (* which executor runs this stage. The filter, the accessors and the law
           are identical whichever it is — the tag is visible only through
           [Config.pp] and the streaming emission cadence *) }

and stage_exec =
  | Sdirect  (** the C dot-product kernel *)
  | Sols of ols_plan  (** overlap-save on the block grid of [ols_geom] *)
  | Sgemm of (float, Nx.float64_elt) Nx.t Lazy.t
      (** the banded matrix product, over the [P; L] arrangement of the bank
          ([gemm_bank] with [blocks = 1]); forced on the first prepared kernel,
          and not domain-safe, like [sgemm] *)

type config =
  { sample_rate: int
  ; target: int
  ; quality: quality
  ; l: int (* output samples per L/M block, gcd-reduced *)
  ; m: int (* input samples per L/M block *)
  ; latency: int
        (* composite group delay in input samples: [K] for one stage, [K1 +
           K2*M1/L1] — integral by construction — for two *)
  ; stages: stage_plan list (* one or two, input to output order *) }

let is_identity c = c.l = 1 && c.m = 1 && c.latency = 0

let spec_of_quality = function
  | `Fast ->
      {attenuation= 100.; passband= 0.913}
  | `High ->
      {attenuation= 126.; passband= 0.913}
  | `Best ->
      {attenuation= 175.; passband= 0.913}
  | `Custom spec ->
      spec

let pp_bytes fmt bytes =
  let scaled v unit =
    if Float.is_integer v then Stdlib.Format.fprintf fmt "%.0f %s" v unit
    else Stdlib.Format.fprintf fmt "%.1f %s" v unit
  in
  if bytes >= 1024. *. 1024. *. 1024. then
    scaled (bytes /. (1024. *. 1024. *. 1024.)) "GB"
  else if bytes >= 1024. *. 1024. then scaled (bytes /. (1024. *. 1024.)) "MB"
  else scaled (bytes /. 1024.) "KB"

(* {1 The cascade planner}

   A single stage pays roughly [c * max(1, M/L) / (1 - passband)] multiplies per
   output — the sharp transition to the output Nyquist, scaled by the full rate
   span. When the span admits an intermediate rate, splitting into a
   wide-transition stage and a sharp stage that runs at the lowest rate in the
   chain removes most of the span factor (measured on the shipped `High
   geometries: 523 -> 357 MACs/output at 44.1 -> 16 kHz, 1137 -> 505 at 48 -> 8
   kHz, 191 -> 88 at 8 -> 48 kHz). The sharp stage itself may additionally be
   executed by overlap-save FFT convolution when it lands on an integer factor
   (the [ols_*] machinery above): that is what finally reaches the near-unity
   ratios, whose every all-FIR split prices above the single stage — 44.1 <-> 48
   kHz oversamples by two through the FFT-executed sharp stage and descends
   through a cheap rational stage, gated by the measured [near_unity_ols] rule.

   Spec discipline: each cascade stage is designed at [A + 20*log10 2] dB — two
   independent error sources, amplitude-summed worst case — and the band edges
   are placed so the composite meets the tier spec everywhere, not just at test
   frequencies. With output-side Nyquist [F_N = min(sr, target) / 2] and
   intermediate rate [f_mid]: the sharp stage owns the composite transition
   [passband*F_N, F_N] at the lowest rate that sees it, and the wide stage's
   stopband starts at [f_mid - F_N] — everything that can fold or image onto [0,
   F_N] across the other stage's rate change. Residuals landing inside the
   composite transition band originate in a stopband, so they are at least the
   tightened attenuation down; passband ripple compounds at ~1e-7, orders below
   the 0.01 dB flatness gate.

   The decomposition is a deterministic arithmetic search, costed in MACs per
   final output with exactly the K roundings the designer applies — the same
   integers every time — and a cascade is taken only when strictly cheaper than
   the single stage. Downsampling strips an integer factor first (L1 = 1:
   one-row bank, free latency composition); upsampling mirrors with the sharp
   stage first, and its wide stage may be rational — the composite-delay
   constraint [L1 | K2] (the group delay must stay integral on the input grid)
   is priced into the search, which is what rules the large-L1 splits out.

   MACs alone under-price bank residency: the executor's dot product runs at
   34.3 GFLOP/s (float32) when the bank sits at the L1 edge and 25.5-27.1 when
   the phase walk streams it from L2 (measured on the shipped geometries, Apple
   M4 Pro). Stages whose float32 bank passes 128 KB carry a 1.25x cost factor —
   the measured ratio, taken conservatively — which is what steers ties toward
   L1-resident splits without ever overriding a real arithmetic gap. *)

type stage_geom =
  {gl: int; gm: int; gk: int; gfc: float; gols: (int * int * int) option}

(* [bank_cost l k] is [2K + 1] MACs weighted by the residency factor above; the
   tie-breaker uses the float32 bank size — the throughput-critical
   instantiation. *)
let bank_cost l k =
  let macs = (2. *. Float.of_int k) +. 1. in
  if l * ((2 * k) + 1) * 4 > l1_edge_bytes then macs *. 1.25 else macs

let plan_cascade ~l ~m ~attenuation ~passband ~sample_rate ~cost_bar =
  let att = attenuation +. (20. *. Float.log10 2.) in
  let sr = Float.of_int sample_rate in
  let target_i =
    sample_rate * l / m
    (* exact: [m | sample_rate * l] *)
  in
  let target_f = Float.of_int target_i in
  let f_n = 0.5 *. Float.min sr target_f in
  let pass = passband *. f_n in
  let bank_ok len k =
    Float.of_int len *. ((2. *. Float.of_int k) +. 1.) *. 8.
    <= Float.of_int bank_budget_bytes
  in
  let near_unity = Stdlib.max l m < 2 * Stdlib.min l m in
  let best = ref None in
  let take cost s1 s2 =
    match !best with
    | Some (c0, _, _) when c0 <= cost ->
        ()
    | _ ->
        best := Some (cost, s1, s2)
  in
  (* Direct ÷F strip: wide integer decimator first, sharp rational stage at the
     lowest rate. [F * L = M] would leave the sharp stage rate-preserving while
     the wide stage inherits the very transition the split was meant to avoid —
     always dearer than one stage — so F stops strictly short of the full
     span. *)
  if m > l then
    for f = 2 to Stdlib.min ((m - 1) / l) 128 do
      let f_mid = sr /. Float.of_int f in
      let stop1 = f_mid -. f_n in
      let width1 = (stop1 -. pass) /. (sr /. 2.) in
      let nt1 = kaiser_numtaps att width1 in
      (* the drain-truncation bound needs [K1 * L >= M] (see the kernel);
         lengthening a Kaiser window at fixed beta and cutoff only deepens the
         stopband, so the bump is spec-safe *)
      let k1 =
        Stdlib.max
          (Float.to_int (Float.ceil ((nt1 -. 1.) /. 2.)))
          (ceil_pos m l)
      in
      let g2 = gcd (l * f) m in
      let l2 = l * f / g2 and m2 = m / g2 in
      let interp2 = f_mid *. Float.of_int l2 in
      let width2 = (1. -. passband) *. f_n /. (interp2 /. 2.) in
      let nt2 = kaiser_numtaps att width2 in
      let k2 =
        Stdlib.max 1
          (Float.to_int (Float.ceil ((nt2 -. 1.) /. (2. *. Float.of_int l2))))
      in
      if bank_ok 1 k1 && bank_ok l2 k2 then
        take
          ((bank_cost 1 k1 *. f_mid /. target_f) +. bank_cost l2 k2)
          {gl= 1; gm= f; gk= k1; gfc= (pass +. stop1) /. sr; gols= None}
          {gl= l2; gm= m2; gk= k2; gfc= (pass +. f_n) /. interp2; gols= None}
    done
  else
    (* Direct rational split: sharp stage first at the low-rate end. *)
    for l1 = 2 to 16 do
      for m1 = 1 to l1 - 1 do
        if gcd l1 m1 = 1 && l1 * m < l * m1 then begin
          let f_mid = sr *. Float.of_int l1 /. Float.of_int m1 in
          let interp1 = sr *. Float.of_int l1 in
          let width1 = (1. -. passband) *. f_n /. (interp1 /. 2.) in
          let nt1 = kaiser_numtaps att width1 in
          let k1 =
            Stdlib.max 1
              (Float.to_int
                 (Float.ceil ((nt1 -. 1.) /. (2. *. Float.of_int l1))) )
          in
          let g2 = gcd (l * m1) (m * l1) in
          let l2 = l * m1 / g2 and m2 = m * l1 / g2 in
          let interp2 = f_mid *. Float.of_int l2 in
          let stop2 = f_mid -. f_n in
          let width2 = (stop2 -. pass) /. (interp2 /. 2.) in
          let nt2 = kaiser_numtaps att width2 in
          let k2n =
            Stdlib.max 1
              (Float.to_int
                 (Float.ceil ((nt2 -. 1.) /. (2. *. Float.of_int l2))) )
          in
          (* K2 rounds up onto stage 1's grid so [K1 + K2*M1/L1] is integral *)
          let k2 = l1 * ceil_pos k2n l1 in
          if bank_ok l1 k1 && bank_ok l2 k2 then
            take
              ((bank_cost l1 k1 *. f_mid /. target_f) +. bank_cost l2 k2)
              { gl= l1
              ; gm= m1
              ; gk= k1
              ; gfc= (1. +. passband) *. f_n /. interp1
              ; gols= None }
              { gl= l2
              ; gm= m2
              ; gk= k2
              ; gfc= (pass +. stop2) /. interp2
              ; gols= None }
        end
      done
    done ;
  (* OLS ÷F: wide rational stage first, sharp integer decimator executed by
     overlap-save at the lowest rate — soxr's own planner shape. *)
  if m > l then
    List.iter
      (fun f ->
        if f * l < m then begin
          let f_mid_i = f * target_i in
          let f_mid = Float.of_int f_mid_i in
          let g1 = gcd (f * l) m in
          let l1 = f * l / g1 and m1 = m / g1 in
          let interp1 = sr *. Float.of_int l1 in
          let stop1 = f_mid -. f_n in
          let width1 = (stop1 -. pass) /. (interp1 /. 2.) in
          let nt1 = kaiser_numtaps att width1 in
          let k1 =
            Stdlib.max
              (Stdlib.max 1
                 (Float.to_int
                    (Float.ceil ((nt1 -. 1.) /. (2. *. Float.of_int l1))) ) )
              (ceil_pos m l)
          in
          let width2 = (f_n -. pass) /. (f_mid /. 2.) in
          let nt2 = kaiser_numtaps att width2 in
          let k2n =
            Stdlib.max 1 (Float.to_int (Float.ceil ((nt2 -. 1.) /. 2.)))
          in
          (* [L1 | K2 * M1] keeps the composite delay integral; L1 and M1 are
             coprime, so K2 rounds to a multiple of L1 *)
          let k2 = l1 * ceil_pos k2n l1 in
          match ols_geom ~rate:f_mid_i ~l:1 ~m:f ~k:k2 with
          | Some geom when bank_ok l1 k1 && bank_ok 1 k2 ->
              take
                ( (bank_cost l1 k1 *. f_mid /. target_f)
                +. ols_cost ~l:1 ~m:f ~k:k2 geom )
                { gl= l1
                ; gm= m1
                ; gk= k1
                ; gfc= (pass +. stop1) /. interp1
                ; gols= None }
                { gl= 1
                ; gm= f
                ; gk= k2
                ; gfc= (pass +. f_n) /. f_mid
                ; gols= Some geom }
          | _ ->
              ()
        end )
      [2; 3; 4] ;
  (* OLS ×F: sharp integer interpolator executed by overlap-save at the source
     rate, then a cheap wide-transition rational stage. For near-unity ratios
     this is the only admissible OLS shape (oversample past the transition, then
     descend) and it is gated by the measured decision rule recorded at
     [near_unity_ols]. Ratios within 1 % of unity are excluded outright: this
     shape could represent them within the bank budget (its wide stage needs
     only a few taps per phase), but they are clock-drift correction, which this
     fixed-ratio resampler refuses by design — the creation error, not a plan,
     is the contract there. *)
  let drift_class =
    Float.of_int (Stdlib.max l m) < 1.01 *. Float.of_int (Stdlib.min l m)
  in
  if ((not near_unity) || near_unity_ols) && not drift_class then
    List.iter
      (fun f ->
        if f * m <> l then begin
          let f_mid_i = f * sample_rate in
          let f_mid = Float.of_int f_mid_i in
          let interp1 = f_mid in
          let width1 = (f_n -. pass) /. (interp1 /. 2.) in
          let nt1 = kaiser_numtaps att width1 in
          let k1 =
            Stdlib.max
              (Stdlib.max 1
                 (Float.to_int
                    (Float.ceil ((nt1 -. 1.) /. (2. *. Float.of_int f))) ) )
              (ceil_pos m l)
          in
          let g2 = gcd l (f * m) in
          let l2 = l / g2 and m2 = f * m / g2 in
          let interp2 = f_mid *. Float.of_int l2 in
          let stop2 = f_mid -. f_n in
          let width2 = (stop2 -. pass) /. (interp2 /. 2.) in
          let nt2 = kaiser_numtaps att width2 in
          let k2n =
            Stdlib.max 1
              (Float.to_int
                 (Float.ceil ((nt2 -. 1.) /. (2. *. Float.of_int l2))) )
          in
          let k2 = f * ceil_pos k2n f in
          match ols_geom ~rate:sample_rate ~l:f ~m:1 ~k:k1 with
          | Some geom when bank_ok f k1 && bank_ok l2 k2 ->
              take
                ( (ols_cost ~l:f ~m:1 ~k:k1 geom *. f_mid /. target_f)
                +. bank_cost l2 k2 )
                { gl= f
                ; gm= 1
                ; gk= k1
                ; gfc= (pass +. f_n) /. interp1
                ; gols= Some geom }
                { gl= l2
                ; gm= m2
                ; gk= k2
                ; gfc= (pass +. stop2) /. interp2
                ; gols= None }
          | _ ->
              ()
        end )
      [2; 3; 4] ;
  match !best with
  | Some (cost, s1, s2) when cost < cost_bar ->
      Some (cost, s1, s2, kaiser_beta att)
  | _ ->
      None

(* Small-L stages are blocked in the tensor formulation: the natural patch per
   phase cycle is [2K + 1 + floor((L-1)*M/L)] samples advancing [M] — kernel far
   wider than the stride when L is small — and [Nx.extract_patches] pays per
   gathered element (measured ~4-5 ns/element: a 381-tap /2 decimator was
   gathering 29x the input and spending 5.5 ms per second of audio against 0.02
   ms of matmul; the x2 sharp stage of the FFT-executed near-unity plans is the
   same shape from the other side, 201 taps advancing one input). Grouping [B]
   phase cycles per patch is the same conversion expressed as the unreduced [B*L
   / B*M] resampler — the columns of the [P; B*L] matrix are the same bank rows,
   shifted by the per-output integer advance — and it divides the gathered
   volume by ~B*M / (taps + (B-1)*M) while the extra multiplies land in the
   GEMM, which is the fast path. [B = max 1 (64 / L)] puts the gather within a
   factor of two of the input size for every shipped geometry and leaves L >= 64
   stages on the natural one-cycle arrangement. *)
let gemm_block = 64

let gemm_blocks l = Stdlib.max 1 (gemm_block / l)

(* The executor a plan assigns a stage, before the stage is designed: the OLS
   variant carries the block geometry [ols_geom] returned. *)
type exec_spec = Xdirect | Xols of (int * int * int) | Xgemm

let make_stage ?(exec = Xdirect) ~l ~m ~k ~fc ~beta () =
  let h = design_prototype ~l ~k ~fc ~beta in
  let taps = (2 * k) + 1 in
  let b = bank_of_prototype ~l ~k h in
  let bank = Nx.create Nx.float64 [|l; taps|] b in
  let proto = Nx.create Nx.float64 [|Array.length h|] h in
  { sl= l
  ; sm= m
  ; sk= k
  ; sproto= proto
  ; sbank= bank
  ; svisit= visit_bank ~l ~m ~taps bank
  ; sgemm= lazy (gemm_bank ~l ~m ~k ~blocks:(gemm_blocks l) (Nx.to_array bank))
  ; sexec=
      ( match exec with
      | Xdirect ->
          Sdirect
      | Xgemm ->
          Sgemm (lazy (gemm_bank ~l ~m ~k ~blocks:1 (Nx.to_array bank)))
      | Xols (n, b, delta) ->
          Sols
            { on= n
            ; ob= b
            ; odelta= delta
            ; oh=
                lazy
                  (let len = if l > 1 then n * l else n in
                   let padded = Nx.pad [|(0, len - Array.length h)|] 0. proto in
                   let spec = Nx.rfft Nx.complex128 padded in
                   if m > 1 then
                     Nx.mul_s spec {Complex.re= 1. /. Float.of_int m; im= 0.}
                   else spec ) } ) }

module Config = struct
  type t = config

  let create ?(quality = `High) ~sample_rate ~target () =
    if sample_rate < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot resample from %d Hz (sample_rate must be at least 1)"
           sample_rate ) ;
    if target < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot resample to %d Hz (target must be at least 1)" target ) ;
    let {attenuation; passband} = spec_of_quality quality in
    if
      not
        ( Float.is_finite attenuation
        && attenuation >= 40. && attenuation <= 200. )
    then
      invalid_arg
        (Printf.sprintf
           "create: cannot design a filter with %g dB of stop-band rejection \
            (attenuation must be finite, in [40, 200])"
           attenuation ) ;
    if not (Float.is_finite passband && passband >= 0.5 && passband <= 0.99)
    then
      invalid_arg
        (Printf.sprintf
           "create: cannot preserve %g of the band (passband must be finite, \
            in [0.5, 0.99])"
           passband ) ;
    let g = gcd sample_rate target in
    let l = target / g and m = sample_rate / g in
    if l = 1 && m = 1 then
      { sample_rate
      ; target
      ; quality
      ; l= 1
      ; m= 1
      ; latency= 0
      ; stages=
          [ { sl= 1
            ; sm= 1
            ; sk= 0
            ; sproto= Nx.create Nx.float64 [|1|] [|1.|]
            ; sbank= Nx.create Nx.float64 [|1; 1|] [|1.|]
            ; svisit= Nx.create Nx.float64 [|1; 1|] [|1.|]
            ; sgemm= lazy (Nx.create Nx.float64 [|1; 1|] [|1.|])
            ; sexec= Sdirect } ] }
    else begin
      let width = (1. -. passband) /. Float.of_int (Stdlib.max l m) in
      let ntaps = kaiser_numtaps attenuation width in
      (* K in float first: the budget check must precede any conversion that
         could overflow or any array that could not be allocated *)
      let k_f = Float.ceil ((ntaps -. 1.) /. (2. *. Float.of_int l)) in
      let bank_bytes = Float.of_int l *. ((2. *. k_f) +. 1.) *. 8. in
      let single_fits = bank_bytes <= Float.of_int bank_budget_bytes in
      let k_single = Stdlib.max 1 (Float.to_int k_f) in
      let cost_single =
        if single_fits then bank_cost l k_single else Float.infinity
      in
      let single exec =
        let fc = (1. +. passband) /. (2. *. Float.of_int (Stdlib.max l m)) in
        let beta = kaiser_beta attenuation in
        { sample_rate
        ; target
        ; quality
        ; l
        ; m
        ; latency= k_single
        ; stages= [make_stage ~exec ~l ~m ~k:k_single ~fc ~beta ()] }
      in
      (* The one stage runs as a matrix product when the ratio is phase-rich
         against a narrow window: no split can beat it there, and the plan keeps
         the single stage's latency and prototype exactly. *)
      let gemm_single =
        single_fits && l >= gemm_min_phases && m <= (2 * k_single) + 1
      in
      (* A pure ×F or ÷F conversion may run its one stage — the same designed
         filter, so the same latency and accessors to the bit — by overlap-save
         when the model prices the transform under the dense dot products. *)
      let single_ols =
        if
          single_fits
          && ((m = 1 && l >= 2 && l <= 4) || (l = 1 && m >= 2 && m <= 4))
        then
          match ols_geom ~rate:sample_rate ~l ~m ~k:k_single with
          | Some geom ->
              let cost = ols_cost ~l ~m ~k:k_single geom in
              if cost < cost_single then Some (cost, geom) else None
          | None ->
              None
        else None
      in
      let cost_bar =
        match single_ols with Some (c, _) -> c | None -> cost_single
      in
      let of_geom = function None -> Xdirect | Some g -> Xols g in
      match
        if gemm_single then None
        else plan_cascade ~l ~m ~attenuation ~passband ~sample_rate ~cost_bar
      with
      | Some (_, g1, g2, beta) ->
          (* the two invariants the kernel composition rests on, restated as
             executable checks: integral composite delay, and the
             drain-truncation bound (mid-stream emission can never pass the
             composite ceil) *)
          assert (g2.gk * g1.gm mod g1.gl = 0) ;
          assert (g1.gk * l >= m) ;
          let s1 =
            make_stage ~exec:(of_geom g1.gols) ~l:g1.gl ~m:g1.gm ~k:g1.gk
              ~fc:g1.gfc ~beta ()
          in
          let s2 =
            make_stage ~exec:(of_geom g2.gols) ~l:g2.gl ~m:g2.gm ~k:g2.gk
              ~fc:g2.gfc ~beta ()
          in
          { sample_rate
          ; target
          ; quality
          ; l
          ; m
          ; latency= g1.gk + (g2.gk * g1.gm / g1.gl)
          ; stages= [s1; s2] }
      | None -> (
          (* the raise names the whole blocker — the single-stage bank is over
             budget and no two-stage split brings it under. The drift hint is
             reserved for ratios that actually look like drift: a wide ratio can
             land here under [`Custom], and telling that caller about clock-slew
             would only mislead *)
          if not single_fits then
            invalid_arg
              (Stdlib.Format.asprintf
                 "create: cannot resample %d Hz to %d Hz (%d phases need a %a \
                  bank; the budget is %a, and no two-stage split brings it \
                  under)%s"
                 sample_rate target l pp_bytes bank_bytes pp_bytes
                 (Float.of_int bank_budget_bytes)
                 ( if
                     Float.of_int (Stdlib.max l m)
                     < 1.01 *. Float.of_int (Stdlib.min l m)
                   then
                     " hint: near-unity conversion is clock-drift correction, \
                      which the fixed-ratio resampler does not do"
                   else "" ) ) ;
          if gemm_single then single Xgemm
          else
            match single_ols with
            | Some (_, geom) ->
                single (Xols geom)
            | None ->
                single Xdirect )
    end

  let sample_rate c = c.sample_rate

  let target c = c.target

  let quality c = c.quality

  let rate c = {Pipeline.Rate.num= c.l; den= c.m}

  let latency c = c.latency

  let output_latency c =
    let num = c.latency * c.l in
    if num = 0 then {Pipeline.Rate.num= 0; den= 1}
    else
      let g = gcd num c.m in
      {Pipeline.Rate.num= num / g; den= c.m / g}

  let output_frames c ~n =
    if n < 0 then
      invalid_arg
        (Printf.sprintf
           "output_frames: cannot resample a signal of length %d (length must \
            be non-negative)"
           n ) ;
    if n > 0 && n > Stdlib.max_int / c.l then
      invalid_arg
        (Printf.sprintf
           "output_frames: cannot resample a signal of length %d (n * %d \
            overflows)"
           n c.l ) ;
    ceil_pos (n * c.l) c.m

  let prototype dtype c =
    match c.stages with
    | [s] ->
        Nx.cast dtype (Nx.copy s.sproto)
    | [s1; s2] ->
        (* the equivalent-response lowpass of the plan: the two stage prototypes
           zero-stuffed onto their least common interpolation grid and
           convolved. Its frequency response is the product of the stage
           responses — the composite magnitude response that inspection and
           plots need — linear-phase, centered at [latency] input samples, and
           scaled to the unit-passband convention: unit DC gain on the common
           grid, so it sums to that grid's interpolation factor [lc / q] — equal
           to [L] only when the common grid is the composite's own ([`Fast] 8 →
           48 kHz sums to 78 with [L] = 6). Built by float convolution, so its
           symmetry is exact only to rounding, unlike a single-stage
           prototype's. *)
        let p0 = s1.sl * s2.sl and q0 = s1.sm in
        let gq = gcd p0 q0 in
        let p = p0 / gq and q = q0 / gq in
        let a_num = s1.sl * q in
        let lc = a_num / gcd a_num p * p in
        let stuff1 = lc / a_num and stuff2 = lc / p in
        let h1 = Nx.to_array s1.sproto and h2 = Nx.to_array s2.sproto in
        let n1 = Array.length h1 and n2 = Array.length h2 in
        let out =
          Array.make ((stuff1 * (n1 - 1)) + (stuff2 * (n2 - 1)) + 1) 0.
        in
        for i = 0 to n1 - 1 do
          let base = stuff1 * i and hi = h1.(i) in
          for j = 0 to n2 - 1 do
            out.(base + (stuff2 * j)) <-
              out.(base + (stuff2 * j)) +. (hi *. h2.(j))
          done
        done ;
        let scale = Float.of_int (lc / q) /. Float.of_int (s1.sl * s2.sl) in
        Array.iteri (fun i v -> out.(i) <- v *. scale) out ;
        Nx.cast dtype (Nx.create Nx.float64 [|Array.length out|] out)
    | _ ->
        assert false

  let pp fmt c =
    let quality fmt = function
      | `Fast ->
          Stdlib.Format.pp_print_string fmt "fast"
      | `High ->
          Stdlib.Format.pp_print_string fmt "high"
      | `Best ->
          Stdlib.Format.pp_print_string fmt "best"
      | `Custom {attenuation; passband} ->
          Stdlib.Format.fprintf fmt "custom(%g dB, %g)" attenuation passband
    in
    (* A stage prints the filter length its executor touches per output — [2K +
       1] for the direct dot product and for the matrix product's banded column,
       the whole [2*K*L + 1] prototype for an OLS stage (frequency-domain
       convolution runs the full filter) — and names the executor where it is
       not the dot product: OLS stages add their transform length, [N*F] points
       for ×F interpolators and [N] points for ÷F decimators. *)
    let stage_taps fmt s =
      match s.sexec with
      | Sdirect ->
          Stdlib.Format.fprintf fmt "%d" ((2 * s.sk) + 1)
      | Sgemm _ ->
          Stdlib.Format.fprintf fmt "%d(gemm)" ((2 * s.sk) + 1)
      | Sols o ->
          Stdlib.Format.fprintf fmt "%d(ols,N=%d)"
            ((2 * s.sk * s.sl) + 1)
            (o.on * s.sl)
    in
    if is_identity c then
      Stdlib.Format.fprintf fmt "resample(%d Hz, identity)" c.sample_rate
    else
      match c.stages with
      | [s] ->
          Stdlib.Format.fprintf fmt
            "resample(%d -> %d Hz, quality=%a, L/M=%d/%d, taps=%a, latency=%d)"
            c.sample_rate c.target quality c.quality c.l c.m stage_taps s
            c.latency
      | [s1; s2] ->
          Stdlib.Format.fprintf fmt
            "resample(%d -> %d Hz, quality=%a, L/M=%d/%d, stages=%d/%d:%a >> \
             %d/%d:%a, latency=%d)"
            c.sample_rate c.target quality c.quality c.l c.m s1.sl s1.sm
            stage_taps s1 s2.sl s2.sm stage_taps s2 c.latency
      | _ ->
          assert false

  let equal a b =
    a.sample_rate = b.sample_rate
    && a.target = b.target
    &&
    match (a.quality, b.quality) with
    | `Fast, `Fast | `High, `High | `Best, `Best ->
        true
    | `Custom x, `Custom y ->
        Float.equal x.attenuation y.attenuation
        && Float.equal x.passband y.passband
    | _ ->
        false
end

(* {1 The C seam} *)

(* One call per chunk per stage; the stub does all slicing internally, validates
   every extent against the arrays it was actually handed, and releases the
   runtime lock around the bulk work (hence no [@@noalloc]). Argument order:
   bank, history, scratch, input, output, n, n_out, channels, K, L, M, row0, s0,
   y_off and y_stride (the destination window: channel [c]'s run of [n_out]
   samples starts at [y_off + c*y_stride]), visit (the bank's layout: visit
   order when true, phase-major when false), is_flush. *)
external resample_step_c :
     (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> bool
  -> bool
  -> unit = "soundml_resample_step_bc" "soundml_resample_step"

(* The block identity of an overlap-save stage, between the two transforms:
   spectra [lines; N/2 + 1] and the plan spectrum in, the half grid of the
   length-[W] inverse transform out ([W] is [N*L] for a ×L stage, [N/M] for a ÷M
   one, [N] otherwise). Argument order: spectra, plan spectrum, destination,
   lines, N, L, M. Every line is shaped from its own line alone, in an order
   independent of the line count. *)
external resample_shape_c :
     (Complex.t, Bigarray.complex64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> (Complex.t, Bigarray.complex64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> (Complex.t, Bigarray.complex64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> int
  -> int
  -> int
  -> int
  -> unit = "soundml_resample_shape_bc" "soundml_resample_shape"

(* [stage_out_bound sp ~n] dominates the number of stage outputs any single call
   can emit for a feed of [n] samples, including the drain (which is the [n = 0]
   instance: the direct executor drains its [K]-sample lookahead, the OLS
   executor its buffered carry plus one padded block, the GEMM executor the
   calls its remaining outputs fall in). Sizes the hand-off buffer, the
   downstream scratch and the advisory pipeline bound. *)
let stage_out_bound sp ~n =
  match sp.sexec with
  | Sdirect ->
      ceil_pos (Stdlib.max n sp.sk * sp.sl) sp.sm + 1
  | Sgemm _ ->
      let rows = gemm_rows_of ~m:sp.sm in
      let per_call = rows * sp.sl and b = rows * sp.sm in
      let calls_step = ((Stdlib.max 1 n - 1) / b) + 2 in
      let calls_drain = (sp.sk / b) + 2 in
      Stdlib.max calls_step calls_drain * per_call
  | Sols o ->
      let per_block = if sp.sl > 1 then o.ob * sp.sl else o.ob / sp.sm in
      let blocks_step = ((Stdlib.max 1 n - 1) / o.ob) + 2 in
      let blocks_drain = (((2 * sp.sk) + o.odelta) / o.ob) + 2 in
      (Stdlib.max blocks_step blocks_drain * per_block) + (2 * sp.sm) + 2

(* {1 Incremental kernel} *)

module Kernel = struct
  type 'a direct_state =
    { bank: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
    ; visit: bool
          (* [bank]'s layout: [svisit] when the instantiated bank passes the L1
             edge (the walk then streams it forward), [sbank] under it (the
             phase walk is free on a resident bank, and the executor's
             phase-major loop carries no row counter) *)
    ; hist: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* [channels * 2K], planar *)
    ; scratch: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* [2K + max max_in K], one lane shared across channels *) }

  type 'a ols_state =
    { op: ols_plan
    ; ohs: Nx.complex128_t
          (* the config's spectrum, forced once per kernel family *)
    ; carry_t: (float, 'a) Nx.t (* [channels; N] — keeps [carry] alive *)
    ; carry: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* the grid tail: stage-input samples [next block window start ..
             fed), planar per channel; starts as the [2K + delta] virtual zeros
             before the stream *)
    ; mutable oblocks: int (* blocks executed so far *) }

  type 'a gemm_state =
    { gg: (float, 'a) Nx.t
          (* the config's [P; L] bank arrangement at the kernel dtype, forced
             and cast once per kernel *)
    ; gr: int (* the block-rows one call carries *)
    ; gp: int (* P: the stage inputs one block-row reads *)
    ; gb: int (* B = gr * M: the stage inputs one call advances *)
    ; gspan: int (* B + 2K: the stage inputs one call reads *)
    ; ga: (float, 'a) Nx.t
          (* the [gr; P] gather, the product's left operand; one call's worth,
             reused by every call *)
    ; gaa: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
    ; gcarry_t: (float, 'a) Nx.t (* [channels; B + 2K] — keeps [gcarry] alive *)
    ; gcarry: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* the window tail: stage inputs [next call window start .. fed),
             planar per channel; starts as the [K] virtual zeros before the
             stream *)
    ; mutable gcalls: int (* calls executed so far *) }

  type 'a exec_state =
    | Dx of 'a direct_state
    | Ox of 'a ols_state
    | Gx of 'a gemm_state

  type 'a stage_state =
    { sp: stage_plan
    ; ex: 'a exec_state
    ; mutable fed: int (* stage-input samples consumed so far *)
    ; mutable emitted: int (* stage-output samples emitted so far *) }

  type 'a t =
    { cfg: config
    ; dtype: (float, 'a) Nx.dtype
    ; channels: int
    ; max_block: int
    ; st: 'a stage_state array (* one or two, input to output order *)
    ; mid: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* the stage-1 -> stage-2 hand-off, planar [channels; mid_cap]; empty
             for single-stage plans. Allocated here, once: a direct [step]
             allocates exactly the output tensor (an OLS step additionally
             allocates its transform transients — the documented deviation until
             nx grows destination-passing) *)
    ; mutable drained: bool
    ; mutable leading: int array (* leading shape of the last chunk *) }

  (* [ready sp fed] is the number of stage outputs computable once [fed] input
     samples have arrived: output [i] needs input [floor (i*M/L) + K], so every
     [i] with [floor (i*M/L) <= fed - 1 - K] — that is [ceil ((fed - K) * L /
     M)] of them. Feeding the [K] virtual zeros of [flush] makes this exactly
     [ceil (fed * L / M)]: the deterministic-length contract per stage, and —
     because [L1*L2 / (M1*M2)] equals [L / M] exactly — the composite drain
     over-produces by at most one sample, which the flush truncates. *)
  let ready sp fed = ceil_pos ((fed - sp.sk) * sp.sl) sp.sm

  (* The OLS availability arithmetic — every quantity a function of the plan
     constants and totals alone, which is the partition-independence of the grid
     stated as code. Block [b] is executable once its window [b*B - 2K - delta
     .. +N) has fully arrived: [fed >= b*B + N - 2K - delta]. Its last
     computable output: the valid (wrap-free) region of the circular result is s
     in [2K, N), output [i] sits at s = i*M + 3K + delta - b*B (decimated grids
     read s = i + 3KF - F*b*B on the interpolated grid), so block [b] extends
     the emitted run to [hi b] and the runs of consecutive blocks tile the
     output axis without gap or overlap. *)
  let ols_blocks sp o fed =
    let need0 = o.on - (2 * sp.sk) - o.odelta in
    if fed < need0 then 0 else ((fed - need0) / o.ob) + 1

  let ols_hi sp o b =
    if sp.sl > 1 then (sp.sl * ((b * o.ob) + o.on - (3 * sp.sk))) - 1
    else ((b * o.ob) + o.on - (3 * sp.sk) - o.odelta - 1) / sp.sm

  let ols_avail sp o fed =
    let nb = ols_blocks sp o fed in
    if nb = 0 then 0 else ols_hi sp o (nb - 1) + 1

  (* The GEMM availability arithmetic, the same shape: call [b] reads stage
     inputs [b*B - K, (b+1)*B + K), so it is executable once [fed >= (b+1)*B +
     K], and every executed call emits its whole [R*L] run. Emission therefore
     stays [K*L/M] outputs behind the direct executor's, which is what keeps the
     composite truncation inside the drain. *)
  let gemm_calls sp gs fed =
    if fed < gs.gb + sp.sk then 0 else (fed - sp.sk) / gs.gb

  let gemm_avail sp gs fed = gemm_calls sp gs fed * gs.gr * sp.sl

  (* [avail st fed] is the stage-output count computable at [fed] input samples:
     the direct kernel emits as soon as an output's window closes, the OLS and
     GEMM executors when the block or call containing it completes. *)
  let avail st fed =
    match st.ex with
    | Dx _ ->
        ready st.sp fed
    | Ox os ->
        ols_avail st.sp os.op fed
    | Gx gs ->
        gemm_avail st.sp gs fed

  let prepare cfg dtype ~channels ~max_block =
    check_dtype "prepare" dtype ;
    if channels < 1 then
      invalid_arg
        (Printf.sprintf
           "prepare: cannot resample %d channels (channels must be at least 1)"
           channels ) ;
    if max_block < 1 then
      invalid_arg
        (Printf.sprintf
           "prepare: cannot accept blocks of %d samples (max_block must be at \
            least 1)"
           max_block ) ;
    let mk max_in sp =
      let ex =
        match sp.sexec with
        | Sgemm bank ->
            let p_len = (2 * sp.sk) + 1 + ((sp.sl - 1) * sp.sm / sp.sl) in
            let rows = gemm_rows_of ~m:sp.sm in
            let span = (rows * sp.sm) + (2 * sp.sk) in
            let ga = Nx.empty dtype [|rows; p_len|] in
            let gcarry_t = Nx.zeros dtype [|channels; span|] in
            Gx
              { gg= Nx.cast dtype (Lazy.force bank)
              ; gr= rows
              ; gp= p_len
              ; gb= rows * sp.sm
              ; gspan= span
              ; ga
              ; gaa= array1_of ga
              ; gcarry_t
              ; gcarry= array1_of gcarry_t
              ; gcalls= 0 }
        | Sdirect ->
            let hist = array1_of (Nx.zeros dtype [|channels * 2 * sp.sk|]) in
            let elt = Bigarray.kind_size_in_bytes (Bigarray.Array1.kind hist) in
            let visit = sp.sl * ((2 * sp.sk) + 1) * elt > l1_edge_bytes in
            Dx
              { bank=
                  array1_of
                    (Nx.cast dtype (if visit then sp.svisit else sp.sbank))
              ; visit
              ; hist
              ; scratch=
                  array1_of
                    (Nx.empty dtype [|(2 * sp.sk) + Stdlib.max max_in sp.sk|])
              }
        | Sols o ->
            let carry_t = Nx.zeros dtype [|channels; o.on|] in
            Ox
              { op= o
              ; ohs= Lazy.force o.oh
              ; carry_t
              ; carry= array1_of carry_t
              ; oblocks= 0 }
      in
      {sp; ex; fed= 0; emitted= 0}
    in
    match cfg.stages with
    | [s] ->
        { cfg
        ; dtype
        ; channels
        ; max_block
        ; st= [|mk max_block s|]
        ; mid= array1_of (Nx.empty dtype [|0|])
        ; drained= false
        ; leading= [|channels|] }
    | [s1; s2] ->
        (* one call's worth of stage-1 output, whatever its executor: steps
           bounded by [max_block], the drain by the stage's lookahead *)
        let cap = stage_out_bound s1 ~n:max_block in
        { cfg
        ; dtype
        ; channels
        ; max_block
        ; st= [|mk max_block s1; mk cap s2|]
        ; mid= array1_of (Nx.empty dtype [|channels * cap|])
        ; drained= false
        ; leading= [|channels|] }
    | _ ->
        assert false

  (* [direct_run k st dx ~n ~n_out ~y_off ~y_stride ~is_flush x y] drives the C
     executor for one stage over [n] stage-input samples ([x] when streaming,
     virtual zeros when draining), writing [n_out] freshly computed outputs per
     channel into [y] at [y_off], channel stride [y_stride]. The state of the
     first output [i = emitted] is exact integer arithmetic: [t = i * M], bank
     row [i mod L] in visit order (the executor reconstructs the phase [t mod L]
     from it; in phase-major layout it recomputes the row too), window start [t
     / L + K - fed] in scratch coordinates. Advances [emitted]; the caller
     advances [fed] — a drain feeds no real samples. *)
  let direct_run k st dx ~n ~n_out ~y_off ~y_stride ~is_flush x y =
    let t = st.emitted * st.sp.sm in
    let row0 = st.emitted mod st.sp.sl in
    let s0 = (t / st.sp.sl) + st.sp.sk - st.fed in
    resample_step_c dx.bank dx.hist dx.scratch x y n n_out k.channels st.sp.sk
      st.sp.sl st.sp.sm row0 s0 y_off y_stride dx.visit is_flush ;
    st.emitted <- st.emitted + n_out

  type 'a feed_src =
    | From of (float, 'a, Bigarray.c_layout) Bigarray.Array1.t * int
      (* packed planar source and its per-channel stride *)
    | Silence

  (* [ols_run k st os ~src ~n ~n_out ~y_off ~y_stride y] feeds [n] stage-input
     samples into an OLS stage, executes every block that completes — in order,
     exactly once each, always the same call shape — and writes this call's
     [n_out] outputs per channel into [y] at [y_off], channel stride [y_stride].
     The one caller-visible asymmetry against [direct_run]: [n_out] may be
     smaller than what the executed blocks make available only on the truncating
     drain, where the surplus is exactly the beyond-[output_frames] tail and is
     discarded. *)
  let ols_run k st os ~src ~n ~n_out ~y_off ~y_stride y =
    let sp = st.sp and o = os.op in
    let ch = k.channels in
    let kk = sp.sk in
    let lead = (2 * kk) + o.odelta in
    let pend = st.fed + lead - (os.oblocks * o.ob) in
    let alen = pend + n in
    if alen < o.on then begin
      (* no block completes: the samples only extend the carry *)
      ( match src with
      | From (s, stride) ->
          for c = 0 to ch - 1 do
            Bigarray.Array1.blit
              (Bigarray.Array1.sub s (c * stride) n)
              (Bigarray.Array1.sub os.carry ((c * o.on) + pend) n)
          done
      | Silence ->
          for c = 0 to ch - 1 do
            Bigarray.Array1.fill
              (Bigarray.Array1.sub os.carry ((c * o.on) + pend) n)
              0.
          done ) ;
      st.emitted <- st.emitted + n_out
    end
    else begin
      let t = ((alen - o.on) / o.ob) + 1 in
      (* assemble the grid span [window of block oblocks .. fed + n) as one
         fresh planar tensor: carry, then this call's samples *)
      let av_t = Nx.empty k.dtype [|ch; alen|] in
      let av = array1_of av_t in
      for c = 0 to ch - 1 do
        Bigarray.Array1.blit
          (Bigarray.Array1.sub os.carry (c * o.on) pend)
          (Bigarray.Array1.sub av (c * alen) pend) ;
        match src with
        | From (s, stride) ->
            Bigarray.Array1.blit
              (Bigarray.Array1.sub s (c * stride) n)
              (Bigarray.Array1.sub av ((c * alen) + pend) n)
        | Silence ->
            Bigarray.Array1.fill
              (Bigarray.Array1.sub av ((c * alen) + pend) n)
              0.
      done ;
      let w = if sp.sl > 1 then o.on * sp.sl else o.on / sp.sm in
      (* [transform_spec x] is the block identity from the half spectrum on: the
         plan spectrum applied on the inverse transform's half grid -> [irfft]
         at the kernel dtype (last axis [w]). Leading axes are transform lines,
         shaped one from the other's data alone.

         The spectrum reaches it two ways — one block's [rfft], or one batched
         [rfft] over many gathered blocks — and the two agree bit for bit. *)
      let transform_spec x =
        let lead_shape = Array.sub (Nx.shape x) 0 (Nx.ndim x - 1) in
        let lines = Array.fold_left ( * ) 1 lead_shape in
        let shaped =
          Nx.empty Nx.complex128 (Array.append lead_shape [|(w / 2) + 1|])
        in
        resample_shape_c (array1_of x) (array1_of os.ohs) (array1_of shaped)
          lines o.on sp.sl sp.sm ;
        Nx.irfft k.dtype ~n:w shaped
      in
      let transform v = transform_spec (Nx.rfft Nx.complex128 v) in
      (* [framed_spec ~j0 t0] is the half spectrum of the [t0] overlap-save
         blocks starting at block [j0]: the blocks are gathered into a
         contiguous [ch; t0; on] scratch and carried by one batched transform.
         The gather moves exactly the bytes a strided read would have moved, and
         the batch is bit-identical to the per-block loop under the standing
         probe. *)
      let framed_spec ~j0 t0 =
        let f = Nx.empty k.dtype [|ch; t0; o.on|] in
        let fa = array1_of f in
        for c = 0 to ch - 1 do
          for j = 0 to t0 - 1 do
            Bigarray.Array1.blit
              (Bigarray.Array1.sub av ((c * alen) + ((j0 + j) * o.ob)) o.on)
              (Bigarray.Array1.sub fa (((c * t0) + j) * o.on) o.on)
          done
        done ;
        Nx.rfft Nx.complex128 f
      in
      (* emit: block [b]'s run extends the output to [ols_hi b]; copy each
         block's contiguous kept span to its place in [y] *)
      let out_pos = ref 0 in
      let emit r_arr ~stack ~j0 =
        for j = 0 to stack - 1 do
          let b = os.oblocks + j0 + j in
          let i0 = st.emitted + !out_pos in
          let cnt = Stdlib.min (ols_hi sp o b + 1 - i0) (n_out - !out_pos) in
          if cnt > 0 then begin
            let pos =
              if sp.sl > 1 then i0 + (sp.sl * ((3 * kk) - (b * o.ob)))
              else ((i0 * sp.sm) + (3 * kk) + o.odelta - (b * o.ob)) / sp.sm
            in
            for c = 0 to ch - 1 do
              Bigarray.Array1.blit
                (Bigarray.Array1.sub r_arr ((((c * stack) + j) * w) + pos) cnt)
                (Bigarray.Array1.sub y
                   (y_off + (c * y_stride) + (i0 - st.emitted))
                   cnt )
            done ;
            out_pos := !out_pos + cnt
          end
        done
      in
      if ols_batch then begin
        (* stacked transforms over this call's blocks, in tiles of at most
           [ols_tile_lines] lines — bit-identical to the per-block loop (and to
           any other stacking) under the standing probe, so the tile bound only
           caps the per-call transient, never the result *)
        let tile = Stdlib.max 1 (ols_tile_lines / ch) in
        if t <= tile then
          (* the common shape — every step and every clip under ~15 s mono: one
             stack, no tiling views, exactly the pre-tiling call *)
          emit (array1_of (transform_spec (framed_spec ~j0:0 t))) ~stack:t ~j0:0
        else begin
          let j0 = ref 0 in
          while !j0 < t do
            let tsz = Stdlib.min tile (t - !j0) in
            emit
              (array1_of (transform_spec (framed_spec ~j0:!j0 tsz)))
              ~stack:tsz ~j0:!j0 ;
            j0 := !j0 + tsz
          done
        end
      end
      else
        for j = 0 to t - 1 do
          let v = Nx.shrink [|(0, ch); (j * o.ob, (j * o.ob) + o.on)|] av_t in
          emit (array1_of (transform v)) ~stack:1 ~j0:j
        done ;
      assert (!out_pos = n_out) ;
      (* the grid tail becomes the next carry *)
      let pend' = alen - (t * o.ob) in
      for c = 0 to ch - 1 do
        Bigarray.Array1.blit
          (Bigarray.Array1.sub av ((c * alen) + (t * o.ob)) pend')
          (Bigarray.Array1.sub os.carry (c * o.on) pend')
      done ;
      os.oblocks <- os.oblocks + t ;
      st.emitted <- st.emitted + n_out
    end

  (* [gemm_run k st gs ~src ~n ~n_out ~y_off ~y_stride y] feeds [n] stage-input
     samples into a GEMM stage, executes every call that completes — in order,
     exactly once each, always the same [R; P] by [P; L] product — and writes
     this call's [n_out] outputs per channel into [y] at [y_off], channel stride
     [y_stride]. Like [ols_run], [n_out] falls short of what the executed
     calls make available only on the truncating drain, where the surplus is
     exactly the beyond-[output_frames] tail. *)
  let gemm_run k st gs ~src ~n ~n_out ~y_off ~y_stride y =
    let sp = st.sp in
    let ch = k.channels in
    let kk = sp.sk in
    let span = gs.gspan in
    (* the window samples already held: the [K] virtual zeros of the stream
       start, then everything fed past the current call's window start *)
    let pend = st.fed + kk - (gs.gcalls * gs.gb) in
    let alen = pend + n in
    (* [gather c j] lays call [j]'s block-rows into the product's left operand:
       channel [c]'s stage inputs [j*B, j*B + B + 2K), cut into the [R] windows
       of [P] samples advancing [M] that the rows read. Window offsets below
       [held] are still in the carry, the rest are in this call's input, so a
       row crossing that point is two runs and every other row is one. *)
    let gather c j =
      let off = j * gs.gb in
      let held = Stdlib.min span (Stdlib.max 0 (pend - off)) in
      for r = 0 to gs.gr - 1 do
        let w = r * sp.sm and dst = r * gs.gp in
        let kept = Stdlib.min gs.gp (Stdlib.max 0 (held - w)) in
        if kept > 0 then
          Bigarray.Array1.blit
            (Bigarray.Array1.sub gs.gcarry ((c * span) + off + w) kept)
            (Bigarray.Array1.sub gs.gaa dst kept) ;
        let rest = gs.gp - kept in
        if rest > 0 then
          let into = Bigarray.Array1.sub gs.gaa (dst + kept) rest in
          match src with
          | From (s, stride) ->
              Bigarray.Array1.blit
                (Bigarray.Array1.sub s
                   ((c * stride) + off + w + kept - pend)
                   rest )
                into
          | Silence ->
              Bigarray.Array1.fill into 0.
      done
    in
    let t = if alen < span then 0 else ((alen - span) / gs.gb) + 1 in
    if t > 0 then begin
      let per_call = gs.gr * sp.sl in
      let out_pos = ref 0 in
      for j = 0 to t - 1 do
        let b = gs.gcalls + j in
        let i0 = st.emitted + !out_pos in
        let cnt = Stdlib.min (((b + 1) * per_call) - i0) (n_out - !out_pos) in
        if cnt > 0 then begin
          let pos = i0 - (b * per_call) in
          for c = 0 to ch - 1 do
            gather c j ;
            let res = array1_of (Nx.matmul gs.ga gs.gg) in
            Bigarray.Array1.blit
              (Bigarray.Array1.sub res pos cnt)
              (Bigarray.Array1.sub y
                 (y_off + (c * y_stride) + (i0 - st.emitted))
                 cnt )
          done ;
          out_pos := !out_pos + cnt
        end
      done ;
      assert (!out_pos = n_out)
    end ;
    (* the window tail becomes the next carry: what the executed calls left
       behind, then whatever of this call's samples follows it *)
    let base = t * gs.gb in
    let pend' = alen - base in
    for c = 0 to ch - 1 do
      let held = Stdlib.min pend' (Stdlib.max 0 (pend - base)) in
      (* what the executed calls left behind moves down by their advance; with
         no call executed it is already in place. The move is strictly leftward,
         so a forward copy is in order *)
      if base > 0 then
        for i = 0 to held - 1 do
          Bigarray.Array1.unsafe_set gs.gcarry
            ((c * span) + i)
            (Bigarray.Array1.unsafe_get gs.gcarry ((c * span) + base + i))
        done ;
      let rest = pend' - held in
      if rest > 0 then
        let s0 = base + held - pend in
        let dst = Bigarray.Array1.sub gs.gcarry ((c * span) + held) rest in
        match src with
        | From (s, stride) ->
            Bigarray.Array1.blit
              (Bigarray.Array1.sub s ((c * stride) + s0) rest)
              dst
        | Silence ->
            Bigarray.Array1.fill dst 0.
    done ;
    gs.gcalls <- gs.gcalls + t ;
    st.emitted <- st.emitted + n_out

  (* [feed k st ~n ~n_out ~y_off ~y_stride x y] advances stage [st] over [n]
     real samples read planar from [x] (per-channel stride [n]), emitting
     [n_out] outputs per channel into [y] at [y_off], channel stride [y_stride];
     [feed0] is the no-emission instance (the call still threads state). [drain]
     extends the stage with virtual silence and emits its remaining
     [n_out]-sample tail. *)
  let feed k st ~n ~n_out ~y_off ~y_stride x y =
    match st.ex with
    | Dx d ->
        direct_run k st d ~n ~n_out ~y_off ~y_stride ~is_flush:false x y
    | Ox os ->
        ols_run k st os ~src:(From (x, n)) ~n ~n_out ~y_off ~y_stride y
    | Gx gs ->
        gemm_run k st gs ~src:(From (x, n)) ~n ~n_out ~y_off ~y_stride y

  let feed0 k st ~n x =
    match st.ex with
    | Dx d ->
        direct_run k st d ~n ~n_out:0 ~y_off:0 ~y_stride:0 ~is_flush:false x
          d.scratch
    | Ox os ->
        ols_run k st os
          ~src:(From (x, n))
          ~n ~n_out:0 ~y_off:0 ~y_stride:0 os.carry
    | Gx gs ->
        gemm_run k st gs
          ~src:(From (x, n))
          ~n ~n_out:0 ~y_off:0 ~y_stride:0 gs.gcarry

  let drain k st ~n_out ~y_off ~y_stride y =
    match st.ex with
    | Dx d ->
        direct_run k st d ~n:st.sp.sk ~n_out ~y_off ~y_stride ~is_flush:true
          d.hist y
    | Gx gs ->
        (* exactly enough virtual zeros to complete the calls that cover the
           remaining outputs — a function of totals alone *)
        let per_call = gs.gr * st.sp.sl in
        let target = st.emitted + n_out in
        let b = ref gs.gcalls in
        while (!b + 1) * per_call < target do
          incr b
        done ;
        let zeros = ((!b + 1) * gs.gb) + st.sp.sk - st.fed in
        gemm_run k st gs ~src:Silence ~n:zeros ~n_out ~y_off ~y_stride y
    | Ox os ->
        (* exactly enough virtual zeros to complete the blocks that cover the
           remaining outputs — a function of totals alone *)
        let o = os.op in
        let target = st.emitted + n_out in
        let b = ref os.oblocks in
        while ols_hi st.sp o !b < target - 1 do
          incr b
        done ;
        let zeros = (!b * o.ob) + o.on - (2 * st.sp.sk) - o.odelta - st.fed in
        ols_run k st os ~src:Silence ~n:zeros ~n_out ~y_off ~y_stride y

  (* [outputs k n] is what a [n]-sample chunk emits per channel, and
     [drain_outputs k] what the tail behind it does. Both are functions of the
     stage states and the plan alone — no executor runs — so a destination can
     be sized, and placed, before any sample moves. *)
  let outputs k n =
    match k.st with
    | [|s|] ->
        avail s (s.fed + n) - s.emitted
    | [|s1; s2|] ->
        let n1 = avail s1 (s1.fed + n) - s1.emitted in
        avail s2 (s2.fed + n1) - s2.emitted
    | _ ->
        assert false

  let drain_outputs k =
    match k.st with
    | [|s|] ->
        ceil_pos (s.fed * s.sp.sl) s.sp.sm - s.emitted
    | [|s1; s2|] ->
        ceil_pos (s1.fed * k.cfg.l) k.cfg.m - s2.emitted
    | _ ->
        assert false

  (* [run k chunk ~n ~y_off ~y_stride y] pushes [n] samples through the plan,
     writing the [outputs k n] results per channel into [y] at [y_off], channel
     stride [y_stride]. A cascade lands stage 1 in the hand-off buffer: its
     output sequence is partition-independent (stage 1 is chunk-invariant), and
     stage 2 is invariant to how that sequence reaches it — the composite keeps
     the law by composition, not by re-proof. *)
  let run k chunk ~n ~y_off ~y_stride y =
    let x = array1_of chunk in
    match k.st with
    | [|s|] ->
        let n_out = avail s (s.fed + n) - s.emitted in
        if n_out = 0 then feed0 k s ~n x
        else feed k s ~n ~n_out ~y_off ~y_stride x y ;
        s.fed <- s.fed + n
    | [|s1; s2|] ->
        let n1 = avail s1 (s1.fed + n) - s1.emitted in
        if n1 = 0 then feed0 k s1 ~n x
        else feed k s1 ~n ~n_out:n1 ~y_off:0 ~y_stride:n1 x k.mid ;
        s1.fed <- s1.fed + n ;
        if n1 > 0 then begin
          let n2 = avail s2 (s2.fed + n1) - s2.emitted in
          if n2 = 0 then feed0 k s2 ~n:n1 k.mid
          else feed k s2 ~n:n1 ~n_out:n2 ~y_off ~y_stride k.mid y ;
          s2.fed <- s2.fed + n1
        end
    | _ ->
        assert false

  (* [drain_run k ~y_off ~y_stride y] writes the [drain_outputs k] tail. A
     cascade drains in stage order and truncates: stage 1 flushes its exact ceil
     tail into the hand-off buffer, the tail streams through stage 2, stage 2
     flushes its own tail — and the composite stream is cut to [output_frames].
     The cut only ever lands in this drain: [ceil (ceil (n*L1/M1) * L2/M2) >=
     ceil (n*L/M)] (the stage rationals multiply to exactly [L/M]), and
     mid-stream emission never passes the composite ceil because [K1*L >= M] —
     checked at create — keeps [emitted <= fed*L/M + 1 - K1*L/M], and an OLS
     stage withholds at least as much (a block only emits outputs whose full
     lookahead lies inside it). Both counts are functions of the totals alone,
     so the cut is partition-independent. *)
  let drain_run k ~y_off ~y_stride y =
    match k.st with
    | [|s|] ->
        let n_out = ceil_pos (s.fed * s.sp.sl) s.sp.sm - s.emitted in
        if n_out > 0 then drain k s ~n_out ~y_off ~y_stride y
    | [|s1; s2|] ->
        let keep = ceil_pos (s1.fed * k.cfg.l) k.cfg.m - s2.emitted in
        if keep > 0 then begin
          let n1 = ceil_pos (s1.fed * s1.sp.sl) s1.sp.sm - s1.emitted in
          if n1 > 0 then drain k s1 ~n_out:n1 ~y_off:0 ~y_stride:n1 k.mid ;
          let n2a = Stdlib.min (avail s2 (s2.fed + n1) - s2.emitted) keep in
          if n2a = 0 then begin
            if n1 > 0 then
              (* thread the tail into stage-2 state: the stage-2 drain below
                 still reads it *)
              feed0 k s2 ~n:n1 k.mid
          end
          else feed k s2 ~n:n1 ~n_out:n2a ~y_off ~y_stride k.mid y ;
          s2.fed <- s2.fed + n1 ;
          let n2b = keep - n2a in
          if n2b > 0 then drain k s2 ~n_out:n2b ~y_off:(y_off + n2a) ~y_stride y
        end
    | _ ->
        assert false

  let step k chunk =
    if k.drained then
      invalid_arg
        "step: cannot feed a drained kernel (flush consumed the tail; reset \
         before reusing)" ;
    check_rank "step" chunk ;
    let n = last_dim chunk in
    if n > k.max_block then
      invalid_arg
        (Printf.sprintf "step: cannot feed a %d-sample chunk (max_block is %d)"
           n k.max_block ) ;
    let lead = leading_shape chunk in
    let channels = Array.fold_left ( * ) 1 lead in
    if channels <> k.channels then
      invalid_arg
        (Printf.sprintf
           "step: cannot feed %d-channel chunks (the kernel was prepared for \
            %d %s)"
           channels k.channels
           (if k.channels = 1 then "channel" else "channels") ) ;
    k.leading <- lead ;
    if n = 0 then None
    else
      let n_out = outputs k n in
      if n_out = 0 then begin
        (* nothing computable yet: the call only threads the state *)
        run k chunk ~n ~y_off:0 ~y_stride:0 k.mid ;
        None
      end
      else begin
        let out = Nx.empty k.dtype (Array.append k.leading [|n_out|]) in
        run k chunk ~n ~y_off:0 ~y_stride:n_out (array1_of out) ;
        Some out
      end

  let flush k =
    if k.drained then None
    else begin
      k.drained <- true ;
      let n_out = drain_outputs k in
      if n_out = 0 then None
      else begin
        let out = Nx.empty k.dtype (Array.append k.leading [|n_out|]) in
        drain_run k ~y_off:0 ~y_stride:n_out (array1_of out) ;
        Some out
      end
    end

  let reset k =
    Array.iter
      (fun s ->
        ( match s.ex with
        | Dx d ->
            Bigarray.Array1.fill d.hist 0.
        | Ox os ->
            Bigarray.Array1.fill os.carry 0. ;
            os.oblocks <- 0
        | Gx gs ->
            Bigarray.Array1.fill gs.gcarry 0. ;
            gs.gcalls <- 0 ) ;
        s.fed <- 0 ;
        s.emitted <- 0 )
      k.st ;
    k.drained <- false ;
    k.leading <- [|k.channels|]
end

(* {1 Offline} *)

let apply c x =
  check_rank "apply" x ;
  check_dtype "apply" (Nx.dtype x) ;
  if is_identity c then x
  else
    let n = last_dim x in
    let lead = leading_shape x in
    let total = Config.output_frames c ~n in
    let channels = Array.fold_left ( * ) 1 lead in
    if channels = 0 || n = 0 then
      Nx.zeros (Nx.dtype x) (Array.append lead [|total|])
    else begin
      let k = Kernel.prepare c (Nx.dtype x) ~channels ~max_block:n in
      (* the whole conversion is one step and its drain, both writing their own
         run of the result: the offline call owns the destination, so the two
         land in place instead of being joined afterwards *)
      let out = Nx.empty (Nx.dtype x) (Array.append lead [|total|]) in
      let y = array1_of out in
      let stepped = Kernel.outputs k n in
      Kernel.run k x ~n ~y_off:0 ~y_stride:total y ;
      assert (Kernel.drain_outputs k = total - stepped) ;
      Kernel.drain_run k ~y_off:stepped ~y_stride:total y ;
      out
    end

(* {1 Offline, tensor formulation}

   The GEMM surface: the same conversion as [apply], written as dense tensor
   expressions over the same config-owned filters — one patches-times-matrix
   product per plan stage, so the two surfaces always compute the same filter
   architecture. Outputs of a stage are grouped into blocks of [L] (one full
   phase cycle): block [b] holds outputs [b*L + r], [r] in [0..L-1], and output
   [b*L + r] reads the input window starting at [b*M + floor(r*M/L) - K]. One
   patch of [P = 2K + 1 + floor((L-1)*M/L)] samples per block covers all [L]
   windows, so the stage is patches [n_frames; P] times a [P; L] arrangement of
   the bank — the strided formulation whose arithmetic redundancy is [P / taps].
   The matrix product sums in whatever order the backend blocks it, which is
   exactly why this surface is documented as numerically distinct and carries no
   partitioning law. The [P; L] matrix itself is [gemm_bank] above, built once
   per config on the first call.

   A cascade runs two such stages over the plan's two banks: stage 2 consumes
   stage 1's full flushed stream (exactly [ceil (n*L1/M1)] samples — the
   per-stage length contract) and computes exactly the composite [ceil (n*L/M)]
   outputs, the same truncation the kernel's drain applies. *)

let gemm_stage ~sl ~sm ~sk sgemm ~total dtype lead x =
  let taps = (2 * sk) + 1 in
  let p_len = taps + ((sl - 1) * sm / sl) in
  let frames = ceil_pos total sl in
  let n = last_dim x in
  (* [K] zeros on the left (the delay-compensated window of output 0 starts at
     input [-K]); on the right, exactly enough for the last block's patch. The
     clamp only ever discards surplus signal, and the frame axis is cut back to
     [frames] below either way. *)
  let right = Stdlib.max 0 (((frames - 1) * sm) + p_len - (sk + n)) in
  let pad_spec =
    Array.init
      (Array.length lead + 1)
      (fun i -> if i = Array.length lead then (sk, right) else (0, 0))
  in
  let padded = Nx.pad pad_spec 0. x in
  let patches =
    Nx.extract_patches ~kernel_size:[|p_len|] ~stride:[|sm|] ~dilation:[|1|]
      ~padding:[|(0, 0)|]
      padded
  in
  (* [lead ++ [P; frames']] with [frames' >= frames]; keep [frames] *)
  let rank = Nx.ndim patches in
  let patches =
    Nx.shrink
      (Array.init rank (fun i ->
           if i = rank - 1 then (0, frames) else (0, Nx.dim i patches) ) )
      patches
  in
  let axes =
    List.init rank (fun i ->
        if i = rank - 2 then rank - 1 else if i = rank - 1 then rank - 2 else i )
  in
  let y =
    Nx.matmul (Nx.transpose ~axes patches) (Nx.cast dtype (Lazy.force sgemm))
  in
  let y = Nx.reshape (Array.append lead [|frames * sl|]) y in
  Nx.shrink
    (Array.init (Nx.ndim y) (fun i ->
         if i = Nx.ndim y - 1 then (0, total) else (0, Nx.dim i y) ) )
    y

let apply_gemm c x =
  check_rank "apply_gemm" x ;
  check_dtype "apply_gemm" (Nx.dtype x) ;
  if is_identity c then x
  else
    let dtype = Nx.dtype x in
    let n = last_dim x in
    let lead = leading_shape x in
    let total = Config.output_frames c ~n in
    let channels = Array.fold_left ( * ) 1 lead in
    if channels = 0 || n = 0 then Nx.zeros dtype (Array.append lead [|total|])
    else
      (* small-L stages run blocked (see [gemm_block]): the same conversion in
         the patch geometry of the unreduced [B*L / B*M] form *)
      let run_stage s ~total x =
        let blocks = gemm_blocks s.sl in
        gemm_stage ~sl:(blocks * s.sl) ~sm:(blocks * s.sm) ~sk:s.sk s.sgemm
          ~total dtype lead x
      in
      match c.stages with
      | [s] ->
          run_stage s ~total x
      | [s1; s2] ->
          let t1 = ceil_pos (n * s1.sl) s1.sm in
          run_stage s2 ~total (run_stage s1 ~total:t1 x)
      | _ ->
          assert false

(* {1 Pipeline stage} *)

type 'a stage_state =
  { cfg: config
  ; channels: int
  ; bound: int option
  ; mutable kern: 'a Kernel.t option
        (* prepared on the first chunk: the chunk witnesses the element dtype,
           which the dynamic format only carries existentially *) }

let stage_kernel s chunk =
  match s.kern with
  | Some k ->
      k
  | None ->
      let max_block =
        match s.bound with
        | Some b ->
            b
        | None ->
            (* unbounded chunks: the offline driver pushes exactly once *)
            Stdlib.max 1 (last_dim chunk)
      in
      let k =
        Kernel.prepare s.cfg (Nx.dtype chunk) ~channels:s.channels ~max_block
      in
      s.kern <- Some k ;
      k

(* [stage_bound cfg b] is the advisory per-chunk output bound the stage threads
   downstream — downstream kernels size their [max_block] from it, so it must
   dominate every step and the drain, whatever each stage's executor: for direct
   stages that is [ceil (b*L/M) + 1] per length-[b] chunk, for FFT-executed
   stages at least one full block and for GEMM-executed stages at least one full
   call — a step that completes blocks or calls emits them whole, burst emission
   being those executors' cadence. The GEMM cadence is the coarsest of the
   three: one call spans [R*M] stage inputs, so a stage emits nothing until that
   many have arrived and then emits [R*L] outputs at once. A cascade compounds
   the two per-stage bounds, and its drain — stage-1 tail through stage 2 plus
   stage-2's own tail, delivered as one chunk — can exceed both, so it enters
   the max explicitly. *)
let stage_bound cfg b =
  match cfg.stages with
  | [s] ->
      stage_out_bound s ~n:b
  | [s1; s2] ->
      let through n = stage_out_bound s2 ~n:(stage_out_bound s1 ~n) in
      let drain =
        stage_out_bound s2 ~n:(stage_out_bound s1 ~n:0)
        + stage_out_bound s2 ~n:0
      in
      Stdlib.max (through b + 1) drain
  | _ ->
      assert false

let stage cfg =
  let witness = ref None in
  let concat = function
    | [] -> (
      match !witness with
      | Some (dtype, leading) ->
          Nx.zeros dtype (Array.append leading [|0|])
      | None ->
          invalid_arg
            "stage: cannot concatenate zero chunks before any chunk fixed the \
             element dtype" )
    | parts ->
        Nx.concatenate ~axis:(-1) parts
  in
  let prepare fmt =
    let ips = Pipeline.Format.items_per_second fmt in
    if not (Pipeline.Rate.equal ips {num= cfg.sample_rate; den= 1}) then
      invalid_arg
        (Stdlib.Format.asprintf
           "prepare: cannot resample a stream at %a items/s (the configuration \
            converts from %d Hz)"
           Pipeline.Rate.pp ips cfg.sample_rate ) ;
    { cfg
    ; channels= Pipeline.Format.channels fmt
    ; bound= Pipeline.Format.max_items fmt
    ; kern= None }
  in
  let step s chunk =
    witness := Some (Nx.dtype chunk, leading_shape chunk) ;
    Kernel.step (stage_kernel s chunk) chunk
  in
  let flush s =
    match s.kern with None -> [] | Some k -> Option.to_list (Kernel.flush k)
  in
  let reset s = Option.iter Kernel.reset s.kern in
  if is_identity cfg then
    (* transparent to the accounting: latency 0, rate 1/1, format untouched —
       but the chunk-ownership contract still demands a fresh output, so the
       identity kernel copies through the same executor *)
    Pipeline.kernel ~flush ~reset ~concat ~prepare ~step ()
  else
    let rate = {Pipeline.Rate.num= cfg.l; den= cfg.m} in
    let out_format fmt =
      let ips = Pipeline.Rate.(Pipeline.Format.items_per_second fmt * rate) in
      let bound =
        Option.map (fun b -> stage_bound cfg b) (Pipeline.Format.max_items fmt)
      in
      fmt
      |> Pipeline.Format.with_items_per_second ips
      |> Pipeline.Format.with_max_items bound
    in
    Pipeline.kernel ~latency:cfg.latency ~rate ~out_format ~flush ~reset ~concat
      ~prepare ~step ()
