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

   The C executor (resample_stubs.c) computes each output as one dot product
   over [history ++ chunk] with a fixed summation order, so offline equals
   streaming bit for bit on every partitioning — the pipeline law is structural,
   not tested-in. A cascade chains two such executors: stage 1 is
   chunk-invariant, so the sample sequence entering stage 2 does not depend on
   the input partitioning, and stage 2 is invariant to how that sequence is
   partitioned — the same argument, applied twice, makes the composite
   chunk-invariant. OCaml orchestrates at chunk granularity only: exact integer
   phase state, no float accumulator anywhere. *)

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

(* [gemm_bank ~l ~m ~k bank] is the [P; L] matrix [G] of the tensor formulation
   ([apply_gemm]), with [P = 2K + 1 + floor((L-1)*M/L)] and [G[q, r] =
   bank[(r*M) mod L, q - floor(r*M/L)]] where in range and zero elsewhere:
   column [r] is the phase-[(r*M) mod L] row of the polyphase bank, shifted down
   by the integer input advance [floor(r*M/L)] of output [r] within the block.
   [bank] is the phase-major array of [bank_of_prototype]. *)
let gemm_bank ~l ~m ~k bank =
  let taps = (2 * k) + 1 in
  let p_len = taps + ((l - 1) * m / l) in
  let g = Array.make (p_len * l) 0. in
  for r = 0 to l - 1 do
    let d = r * m / l and ph = r * m mod l in
    for s = 0 to taps - 1 do
      g.(((d + s) * l) + r) <- bank.((ph * taps) + s)
    done
  done ;
  Nx.create Nx.float64 [|p_len; l|] g

(* {1 Configuration} *)

let bank_budget_bytes = 8 * 1024 * 1024

(* The L1 residency edge (128 KB of data L1 on the reference machine's
   performance cores), shared by the planner's cost model and the bank-layout
   choice at [Kernel.prepare]. *)
let l1_edge_bytes = 128 * 1024

(* One polyphase stage of a plan: factors [sl / sm], group delay [sk] in
   stage-input samples, and the designed filter in its three layouts. *)
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
           piece of state in this module *) }

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
   kHz, 191 -> 88 at 8 -> 48 kHz). Near-unity ratios admit no useful
   intermediate — every split of 44.1 <-> 48 kHz prices above the single stage —
   so they keep the single-stage design, bit for bit.

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

type stage_geom = {gl: int; gm: int; gk: int; gfc: float}

(* [bank_cost l k] is [2K + 1] MACs weighted by the residency factor above; the
   tie-breaker uses the float32 bank size — the throughput-critical
   instantiation. *)
let bank_cost l k =
  let macs = (2. *. Float.of_int k) +. 1. in
  if l * ((2 * k) + 1) * 4 > l1_edge_bytes then macs *. 1.25 else macs

let plan_cascade ~l ~m ~attenuation ~passband ~sample_rate ~cost_single =
  let att = attenuation +. (20. *. Float.log10 2.) in
  let sr = Float.of_int sample_rate in
  let target_f = sr *. Float.of_int l /. Float.of_int m in
  let f_n = 0.5 *. Float.min sr target_f in
  let pass = passband *. f_n in
  let bank_ok len k =
    Float.of_int len *. ((2. *. Float.of_int k) +. 1.) *. 8.
    <= Float.of_int bank_budget_bytes
  in
  let best = ref None in
  let take cost s1 s2 =
    match !best with
    | Some (c0, _, _) when c0 <= cost ->
        ()
    | _ ->
        best := Some (cost, s1, s2)
  in
  if m > l then
    (* [F * L = M] would leave the sharp stage rate-preserving while the wide
       stage inherits the very transition the split was meant to avoid — always
       dearer than one stage — so F stops strictly short of the full span *)
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
          {gl= 1; gm= f; gk= k1; gfc= (pass +. stop1) /. sr}
          {gl= l2; gm= m2; gk= k2; gfc= (pass +. f_n) /. interp2}
    done
  else
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
              {gl= l1; gm= m1; gk= k1; gfc= (1. +. passband) *. f_n /. interp1}
              {gl= l2; gm= m2; gk= k2; gfc= (pass +. stop2) /. interp2}
        end
      done
    done ;
  match !best with
  | Some (cost, s1, s2) when cost < cost_single ->
      Some (s1, s2, kaiser_beta att)
  | _ ->
      None

(* One-phase decimators (L = 1) are blocked in the tensor formulation: the
   natural patch per output is [2K + 1] samples at stride M — kernel far wider
   than the stride — and [Nx.extract_patches] pays per gathered element
   (measured ~4-5 ns/element: a 381-tap /2 decimator was gathering 29x the input
   and spending 5.5 ms per second of audio against 0.02 ms of matmul). Grouping
   [B] outputs per patch is the same conversion expressed as the unreduced [B /
   B*M] resampler — every column of the [2K + 1 + (B-1)*M; B] matrix is the one
   bank row, shifted by M — and it divides the gathered volume by ~B*M /
   (2K+1+(B-1)*M) while the extra multiplies land in the GEMM, which is the fast
   path. B = 64 puts the gather within a factor of two of the input size for
   every shipped geometry. *)
let gemm_block = 64

let make_stage ~l ~m ~k ~fc ~beta =
  let h = design_prototype ~l ~k ~fc ~beta in
  let taps = (2 * k) + 1 in
  let b = bank_of_prototype ~l ~k h in
  let bank = Nx.create Nx.float64 [|l; taps|] b in
  { sl= l
  ; sm= m
  ; sk= k
  ; sproto= Nx.create Nx.float64 [|Array.length h|] h
  ; sbank= bank
  ; svisit= visit_bank ~l ~m ~taps bank
  ; sgemm=
      lazy
        ( if l = 1 then
            gemm_bank ~l:gemm_block ~m:(gemm_block * m) ~k (Nx.to_array bank)
          else gemm_bank ~l ~m ~k (Nx.to_array bank) ) }

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
            ; sgemm= lazy (Nx.create Nx.float64 [|1; 1|] [|1.|]) } ] }
    else begin
      let width = (1. -. passband) /. Float.of_int (Stdlib.max l m) in
      let ntaps = kaiser_numtaps attenuation width in
      (* K in float first: the budget check must precede any conversion that
         could overflow or any array that could not be allocated *)
      let k_f = Float.ceil ((ntaps -. 1.) /. (2. *. Float.of_int l)) in
      let bank_bytes = Float.of_int l *. ((2. *. k_f) +. 1.) *. 8. in
      let single_fits = bank_bytes <= Float.of_int bank_budget_bytes in
      let cost_single =
        if single_fits then bank_cost l (Stdlib.max 1 (Float.to_int k_f))
        else Float.infinity
      in
      match
        plan_cascade ~l ~m ~attenuation ~passband ~sample_rate ~cost_single
      with
      | Some (g1, g2, beta) ->
          (* the two invariants the kernel composition rests on, restated as
             executable checks: integral composite delay, and the
             drain-truncation bound (mid-stream emission can never pass the
             composite ceil) *)
          assert (g2.gk * g1.gm mod g1.gl = 0) ;
          assert (g1.gk * l >= m) ;
          let s1 = make_stage ~l:g1.gl ~m:g1.gm ~k:g1.gk ~fc:g1.gfc ~beta in
          let s2 = make_stage ~l:g2.gl ~m:g2.gm ~k:g2.gk ~fc:g2.gfc ~beta in
          { sample_rate
          ; target
          ; quality
          ; l
          ; m
          ; latency= g1.gk + (g2.gk * g1.gm / g1.gl)
          ; stages= [s1; s2] }
      | None ->
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
          let k = Stdlib.max 1 (Float.to_int k_f) in
          let fc = (1. +. passband) /. (2. *. Float.of_int (Stdlib.max l m)) in
          let beta = kaiser_beta attenuation in
          { sample_rate
          ; target
          ; quality
          ; l
          ; m
          ; latency= k
          ; stages= [make_stage ~l ~m ~k ~fc ~beta] }
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
    if is_identity c then
      Stdlib.Format.fprintf fmt "resample(%d Hz, identity)" c.sample_rate
    else
      match c.stages with
      | [s] ->
          Stdlib.Format.fprintf fmt
            "resample(%d -> %d Hz, quality=%a, L/M=%d/%d, taps=%dx%d, \
             latency=%d)"
            c.sample_rate c.target quality c.quality c.l c.m c.l
            ((2 * s.sk) + 1)
            c.latency
      | [s1; s2] ->
          Stdlib.Format.fprintf fmt
            "resample(%d -> %d Hz, quality=%a, L/M=%d/%d, stages=%d/%d:%dx%d \
             >> %d/%d:%dx%d, latency=%d)"
            c.sample_rate c.target quality c.quality c.l c.m s1.sl s1.sm s1.sl
            ((2 * s1.sk) + 1)
            s2.sl s2.sm s2.sl
            ((2 * s2.sk) + 1)
            c.latency
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
   visit (the bank's layout: visit order when true, phase-major when false),
   is_flush. *)
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
  -> bool
  -> bool
  -> unit = "soundml_resample_step_bc" "soundml_resample_step"

(* {1 Incremental kernel} *)

module Kernel = struct
  type 'a stage_state =
    { sp: stage_plan
    ; bank: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
    ; visit: bool
          (* [bank]'s layout: [svisit] when the instantiated bank passes the L1
             edge (the walk then streams it forward), [sbank] under it (the
             phase walk is free on a resident bank, and the executor's
             phase-major loop carries no row counter) *)
    ; hist: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* [channels * 2K], planar *)
    ; scratch: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* [2K + max max_in K], one lane shared across channels *)
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
             for single-stage plans. Allocated here, once: [step] allocates
             exactly the output tensor *)
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
      let hist = array1_of (Nx.zeros dtype [|channels * 2 * sp.sk|]) in
      let elt = Bigarray.kind_size_in_bytes (Bigarray.Array1.kind hist) in
      let visit = sp.sl * ((2 * sp.sk) + 1) * elt > l1_edge_bytes in
      { sp
      ; bank= array1_of (Nx.cast dtype (if visit then sp.svisit else sp.sbank))
      ; visit
      ; hist
      ; scratch=
          array1_of (Nx.zeros dtype [|(2 * sp.sk) + Stdlib.max max_in sp.sk|])
      ; fed= 0
      ; emitted= 0 }
    in
    match cfg.stages with
    | [s] ->
        { cfg
        ; dtype
        ; channels
        ; max_block
        ; st= [|mk max_block s|]
        ; mid= array1_of (Nx.zeros dtype [|0|])
        ; drained= false
        ; leading= [|channels|] }
    | [s1; s2] ->
        (* one call's worth of stage-1 output: at most [ceil (n*L1/M1) + 1]
           samples per channel, [n] bounded by [max_block] on steps and by [K1]
           at the drain *)
        let cap = ceil_pos (Stdlib.max max_block s1.sk * s1.sl) s1.sm + 1 in
        { cfg
        ; dtype
        ; channels
        ; max_block
        ; st= [|mk max_block s1; mk cap s2|]
        ; mid= array1_of (Nx.zeros dtype [|channels * cap|])
        ; drained= false
        ; leading= [|channels|] }
    | _ ->
        assert false

  (* [run k st ~n ~n_out ~is_flush x y] drives the executor for one stage over
     [n] stage-input samples ([x] when streaming, virtual zeros when draining),
     writing [n_out] freshly computed planar outputs into [y]. The state of the
     first output [i = emitted] is exact integer arithmetic: [t = i * M], bank
     row [i mod L] in visit order (the executor reconstructs the phase [t mod L]
     from it; in phase-major layout it recomputes the row too), window start [t
     / L + K - fed] in scratch coordinates. Advances [emitted]; the caller
     advances [fed] — a drain feeds no real samples. *)
  let run k st ~n ~n_out ~is_flush x y =
    let t = st.emitted * st.sp.sm in
    let row0 = st.emitted mod st.sp.sl in
    let s0 = (t / st.sp.sl) + st.sp.sk - st.fed in
    resample_step_c st.bank st.hist st.scratch x y n n_out k.channels st.sp.sk
      st.sp.sl st.sp.sm row0 s0 st.visit is_flush ;
    st.emitted <- st.emitted + n_out

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
    else begin
      let x = array1_of chunk in
      match k.st with
      | [|s|] ->
          let n_out = ready s.sp (s.fed + n) - s.emitted in
          let out =
            if n_out = 0 then begin
              (* nothing computable yet: the call only threads the history *)
              run k s ~n ~n_out:0 ~is_flush:false x s.scratch ;
              None
            end
            else begin
              let out = Nx.empty k.dtype (Array.append k.leading [|n_out|]) in
              run k s ~n ~n_out ~is_flush:false x (array1_of out) ;
              Some out
            end
          in
          s.fed <- s.fed + n ;
          out
      | [|s1; s2|] ->
          (* stage 1 lands in the hand-off buffer. Its output sequence is
             partition-independent (stage 1 is chunk-invariant), and stage 2 is
             invariant to how that sequence reaches it — the composite keeps the
             law by composition, not by re-proof *)
          let n1 = ready s1.sp (s1.fed + n) - s1.emitted in
          run k s1 ~n ~n_out:n1 ~is_flush:false x k.mid ;
          s1.fed <- s1.fed + n ;
          if n1 = 0 then None
          else begin
            let n2 = ready s2.sp (s2.fed + n1) - s2.emitted in
            let out =
              if n2 = 0 then begin
                run k s2 ~n:n1 ~n_out:0 ~is_flush:false k.mid k.mid ;
                None
              end
              else begin
                let out = Nx.empty k.dtype (Array.append k.leading [|n2|]) in
                run k s2 ~n:n1 ~n_out:n2 ~is_flush:false k.mid (array1_of out) ;
                Some out
              end
            in
            s2.fed <- s2.fed + n1 ;
            out
          end
      | _ ->
          assert false
    end

  let flush k =
    if k.drained then None
    else begin
      k.drained <- true ;
      match k.st with
      | [|s|] ->
          let n_out = ready s.sp (s.fed + s.sp.sk) - s.emitted in
          if n_out = 0 then None
          else begin
            let out = Nx.empty k.dtype (Array.append k.leading [|n_out|]) in
            run k s ~n:s.sp.sk ~n_out ~is_flush:true s.hist (array1_of out) ;
            Some out
          end
      | [|s1; s2|] ->
          (* drain in stage order, then truncate: stage 1 flushes its exact ceil
             tail into the hand-off buffer, the tail streams through stage 2,
             stage 2 flushes its own tail — and the composite stream is cut to
             [output_frames]. The cut only ever lands in this drain: [ceil (ceil
             (n*L1/M1) * L2/M2) >= ceil (n*L/M)] (the stage rationals multiply
             to exactly [L/M]), and mid-stream emission never passes the
             composite ceil because [K1*L >= M] — checked at create — keeps
             [emitted <= fed*L/M + 1 - K1*L/M]. Both counts are functions of the
             totals alone, so the cut is partition-independent. *)
          let total = ceil_pos (s1.fed * k.cfg.l) k.cfg.m in
          let keep = total - s2.emitted in
          if keep = 0 then None
          else begin
            let n1 = ceil_pos (s1.fed * s1.sp.sl) s1.sp.sm - s1.emitted in
            if n1 > 0 then
              run k s1 ~n:s1.sp.sk ~n_out:n1 ~is_flush:true s1.hist k.mid ;
            let n2a =
              Stdlib.min (ready s2.sp (s2.fed + n1) - s2.emitted) keep
            in
            let piece_a =
              if n2a = 0 then begin
                if n1 > 0 then
                  (* thread the tail into stage-2 history: the stage-2 drain
                     below still reads it *)
                  run k s2 ~n:n1 ~n_out:0 ~is_flush:false k.mid k.mid ;
                None
              end
              else begin
                let out = Nx.empty k.dtype (Array.append k.leading [|n2a|]) in
                run k s2 ~n:n1 ~n_out:n2a ~is_flush:false k.mid (array1_of out) ;
                Some out
              end
            in
            s2.fed <- s2.fed + n1 ;
            let n2b = keep - n2a in
            let piece_b =
              if n2b = 0 then None
              else begin
                let out = Nx.empty k.dtype (Array.append k.leading [|n2b|]) in
                run k s2 ~n:s2.sp.sk ~n_out:n2b ~is_flush:true s2.hist
                  (array1_of out) ;
                Some out
              end
            in
            match (piece_a, piece_b) with
            | None, None ->
                None
            | (Some _ as one), None | None, (Some _ as one) ->
                one
            | Some a, Some b ->
                Some (Nx.concatenate ~axis:(-1) [a; b])
          end
      | _ ->
          assert false
    end

  let reset k =
    Array.iter
      (fun s ->
        Bigarray.Array1.fill s.hist 0. ;
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
      (* sequence step before flush *)
      let stepped = Kernel.step k x in
      let drained = Kernel.flush k in
      match Option.to_list stepped @ Option.to_list drained with
      | [] ->
          Nx.zeros (Nx.dtype x) (Array.append lead [|total|])
      | [one] ->
          one
      | many ->
          Nx.concatenate ~axis:(-1) many
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
      (* one-phase stages run blocked (see [gemm_block]): the same conversion in
         the patch geometry of the unreduced [B / B*M] form *)
      let run_stage s ~total x =
        if s.sl = 1 then
          gemm_stage ~sl:gemm_block ~sm:(gemm_block * s.sm) ~sk:s.sk s.sgemm
            ~total dtype lead x
        else gemm_stage ~sl:s.sl ~sm:s.sm ~sk:s.sk s.sgemm ~total dtype lead x
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
   dominate every step and the drain. One stage: a length-[b] chunk emits at
   most [ceil (b*L/M) + 1] (the +1 absorbs phase alignment; the drain, at most
   [ceil (K*L/M) + 1], is covered by the generic latency widening). A cascade
   compounds the two per-stage bounds, and its drain — stage-1 tail through
   stage 2 plus stage-2's own tail, delivered as one chunk — can exceed both, so
   it enters the max explicitly. *)
let stage_bound cfg b =
  match cfg.stages with
  | [s] ->
      ceil_pos (b * s.sl) s.sm + 1
  | [s1; s2] ->
      let through n =
        ceil_pos ((ceil_pos (n * s1.sl) s1.sm + 1) * s2.sl) s2.sm
      in
      let drain = through s1.sk + ceil_pos (s2.sk * s2.sl) s2.sm + 2 in
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
