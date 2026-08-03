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

(** Sample-rate conversion.

    One resampler serves the whole library. A {!Config.t} fixes the conversion
    and precomputes the polyphase filter bank once; {!apply} is the offline
    whole-tensor form, {!Kernel} is the incremental form behind it, and
    {!stage} exposes the same kernel as a {!Pipeline} stage. Offline is the
    one-chunk instance of streaming: all three faces drive one executor, so
    they cannot disagree — bit for bit, on every partitioning.

    The converter is an exact-rational polyphase windowed-sinc (Kaiser),
    linear-phase. [target / sample_rate] is reduced by gcd to [L / M]; output
    sample [n] sits at input time [n * M / L], phases are exact by integer
    arithmetic, and no drift is representable. Output length is exactly
    [ceil (n * L / M)] and the group delay is compensated: there is no leading
    silence and no trailing padding. The conversion is between two {e nominal}
    rates; clock-drift compensation and variable ratios are deliberately not
    this function's job — those need a variable-ratio converter, a different
    feature with different state. *)

(** The type for custom filter specifications. [attenuation] is the stop-band
    rejection target in dB; [passband] is the flat-to-0-dB bandwidth preserved,
    as a fraction of the {e lower} of the two Nyquist frequencies. The Kaiser
    shape parameter and the filter length are derived (never asked for): cost
    grows linearly with [attenuation] and inversely with [1 - passband]. *)
type spec = {attenuation: float; passband: float}

(** The type for quality presets. All three named presets keep the passband
    flat to [0.913] of the lower Nyquist (SoXR's published 0 dB point) and
    differ only in stop-band rejection — each documents its numbers:

    - [`Fast] — 100 dB design attenuation (16-bit-transparent class, between
      SoXR's MQ and HQ). Roughly 148 taps per output at 44.1↔48 kHz.
    - [`High] — 126 dB design attenuation: SoXR's HQ ("20-bit") spec, the
      default, and the spec of librosa's default [soxr_hq]. ~190 taps.
    - [`Best] — 175 dB design attenuation: SoXR's VHQ ("28-bit") spec. ~268
      taps. Only meaningful at float64: float32 arithmetic floors near −133 dB
      THD+N regardless of filter length — the same reason SoXR runs HQ
      single-precision and forces VHQ to double.
    - [`Custom spec] — the escape hatch, validated at {!Config.create}. *)
type quality = [`Fast | `High | `Best | `Custom of spec]

(** {1 Configuration} *)

module Config : sig
  (** The type for validated, immutable resampling configurations. Creation
      reduces the ratio, designs the prototype filter (float64) and lays the
      polyphase bank out phase-major; configs are cheap to share, and one
      config serves any number of {!apply} calls and prepared kernels —
      construction is a noticeable fraction of resampling a short clip, so
      corpus jobs must reuse it. *)
  type t

  val create : ?quality:quality -> sample_rate:int -> target:int -> unit -> t
  (** [create ~sample_rate ~target ()] is the conversion from [sample_rate] to
      [target] hertz. [quality] defaults to [`High].

      When [target = sample_rate] the configuration is the identity: no filter
      is designed, {!latency} is [0], and every face of this module passes
      audio through untouched.

      Raises [Invalid_argument] if [sample_rate < 1] or [target < 1]; if a
      [`Custom] spec is invalid ([attenuation] not finite or outside
      [\[40., 200.\]], [passband] not finite or outside [\[0.5, 0.99\]]); or
      if the reduced ratio needs a coefficient bank above the documented
      budget (8 MB) — the near-unity case, e.g. 44100 → 44099. The message
      names L, the bank size, and states that near-unity conversion is
      clock-drift correction, which this fixed-ratio resampler does not do.
      Every pair of standard rates fits with margin: the worst,
      11025 → 192000, needs under 4 MB. *)

  val sample_rate : t -> int
  (** [sample_rate c] is the source rate in hertz. *)

  val target : t -> int
  (** [target c] is the destination rate in hertz. *)

  val quality : t -> quality
  (** [quality c] is the quality preset [c] was created with. *)

  val rate : t -> Pipeline.Rate.t
  (** [rate c] is the exact reduced ratio [L / M] — output samples per input
      sample. [Rate.{num = L; den = M}]. *)

  val latency : t -> int
  (** [latency c] is the group delay in {e input} samples — exact and
      integral, because the prototype length is rounded up to [2*K*L + 1] so
      the linear-phase delay [K] lands on the input grid. [0] for the
      identity. At [`High] 44.1 → 48 kHz, [K = 95] (~2.2 ms). *)

  val output_latency : t -> Pipeline.Rate.t
  (** [output_latency c] is the group delay in output samples, exact:
      [K * L / M] as a rational. Latency is queryable in both domains because
      every conversion between them rounds. *)

  val output_frames : t -> n:int -> int
  (** [output_frames c ~n] is the number of output samples a length-[n] input
      produces: [ceil (n * L / M)], exactly — the deterministic-length
      contract every face of this module honours.

      Raises [Invalid_argument] if [n < 0]. *)

  val prototype : (float, 'a) Nx.dtype -> t -> (float, 'a) Nx.t
  (** [prototype dtype c] is a fresh copy of the designed lowpass, rank-one,
      length [2*K*L + 1] — for inspection, response plots and tests. Kernels
      use the config-owned bank without copying; this accessor copies. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt c] prints [c] on [fmt] in a compact single-line form. *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff [a] and [b] were created from the same rates
      and quality; [`Custom] fields are compared with [Float.equal]. *)
end

(** {1 Offline} *)

val apply : Config.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [apply c x] resamples [x] along its last axis; leading axes broadcast, so
    a batch of clips is one call. Output length is {!Config.output_frames}
    exactly; sample [i] of the output corresponds to input time [i * M / L]
    (group delay compensated); the signal is treated as silence outside its
    extent (zeros — the linear-convolution convention shared by torchaudio and
    scipy [resample_poly]).

    [apply] is the one-chunk instance of {!Kernel}: one [step] on the whole
    signal plus the drain, bit-identical to any chunked execution. For the
    identity configuration it is the identity — it returns [x] itself,
    documented deviation from the fresh-tensor rule.

    Raises [Invalid_argument] if [x] has rank zero, or if the dtype of [x] is
    neither float32 nor float64. *)

val apply_gemm : Config.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [apply_gemm c x] is the same conversion as {!apply} — same config-owned
    filter, same exact output length, same group-delay compensation, same
    zeros-outside-the-extent convention — computed as one dense tensor
    expression: the input is cut into strided patches ([Nx.extract_patches])
    and multiplied against the filter bank laid out as a single matrix
    ([Nx.matmul]). Because it is built from Nx operations end to end, it is
    differentiable and device-eligible wherever those operations are.

    It is numerically {e distinct} from {!apply}: the matrix product sums each
    output in a different order than the executor's fixed dot product, so the
    two surfaces agree to within a few ULP of the signal peak — growing with
    the reduced filter's length; measured worst case: under 10 ULP across
    presets and dtypes at the common conversions (150-800-tap filters), under
    30 ULP across the full standard-rate matrix (extreme pairs run 1 500+
    taps) — and are {e never} bit-identical by contract. It is offline-only and carries no partitioning
    law: it is not a {!Pipeline} stage, it has no streaming form, and it is
    deliberately not a flag on {!apply} — a flag that changes bits would fork
    the library's one resampler in place. When bit-stability across
    partitionings matters, use {!apply} or {!Kernel}; when offline throughput
    on wide-matmul hardware or differentiability matters, use this.

    For the identity configuration it returns [x] itself, like {!apply}.

    Raises [Invalid_argument] if [x] has rank zero, or if the dtype of [x] is
    neither float32 nor float64. *)

(** {1 Incremental kernel}

    The Mealy kernel behind every face. [step] consumes an arbitrary-length
    chunk (any length up to [max_block] — there is no chunk-size negotiation
    and no divisibility requirement) and emits every output sample that became
    computable; [flush] extends the signal with virtual silence, emits the
    remaining [ceil] tail, and truncates to the exact total. Output chunk
    lengths vary call to call; that is normal operation. *)

module Kernel : sig
  (** The type for prepared kernel states. Mutable; single-owner; not
      domain-safe. One state carries all channels — never one object per
      channel. *)
  type 'a t

  val prepare :
    Config.t -> (float, 'a) Nx.dtype -> channels:int -> max_block:int -> 'a t
  (** [prepare c dtype ~channels ~max_block] is a fresh kernel state for
      chunks of at most [max_block] samples of [channels]-channel [dtype]
      audio. Allocates everything: the per-channel history ([2K] samples) and
      the dtype-instantiated bank (cast from the config's float64 design once,
      here — never cached narrower than the state's dtype).

      Raises [Invalid_argument] if [channels < 1] or [max_block < 1], or if
      [dtype] is neither float32 nor float64 — the two sample types the
      executor carries. *)

  val step : 'a t -> (float, 'a) Nx.t -> (float, 'a) Nx.t option
  (** [step k chunk] feeds [chunk] — time axis last — and is the newly
      computable output samples, if any. [chunk] is borrowed: the kernel
      copies what it must retain, and the returned tensor aliases neither
      [chunk] nor kernel state. At most [ceil (len * L / M) + 1] samples are
      returned for a length-[len] chunk.

      Raises [Invalid_argument] if [chunk] is longer than [max_block], if a
      leading axis disagrees with [channels], or if [k] was drained by
      {!flush} — {!reset} it before feeding a new signal. *)

  val flush : 'a t -> (float, 'a) Nx.t option
  (** [flush k] drains the kernel: emits the delayed tail so that the
      concatenation of every [step] output plus [flush] equals {!apply} on the
      concatenated input — bit-identically. Draining consumes the tail; a
      second [flush] is [None]. *)

  val reset : 'a t -> unit
  (** [reset k] restores [k] to its freshly prepared state without
      reallocating (zero the history, keep the plan). *)
end

(** {1 Pipeline stage} *)

val stage : Config.t -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage c] is {!Kernel} as a pipeline stage: causal — hence polymorphic in
    ['k] — with latency [Config.latency c] input items and rate
    [Config.rate c]. Its [out_format] scales items per second by exactly
    [L / M] (44100 → 48000 threads through as exactly 48000) and widens the
    per-chunk bound to [ceil (bound * L / M) + 1]. [prepare] validates that
    the incoming format's items per second equal [sample_rate] and raises
    [Invalid_argument] otherwise — at prepare time, never mid-stream.

    For the identity configuration the stage is transparent to the latency and
    rate accounting (latency [0], rate [1/1]) and forwards chunks with one
    copy (the chunk-ownership contract forbids returning the borrowed input).
    An elided resampler that still claimed latency would corrupt every
    downstream alignment. *)
