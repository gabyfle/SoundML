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

(** Constant-Q and variable-Q transforms.

    A {!Config.t} fixes the filter geometry once — the frequency ladder, the
    per-bin bandwidths, the octave plan and every filter kernel —
    {!transform} runs it over a whole signal and {!Kernel} runs the same plan
    incrementally over a stream. One configuration serves both
    transforms: the constant-Q transform is the variable-Q transform at
    [gamma = 0], so [`Constant_q] and the [gamma] variants differ only in how
    each filter's bandwidth is offset.

    Bin [k] is a complex sinusoid at [f_k = fmin * 2 ^ (k / bins_per_octave)]
    (shifted by [tuning] bins), windowed over [Q_k * sample_rate / f_k]
    samples. Holding the quality factor [Q] constant across bins makes the
    time-frequency resolution geometric — the musically natural analysis, one
    bin per equal-tempered step. The [gamma] offset widens the low bins beyond
    constant-Q so that their filters stay short enough to resolve onsets.

    The transform is evaluated the way the literature builds it: the filters
    are applied as a frequency-domain kernel against a short FFT, and each
    octave down runs on a signal decimated by two — so the cost is a handful of
    small transforms rather than one long convolution per bin. That recursion
    rides the library's own resampler ({!Resample}, the [`High] preset); see
    {!section:parity} for exactly what that costs.

    Complex-valued entry points take the spectrum dtype as an explicit
    dtype-first witness, exactly like {!Stft.transform}. Whatever the witness,
    the filter bank, the FFTs and the projection all run in double precision
    and round once into the requested storage. The real-valued convenience
    ({!power_spectrum}) is dtype-preserving and never exposes a complex dtype.

    Inverse transforms, invertible non-stationary Gabor frames, the hybrid and
    pseudo constant-Q approximations, tuning estimation, sparsified filter
    bases, intervals other than equal temperament and a {!Pipeline} stage for
    the kernel are all deliberately outside this module. So is the sliced
    real-time constant-Q transform of Holighaus, Doerfler, Velasco & Grill
    2013: it buys low latency by analysing overlapping slices of the signal —
    a different transform with different coefficients, not a lower-latency
    execution of this one. {!Kernel} streams {e this} plan, and pays the
    lookahead constant-Q analysis costs ({!Config.latency}). *)

(** {1 Configuration} *)

module Config : sig
  (** The type for the bandwidth offset of the variable-Q generalisation. Bin
      [k] gets bandwidth [alpha_k * f_k + gamma_k], with [alpha_k] the relative
      bandwidth of the ladder.

      - [`Constant_q] — [gamma = 0]: the constant-Q transform, every filter at
        the same quality factor.
      - [`Erb] — [gamma_k = alpha_k * 24.7 / 0.108]: bandwidths proportional
        to the equivalent rectangular bandwidth of the auditory filter at
        [f_k] (Glasberg & Moore 1990). This is the usual variable-Q setting.
      - [`Fixed g] — offsets every bandwidth by the same [g] hertz; [g] must
        be finite and non-negative. *)
  type gamma = [`Constant_q | `Erb | `Fixed of float]

  (** The type for validated, immutable transform configurations. Creation
      does every signal-independent computation: the frequency ladder, the
      octave plan, one filter kernel per octave in double precision and the
      length normalisers. Configurations are cheap to share and compare, and
      the kernels are never exposed. Construction is {e not} free — a 252-bin
      configuration builds seven kernels of 36 filters each — so corpus jobs
      must build the configuration once and reuse it. *)
  type t

  val create :
       ?fmin:float
    -> ?bins_per_octave:int
    -> ?gamma:gamma
    -> ?tuning:float
    -> ?filter_scale:float
    -> ?norm:float
    -> ?window:Window.t
    -> ?scale:bool
    -> ?hop:int
    -> ?pad:[`Reflect | `Constant of float | `Edge]
    -> n_bins:int
    -> sample_rate:int
    -> unit
    -> t
  (** [create ~n_bins ~sample_rate ()] is a transform configuration covering
      [n_bins] bins upward from [fmin] at [sample_rate].

      [fmin] is the centre frequency of the lowest bin in hertz and defaults
      to [32.70319566257483] (C1); [bins_per_octave] defaults to [12], one bin
      per equal-tempered semitone. [tuning] shifts the whole ladder by that
      many bins ([0.] by default) and is always explicit — it is never
      estimated from the signal, see {!section:parity}.

      [gamma] selects the bandwidth offset and defaults to [`Constant_q].
      [filter_scale] multiplies every filter's support ([1.] by default;
      smaller values buy time resolution with frequency resolution), [norm] is
      the exponent of the norm each filter is scaled to in the time domain
      ([1.] by default), and [window] is the filter envelope, instantiated in
      its periodic form ({!Window.Hann} by default).

      [scale] defaults to [true] and divides each bin by the square root of
      its filter length, so bins are comparable across the ladder and a
      unit-amplitude sinusoid reads the same magnitude at every frequency;
      [false] leaves the raw filter responses, whose magnitude grows with
      filter length.

      [hop] is the frame advance in samples at [sample_rate] and defaults to
      [512]. It also drives the octave recursion: the transform halves the
      signal rate between octaves only while the hop stays even, so an odd hop
      analyses every octave at [sample_rate] and a hop divisible by a high
      power of two additionally licenses an early decimation before the first
      octave. [pad] is the boundary extension of the per-octave analysis and
      defaults to [`Constant 0.] — zero padding, the boundary convention of
      the reference constant-Q transform (unlike its STFT).

      Raises [Invalid_argument] if [n_bins], [bins_per_octave], [hop] or
      [sample_rate] is smaller than [1]; if [fmin] is not finite and positive;
      if [tuning] is not finite; if [filter_scale] or [norm] is not finite and
      positive; if [`Fixed g] is not finite and non-negative; if the shape
      parameter of [window] is invalid as documented for {!Window.make}; if
      the frequency ladder overflows; if [filter_scale] is so small that some
      octave's filters span less than one sample; or if any filter's main lobe
      would cross the Nyquist frequency — the message names the offending bin,
      its centre and its reach. That last check is the admissibility condition
      of the octave recursion: a filter reaching past Nyquist cannot be
      resolved at any rate the recursion visits. *)

  val n_bins : t -> int
  (** [n_bins c] is the number of frequency bins. *)

  val bins_per_octave : t -> int
  (** [bins_per_octave c] is the number of bins per octave. *)

  val sample_rate : t -> int
  (** [sample_rate c] is the sample rate in hertz the filters were built
      for. *)

  val hop : t -> int
  (** [hop c] is the frame advance in samples at [sample_rate]. *)

  val fmin : t -> float
  (** [fmin c] is the centre frequency of the lowest bin in hertz, before the
      [tuning] shift. *)

  val gamma : t -> gamma
  (** [gamma c] is the bandwidth offset mode. *)

  val tuning : t -> float
  (** [tuning c] is the ladder shift in bins. *)

  val filter_scale : t -> float
  (** [filter_scale c] is the filter support multiplier. *)

  val norm : t -> float
  (** [norm c] is the exponent of the time-domain filter norm. *)

  val window : t -> Window.t
  (** [window c] is the filter envelope specification. *)

  val scale : t -> bool
  (** [scale c] is [true] iff bins are divided by the square root of their
      filter length. *)

  val pad : t -> [`Reflect | `Constant of float | `Edge]
  (** [pad c] is the boundary extension mode. *)

  val n_octaves : t -> int
  (** [n_octaves c] is the number of octaves the recursion visits,
      [ceil (n_bins / bins_per_octave)]. The lowest one is partial whenever
      [bins_per_octave] does not divide [n_bins]. *)

  val cutoff : t -> float
  (** [cutoff c] is the highest frequency any filter's main lobe reaches, in
      hertz: [max_k (f_k * (1 + enbw / (2 Q_k)) + gamma_k / 2)], with [enbw]
      the equivalent noise bandwidth of the window in FFT bins. {!create}
      rejects a configuration whose [cutoff] exceeds the Nyquist frequency,
      and the margin between the two is what licenses the early decimation. *)

  val latency : t -> int
  (** [latency c] is the analysis lookahead of the whole plan in input samples
      at [sample_rate]: the number of samples past a frame's grid position
      that must arrive before {!Kernel} can emit that frame's column. Column
      [p] is complete once [p * hop + latency c] input samples have been fed.

      It is the largest per-octave lookahead,

      {[ latency = max_i (D_i * (L_i - 1) + K * (D_i - 1) + 1) ]}

      over the octave plan, where [D_i = 2 ^ (early + halvings_i)] is how many
      2:1 decimations octave [i] sits under, [L_i] is that octave's centered
      analysis lookahead in decimated samples ([n_fft_i / 2], the
      {!Stft.Config.latency} of its analysis), and [K] is the group delay of
      one exact 2:1 conversion in its own input samples ([190] at the [`High]
      preset the recursion runs).

      Both terms are physics, not implementation slack. [D_i * L_i] is half
      the octave's longest filter measured in input samples: constant-Q
      analysis of a bin at [f] with quality factor [Q] spans [Q / f] seconds,
      so its centered frame cannot close before [Q / (2 f)] seconds of
      lookahead — [265] ms at the [84]-bin C1 default — and the plan rounds
      that up to the power-of-two transform length the octave is analysed at.
      [K * (D_i - 1)] is the composed group delay of the decimation chain
      above the octave, the price of analysing the low bins at a low rate.

      The default [84]-bin C1 ladder at 22.05 kHz declares [20099] samples,
      [0.91] s: [8192] of them the bottom octave's analysis half-window
      carried up to the input rate, the remaining [11907] the composed delay
      of the six 2:1 stages below it. A plan that never decimates — an odd
      [hop], where every octave runs at [sample_rate] — pays the analysis
      half-window alone: [8192] samples, [0.37] s, at the same default ladder.

      The offline {!transform} is not subject to this: it has the whole signal
      in hand. Latency is what an incremental analysis of the same plan owes
      its own frame grid. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt c] prints [c] on [fmt] in a compact single-line form. *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff [a] and [b] were created from the same
      parameters; float parameters are compared with [Float.equal]. *)
end

(** {1 The frequency ladder} *)

val frequencies : (float, 'a) Nx.dtype -> Config.t -> (float, 'a) Nx.t
(** [frequencies dtype c] is the centre frequency of each bin in hertz, as a
    rank-one tensor of [Config.n_bins c] entries.

    Bin [k] of octave [o] — [k = o * bins_per_octave + j] — sits at
    [fmin * 2 ^ (tuning / bins_per_octave) * 2 ^ o * 2 ^ (j / bins_per_octave)].
    The ladder is evaluated octave by octave, so a bin and its octave
    transposition differ by exactly a factor of two and the reported
    frequencies are exactly the ones the filters are built at. Evaluating
    [2 ^ (k / bins_per_octave)] in one step instead — as the reference
    implementation's
    [cqt_frequencies] helper does, though its [vqt] does not — moves the last
    three bits of some bins. *)

val filter_lengths : (float, 'a) Nx.dtype -> Config.t -> (float, 'a) Nx.t
(** [filter_lengths dtype c] is the {e fractional} support of each filter in
    samples at [Config.sample_rate c], as a rank-one tensor:
    [Q_k * sample_rate / (f_k + gamma_k / alpha_k)] with
    [Q_k = filter_scale / alpha_k]. A filter is realised over
    [floor (l / 2) + ceil (l / 2)] samples of that support, and the transform
    length of an octave is the least power of two at or above its longest
    filter.

    When the plan decimates early, the octaves run at a lower rate and their
    filters are correspondingly shorter; these are the lengths at the
    configured rate, which is also the scale the [scale] normalisation and
    the reference implementation's [filters.wavelet_lengths] report. *)

(** {1 The frame grid}

    Frames sit on the grid of the {e highest} octave: frame [p] is centred on
    sample [p * hop] of the input. Every octave analyses the same span at its
    own decimated rate and hop, and the transform keeps the frames all of them
    produced, so the count is the minimum over the octave plan. *)

val frames : Config.t -> n:int -> int
(** [frames c ~n] is the number of frames a length-[n] signal produces.

    Raises [Invalid_argument] if [n < 0]. *)

(** {1 Offline} *)

val transform :
     (Complex.t, 'c) Nx.dtype
  -> Config.t
  -> (float, 'a) Nx.t
  -> (Complex.t, 'c) Nx.t
(** [transform cdtype c x] is the constant-Q (or variable-Q) transform of [x],
    shaped [[...; n_bins; frames]]. The time axis is the last axis of [x];
    leading axes broadcast, so a batch of clips is one call.

    Only one octave's spectrum and projection exist at a time — the whole
    filter bank is never multiplied against the whole signal — but the
    decimation chain runs the resampler's dense offline form
    ({!Resample.apply_gemm}), whose working set is a few times the signal it
    converts. Peak memory for the default seven-octave ladder measures around
    ten times a float32 input.

    Raises [Invalid_argument] if [x] has rank zero, or if the plan decimates
    and the dtype of [x] is neither float32 nor float64: the recursion runs
    the resampler at the caller's sample type. *)

val power_spectrum :
  ?power:float -> Config.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [power_spectrum c x] is [|transform c x| ^ power], shaped
    [[...; n_bins; frames]], in the dtype of [x]. [power] defaults to [2.]
    (the power spectrum; [1.] is the magnitude spectrum, the input
    {!Chroma.of_cqt} expects). The complex intermediate never escapes: it is
    stored at the component width matching the dtype of [x], and magnitudes
    land directly in that dtype.

    Raises [Invalid_argument] if [x] has rank zero, or for the dtype reason
    documented on {!transform}. *)

(** {1:kernel Incremental kernel}

    The same octave plan driven chunk by chunk. [step] consumes an
    arbitrary-length chunk of samples — the time axis last, leading axes
    broadcast — and emits the frame columns that became complete; [flush]
    extends the signal with virtual silence and emits the remaining columns.
    A column leaves only when every octave has produced its frame at that
    grid position, so the emitted sequence is the same frame grid
    {!frames} counts, gated on the slowest octave.

    {2 The partition law}

    Concatenating every emitted chunk along the time axis is bit-identical to
    the one-chunk instance of the same kernel — one [step] on the whole signal
    plus the drain — for {e every} partitioning of the input, including
    one-sample chunks, chunks that straddle the frame cadence and interspersed
    empty ones. The law is anchored on the kernel's own whole-signal instance,
    the way {!Resample} words it: every sub-kernel in the composition
    ({!Resample.Kernel} for the decimation chain, {!Stft.Kernel} for each
    octave's analysis) carries that law already, the inter-stage scalars are
    elementwise, and frames are projected one at a time indexed by frame
    number — never by arrival shape — so no emitted value can depend on where
    a chunk boundary fell.

    {2 Against the offline transform}

    The kernel and {!transform} evaluate the same plan by different
    arithmetic: {!transform} decimates through {!Resample.apply_gemm} and
    projects the whole octave in one matrix product, neither of which carries
    a partition law, while the kernel decimates through {!Resample.Kernel} and
    projects frame by frame. Both are the same filter bank; their summation
    orders differ. Agreement is therefore documented, not promised: the test
    suite pins it at [2e-6] of the frame peak in float32 and [1e-13] in
    float64 across its configuration grid, where the worst measured values are
    [9.8e-7] and [1.7e-14] — the latter at a resampler-free plan, whose long
    undecimated transforms give the projection its longest reductions. Use
    {!transform} when a signal is in hand and the offline goldens are the
    reference; use {!Kernel} when the signal arrives over time.

    {2 Cost}

    Every [step] carries the dispatch cost of the small per-octave tensor
    calls it makes, and the per-frame projection replaces one wide matrix
    product with one narrow product per frame. Chunk size is a throughput
    knob, never a correctness one — the law guarantees identical bits at
    every chunking. *)

module Kernel : sig
  (** The type for prepared kernel states. Mutable; single-owner; not
      domain-safe. One state carries all channels — never one object per
      channel. *)
  type ('a, 'c) t

  val prepare :
       (Complex.t, 'c) Nx.dtype
    -> Config.t
    -> (float, 'a) Nx.dtype
    -> channels:int
    -> max_block:int
    -> ('a, 'c) t
  (** [prepare cdtype c dtype ~channels ~max_block] is a fresh kernel state
      for chunks of at most [max_block] samples of [channels]-channel [dtype]
      audio, producing [cdtype] columns. It allocates the whole composition:
      one resampler state per early decimation, and per octave one STFT state,
      one resampler state where the plan halves, and the column buffer that
      holds what that octave ran ahead of the slowest one — at most
      [ceil ((Config.latency c - lookahead) / hop) + 1] columns each.

      Raises [Invalid_argument] if [channels < 1] or [max_block < 1], or if
      the plan decimates and [dtype] is neither float32 nor float64: the
      recursion runs the resampler at the caller's sample type, exactly as
      {!transform} does. *)

  val step : ('a, 'c) t -> (float, 'a) Nx.t -> (Complex.t, 'c) Nx.t option
  (** [step k chunk] feeds [chunk] and is the newly completed columns, if any,
      shaped [[...; n_bins; columns]]. [chunk] is borrowed: the kernel copies
      what it must retain, and the returned tensor aliases neither [chunk] nor
      kernel state. A chunk that completes no column — the common case below
      the cadence — is [None], and a chunk that completes several emits them
      as one tensor.

      Raises [Invalid_argument] if [chunk] has rank zero, if it is longer than
      [max_block], or if [k] was drained by {!flush} — {!reset} it before
      feeding a new signal. *)

  val flush : ('a, 'c) t -> (Complex.t, 'c) Nx.t option
  (** [flush k] drains the kernel: it installs each octave's right boundary
      extension, drains the decimation chain and is the remaining columns, if
      any. Over a whole stream the emitted columns total [frames c ~n] for the
      [n] samples fed; octaves that ran ahead of the slowest keep their
      surplus columns unemitted, which is the same trim {!transform} applies.
      Draining consumes the tail — a second [flush] is [None]; {!reset} the
      kernel before reusing it. *)

  val reset : ('a, 'c) t -> unit
  (** [reset k] restores [k] to its freshly prepared state. *)
end

(** {1:parity Parity with librosa 0.11}

    Magnitudes match [librosa.cqt] and [librosa.vqt] at matching explicit
    settings, enforced against committed golden vectors in the test suite. The
    deliberate departures, all of them measured:

    - {b The octave decimation is this library's resampler.} librosa steps
      down an octave with SoX Resampler at its [soxr_hq] tier; the recursion
      here uses {!Resample} at [`High], the preset designed to that same
      specification, through its dense offline form ({!Resample.apply_gemm}):
      the same filter, the same compensated group delay and the same output
      length as the streaming executor, without the partitioning law this
      module has no use for. Two resamplers meeting one specification are
      not the same filter, so magnitudes differ: measured at most [6.3e-5] of
      the frame peak over 30 s music clips and [1.5e-4] over short broadband
      signals. Nothing closes that gap — substituting soxr's own higher
      [soxr_vhq] tier for [soxr_hq] moves the same cases by the same order
      ([7.7e-5] and [4.1e-4]), so the residual is what any two conforming
      resamplers differ by, not a quality shortfall. Configurations that never
      decimate carry no such term: the top octave is always resampler-free,
      and an odd [hop] keeps the whole transform resampler-free, where
      agreement is [1.2e-7] of peak in double precision.
    - {b The filter bank is double precision.} librosa builds its wavelet
      basis in single precision even when asked for a double-precision
      transform. Building it in double moves magnitudes by around [1e-7] of
      peak — on the accurate side.
    - {b No sparsity thresholding.} librosa discards the smallest entries of
      each filter's spectrum, by default enough to account for 1% of its mass,
      and projects through the sparse remainder. The projection here is a
      dense product against the exact filter response: no basis energy is
      thrown away. librosa's default moves its own magnitudes by about
      [5e-3] of peak on music, so parity is stated against [sparsity=0].
    - {b Tuning is explicit.} [tuning] defaults to [0.] and is never estimated
      from the signal; librosa's transforms estimate it by default. *)

(** {1:references References}

    - J. C. Brown, {e Calculation of a constant Q spectral transform}, JASA
      89(1), 1991 — the transform and its geometric frequency ladder.
    - J. C. Brown & M. S. Puckette, {e An efficient algorithm for the
      calculation of a constant Q transform}, JASA 92(5), 1992 — the
      frequency-domain kernel this module projects through.
    - C. Schoerkhuber & A. Klapuri, {e Constant-Q transform toolbox for music
      processing}, SMC 2010 — the octave-by-octave recursion over a decimated
      signal, and the variable-Q generalisation.
    - N. Holighaus, M. Doerfler, G. A. Velasco & T. Grill, {e A framework for
      invertible, real-time constant-Q transforms}, IEEE TASLP 21(4), 2013 —
      the sliced transform, a different construction with lower latency than
      the one implemented here. *)
