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

(** Linear recursive filtering: scipy's transfer-function and second-order
    section filters with their initial and final states, their steady states
    and zero-phase forms, Butterworth design, and librosa's pre-emphasis.

    {2 Coefficients}

    A filter is given by float64 coefficient tensors, the representation
    scipy's designs return. A {e transfer function} is a pair [(b, a)] of
    rank-one tensors, the numerator and denominator of

    {[ H(z) = (b0 + b1 z^-1 + ... + b(nb-1) z^-(nb-1))
            / (a0 + a1 z^-1 + ... + a(na-1) z^-(na-1)) ]}

    with [b] and [a] non-empty and [a0] nonzero: both are divided by [a0]
    before use, and the shorter is zero-filled to [max nb na] taps. A cascade
    of {e second-order sections} is one tensor shaped [[n_sections; 6]] whose
    rows [[b0; b1; b2; a0; a1; a2]] are transfer functions of order two with
    [a0 = 1], applied in row order. {!butter} designs the first form and
    {!butter_sos} the second; any other design is the caller's.

    Above order three or so, prefer sections. Multiplying out the roots into a
    single polynomial is numerically sensitive — scipy's own documentation
    says so — and a direct-form filter amplifies the rounding of its
    coefficients and of its arithmetic: rounding each product of its
    multiply-adds on its own, rather than contracting it, moves a sixth-order
    50 Hz highpass at 16 kHz by [4.2e-6] of its largest output as a transfer
    function, and by [6.3e-14] as sections.

    {2 State}

    The {e state} of a filter is its memory between two samples: the
    [order = max nb na - 1] delays of the transposed direct form for a
    transfer function, shaped [[...; order]], and the two delays of each
    section for a cascade, shaped [[...; n_sections; 2]] — always float64,
    whatever the dtype of the signal. A [~zi] starts a filter from that state
    instead of from rest (all zeros), and the [_zf] variants return the state
    after the last sample beside the output, so that blocks chain bit for bit
    at both dtypes:

    {[
      let y1, z1 = lfilter_zf ~zi b a x1 in
      let y2 = lfilter ~zi:z1 b a x2 in
      (* concat y1 y2 = lfilter ~zi b a (concat x1 x2), every bit *)
    ]}

    A [~zi] broadcasts against the leading axes of the signal: it may omit
    leading axes or give them size one, so a rank-one state starts every
    channel alike. {!lfilter_zi} and {!sosfilt_zi} are the steady states of a
    unit step, which a caller scales by a signal's first samples —
    [Nx.mul (lfilter_zi b a) x0] with [x0] shaped [[...; 1]] — to start a
    filter as if the signal had always been at that value.

    {2 Arithmetic}

    {!lfilter} runs the transposed direct form II, the structure of
    [scipy.signal.lfilter] (and of MATLAB's [filter]): per sample,

    {[
      y = z0 + b0 x
      z(k) = z(k+1) + b(k+1) x - a(k+1) y      (k < order - 1)
      z(order-1) = b(order) x - a(order) y
    ]}

    and {!sosfilt} runs the same structure per section, each sample through
    every section before the next sample. The interior is float64 at both
    dtypes and the output is rounded once into the dtype of the signal.
    Signals are [[...; samples]] in float32 or float64, the time axis last;
    leading axes broadcast, so a batch of channels is one call, and rows run
    across worker domains without changing a bit.

    {2 Streaming}

    {!lfilter_stage} and {!sosfilt_stage} stream the two filters, and
    {!preemphasis_stage} and {!deemphasis_stage} the emphasis filters; each
    carries its state from chunk to chunk and is its offline function bit for
    bit, at every partitioning of the signal:

    {[
      Pipeline.run ~source (lfilter_stage ?zi b a) x = lfilter ?zi b a x
    ]}

    The zero-phase {!filtfilt} and {!sosfiltfilt} run the filter backward
    from the end of the signal and are offline only; {!Pipeline.offline_only}
    wraps them when a pipeline needs them.

    {2 Parity}

    Numerical parity with scipy 1.18 ([lfilter], [lfilter_zi], [sosfilt],
    [sosfilt_zi], [filtfilt], [sosfiltfilt], [butter]) and librosa 1.0
    ([librosa.effects.preemphasis] and [deemphasis]) is enforced against
    committed golden vectors. Every recurrence keeps the reference's order of
    operations and its rounding of each product, including the multiply-adds
    its compiler contracts on arm64; the steady states sum their coefficients
    in numpy's pairwise order, and the design reproduces numpy's complex
    multiply, division and modulus. The vectors are generated on arm64, where
    every filtering, steady-state, zero-phase and emphasis case agrees bit for
    bit, but for the FIR filters below. An x86_64 build, which contracts no
    multiply-add where the reference's arm64 build does, stands at the
    contraction's distance from them: up to [1.2e-13] of the largest output
    over the transfer functions of the vectors, which keep to designs of order
    four or less, [3.3e-12] over the sections, [3.6e-14] over the zero-phase
    cases, and nothing over the steady states and the emphasis filters, whose
    arithmetic contracts to the same bits. The deviations that remain are
    these.

    - The interior and the state are float64 at both dtypes, and the output
      is rounded once into the dtype of the signal. scipy returns float64 for
      a float32 signal filtered by float64 coefficients, and librosa filters a
      float32 signal in float32 with the coefficient rounded to float32: its
      float32 de-emphasis of a second of the golden signal at 22.05 kHz
      strays from its own float64 result by [6.9e-7] of the peak, a dozen
      float32 units in the last place, where this one keeps to the rounding
      of the output.
    - A state broadcasts as above and a cascade's state is
      [[...; n_sections; 2]], the time-axis-last layout of the library;
      scipy's is [(n_sections, ..., 2, ...)], and its [zi] must have every
      dimension of the signal.
    - A transfer function with a single denominator coefficient (an FIR
      filter) runs the same recurrence, where scipy convolves, in an order
      that depends on whether the signal is longer than the filter, which a
      stream cannot know. On signals longer than the filter the two agree bit
      for bit at two taps — pre-emphasis — and at three in the vectors, and
      otherwise round each output differently by a few units in the last
      place of its terms: [3.4e-16] of the largest output at sixteen taps.
      Where the terms cancel, short signals show more: [2.1e-16] of the
      output for pre-emphasis on two samples, [7.0e-15] for five taps with
      [a0 = 2.5] on one sample and a state.
    - An empty signal gives an empty output and, from a [_zf] variant, its
      initial state back. scipy raises on an empty FIR signal and returns an
      uninitialised state for an empty IIR one.
    - {!deemphasis_zf} without [~zi] returns the state that continues its
      output (see there); librosa returns the state of an intermediate
      recursion, from which a next block cannot continue.
    - The band transforms of {!butter} take the complex square root by
      Friedland's algorithm, where numpy calls the platform's [csqrt], whose
      last bit is the platform's own. Every design of the vectors is scipy's
      bit for bit on arm64. Lowpass and highpass designs take no square
      root; of 515 random band designs of orders 1 to 16 at eight sample
      rates from 2 to 96000, 37 take one that rounds otherwise than Apple's
      and differ from scipy by up to [8.7e-15] of the largest coefficient as
      a transfer function, [2.0e-15] as sections. An x86_64 build differs
      from the vectors by up to [2.9e-15].
    - Cutoffs are in hertz under a mandatory [~sample_rate], and the band
      type is a variant that carries its cutoffs; [~sample_rate:2] gives
      scipy's normalised frequencies exactly.
    - Non-finite coefficients and a non-finite emphasis coefficient raise
      [Invalid_argument]; the references filter with them. So do a NaN
      cutoff, from which scipy's [butter] designs a transfer function of NaN
      coefficients, and, in {!sosfilt_zi}, a section whose [a0] is not [1]
      or a cascade without sections, where scipy normalises the section or
      returns an empty state.

    Citations. Smith, J. O., {e Introduction to Digital Filters with Audio
    Applications}, CCRMA, 2007 (the transposed direct forms, series
    second-order sections). Oppenheim, A. V. and Schafer, R. W.,
    {e Discrete-Time Signal Processing}, 3rd ed., 2010, §7.2 (the bilinear
    transformation) and Appendix B (Butterworth filters). Gustafsson, F.,
    "Determining the initial states in forward-backward filtering",
    {e IEEE Transactions on Signal Processing} 44(4), 1996. Friedland, P.,
    "Algorithm 312: Absolute value and square root of a complex number",
    {e Communications of the ACM} 10(10), 1967. *)

(** {1 Types} *)

(** The type for filter coefficients: a rank-one numerator or denominator,
    or a [[n_sections; 6]] cascade of second-order sections. *)
type coefficients = (float, Nx.float64_elt) Nx.t

(** The type for filter states: [[...; order]] for a transfer function,
    [[...; n_sections; 2]] for a cascade, [[...; 1]] for the emphasis
    filters. *)
type state = (float, Nx.float64_elt) Nx.t

(** The type for Butterworth band types, cutoffs in hertz. *)
type band =
  [ `Lowpass of float  (** passes below the cutoff *)
  | `Highpass of float  (** passes above the cutoff *)
  | `Bandpass of float * float  (** passes between the two edges *)
  | `Bandstop of float * float  (** rejects between the two edges *) ]

(** {1:design Design} *)

val butter : order:int -> sample_rate:int -> band -> coefficients * coefficients
(** [butter ~order ~sample_rate band] is the digital Butterworth filter of
    order [order] for [band] as a transfer function [(b, a)], each of
    [order + 1] coefficients — [2 order + 1] for the band types —, after
    [scipy.signal.butter ~output:"ba"]. [order = 0] is the identity
    [([1.], [1.])].

    The design is scipy's: the analog prototype's poles on the unit circle,
    the frequency transformation to [band] at the cutoffs pre-warped by
    [4 tan (pi f / sample_rate)], the bilinear map at a sample rate of two,
    and the poles and zeros multiplied out — each value computed with the
    operations numpy computes it with. A band edge is a [-3] dB point of the
    digital filter. Above order three or so, {!butter_sos} is the numerically
    sound form (see the module's introduction).

    Raises [Invalid_argument] if [order < 0], if [sample_rate < 1], if a
    cutoff is not finite or does not lie strictly between [0] and
    [sample_rate / 2], or if a band's lower edge does not lie below its upper
    edge. *)

val butter_sos : order:int -> sample_rate:int -> band -> coefficients
(** [butter_sos ~order ~sample_rate band] is {!butter}'s filter as a cascade
    of [ceil (order / 2)] second-order sections — [order] for the band types
    —, shaped [[n_sections; 6]], after
    [scipy.signal.butter ~output:"sos"]: the same zeros, poles and gain,
    paired by scipy's [zpk2sos] with its default pairing, poles closest to
    the unit circle last, each with its nearest zeros, the gain in the first
    section. [order = 0] is the single section [[1; 0; 0; 1; 0; 0]].

    Raises [Invalid_argument] under the conditions of {!butter}. *)

(** {1:tf Transfer functions} *)

val lfilter :
     ?zi:state
  -> coefficients
  -> coefficients
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [lfilter ?zi b a x] is [x] filtered along its last axis by the transfer
    function [b / a], in the dtype and shape of [x], after
    [scipy.signal.lfilter]: the transposed direct form II, started from
    [zi] — shaped [[...; order]], broadcast against the leading axes of [x] —
    or from rest. Costs [2 max nb na - 1] multiply-adds per sample.

    Raises [Invalid_argument] if [b] or [a] is not a non-empty rank-one
    tensor of finite values, if [a.(0) = 0], if [x] has rank zero or a dtype
    other than float32 and float64, or if [zi] does not broadcast to
    [[...; order]] over the leading shape of [x]. *)

val lfilter_zf :
     ?zi:state
  -> coefficients
  -> coefficients
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * state
(** [lfilter_zf ?zi b a x] is [(lfilter ?zi b a x, zf)] with [zf] the state
    after the last sample of each row, shaped [[...; order]] over the leading
    shape of [x] — scipy's [lfilter] with [zi] given, which returns both. Fed
    back as the [~zi] of the next block it continues the filter bit for bit.
    An empty [x] returns its initial state.

    Raises [Invalid_argument] under the conditions of {!lfilter}. *)

val lfilter_zi : coefficients -> coefficients -> state
(** [lfilter_zi b a] is the state, shaped [[max nb na - 1]], in which the
    filter [b / a] has settled under a constant input of [1] — its response
    to a unit step started at minus infinity — after
    [scipy.signal.lfilter_zi]: with [y = sum b / sum a] the step's limit, the
    running sums of [b - y a] from the last coefficient back. Scaled by a
    signal's first sample it starts the filter without the transient of a
    start from rest. A gain-only filter has the empty state.

    Raises [Invalid_argument] under the conditions of {!lfilter} on [b] and
    [a], or if the coefficients of [a] sum to [0] — a pole at [z = 1], whose
    step response never settles. *)

val lfilter_stage :
     ?zi:state
  -> coefficients
  -> coefficients
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [lfilter_stage ?zi b a] is {!lfilter} as a pipeline stage: causal, no
    latency, every sample of a chunk filtered and emitted at once, the state
    carried from chunk to chunk in float64. The state is made at the first
    chunk, from [zi] broadcast against its leading shape or from rest, and
    reset to it by {!Pipeline.Stream.reset}. Run on a whole signal it is
    [lfilter ?zi b a], bit for bit, at every partitioning. An empty chunk
    emits an empty chunk.

    Raises [Invalid_argument] under the conditions of {!lfilter} on [b], [a]
    and the trailing axis of [zi]; and, at a chunk, under those on the signal
    and the broadcast of [zi], or if the chunk's leading shape differs from
    the stream's first. *)

val filtfilt :
     ?pad:[`Odd | `Reflect | `Edge]
  -> ?padlen:int
  -> coefficients
  -> coefficients
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [filtfilt ?pad ?padlen b a x] is [x] filtered by [b / a] forward and then
    backward along its last axis, in the dtype and shape of [x]: a filter of
    zero phase whose magnitude response is the square of [b / a]'s, after
    [scipy.signal.filtfilt] with its default method ["pad"]. Offline only.

    Each row is extended by [padlen] samples at each end, each pass starts
    from {!lfilter_zi} scaled by the first sample it reads, and the output is
    the samples under the original row. [pad] is the extension and defaults to
    [`Odd], scipy's default — the signal rotated by half a turn about its end
    samples, [2 x0 - x(k)] —; [`Reflect] mirrors it about its end samples
    (scipy's ["even"]) and [`Edge] repeats them (scipy's ["constant"]).
    [padlen] defaults to [3 (max nb na)], scipy's default, and [~padlen:0]
    filters the row unextended (scipy's [padtype=None]). Gustafsson's method,
    scipy's ["gust"], is not offered.

    Costs twice {!lfilter} over the extended rows, in float64.

    Raises [Invalid_argument] under the conditions of {!lfilter} on [b], [a]
    and [x], or of {!lfilter_zi}, if [padlen < 0], or if a row of [x] is not
    longer than [padlen]. *)

(** {1:sos Second-order sections} *)

val sosfilt : ?zi:state -> coefficients -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [sosfilt ?zi sos x] is [x] filtered along its last axis by the cascade
    [sos], in the dtype and shape of [x], after [scipy.signal.sosfilt]: each
    section a transposed direct form II, each sample through every section in
    turn, the intermediate values in float64, started from [zi] — shaped
    [[...; n_sections; 2]], broadcast against the leading axes of [x] — or
    from rest. Costs five multiply-adds per section per sample.

    Raises [Invalid_argument] if [sos] is not a [[n_sections; 6]] tensor of
    finite values with at least one section, if a section's [a0] is not
    exactly [1], if [x] has rank zero or a dtype other than float32 and
    float64, or if [zi] does not broadcast to [[...; n_sections; 2]] over the
    leading shape of [x]. *)

val sosfilt_zf :
  ?zi:state -> coefficients -> (float, 'a) Nx.t -> (float, 'a) Nx.t * state
(** [sosfilt_zf ?zi sos x] is [(sosfilt ?zi sos x, zf)] with [zf] the state
    after the last sample of each row, shaped [[...; n_sections; 2]] over the
    leading shape of [x]. Fed back as the [~zi] of the next block it continues
    the filter bit for bit. An empty [x] returns its initial state.

    Raises [Invalid_argument] under the conditions of {!sosfilt}. *)

val sosfilt_zi : coefficients -> state
(** [sosfilt_zi sos] is the state, shaped [[n_sections; 2]], in which the
    cascade [sos] has settled under a constant input of [1], after
    [scipy.signal.sosfilt_zi]: each section's {!lfilter_zi} scaled by the DC
    gain of the sections before it.

    Raises [Invalid_argument] under the conditions of {!sosfilt} on [sos], or
    if a section's denominator sums to [0]. *)

val sosfilt_stage :
     ?zi:state
  -> coefficients
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [sosfilt_stage ?zi sos] is {!sosfilt} as a pipeline stage, as
    {!lfilter_stage} is {!lfilter}'s: causal, no latency, the state carried
    in float64, [sosfilt ?zi sos] bit for bit at every partitioning.

    Raises [Invalid_argument] under the conditions of {!sosfilt} on [sos] and
    the trailing axes of [zi]; and, at a chunk, under those on the signal and
    the broadcast of [zi], or if the chunk's leading shape differs from the
    stream's first. *)

val sosfiltfilt :
     ?pad:[`Odd | `Reflect | `Edge]
  -> ?padlen:int
  -> coefficients
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [sosfiltfilt ?pad ?padlen sos x] is {!filtfilt} for the cascade [sos],
    after [scipy.signal.sosfiltfilt]: each pass starts from {!sosfilt_zi}
    scaled by the first sample it reads. [padlen] defaults to scipy's
    [3 (2 n_sections + 1 - min zb za)], with [zb] and [za] the numbers of
    sections whose [b2] and [a2] are zero — eighteen for a fifth-order
    Butterworth lowpass. Offline only.

    Raises [Invalid_argument] under the conditions of {!sosfilt} on [sos] and
    [x], or of {!sosfilt_zi}, if [padlen < 0], or if a row of [x] is not
    longer than [padlen]. *)

(** {1:emphasis Pre-emphasis}

    [preemphasis] is the first difference [y(t) = x(t) - coef x(t-1)], which
    tilts a spectrum up by about 6 dB per octave above [(1 - coef)] times the
    sample rate over [2 pi] — the step speech front ends take before
    framing — and [deemphasis] is its inverse, the one-pole recursion
    [y(t) = x(t) + coef y(t-1)]. Both are transfer functions of order one, so
    their state is shaped [[...; 1]]: {!preemphasis} is
    [lfilter ~zi [1; -coef] [1]] and {!deemphasis} is
    [lfilter ~zi [1] [1; -coef]], bit for bit, when [~zi] is given.

    Without [~zi], both reproduce librosa's start, which reads the first two
    samples: {!preemphasis} starts from the state [2 x0 - x1], so that
    [y0 = 3 x0 - x1] whatever [coef] — not the linear extrapolation of the
    signal librosa's comment names, which would give
    [y0 = x0 - coef (2 x0 - x1)] — and {!deemphasis} runs from rest and
    subtracts [((2 - coef) x0 - x1) / (3 - coef) coef^t] from sample [t]
    afterwards. Without [~zi], a signal of one sample raises, as it does in
    librosa.

    The other front ends' conventions are states: the start from rest of
    torchaudio and python_speech_features is [~zi] of zeros, and Kaldi's and
    HTK's, which treat the sample before the first as equal to it, is
    [~zi:(-coef x0)]. The inverse of [preemphasis ~zi:z x] is
    [deemphasis ~zi:(-z)] — the opposite state, where librosa's documentation
    says the same one. *)

val preemphasis :
  ?coef:float -> ?zi:state -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [preemphasis ?coef ?zi x] is [x] pre-emphasised along its last axis, in
    the dtype and shape of [x], after [librosa.effects.preemphasis]: [coef]
    defaults to [0.97], librosa's and HTK's default, and [zi], shaped
    [[...; 1]], to librosa's start from the first two samples (see above). An
    empty [x] gives an empty result.

    Raises [Invalid_argument] if [coef] is not finite, if [x] has rank zero
    or a dtype other than float32 and float64, if [zi] does not broadcast to
    [[...; 1]] over the leading shape of [x], or if [zi] is absent and the
    rows of [x] hold a single sample. *)

val preemphasis_zf :
  ?coef:float -> ?zi:state -> (float, 'a) Nx.t -> (float, 'a) Nx.t * state
(** [preemphasis_zf ?coef ?zi x] is [(preemphasis ?coef ?zi x, zf)] with
    [zf], shaped [[...; 1]], the state after the last sample, [-coef] times
    it — librosa's [return_zf] — with which the next block continues bit for
    bit. An empty [x] returns its initial state, zeros without [zi].

    Raises [Invalid_argument] under the conditions of {!preemphasis}. *)

val preemphasis_stage :
     ?coef:float
  -> ?zi:state
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [preemphasis_stage ?coef ?zi ()] is {!preemphasis} as a pipeline stage:
    [preemphasis ?coef ?zi] bit for bit at every partitioning, the state
    carried from chunk to chunk.

    With [~zi] it is {!lfilter_stage} of the filter: no latency. Without it,
    the first output depends on the second sample, so the stage reproduces
    librosa's start by holding the first sample back until the second
    arrives, and declares one sample of latency for it: a first chunk of one
    sample emits an empty chunk, the next emits both, and every chunk after
    the start is emitted whole. Its flush emits nothing, and raises
    [Invalid_argument] after a stream of a single sample, as {!preemphasis}
    raises on one.

    Raises [Invalid_argument] under the conditions of {!preemphasis} on
    [coef] and the trailing axis of [zi]; and, at a chunk, under those on the
    signal and the broadcast of [zi], or if the chunk's leading shape differs
    from the stream's first. *)

val deemphasis :
  ?coef:float -> ?zi:state -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [deemphasis ?coef ?zi x] is [x] de-emphasised along its last axis, in
    the dtype and shape of [x], after [librosa.effects.deemphasis]: [coef]
    defaults to [0.97] and [zi], shaped [[...; 1]], to librosa's correction
    of a start from rest (see above), which makes
    [deemphasis (preemphasis x)] give [x] back to the arithmetic. An empty
    [x] gives an empty result.

    Raises [Invalid_argument] under the conditions of {!preemphasis}. *)

val deemphasis_zf :
  ?coef:float -> ?zi:state -> (float, 'a) Nx.t -> (float, 'a) Nx.t * state
(** [deemphasis_zf ?coef ?zi x] is [(deemphasis ?coef ?zi x, zf)] with [zf],
    shaped [[...; 1]], [coef] times the last output: the state with which
    [deemphasis ~zi:zf] continues the output into the next block. With [~zi]
    the continuation is bit for bit. Without it, the next block continues the
    recursion [y(t) = x(t) + coef y(t-1)] that the output satisfies, where the
    one-shot result keeps subtracting its correction from an uncorrected
    recursion: the two agree to the arithmetic, not to the bit (the stage, not
    the chaining, is the bit-exact way to stream it). This is a named
    deviation: librosa returns the uncorrected recursion's state, with which
    its documented block-by-block use strays from the one-shot result by
    [0.13] (of a peak of [5.8]) on the golden signal cut after fifty samples,
    where this state continues it to [1.8e-15]. An empty [x] returns its
    initial state, zeros without [zi].

    Raises [Invalid_argument] under the conditions of {!preemphasis}. *)

val deemphasis_stage :
     ?coef:float
  -> ?zi:state
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [deemphasis_stage ?coef ?zi ()] is {!deemphasis} as a pipeline stage,
    [deemphasis ?coef ?zi] bit for bit at every partitioning — without [~zi]
    too, for the stage carries the uncorrected recursion, the correction
    constants and the sample count, and subtracts the correction as the
    offline function does. Its latency, startup and flush are
    {!preemphasis_stage}'s.

    Raises [Invalid_argument] under the conditions of {!preemphasis_stage}. *)
