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
    per-bin bandwidths, the octave plan and every filter kernel — and
    {!transform} runs it over a signal. One configuration serves both
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
    {!section:parity} for exactly what that costs against librosa 0.11.

    Complex-valued entry points take the spectrum dtype as an explicit
    dtype-first witness, exactly like {!Stft.transform}. Whatever the witness,
    the filter bank, the FFTs and the projection all run in double precision
    and round once into the requested storage. The real-valued convenience
    ({!power_spectrum}) is dtype-preserving and never exposes a complex dtype.

    Inverse transforms, invertible non-stationary Gabor frames, the hybrid and
    pseudo constant-Q approximations, tuning estimation, sparsified filter
    bases, intervals other than equal temperament and a streaming
    {!Pipeline} stage are all deliberately outside this module: the transform
    here is offline and centered. *)

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
      defaults to [`Constant 0.] — the zero padding librosa 0.11's own
      constant-Q transform uses, unlike its STFT.

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
    [2 ^ (k / bins_per_octave)] in one step instead — as librosa 0.11's
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
    librosa 0.11's [filters.wavelet_lengths] report. *)

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
