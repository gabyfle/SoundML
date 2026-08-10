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

(** Short-time Fourier transform.

    A {!Config.t} fixes the analysis geometry once; {!transform} is the
    offline transform, {!invert} and {!griffin_lim} are its synthesis side,
    {!Kernel} is the incremental form behind the analysis, and
    {!stage}/{!power_stage} expose the same kernel as {!Pipeline} stages.
    Offline is the one-chunk instance of streaming: {!transform} drives the
    kernel on a single chunk, so the two cannot disagree.

    Complex-valued entry points take the spectrum dtype as an explicit
    dtype-first witness ([transform Nx.complex64 c x] maps float32 audio to a
    complex64 spectrum); OCaml has no implicit type-level mapping from float
    dtypes to complex ones, so the pairing is the caller's choice. Whatever
    the witness, values are computed in double precision and rounded once
    into the requested storage, exactly as librosa rounds its double interior
    into the complex dtype it pairs with the input. The real-valued
    conveniences ({!power_spectrum}, {!power_stage}) are dtype-preserving and
    never expose a complex dtype.

    Defaults are the Hann window, [hop = fft_size / 4] and centered frames
    over reflect padding; numerical parity with librosa 0.11 is enforced
    against committed golden vectors in the test suite. librosa 0.11 itself
    pads with zeros by default — pass [~pad:(`Constant 0.)] to reproduce its
    defaults exactly. *)

(** {1 Configuration} *)

module Config : sig
  (** The type for validated, immutable STFT configurations. Creation
      precomputes the padded analysis window; configurations are cheap to
      share and compare. *)
  type t

  val create :
       ?window:Window.t
    -> ?win_length:int
    -> ?hop:int
    -> ?alignment:[`Centered | `Left | `Right]
    -> ?pad:[`Reflect | `Constant of float | `Edge]
    -> ?scale:[`None | `Magnitude | `Psd]
    -> fft_size:int
    -> unit
    -> t
  (** [create ~fft_size ()] is an STFT configuration.

      [window] defaults to {!Window.Hann} and is instantiated in its periodic
      form at [win_length] points; [win_length] defaults to [fft_size] and a
      shorter window is centered inside the FFT frame with zeros on both
      sides. [hop] is the frame advance in samples and defaults to
      [fft_size / 4] or [1], whichever is larger.

      [alignment] places the analysis window relative to each frame's grid
      position [p * hop] and defaults to [`Centered]:
      - [`Centered] — the window is centered on the grid position; the signal
        is padded by [fft_size / 2] samples on both sides (librosa
        [center=true]).
      - [`Left] — the window starts at the grid position; no padding (librosa
        [center=false]).
      - [`Right] — the window ends at the grid position; the signal is padded
        by [fft_size - 1] samples on the left. Whether frames stay strictly
        causal depends on [pad], below.

      [pad] selects the boundary extension used wherever the alignment pads
      and defaults to [`Reflect] (mirror without repeating the edge sample,
      numpy's [reflect] mode); [`Constant v] extends with [v] and [`Edge]
      repeats the boundary sample. With [`Right] alignment, [`Constant] and
      [`Edge] keep frames strictly causal — every frame reads only samples at
      or before its grid position — while [`Reflect] mirrors early samples
      into the left border.

      [scale] normalises the analysis window and defaults to [`None]:
      [`Magnitude] divides the window by its sum (spectral peaks read as
      component amplitudes) and [`Psd] divides it by the square root of its
      energy (squared magnitudes read as an unnormalised power spectral
      density).

      Raises [Invalid_argument] if [fft_size < 1], if [hop < 1], if
      [win_length] is not in [\[1, fft_size\]], or if the shape parameter of
      [window] is invalid as documented for {!Window.make}. *)

  val fft_size : t -> int
  (** [fft_size c] is the FFT length in samples. *)

  val hop : t -> int
  (** [hop c] is the frame advance in samples. *)

  val win_length : t -> int
  (** [win_length c] is the analysis-window length in samples. *)

  val window : t -> Window.t
  (** [window c] is the window specification. *)

  val alignment : t -> [`Centered | `Left | `Right]
  (** [alignment c] is the frame alignment. *)

  val pad : t -> [`Reflect | `Constant of float | `Edge]
  (** [pad c] is the boundary extension mode. *)

  val scale : t -> [`None | `Magnitude | `Psd]
  (** [scale c] is the window normalisation mode. *)

  val bins : t -> int
  (** [bins c] is the number of non-redundant frequency bins,
      [fft_size / 2 + 1]. *)

  val latency : t -> int
  (** [latency c] is the analysis lookahead in samples: [fft_size / 2] for
      [`Centered] and [0] for [`Left] and [`Right]. Bounded lookahead is
      latency, not offline-ness — centered analysis streams exactly, it
      merely emits each frame once the samples past its grid position have
      arrived. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt c] prints [c] on [fmt] in a compact single-line form. *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff [a] and [b] were created from the same
      parameters; the padding constant is compared with [Float.equal]. *)
end

(** {1 The frame grid}

    Frame [p] of a signal covers the [fft_size] samples starting at
    [p * hop - k] where [k] is [fft_size / 2] for [`Centered], [0] for
    [`Left] and [fft_size - 1] for [`Right]; positions outside the signal
    read the boundary extension. The functions below define that grid once:
    {!transform}, {!transform_range} and the streaming kernel index into the
    same grid, so offline, chunked and streaming analyses agree frame by
    frame. A zero-length signal produces zero frames regardless of
    alignment. *)

val frames : Config.t -> n:int -> int
(** [frames c ~n] is the number of frames a length-[n] signal produces.

    Raises [Invalid_argument] if [n < 0]. *)

val first_complete : Config.t -> int
(** [first_complete c] is the index of the first frame that reads no
    left-boundary padding; frames before it touch the left border. It depends
    only on the configuration and may exceed [frames c ~n] for short
    signals. *)

val last_complete : Config.t -> n:int -> int
(** [last_complete c ~n] is the index of the first frame that reads
    right-boundary padding of a length-[n] signal; frames from it onward
    touch the right border, and it equals [frames c ~n] when no frame does.

    Raises [Invalid_argument] if [n < 0]. *)

val times :
     (float, 'a) Nx.dtype
  -> Config.t
  -> sample_rate:int
  -> n:int
  -> (float, 'a) Nx.t
(** [times dtype c ~sample_rate ~n] is the time of each frame's grid position
    in seconds, [p * hop / sample_rate] for [p] in [\[0, frames c ~n)], as a
    rank-one tensor.

    Raises [Invalid_argument] if [sample_rate < 1] or [n < 0]. *)

val frequencies :
  (float, 'a) Nx.dtype -> Config.t -> sample_rate:int -> (float, 'a) Nx.t
(** [frequencies dtype c ~sample_rate] is the center frequency of each bin in
    hertz, [k * sample_rate / fft_size] for [k] in [\[0, bins c)], as a
    rank-one tensor.

    Raises [Invalid_argument] if [sample_rate < 1]. *)

(** {1 Offline} *)

val transform :
     (Complex.t, 'c) Nx.dtype
  -> Config.t
  -> (float, 'a) Nx.t
  -> (Complex.t, 'c) Nx.t
(** [transform cdtype c x] is the STFT of [x], shaped [[...; bins; frames]].
    The time axis is the last axis of [x]; leading axes broadcast, so a batch
    of clips is one call.

    [transform] is the one-chunk instance of {!Kernel}: one [step] on the
    whole signal plus the drain. Framing is a zero-copy strided view, so peak
    memory is the output plus one windowed copy of the framed signal.

    Raises [Invalid_argument] if [x] has rank zero. *)

val transform_range :
     (Complex.t, 'c) Nx.dtype
  -> Config.t
  -> p0:int
  -> p1:int
  -> (float, 'a) Nx.t
  -> (Complex.t, 'c) Nx.t
(** [transform_range cdtype c ~p0 ~p1 x] is frames [\[p0, p1)] of
    [transform cdtype c x] — the same grid, the same values — computed
    without evaluating the frames outside the range. Adjacent ranges
    reassemble the full transform exactly, which makes chunked, seeking and
    parallel offline evaluation the same computation.

    Raises [Invalid_argument] if [x] has rank zero or if the range does not
    satisfy [0 <= p0 <= p1 <= frames c ~n]. *)

val power_spectrum :
  ?power:float -> Config.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [power_spectrum c x] is [|transform c x| ^ power], shaped
    [[...; bins; frames]], in the dtype of [x]. [power] defaults to [2.] (the
    power spectrum; [1.] is the magnitude spectrum). The complex intermediate
    never escapes: it is stored at the component width matching the dtype of
    [x], and magnitudes land directly in that dtype.

    Raises [Invalid_argument] if [x] has rank zero. *)

(** {1 Synthesis}

    {!invert} is the least-squares inverse of {!transform}: among all signals
    it returns the one whose transform is closest to the given frames, which
    for frames that are an actual transform is that signal back. It reads the
    same {!Config.t} the analysis used: the window and its length, the hop, the
    transform size, the alignment and the normalisation, which cancels because
    the same window appears on both sides. The padding mode is the one field it
    does not read — the boundary extension the alignment implies is trimmed
    back off rather than recomputed, so frames analysed under any [pad] invert
    the same way.

    Reconstruction is defined wherever the analysis windows overlap-add to
    something nonzero. That is the invertibility criterion both entry points
    check: the squared window, summed over the frame grid, must be nonzero at
    every position. It is strictly weaker than asking that sum to be
    {e constant}, because dividing by the measured envelope corrects any shape
    it has, and it fails exactly when a hop leaves positions no window tap
    reaches. It is not the condition {!Window.cola} tests either: that one
    overlap-adds the window itself rather than its square.

    Positions the analysis window sends to zero carry no information and come
    back as [0] rather than as a division by zero. With [`Left] and [`Right]
    alignment the returned signal includes the frame edges where the envelope
    is one window tail and nothing else: those samples are recovered from a
    vanishing weight, so their conditioning is poor — around [(fft_size / pi)]
    to the fourth power at the very first sample of a Hann analysis — and they
    amplify perturbations of the frames accordingly. That is inherent to
    least-squares synthesis, not a defect of this implementation; [`Centered]
    analysis trims those edges away with its padding. *)

val invert :
     (float, 'a) Nx.dtype
  -> Config.t
  -> ?length:int
  -> (Complex.t, 'c) Nx.t
  -> (float, 'a) Nx.t
(** [invert dtype c z] is the least-squares signal whose transform under [c]
    is [z], in [dtype]. [z] is shaped [[...; bins; frames]] — the shape
    {!transform} produces — and leading axes broadcast, so a batch of spectra
    is one call. Values are computed in double precision and rounded once into
    [dtype], like the analysis.

    Without [length] the result has [(frames - 1) * hop + fft_size] samples
    less the boundary extension of the alignment, the shortest length that
    analyses back to exactly [frames] frames:
    [frames c ~n:(Nx.dim (-1) (invert dtype c z)) = frames]. It is not the only
    such length — every one up to a sample short of the next hop analyses to
    [frames] as well — but it is the one returned when none is asked. With
    [length] the result has exactly that many samples: frames that lie entirely
    past it are never inverted, and a length beyond the frames is zero-filled.

    Round trip: [invert dtype c ~length:n (transform cdtype c x)] recovers [x]
    at every position the overlap-added squared window covers, for every
    padding mode and normalisation. In float64 that is exact to a few units in
    the last place of the signal's peak; in float32 the storage rounding
    dominates. Positions outside that cover — the leading [fft_size - hop]
    samples under [`Left] alignment, say — are the least-squares estimate from
    the frames that do reach them, not the original signal.

    Raises [Invalid_argument] if [z] has rank below two, if its bin axis is
    not [bins c] long, if [length] is negative, or if the configuration is not
    invertible: a window whose square does not overlap-add to a nonzero value
    at every position (a hop wider than the window's support, in particular)
    determines no signal at the positions it misses. *)

val griffin_lim :
     ?n_iter:int
  -> ?momentum:float
  -> ?init:[`Zero_phase | `Phase of (float, 'a) Nx.t]
  -> ?length:int
  -> Config.t
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [griffin_lim c s] reconstructs a signal from the magnitude spectrogram [s]
    — {!power_spectrum}[ ~power:1.] output, or any non-negative
    [[...; bins; frames]] tensor on the geometry of [c] — by iterated phase
    estimation, in the dtype of [s]. The complex spectrum never surfaces.

    Each iteration synthesises the current estimate, re-analyses it with [c],
    and keeps the phase it measures while restoring the magnitudes [s]. That
    alternation decreases the spectral convergence

    [SC = ‖ |transform (griffin_lim …)| - s ‖ / ‖ s ‖]

    — the normalised distance between the magnitudes asked for and the
    magnitudes obtained — monotonically at [momentum = 0]. [momentum]
    defaults to [0.99] and accelerates it by extrapolating along the previous
    step before each phase update; the sequence is then no longer monotone but
    converges markedly faster, and values above [1.] are accepted and may not
    converge at all. [n_iter] defaults to [32] and is run in full: there is no
    early stop, so the cost is exactly [n_iter] analysis-synthesis pairs plus
    one synthesis.

    [init] fixes the starting phase and defaults to [`Zero_phase], all bins
    starting at phase zero; [`Phase p] starts from the phases in [p], in
    radians, shaped like [s]. No randomness is involved at any point — the
    result is a function of the arguments alone, identical across runs and
    machines up to floating-point reproducibility.

    [length] fixes the length of the returned signal exactly as in {!invert},
    and applies only to the final synthesis: the iteration itself always runs
    at the natural length, the one that re-analyses to the frame count of [s].

    Raises [Invalid_argument] under the conditions of {!invert}, and if
    [n_iter < 1], if [momentum] is negative, or if [`Phase p] does not have
    the shape of [s]. *)

(** {1 Incremental kernel}

    The Mealy kernel behind every face of this module. [step] consumes one
    chunk of samples — the time axis last, leading axes broadcast — and emits
    the frames that became complete, carrying the boundary overlap across
    chunks; [flush] installs the right boundary extension and emits the
    remaining frames. Concatenating every emitted chunk along the time axis
    equals {!transform} on the concatenated input, for every partitioning of
    the signal. *)

module Kernel : sig
  (** The type for prepared kernel states. Mutable; single-owner; not
      domain-safe. *)
  type ('a, 'c) t

  val prepare :
       (Complex.t, 'c) Nx.dtype
    -> Config.t
    -> (float, 'a) Nx.dtype
    -> channels:int
    -> max_block:int
    -> ('a, 'c) t
  (** [prepare cdtype c dtype ~channels ~max_block] is a fresh kernel state
      for chunks of at most [max_block] samples of [channels]-channel
      [dtype] audio, producing [cdtype] spectra.

      Raises [Invalid_argument] if [channels < 1] or [max_block < 1]. *)

  val step : ('a, 'c) t -> (float, 'a) Nx.t -> (Complex.t, 'c) Nx.t option
  (** [step k chunk] feeds [chunk] and is the newly completed frames, if any,
      shaped [[...; bins; frames]]. [chunk] is borrowed: the kernel copies
      what it must retain, and the returned tensor aliases neither [chunk]
      nor kernel state. Framing and FFT run batched over the chunk's whole
      time axis.

      Raises [Invalid_argument] if [k] was drained by {!flush} — {!reset} it
      before feeding a new signal — or if a leading axis of [chunk] has size
      zero. *)

  val flush : ('a, 'c) t -> (Complex.t, 'c) Nx.t option
  (** [flush k] drains the kernel: it installs the right boundary extension
      and is the remaining frames, if any. Draining consumes the tail — a
      second [flush] is [None]; {!reset} the kernel before reusing it. *)

  val reset : ('a, 'c) t -> unit
  (** [reset k] restores [k] to its freshly prepared state. *)
end

(** {1 Pipeline stages} *)

val stage :
     (Complex.t, 'c) Nx.dtype
  -> Config.t
  -> ((float, 'a) Nx.t, (Complex.t, 'c) Nx.t, 'k) Pipeline.t
(** [stage cdtype c] is {!Kernel} as a pipeline stage: causal, with
    [Config.latency c] samples of latency and rate [1 / hop] — one spectral
    frame per [hop] input samples. The one padding mode whose border itself
    looks ahead adds to the declared latency: [`Right] alignment with
    [`Reflect] padding reflects the first [fft_size - 1] samples into the
    left border, and the stage declares that lookahead. Chunks are joined by
    concatenation along the time axis; joining zero chunks yields an empty
    single-channel spectrum [[bins; 0]]. The threaded stream format keeps the
    source's element dtype entry: the spectrum's own dtype is [cdtype],
    carried statically by the stage's type. *)

val power_stage :
     ?power:float
  -> Config.t
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [power_stage c] is {!power_spectrum} as a pipeline stage — real audio in,
    real spectra out, dtype-preserving, with the latency and rate of
    {!stage}. [power] defaults to [2.]. Chunks are joined like {!stage}'s:
    joining zero chunks yields an empty single-channel spectrum [[bins; 0]].
    Compose it downstream of causal stages without ever naming a complex
    dtype. *)
