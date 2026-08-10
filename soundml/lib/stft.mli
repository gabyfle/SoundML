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
    {!Kernel} and {!Synthesis} are the incremental forms of the two
    directions, and {!stage}/{!power_stage}/{!synthesis_stage} expose those
    kernels as {!Pipeline} stages. Offline is the one-chunk instance of
    streaming, in both directions: {!transform} drives {!Kernel} on a single
    chunk and {!invert} is what {!Synthesis} totals over a stream, so no face
    of this module can disagree with another.

    Complex-valued entry points take the spectrum dtype as an explicit
    dtype-first witness ([transform Nx.complex64 c x] maps float32 audio to a
    complex64 spectrum); OCaml has no implicit type-level mapping from float
    dtypes to complex ones, so the pairing is the caller's choice. Whatever
    the witness, values are computed in double precision and rounded once
    into the requested storage. The real-valued
    conveniences ({!power_spectrum}, {!power_stage}) are dtype-preserving and
    never expose a complex dtype.

    Defaults are the Hann window, [hop = fft_size / 4] and centered frames
    over reflect padding; numerical parity with librosa 0.11 — including the
    double interior above — is enforced against committed golden vectors in
    the test suite. [`Centered] and [`Left] alignment correspond to its
    [center=True] and [center=False]; librosa 0.11 itself
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
        is padded by [fft_size / 2] samples on both sides.
      - [`Left] — the window starts at the grid position; no padding.
      - [`Right] — the window ends at the grid position; the signal is padded
        by [fft_size - 1] samples on the left. Whether frames stay strictly
        causal depends on [pad], below.

      [pad] selects the boundary extension used wherever the alignment pads
      and defaults to [`Reflect] (mirror without repeating the edge
      sample); [`Constant v] extends with [v] and [`Edge]
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

  val synthesis_latency : t -> int
  (** [synthesis_latency c] is the lookahead incremental synthesis carries
      ({!Synthesis}), in {e output} samples. Writing [L] for the head trim of
      the alignment — [fft_size / 2] under [`Centered], [0] under [`Left] and
      [fft_size - 1] under [`Right] — and [R] for its tail trim
      ([fft_size / 2], [0], [0]), it is

      {[ synthesis_latency = L + max 0 (hop + R - fft_size) ]}

      the positions the alignment trims off the head, plus the positions one
      hop opens past the trailing trim — nonzero only where the hop outruns
      what the frame keeps, and released as soon as the next frame moves that
      trim along. Synthesis withholds no {e frame}: output sample [q] is
      final once frame [(q + synthesis_latency c) / hop] has been consumed,
      so the whole of this number sits on the output side, which is where
      {!synthesis_stage} declares it. *)

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
    the same window appears on both sides. The padding mode is the one field
    {!invert} does not read — the boundary extension the alignment implies is
    trimmed back off rather than recomputed, so frames analysed under any
    [pad] invert the same way. {!griffin_lim} does read it: every iteration
    re-analyses the signal it has just synthesised, and that analysis pads.

    Reconstruction is defined wherever the analysis windows overlap-add to
    something the division can use. That is the invertibility criterion both
    entry points check: the squared window, summed over the frame grid, must
    stay above [1e-10] of its own largest value at every position. It is
    strictly weaker than asking that sum to be {e constant}, because dividing
    by the measured envelope corrects any shape it has, and strictly stronger
    than asking it to be nonzero, because an envelope that clears zero by less
    than the rounding of the two transforms determines its position in name
    only — the check is numerical, as {!Window.cola}'s is, and the floor is
    the whole of the difference. So it fails in two ways: at positions no
    {e nonzero} window tap reaches — a hop past the window's support, or one
    that lands nothing there but zeros of the window itself, as a periodic
    Hann advanced by its own length does at the frame boundary — and at
    positions a strongly tapered window reaches through its far tails alone,
    where the fold is positive but below the floor, as a Kaiser or a Gaussian
    of large shape parameter is at a quarter overlap. It is not the condition
    {!Window.cola} tests either: that one overlap-adds the window itself
    rather than its square.

    Positions the analysis window sends to zero carry no information and come
    back as [0] rather than as a division by zero. The returned signal reaches
    the trailing frame edge under [`Left] and [`Right] alignment and the
    leading one under [`Left] alone: [`Right] extends the signal by
    [fft_size - 1] positions on the left, so its position 0 already receives
    every tap of its residue class, exactly as an interior position does. At
    an edge it reaches, the envelope is one window tail and nothing else, and
    those samples are ill conditioned: a perturbation of the frames enters the
    overlap-add through a single window tap and the envelope through that tap
    squared, so it moves such a sample by about the reciprocal of the tap more
    than it moves an interior one. Beside the vanishing outermost tap of a
    Hann analysis the tap is [(pi / win_length)] squared — the length of the
    window governs it, not the transform size it is padded into — so the
    amplification there is of the order of [(win_length / pi)] squared. A
    [win_length] below [fft_size] moves that edge inward: the
    [(fft_size - win_length) / 2] outermost positions receive no tap at all
    and come back as [0], and the ill conditioned run starts after them. That
    is inherent to least-squares synthesis, not a defect of this
    implementation; [`Centered] analysis trims both edges away with its
    padding. *)

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

    Without [length] and at [frames >= 1] the result has
    [(frames - 1) * hop + fft_size] samples less the boundary extension of the
    alignment; an empty spectrum returns an empty signal, which that formula
    does not describe. Either is the shortest length that analyses back to
    exactly [frames] frames:
    [frames c ~n:(Nx.dim (-1) (invert dtype c z)) = frames]. Neither is the
    only such length, and the run of lengths sharing the frame count depends on
    that count: at [frames >= 1] it is the [hop] consecutive lengths starting
    at the returned one, and at [frames = 0] it is [0] — which analyses to no
    frames because the empty signal has none by definition, not because its
    extension falls short of a frame — together with the positive lengths the
    boundary extension does leave shorter than one: none under [`Centered] and
    [`Right], whose extensions are a frame wide to within one sample, so the
    run is [0] alone, and [1] to [fft_size - 1] under [`Left], which extends
    nothing. The length just past either run analyses to one frame more.

    One geometry escapes the fixed point, and only at [frames = 1]: under
    [`Centered] with an even [fft_size] the extension is the whole frame, so
    the length is [0], which analyses to no frames at all. The lengths that
    analyse to one frame are [1] to [hop - 1] there, and there are none when
    [hop = 1] — the length past the zero-frame run gains two frames at once
    instead of one.

    With [length] the result has exactly that many samples: the frames that
    open before its end are inverted and those that lie entirely past it are
    not, so their contents cannot reach the result, and a length beyond the
    frames is zero-filled.

    Round trip: [invert dtype c ~length:n (transform cdtype c x)] recovers [x]
    at every position some frame reaches through a nonzero window tap, for
    every padding mode and normalisation: the taps that reach a position
    weight the overlap-add and the envelope alike, so one the frame pattern
    covers only in part cancels exactly as an interior one does. Everything
    else comes back as [0]: the positions every tap reaching them sends to
    zero, and the positions past the last frame when [length] runs beyond it.
    A [win_length] below [fft_size] sits centred in the frame at offset
    [(fft_size - win_length) / 2] and is padded with zeros to the frame, so
    the first kind runs along both frame edges — that offset many positions
    at the head, one more under a Hann window, which itself opens at zero,
    and [fft_size - win_length - ((fft_size - win_length) / 2)] at the tail,
    inside the last frame's transform span rather than past it. [`Left]
    returns both runs; [`Right] returns the trailing one only, its left
    extension already carrying the frame pattern past position 0; and
    [`Centered] trims both away with its padding. In float64 the covered
    positions are exact to a few units in the last place of the signal's peak
    and in float32 the storage rounding dominates, but at the partially
    covered edges the amplification above applies to that rounding too: a
    2048-point Hann analysis returns them to around [1e-11] rather than to the
    last place.

    Raises [Invalid_argument] if [z] has rank below two, if its bin axis is
    not [bins c] long, if [length] is negative, or if the configuration is not
    invertible: a window whose square overlap-adds somewhere on the frame grid
    to less than [1e-10] of its largest value there (a hop wider than the
    window's support, in particular) determines no signal at those positions,
    or none the envelope division could carry. *)

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
    and keeps the phase it measures while restoring the magnitudes [s]. The
    re-analysis pads the way {!transform} does, so the whole of [c] is in
    play here — [pad] included, unlike in {!invert}: wherever the alignment
    extends the signal, the extension feeds the next phase estimate and the
    modes reconstruct differently. That alternation decreases the spectral
    convergence

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
    and applies only to the final synthesis: the iteration itself runs at the
    default length of {!invert}, which re-analyses to the frame count of [s].
    Where that length is [0] there is no signal to re-analyse and no iteration
    runs, whatever [n_iter] asks for: the result is then [init] synthesised
    once at [length]. Two spectrograms reach it — one with no frames at all,
    under every geometry, and a single frame under [`Centered] with an even
    [fft_size], the geometry {!invert} documents.

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

(** {1 Incremental synthesis}

    The Mealy kernel behind {!invert}, and the dual of {!Kernel}: [step]
    consumes a batch of spectral frames — the frame axis last, the bin axis
    before it, leading axes broadcasting — and emits the signal samples those
    frames completed; [flush] releases the trailing positions the last frame
    reaches. Concatenating every emitted chunk along the time axis equals
    {!invert} at its default length on the concatenated frames, for every
    partitioning of the frame sequence, bit for bit.

    Synthesis has no lookahead in frame coordinates: padded position [q] takes
    its last tap from frame [q / hop], so it is settled the moment that frame
    arrives, and every frame fed completes one hop of output. Writing [F] for
    the frames fed, [N] for [fft_size], [H] for [hop] and [L], [R] for the
    head and tail trims of the alignment (see {!Config.synthesis_latency}):

    {[
      positions released after F frames = F * H - max 0 (H + R - N)
      samples emitted after F frames    = F * H - synthesis_latency
      samples emitted over a whole stream, flush included
                                        = (F - 1) * H + N - L - R
    ]}

    the last of which is exactly the default length of {!invert}, so the two
    agree on every sample and on how many there are. That is the whole of the
    relationship: feeding all the frames and flushing {e is} [invert] at its
    default length. There is no [length] here and there cannot be — naming one
    retroactively drops frames that opened past it, a decision about frames
    already consumed and already emitted — so a caller who wants
    {!invert}'s [~length] trims the returned signal, which {!invert} describes
    frame by frame.

    Reconstruction itself, its conditioning at the edges and the interior it
    is exact on are {!invert}'s: this kernel computes the same quotient at the
    same positions, and inherits them unchanged. What it adds is the memory
    bound — the state is the last [ceil (fft_size / hop) - 1] windowed frames
    and the held tail, so at most [(fft_size + hop) * ceil (fft_size / hop)]
    doubles per channel, whatever the stream's length and however its frames
    were chunked.

    The streaming form is the weighted overlap-add of R. E. Crochiere,
    {e A weighted overlap-add method of short-time Fourier analysis/synthesis},
    IEEE TASSP 28(1), 1980, over the analysis-synthesis duality of J. B. Allen
    & L. R. Rabiner, {e A unified approach to short-time Fourier analysis and
    synthesis}, Proc. IEEE 65(11), 1977. *)

module Synthesis : sig
  (** The type for prepared synthesis states. Mutable; single-owner; not
      domain-safe. One state carries all channels — never one object per
      channel. *)
  type ('a, 'c) t

  val prepare :
       (float, 'a) Nx.dtype
    -> Config.t
    -> (Complex.t, 'c) Nx.dtype
    -> channels:int
    -> max_block:int
    -> ('a, 'c) t
  (** [prepare dtype c cdtype ~channels ~max_block] is a fresh synthesis
      state for batches of at most [max_block] frames of [channels]-channel
      [cdtype] spectra, producing [dtype] audio. The witnesses read in the
      order of the data, as {!Kernel.prepare}'s do: what comes out, the
      geometry, what goes in.

      Raises [Invalid_argument] if [channels < 1], if [max_block < 1], or if
      [c] is not invertible — the condition {!invert} states, checked once
      here rather than on every batch. *)

  val step : ('a, 'c) t -> (Complex.t, 'c) Nx.t -> (float, 'a) Nx.t option
  (** [step k z] feeds the frames [z], shaped [[...; bins; frames]], and is
      the samples they completed, if any, shaped [[...; samples]] — [hop] per
      frame once the head trim is paid. [z] is borrowed: the kernel copies
      what it must retain, and the returned tensor aliases neither [z] nor
      kernel state. Values are computed in double precision and rounded once
      into [dtype], as {!invert} rounds them.

      Raises [Invalid_argument] if [k] was drained by {!flush} — {!reset} it
      before feeding new frames — if [z] has rank below two or a bin axis
      that is not [bins c] long, or if a leading axis of [z] has size zero. *)

  val flush : ('a, 'c) t -> (float, 'a) Nx.t option
  (** [flush k] drains the kernel: it releases the positions the last frame
      reaches past the hop grid, less the tail trim of the alignment, and is
      those samples, if any — [max 0 (fft_size - hop - R)] of them, none at
      all where the trim already covers them. Draining consumes the tail — a
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

val synthesis_stage :
     (float, 'a) Nx.dtype
  -> Config.t
  -> ((Complex.t, 'c) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [synthesis_stage dtype c] is {!Synthesis} as a pipeline stage: causal,
    rate [hop] — one output sample per hop of every frame that arrives — and
    zero input-side latency, declaring instead {!Config.synthesis_latency}
    samples of {e output} latency, which is precisely what it withholds. Its
    lookahead is therefore [synthesis_latency c / hop] frames, not an integer
    in general, and composing it under {!stage} on the same configuration
    reports, in source samples,

    {[
      latency (stage cdtype c >> synthesis_stage dtype c)
      = latency (stage cdtype c) + Config.synthesis_latency c
    ]}

    — each end declared on the side it acts on, and the composition adding
    them where they meet, by the rule {!Pipeline.kernel} states. The chain is
    the identity on the interior of the round trip, the region {!invert}
    reconstructs: padded positions [\[fft_size - hop, frames * hop)], which in
    signal coordinates is [\[max 0 (fft_size - hop - L), frames * hop - L)].

    Against the truth measured by first emission, the declaration is exact
    under an odd [fft_size] with [`Centered] frames and under [`Right] with a
    padding that does not look ahead, one sample high under an even
    [fft_size], where {!Config.latency} rounds the half frame up, and a bound
    under [`Right] with [`Reflect] padding, whose border lookahead is the same
    [fft_size - 1] samples the head trim already covers and which the sum
    therefore counts twice. [`Left] under-reports, and does so at the analysis
    end: it declares {!Config.latency}[ = 0], the lookahead of its frame grid,
    though its first frame still needs [fft_size] samples to exist — the round
    trip inherits that declaration exactly as the [`Left] analysis stage
    carries it alone.

    Chunks are joined by concatenation along the time axis; joining zero
    chunks yields an empty signal. The threaded stream format carries [dtype]
    as its element dtype.

    Raises [Invalid_argument] if [c] is not invertible, under the condition
    {!invert} states. *)
