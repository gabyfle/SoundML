(** Digital signal processing on Raven tensors.

    {1:conventions Conventions}

    The time axis is always the last axis. Audio is [[channels; frames]]; a
    rank-one tensor is mono audio. Spectral data is
    [[channels; bins; frames]]. Leading axes broadcast: every function
    accepts [[...; channels; frames]] and maps over the leading axes, so a
    batch of clips is one call.

    Functions return fresh tensors unless documented {e view}. Nothing copies
    defensively; tensors are immutable, so nothing mutates its input.

    Sample rates are explicit [~sample_rate] arguments on the functions whose
    result depends on one; rate-agnostic functions do not take it.

    Numerical defaults follow librosa 1.0; every deviation is a named
    option. Preconditions raise [Invalid_argument]; no [result] values in
    numeric paths.

    Incremental processing: every stateful algorithm is a Mealy kernel
    ([prepare]/[step]/[flush]/[reset]) composable as a {!Pipeline} stage.
    Offline functions are the one-chunk instance of the same kernels; the two
    cannot disagree. The per-frame reductions of the features (the mel and
    chroma projections, the spectral, energy, contrast and onset sums) add
    their terms in a fixed order rather than in the shape-dependent order of
    Nx, so a frame's value is the same bits alone or among others, at every
    dtype; {!Cqt} states its own agreement. See {!Pipeline}.

    Parallelism: the library's own kernels split their work across a pool of
    worker domains, which never changes a bit of a result. The workers park
    between kernels and all exit once a whole major collection cycle passes
    without one. A parked domain takes part in every stop-the-world
    collection of the program, so a program that shares the process with
    its own latency-sensitive domains pays for the pool until it retires.

    Audio-file I/O ships separately as [soundml-io]. *)

(** Streaming and offline chunk pipelines: one pipeline value drives both the
    offline driver {!Pipeline.run} and the online driver {!Pipeline.Stream}. *)
module Pipeline = Pipeline

(** Window-function specifications and their instantiation. *)
module Window = Window

(** The short-time Fourier transform: offline {!Stft.transform}, the
    incremental {!Stft.Kernel} and the {!Stft.stage}/{!Stft.power_stage}
    pipeline stages, all driving one frame grid; and its synthesis side,
    {!Stft.invert} for the least-squares inverse, {!Stft.griffin_lim} for
    magnitude-only reconstruction, and the incremental {!Stft.Synthesis} with
    its {!Stft.synthesis_stage}, which totals to {!Stft.invert} over a
    stream. *)
module Stft = Stft

(** Unit and scale conversions: decibels, the mel and MIDI frequency scales,
    and the frame grid. *)
module Convert = Convert

(** Decibel scaling as {!Pipeline} stages, the reference capability-indexed:
    [Db.stage (Value r)] streams, [Db.stage Maximum] is offline by the
    argument's own type. *)
module Db = Db

(** Mel filterbanks: the weight matrix precomputed by {!Mel.Config}, the
    batched {!Mel.apply} projection over power spectra, and the memoryless
    {!Mel.stage} consuming {!Stft.power_stage} output. *)
module Mel = Mel

(** Constant-Q and variable-Q transforms: the filter geometry precomputed by
    {!Cqt.Config}, the offline {!Cqt.transform} over the recursive octave
    plan, the dtype-preserving {!Cqt.power_spectrum}, and the incremental
    {!Cqt.Kernel} streaming the same plan at the declared
    {!Cqt.Config.latency}. *)
module Cqt = Cqt

(** Chroma (pitch-class) projections: the Gaussian-bump matrix precomputed by
    {!Chroma.Config} for linear-frequency spectra, the exact 0/1
    {!Chroma.cqt_projection} for constant-Q bins, the per-frame
    normalisation both share, and the memoryless {!Chroma.stage} and
    {!Chroma.of_cqt_stage} streaming each projection. *)
module Chroma = Chroma

(** Sample-rate conversion: one exact-rational polyphase resampler behind the
    offline {!Resample.apply}, the incremental {!Resample.Kernel} and the
    {!Resample.stage} pipeline stage — bit-identical on every partitioning. *)
module Resample = Resample

(** Time-scale and pitch modification: the phase vocoder behind
    {!Effects.time_stretch}, the spectrum-level {!Effects.phase_vocoder} it is
    built on, its identity phase-locked variant, and {!Effects.pitch_shift},
    which composes the stretch with {!Resample} at the exact rational ratio
    {!Effects.semitones} names — each also a pipeline stage
    ({!Effects.phase_vocoder_stage}, {!Effects.time_stretch_stage},
    {!Effects.pitch_shift_stage}) streaming the same result bit for bit. Also
    silence detection: {!Effects.trim} and {!Effects.split}, and
    {!Effects.split_stage}, which finds the non-silent intervals of a stream
    against a fixed reference as they close. *)
module Effects = Effects

(** {1:features Flat features} *)

val mel_spectrogram :
     Stft.Config.t
  -> Mel.Config.t
  -> ?power:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mel_spectrogram stft mel x] is the mel spectrogram of [x], shaped
    [[...; n_mels; frames]]: {!Stft.power_spectrum} on the analysis geometry
    [stft], projected through the filterbank [mel] by {!Mel.apply} — one
    framing pass, one batched FFT, one magnitude power, one projection.
    The time axis is the last axis of [x]; leading axes broadcast, so a batch
    of clips is one call. [power] defaults to [2.] (the power spectrum; [1.]
    projects magnitudes).

    Raises [Invalid_argument] if the two configurations disagree on
    [fft_size], naming both sizes — the filterbank must consume exactly the
    bins the transform produces — or if [x] has rank zero. *)

val mfcc :
     Stft.Config.t
  -> Mel.Config.t
  -> ?n_mfcc:int
  -> ?lifter:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mfcc stft mel x] is the mel-frequency cepstrum of [x], shaped
    [[...; n_mfcc; frames]]: the orthonormal type-II DCT along the mel axis
    of the log-mel spectrogram, keeping the first [n_mfcc] coefficients.
    [n_mfcc] defaults to [20]. The log-mel spectrogram is
    {!Convert.power_to_db} of [mel_spectrogram stft mel x] with reference
    [1.], floor [1e-10], and the 80 dB dynamic-range clamp under each
    channel's maximum over mel bands and frames — the global maximum of a
    single clip.

    The DCT is orthonormal: the raw type-II transform scaled by
    [1 / sqrt (4 * n_mels)] on coefficient zero and [1 / sqrt (2 * n_mels)]
    elsewhere. [lifter], when positive, applies sinusoidal liftering —
    coefficient [k], counted from zero, is multiplied by
    [1 + (lifter / 2) * sin (pi * (k + 1) / lifter)] — and [0.]
    or absent applies none. The DCT, its scaling and the liftering are one
    matrix, each coefficient folded from its own frame's mel column in a
    fixed order. The log-mel scaling, the DCT and the liftering
    are computed in double precision and rounded to the dtype of [x] once,
    at the boundary; the mel spectrogram beneath carries its own boundary
    roundings, one per operation, exactly as {!mel_spectrogram}.

    Raises [Invalid_argument] if the configurations disagree on [fft_size],
    if [n_mfcc] does not lie in [\[1, n_mels\]], if [lifter] is not finite
    and non-negative, or if [x] has rank zero. *)

val mfcc_stage :
     Mel.Config.t
  -> ?n_mfcc:int
  -> ?lifter:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mfcc_stage mel ()] is {!mfcc}'s cepstrum as a memoryless pipeline
    stage: rate 1:1 in frames, latency zero and format-preserving. It consumes
    mel power-spectrum chunks — compose
    [Stft.power_stage s >> Mel.stage mel >> mfcc_stage mel ()] for the
    streaming form of [mfcc s mel] — mapping [[...; n_mels; frames]] to
    [[...; n_mfcc; frames]]; an empty chunk maps to an empty chunk.
    [n_mfcc] and [lifter] are those of {!mfcc}, validated when the stage is
    built, and every partitioning of a mel spectrogram yields the same bits.

    The stage computes {!mfcc}'s interior exactly — the log-mel with
    reference [1.] and floor [1e-10], the orthonormal DCT, the liftering, all
    in double precision and rounded once — with one documented deviation: it
    omits the 80 dB dynamic-range clamp, which sits under the whole signal's
    maximum and is a reduction no per-frame stage can compute. The chain is
    therefore bit-identical to {!mfcc} wherever the clamp does not bind.

    Raises [Invalid_argument] if [n_mfcc] does not lie in
    [\[1, Mel.Config.n_mels mel\]] or if [lifter] is not finite and
    non-negative. The stream {!Pipeline.Format} does not carry a band count,
    so a chunk whose mel axis is not [n_mels] long is reported at that chunk,
    not at prepare time. *)

val chroma_stft :
     Stft.Config.t
  -> Chroma.Config.t
  -> ?power:float
  -> ?norm:Chroma.norm
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [chroma_stft stft chroma x] is the chromagram of [x], shaped
    [[...; n_chroma; frames]]: {!Stft.power_spectrum} on the analysis geometry
    [stft], projected onto pitch classes by {!Chroma.apply}. The time axis is
    the last axis of [x]; leading axes broadcast, so a batch of clips is one
    call. [power] defaults to [2.] (the power spectrum; [1.] projects
    magnitudes) and [norm] to [`Inf], so each frame's strongest pitch class
    reads one.

    Raises [Invalid_argument] if the two configurations disagree on
    [fft_size], naming both sizes — the filterbank must consume exactly the
    bins the transform produces — if [x] has rank zero, or if [norm] is
    [`P p] with [p] not finite and positive. *)

val chroma_cqt :
     Cqt.Config.t
  -> ?n_chroma:int
  -> ?norm:Chroma.norm
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [chroma_cqt cqt x] is the constant-Q chromagram of [x], shaped
    [[...; n_chroma; frames]]: {!Cqt.power_spectrum}[ ~power:1.] on the filter
    geometry [cqt], folded onto pitch classes by {!Chroma.of_cqt}. [n_chroma]
    defaults to [12] and [norm] to [`Inf]. Because constant-Q bins already sit
    on the equal-tempered ladder, the fold is an exact 0/1 assignment — no
    interpolation and no frequency-domain filterbank.

    Raises [Invalid_argument] if [n_chroma] does not divide
    [Cqt.Config.bins_per_octave cqt], if [x] has rank zero, if [norm] is
    [`P p] with [p] not finite and positive, or for the dtype reason
    documented on {!Cqt.transform}. *)

val resample :
     ?quality:Resample.quality
  -> sample_rate:int
  -> target:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [resample ~sample_rate ~target x] is
    [Resample.(apply (Config.create ~sample_rate ~target ()) x)]. The config
    is kept for later calls at the same rates and quality, with the few
    others built most recently (up to 8 MB of designed filters), so a
    conversion repeated over a corpus is designed once. *)

(** {2:features_spectral Spectral-shape features}

    Each function summarises every frame of an already-computed {e magnitude}
    spectrogram into one value; each has a stateless {!Pipeline} stage. *)

val spectral_centroid :
     ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [spectral_centroid ~sample_rate s] is the spectral centroid of the
    magnitude spectrogram [s] — {!Stft.power_spectrum}[ ~power:1.] output, or
    any non-negative [[...; bins; frames]] tensor — shaped [[...; 1; frames]]
    (the bin axis kept, reduced to one): each frame, normalised to unit
    magnitude sum,
    is read as a distribution over the bin frequencies and reduced to its
    mean [Σₖ freq k * S k t / Σⱼ S j t], in hertz. Leading axes broadcast, so
    a batch of spectrograms is one call; values are computed in double
    precision and rounded to the dtype of [s] once, at the boundary.

    The bin frequencies default to the FFT grid implied by the bin count —
    bin [k] of a [[...; bins; frames]] spectrogram sits at
    [k * sample_rate / (2 * (bins - 1))], evaluated as one reciprocal and one
    multiply per bin — and [freqs], a rank-one tensor of [bins] frequencies,
    replaces that grid for non-uniformly spaced bins (the per-frame frequency
    {e matrix} of a reassigned spectrogram is not supported). When
    [freqs] is given, [sample_rate] is validated but not otherwise consulted.

    Frames whose magnitudes sum below the smallest positive normal double are
    left unnormalised (the underflow guard), so an all-zero frame has
    centroid [0]; the guard is evaluated in double precision, so on float32
    input it sits far below float32's own tiny threshold — a deviation from
    the reference implementation run natively on float32 data, visible only
    on subnormal-magnitude frames. A spectrogram with no frames, or a zero-size axis, maps to zeros
    of the result shape.

    Raises [Invalid_argument] if [s] has rank below two or holds a negative
    or NaN value (a magnitude spectrogram is non-negative), if
    [sample_rate < 1], if [freqs] is absent and [s] has fewer than two bins
    (a one-bin spectrogram implies no FFT grid), or if [freqs] is present and
    is not rank-one with one frequency per bin. *)

val spectral_centroid_stage :
     ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [spectral_centroid_stage ~sample_rate ()] is {!spectral_centroid} as a
    memoryless pipeline stage: rate 1:1 in frames, latency zero and
    format-preserving, consuming {!Stft.power_stage}[ ~power:1.] output
    ([[...; bins; frames]] magnitude chunks) and emitting [[...; 1; frames]]
    chunks. An empty chunk maps to an empty chunk. Parameters are validated
    when the stage is built, raising [Invalid_argument] exactly as
    {!spectral_centroid} does; only the shape-dependent checks — the pairing
    of [freqs] with the bin count, and the non-negativity of the data — wait
    for the first chunk, since the stream {!Pipeline.Format} carries no bin
    count. *)

val spectral_bandwidth :
     ?p:float
  -> ?freqs:(float, 'a) Nx.t
  -> ?centroid:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [spectral_bandwidth ~sample_rate s] is the [p]-th-order spectral
    bandwidth of the magnitude spectrogram [s], shaped [[...; 1; frames]]:
    the deviation of the bin frequencies around each frame's centroid,
    [(Σₖ S~ k t * |freq k - centroid t| ^ p) ^ (1/p)] in hertz, where [S~]
    is the frame-normalised spectrogram. [p] defaults to [2.], the frequency
    standard deviation. Frames are always normalised to unit magnitude sum,
    with the underflow guard of {!spectral_centroid}; an unnormalised variant
    is not exposed.

    [centroid] supplies precomputed per-frame centroid frequencies, shaped
    [[...; 1; frames]] and broadcastable against [s], sparing the second pass
    when {!spectral_centroid} was already computed on the same grid; it is
    consumed as given, so a single-precision centroid carries its rounding
    into the deviations. When absent the centroid is computed internally, in
    double precision. [freqs] and [sample_rate] behave exactly as in
    {!spectral_centroid}.

    Raises [Invalid_argument] as {!spectral_centroid} does, if [p] is not
    finite and positive, or if [centroid] is present without one row per
    frame of [s] ([[...; 1; frames]]). *)

val spectral_bandwidth_stage :
     ?p:float
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [spectral_bandwidth_stage ~sample_rate ()] is {!spectral_bandwidth} as a
    memoryless pipeline stage, exactly as {!spectral_centroid_stage}. The
    centroid is always computed internally — a precomputed one cannot follow
    the chunk stream. *)

val spectral_rolloff :
     ?roll_percent:float
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [spectral_rolloff ~sample_rate s] is the roll-off frequency of each frame
    of the magnitude spectrogram [s], shaped [[...; 1; frames]]: the
    frequency of the lowest bin at which the cumulative magnitude over the
    bins reaches [roll_percent] of the frame's total, in hertz.
    [roll_percent] defaults to [0.85]; values near [1.] approximate each
    frame's maximum frequency and values near [0.] its minimum. An all-zero
    frame rolls off at the first bin's frequency.

    [freqs] and [sample_rate] behave exactly as in {!spectral_centroid};
    [freqs] must be sorted in increasing order (assumed, not checked).

    Raises [Invalid_argument] as {!spectral_centroid} does, or if
    [roll_percent] does not lie strictly between [0] and [1]. *)

val spectral_rolloff_stage :
     ?roll_percent:float
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [spectral_rolloff_stage ~sample_rate ()] is {!spectral_rolloff} as a
    memoryless pipeline stage, exactly as {!spectral_centroid_stage}. *)

val spectral_flatness :
  ?amin:float -> ?power:float -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [spectral_flatness s] is the spectral flatness of the magnitude
    spectrogram [s], shaped [[...; 1; frames]]: the geometric mean of
    [max amin (S k t ^ power)] over the bins divided by its arithmetic mean —
    [1.] for a perfectly flat (noise-like) frame, near [0.] for a tonal one.
    [power] defaults to [2.]: the magnitudes are squared to the power
    spectrum before averaging; pass [~power:1.] for a spectrogram that is
    already a power spectrum. [amin] defaults to [1e-10]
    and floors the powered magnitudes, keeping the logarithms finite.
    Flatness is rate-agnostic — no frequency grid is involved.

    Raises [Invalid_argument] if [s] has rank below two or holds a negative
    or NaN value, or if [amin] or [power] is not finite and positive (the
    domain is enforced here, not merely documented). *)

val spectral_flatness_stage :
     ?amin:float
  -> ?power:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [spectral_flatness_stage ()] is {!spectral_flatness} as a memoryless
    pipeline stage, exactly as {!spectral_centroid_stage}. *)

(** {2:features_energy Energy features}

    Frame-rate energies over framed raw audio; each flat function is the
    one-chunk instance of the streaming carry behind its {!Pipeline} stage. *)

val rms : ?frame_length:int -> ?hop:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [rms x] is the root-mean-square energy of each frame of the audio [x],
    shaped [[...; 1; frames]]: the square root of the mean squared sample
    over every [frame_length]-sample window advancing by [hop]. The time
    axis is the last axis of [x]; leading axes broadcast, so a batch of
    clips is one call.

    [frame_length] defaults to [2048] and [hop] to [512]. The analysis is
    centered: frame [p] is centered at sample [p * hop], the signal extended
    by [frame_length / 2] zeros on each side (constant zero padding; other
    pad modes and uncentered analysis are not exposed). The mean and root are
    computed in double precision and rounded to the dtype of [x] once, at the
    boundary — the result dtype follows the input, where the reference
    implementation returns float32 regardless. An empty signal produces no
    frames, [[...; 1; 0]] — a deviation: the reference implementation instead
    pads an all-zero frame for even frame lengths and rejects empty input
    for odd ones.

    Raises [Invalid_argument] if [frame_length] or [hop] is smaller than
    [1], or if [x] has rank zero. *)

val rms_of_spectrogram :
  ?frame_length:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [rms_of_spectrogram s] is the frame energy recovered from the magnitude
    spectrogram [s], mapping [[...; bins; frames]] to [[...; 1; frames]] —
    the spectrogram-input companion of {!rms}. By Parseval's identity for the
    one-sided spectrum layout, the frame power is twice the sum of the squared
    magnitudes over the bins — the DC bin halved, and the Nyquist bin
    halved for even frame lengths — divided by [frame_length] squared.
    Windowed spectra weigh the estimate by their window; the exact companion
    of {!rms} is a rectangular-window, uncentered spectrogram.
    [frame_length] defaults to [2048]. The sum is computed in double
    precision and rounded to the dtype of [s] once, at the boundary.

    Raises [Invalid_argument] if [frame_length] is smaller than [1], if [s]
    has rank below two, or if its second-to-last axis does not hold
    [frame_length / 2 + 1] bins. *)

val rms_stage :
     ?frame_length:int
  -> ?hop:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [rms_stage ()] is {!rms} as a pipeline stage: causal, with
    [frame_length / 2] samples of latency and rate [1/hop] — audio chunks
    in, energy chunks [[...; 1; frames]] out, dtype-preserving. Offline
    {!rms} is the one-chunk instance of the same carry; the two cannot
    disagree.

    Raises [Invalid_argument] if [frame_length] or [hop] is smaller than
    [1]. *)

val zero_crossing_rate :
     ?frame_length:int
  -> ?hop:int
  -> ?threshold:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [zero_crossing_rate x] is the zero-crossing rate of each frame of the
    audio [x], shaped [[...; 1; frames]]: the number of sign changes
    between consecutive samples within every [frame_length]-sample window
    advancing by [hop], divided by [frame_length]. The time axis is the
    last axis of [x]; leading axes broadcast, so a batch of clips is one
    call.

    [frame_length] defaults to [2048], [hop] to [512] and [threshold] to
    [1e-10]. The analysis is centered: frame [p] is centered at sample
    [p * hop], the signal extended by [frame_length / 2] edge copies on each
    side (edge copies are the only padding; uncentered analysis is not
    exposed). Signs follow a clamped convention: samples in
    [[-threshold, threshold]] count as positive zero, so a sample is
    negative iff it lies strictly below [-threshold], and a crossing is a
    change of that sign between a sample and its predecessor — the first
    position of a frame carries no crossing. The count is exact in any
    dtype; the division is computed in double precision and rounded to the
    dtype of [x] once, at the boundary — the result dtype follows the input,
    where the reference implementation returns float64 regardless. An empty
    signal produces no frames, [[...; 1; 0]] — a deviation: the reference
    implementation rejects empty input.

    Raises [Invalid_argument] if [frame_length] or [hop] is smaller than
    [1], if [threshold] is not finite and non-negative, or if [x] has rank
    zero. *)

val zero_crossing_rate_stage :
     ?frame_length:int
  -> ?hop:int
  -> ?threshold:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [zero_crossing_rate_stage ()] is {!zero_crossing_rate} as a pipeline
    stage: causal, with [frame_length / 2] samples of latency and rate
    [1/hop] — audio chunks in, rate chunks [[...; 1; frames]] out,
    dtype-preserving. Offline {!zero_crossing_rate} is the one-chunk
    instance of the same carry; the two cannot disagree.

    Raises [Invalid_argument] if [frame_length] or [hop] is smaller than
    [1], or if [threshold] is not finite and non-negative. *)

(** {2:features_delta Delta features}

    The local derivative of a feature sequence along its frame axis —
    typically of {!mfcc} or a log-mel spectrogram, whose deltas and
    delta-deltas speech front ends stack beside the static features. *)

(** The border of {!delta}: how the frames within [width / 2] of either end
    are differentiated.
    - [`Interp] (librosa's default) fits the first and last full windows and
      evaluates those fits at the frames they cover.
    - [`Nearest] extends the sequence by copies of its first and last frames,
      the edge rule of Kaldi's and python_speech_features' deltas. *)
type delta_mode = [`Interp | `Nearest]

val delta :
     ?width:int
  -> ?order:int
  -> ?mode:delta_mode
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [delta x] is the [order]-th derivative of [x] along its last axis, in the
    shape and dtype of [x], as [librosa.feature.delta] computes it: the
    Savitzky-Golay filter that fits a degree-[order] polynomial by least
    squares to each window of [width] frames and differentiates it [order]
    times at the window's centre. [width] defaults to [9], [order] to [1]
    ([2] gives delta-deltas) and [mode] to [`Interp]. Leading axes
    broadcast, so a batch of feature matrices is one call.

    The derivative of a degree-[order] fit is constant over its window, so
    each output frame is one fixed weighted sum of [width] input frames —
    weight [k / (sum of j^2)] at frame offset [k] when [order = 1] —
    accumulated in double precision in window order and rounded to the dtype
    of [x] once. Under [`Interp], the first and last [width / 2] outputs
    therefore repeat the outputs of the first and last full windows.

    Raises [Invalid_argument] if [width] is not an odd integer of at least
    [3], if [order] does not lie in [\[1, width - 1\]], if [x] has rank zero
    or a dtype other than float32 and float64, or if [mode] is [`Interp] and
    [x] holds fewer than [width] frames — librosa's own restriction. *)

val delta_stage :
     ?width:int
  -> ?order:int
  -> ?mode:delta_mode
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [delta_stage ()] is {!delta} as a pipeline stage: rate 1:1 in frames and
    format-preserving, consuming and emitting [[...; features; frames]]
    chunks — compose
    [Stft.power_stage s >> Mel.stage mel >> mfcc_stage mel () >> delta_stage ()]
    for streamed MFCC deltas. Every partitioning of a sequence yields the
    bits {!delta} computes on the whole of it.

    An output frame is emitted once every frame its window reads has
    arrived, so the stage withholds [width / 2] frames, and under [`Interp]
    it also holds its first outputs until the first full window has arrived:
    it declares a latency of [width - 1] frames under [`Interp] and
    [width / 2] under [`Nearest]. The flush emits the last [width / 2]
    outputs; under [`Interp] it raises [Invalid_argument] if the stream held
    fewer than [width] frames, as {!delta} raises on such a sequence.

    Raises [Invalid_argument] if [width] or [order] is invalid as for
    {!delta}, and, at the chunk concerned, if a chunk has rank zero, a dtype
    other than float32 and float64, or a leading shape other than the first
    chunk's. *)

(** {2:features_pcen Per-channel energy normalization}

    PCEN: the adaptive gain control that keyword spotters and far-field
    speech front ends apply to a mel spectrogram in place of the log. *)

val pcen :
     ?hop:int
  -> ?gain:float
  -> ?bias:float
  -> ?power:float
  -> ?time_constant:float
  -> ?eps:float
  -> ?b:float
  -> ?max_size:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [pcen ~sample_rate s] is the per-channel energy normalization of the
    non-negative spectrogram [s] — [[...; bands; frames]], or one band
    [[frames]] — in the shape and dtype of [s], as [librosa.pcen] computes
    it. Each band [E] is smoothed over time and divided by a power of its
    smoothed self, then compressed:

    {[
      M t = b * E t + (1 - b) * M (t - 1)
      pcen t = (E t / (eps + M t) ** gain + bias) ** power - bias ** power
    ]}

    with [M] started as if every frame before the first had magnitude [1]
    (scipy's [lfilter_zi]). With [max_size > 1], [M] smooths instead the
    maximum of [E] over the [max_size] bands around each band, reflected at
    the edges of the band axis. [power = 0.] compresses by
    [log1p (E t / (eps + M t) ** gain)] instead, and [bias = 0.] gives
    [(E t / (eps + M t) ** gain) ** power], as librosa does.

    The smoothing coefficient [b] defaults to the one whose squared response
    falls to half at [time_constant] seconds of frames advancing [hop]
    samples at [sample_rate]: [(sqrt (1 + 4 t ** 2) - 1) / (2 t ** 2)] with
    [t = time_constant * sample_rate / hop]. [hop] defaults to [512], [gain]
    to [0.98], [bias] to [2.], [power] to [0.5], [time_constant] to [0.4],
    [eps] to [1e-6] and [max_size] to [1]. The defaults are tuned for
    magnitudes at the scale of 32-bit integer audio: librosa's examples
    scale a magnitude mel spectrogram of float audio by [2 ** 31] first.

    Every value is computed in double precision — the smoothing exactly as
    scipy's [lfilter] runs it — and rounded to the dtype of [s] once.

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], if
    [gain], [bias] or [power] is negative or not finite, if [eps] or
    [time_constant] is not finite and positive, if [b] does not lie in
    [(0, 1\]], if [max_size] is below [1], if [s] has rank zero, or rank one
    with [max_size > 1], or a dtype other than float32 and float64. *)

val pcen_stage :
     ?hop:int
  -> ?gain:float
  -> ?bias:float
  -> ?power:float
  -> ?time_constant:float
  -> ?eps:float
  -> ?b:float
  -> ?max_size:int
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [pcen_stage ~sample_rate ()] is {!pcen} as a pipeline stage: causal,
    rate 1:1 in frames, latency zero and format-preserving, carrying one
    smoothing state per band across chunks. It consumes [[...; bands;
    frames]] chunks — compose
    [Stft.power_stage ~power:1. s >> Mel.stage mel >> pcen_stage ~sample_rate ()]
    for a streamed PCEN front end — and every partitioning of a spectrogram
    yields the bits {!pcen} computes on the whole of it.

    Raises [Invalid_argument] for the parameters as {!pcen} does, and, at
    the chunk concerned, for a chunk {!pcen} would refuse or one whose
    leading shape differs from the first chunk's. *)

(** {2:features_contrast Spectral contrast and onset strength}

    Octave-band spectral contrast and the spectral-flux onset envelope, each
    with its {!Pipeline} stage. *)

val spectral_contrast :
     Stft.Config.t
  -> ?n_bands:int
  -> ?f_min:float
  -> ?quantile:float
  -> ?linear:bool
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [spectral_contrast c ~sample_rate x] is the spectral contrast of [x],
    shaped [[...; n_bands + 1; frames]]: per frame of the magnitude spectrum
    on the analysis geometry [c], the peak-to-valley difference of each of
    [n_bands + 1] octave-spaced frequency bands (Jiang et al., 2002). The
    time axis is the last axis of [x]; leading axes broadcast, so a batch of
    clips is one call. This face consumes raw audio;
    {!spectral_contrast_of_spectrogram} consumes an already-computed
    magnitude spectrogram. A custom frequency grid is not exposed: the band
    plan always uses the FFT bin grid of [c].

    Band [b] collects the FFT bins whose center frequency lies in
    [[edge b, edge (b + 1)]] inclusive, where [edge 0] is [0] Hz and
    [edge j] is [f_min * 2^(j-1)] beyond — band [0] spans [[0, f_min]] and
    each later band one octave. Every band above the first is extended one
    bin below its lower edge, the top band runs to the top of the spectrum,
    and every band below the top yields its highest bin back to its
    successor. Per band and frame, the valley is the mean of the [take]
    smallest magnitudes and the peak the mean of the [take] largest, where
    [take] is [quantile] of the extended band's bin count, rounded half to
    even and never below one bin.

    [n_bands] defaults to [6], [f_min] to [200.] Hz and [quantile] to
    [0.02]. [linear] defaults to [false]: the contrast is
    [power_to_db peak - power_to_db valley] with reference [1.], floor
    [1e-10] and the 80 dB dynamic-range clamp under each channel's maximum
    over bands and frames — the global maximum of a single clip — a
    whole-signal reduction.
    [~linear:true] is the plain difference [peak - valley] instead. The
    interior runs in double precision and rounds to the dtype of [x] once,
    at the boundary.

    Raises [Invalid_argument] if [n_bands < 1], if [f_min] is not finite and
    positive, if [quantile] does not lie strictly between [0] and [1], if
    [sample_rate < 1], if the top band would start at or above the Nyquist
    frequency, if some band spans no FFT bin — rejected outright, where the
    reference implementation degrades to NaN means; raise [fft_size] or lower
    [n_bands] instead — or if [x] has rank zero. *)

val spectral_contrast_of_spectrogram :
     ?n_bands:int
  -> ?f_min:float
  -> ?quantile:float
  -> ?linear:bool
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [spectral_contrast_of_spectrogram ~sample_rate s] is {!spectral_contrast}
    over an already-computed magnitude spectrogram [s] —
    {!Stft.power_spectrum}[ ~power:1.] output, or any non-negative
    [[...; bins; frames]] tensor — mapping [[...; bins; frames]] to
    [[...; n_bands + 1; frames]]: the spectrogram-input mode, as
    {!rms_of_spectrogram} is to {!rms}. The band plan is built on the FFT bin
    grid the bin count implies ([fft_size = 2 * (bins - 1)]); the parameters,
    the contrast and the clamped logarithmic path are exactly those of
    {!spectral_contrast}, so applying this function to
    [Stft.power_spectrum ~power:1. c x] is [spectral_contrast c ~sample_rate x].

    Raises [Invalid_argument] as {!spectral_contrast} does on invalid
    parameters, if [s] has rank below two or holds a negative or NaN value (a
    magnitude spectrogram is non-negative), or if [s] has fewer than two bins
    (a one-bin spectrogram implies no FFT grid). *)

val spectral_contrast_stage :
     Stft.Config.t
  -> ?n_bands:int
  -> ?f_min:float
  -> ?quantile:float
  -> ?linear:bool
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [spectral_contrast_stage c ~sample_rate ()] is {!spectral_contrast} as a
    memoryless pipeline stage: rate 1:1 in frames, latency zero and
    format-preserving. It consumes {e magnitude}-spectrum chunks — compose it
    downstream of [Stft.power_stage ~power:1.] — mapping [[...; bins; frames]]
    to [[...; n_bands + 1; frames]]; an empty chunk maps to an empty chunk.
    The parameters and the band plan are those of {!spectral_contrast},
    validated when the stage is built, with one documented deviation: on the
    logarithmic path the stage omits the 80 dB clamp, which is a whole-signal
    reduction no per-frame stage can compute — the unclamped difference is
    identical wherever the flat function's clamp does not bind.

    Raises [Invalid_argument] as {!spectral_contrast} does on invalid
    parameters. The stream {!Pipeline.Format} does not carry a bin count, so
    a bins mismatch is reported at the first chunk, not at prepare time. *)

val onset_strength :
     Stft.Config.t
  -> Mel.Config.t
  -> ?lag:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [onset_strength stft mel x] is the spectral-flux onset strength envelope
    of [x], shaped [[...; frames]] — the log-power mel flux chain: the mel
    axis mean of [max 0 (D[f, t] - D[f, t - lag])] over the log-power mel
    spectrogram [D], which is [mel_spectrogram stft mel x] through
    [Convert.power_to_db] with reference [1.], floor [1e-10] and the 80 dB
    dynamic-range clamp under each channel's maximum over mel bands and
    frames — the global maximum of a single clip — a whole-signal
    reduction. The time axis is the last axis of [x]; leading
    axes broadcast, so a batch of clips is one call.

    [lag] is the frame lag of the difference and defaults to [1]. The first
    [lag] envelope values are zero (the lag compensation), and centered
    analysis shifts the envelope right by a further [fft_size / (2 * hop)]
    zeros, trimmed back to the frame grid, so onsets land on the frames whose
    grid position they precede — the shift is derived from the [`Centered]
    alignment of [stft] rather than passed separately, and [`Left] and
    [`Right] alignments apply none. The log-mel scaling and the flux run in
    double precision and round to the dtype of [x] once, at the boundary; the
    mel spectrogram beneath carries its own boundary roundings, exactly as
    [mel_spectrogram].

    Documented deviations from the reference implementation's onset
    envelope, each absent here
    rather than renamed: no [max_size] local maximum filtering along the
    frequency axis and no precomputed [ref] spectrum (the reference is the
    spectrogram itself), no [detrend] DC-removal filter, no [aggregate]
    other than the mean, and no [feature] other than the log-power mel
    chain above.

    Raises [Invalid_argument] if [lag < 1], if the configurations disagree on
    [fft_size], or if [x] has rank zero. *)

val onset_strength_stage :
  ?lag:int -> unit -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [onset_strength_stage ()] is the lagged positive spectral flux as a causal
    pipeline stage: rate 1:1 in frames, latency zero, and one envelope value
    per incoming frame, [[...; bins; frames]] to [[...; frames]]. It consumes
    {e log}-spectral chunks — compose
    [Stft.power_stage c >> Mel.stage m >> Db.stage (Db.Value 1.)] upstream
    for the streaming form of {!onset_strength}'s chain — and carries [lag]
    frames of state across chunks, so every partitioning of a spectrogram
    yields the same envelope. [lag] defaults to [1]; the first [lag] values
    of a stream are zero.

    The stage emits the envelope aligned to its input frames: the centered
    compensation shift of {!onset_strength} is frame bookkeeping on the
    offline grid, not latency, and is left to the caller — and the global
    80 dB clamp inside the flat function's log-mel belongs to the offline
    [Db.clamped_stage], so a causal chain built with [Db.stage (Db.Value 1.)]
    computes the unclamped envelope, identical wherever the clamp does not
    bind.

    Raises [Invalid_argument] if [lag < 1]. *)

(** {2:features_hpss Harmonic/percussive separation}

    Median-filtering separation of sustained partials from transients, in the
    signal, complex-spectrum, spectrogram and mask domains, each spectral face
    with its {!Pipeline} stage. *)

val hpss :
     Stft.Config.t
  -> ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [hpss c x] is the harmonic and percussive parts of the audio [x], each
    shaped like [x]: the analysis of [x] on the geometry [c], the separation
    of {!hpss_of_stft}, and one {!Stft.invert} per component back to the
    length of [x]. The time axis is the last axis of [x]; leading axes
    broadcast, so a batch of clips is one call. This face consumes and
    returns raw audio; {!hpss_of_spectrogram} and {!hpss_of_stft} consume an
    already-computed spectrogram, and {!hpss_masks} returns the masks
    themselves. {!harmonic} and {!percussive} are this function keeping one
    component.

    Separation is median filtering (Fitzgerald, {e Harmonic/percussive
    separation using median filtering}, DAFx 2010) with the soft masks of
    Driedger, Müller and Disch ({e Extending harmonic-percussive separation
    of audio signals}, ISMIR 2014). Sustained partials are horizontal ridges
    of the magnitude spectrogram and transients vertical ones, so a median
    along the frame axis enhances the first and a median along the bin axis
    the second; the two enhanced spectrograms [harm] and [perc] then decide
    the split bin by bin, as {!hpss_masks} defines it.

    [kernel_size] is [(k_h, k_p)] and defaults to [(31, 31)]: [k_h] frames
    for the harmonic median along time, [k_p] bins for the percussive median
    along frequency. [power] is the mask exponent and defaults to [2.];
    [Float.infinity] selects the hard mask. [margin] is [(m_h, m_p)] and
    defaults to [(1., 1.)], the partition. Beyond [(1., 1.)] the two
    components no longer sum to the input, and the difference
    [x - harmonic - percussive] is the residual — the third component of the
    margin split, computed by subtraction rather than returned.

    Raises [Invalid_argument] if either kernel size is below [1], if [power]
    is [nan] or not strictly positive, if either margin is not finite and at
    least [1], if [x] has rank zero, or if the dtype of [x] is neither
    [float32] nor [float64]. *)

val harmonic :
     Stft.Config.t
  -> ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [harmonic c x] is the harmonic part of the audio [x]: [fst (hpss c x)],
    at the same parameters and the same cost — the percussive component is
    computed and dropped. Raises [Invalid_argument] as {!hpss} does. *)

val percussive :
     Stft.Config.t
  -> ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [percussive c x] is the percussive part of the audio [x]:
    [snd (hpss c x)], at the same parameters and the same cost — the harmonic
    component is computed and dropped. Raises [Invalid_argument] as {!hpss}
    does. *)

val hpss_of_spectrogram :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [hpss_of_spectrogram s] is the harmonic and percussive components of an
    already-computed spectrogram [s] — {!Stft.power_spectrum} output, or any
    non-negative [\[...; bins; frames\]] tensor — each shaped and typed like
    [s]: the masks of {!hpss_masks} multiplied into [s]. Phase is not
    involved, so this face separates magnitude and power spectrograms alike;
    it is to {!hpss} what {!spectral_contrast_of_spectrogram} is to
    {!spectral_contrast}.

    At the default margin the components partition [s] exactly:
    [h + p = s] to rounding. Beyond it, [s - (h + p)] is the residual.

    The parameters are those of {!hpss}, and [Invalid_argument] is raised on
    the same conditions, with rank below two — a spectrogram carries a bin
    axis and a frame axis — in place of rank zero. *)

val hpss_of_stft :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (Complex.t, 'c) Nx.t
  -> (Complex.t, 'c) Nx.t * (Complex.t, 'c) Nx.t
(** [hpss_of_stft z] is the harmonic and percussive components of the complex
    spectrum [z] — {!Stft.transform} output, shaped
    [\[...; bins; frames\]] — each shaped and typed like [z]. The masks are
    those of {!hpss_masks} on the magnitude of [z], and each component
    carries the phase of [z] unchanged, so the two spectra invert to signals
    the way [z] itself does. Component width follows the dtype of [z]:
    [complex64] masks in [float32], [complex128] in [float64].

    The unit phase is formed component by component, and a bin of magnitude
    zero takes phase [1 + 0i] rather than a quotient of zeros.

    The parameters are those of {!hpss}, and [Invalid_argument] is raised on
    the same conditions, with rank below two in place of rank zero. *)

val hpss_masks :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [hpss_masks s] is the harmonic and percussive mask pair of the
    spectrogram [s], each shaped and typed like [s]: the weights
    {!hpss_of_spectrogram} multiplies into [s], exposed for callers that
    apply them elsewhere.

    Let [harm] be [s] median-filtered along the frame axis over [k_h] frames
    and [perc] be [s] median-filtered along the bin axis over [k_p] bins. The
    window of index [i] on a line of [n] values is
    [\[i - k / 2, i + k - 1 - k / 2\]] — left-biased for even [k] — and the
    filtered value is rank [k / 2] of that window sorted ascending, the upper
    middle for even [k] and never the average of two. Indices outside the
    line reflect half-sample-symmetrically with period [2 n]: [-1] reads
    [0], [n] reads [n - 1], and the pattern repeats, so a kernel may exceed
    the line length by any amount. The rule holds for every kernel size and
    every line length, boundaries included.

    At finite [power] the harmonic mask is
    [harm^p / (harm^p + (m_h * perc)^p)] and the percussive mask is
    [perc^p / (perc^p + (m_p * harm)^p)], each computed on a pair rescaled by
    its pointwise maximum — which leaves the quotient unchanged and keeps
    both powers representable at any [p]. Where that maximum falls below the
    smallest positive normal of the dtype the quotient is undefined and the
    mask takes [0.5] at the default margin, [0.] otherwise.

    At [power = Float.infinity] the masks are the strict comparisons
    [harm > m_h * perc] and [perc > m_p * harm], carried as the floats [0.]
    and [1.] rather than booleans. Both are zero where the two sides are
    equal, so [h + p = s] does {e not} hold there: such a bin is dropped by
    both components.

    Masks lie in [\[0, 1\]]. At the default margin the finite-[power] pair
    sums to [1.] everywhere and the infinite-[power] pair sums to [1.] away
    from equality; beyond it the pair sums to at most [1.], the deficit being
    the residual's share.

    The parameters are those of {!hpss}, and [Invalid_argument] is raised on
    the same conditions, with rank below two in place of rank zero. *)

val hpss_of_spectrogram_stage :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t * (float, 'a) Nx.t, 'k) Pipeline.t
(** [hpss_of_spectrogram_stage ()] is {!hpss_of_spectrogram} as a pipeline
    stage: rate 1:1 in frames and format-preserving, consuming
    [\[...; bins; frames\]] chunks of a magnitude or power spectrogram —
    compose it downstream of [Stft.power_stage c] — and emitting the
    harmonic and percussive components of the same frames as a pair. Pairs
    are joined component by component along the frame axis, so
    [Pipeline.map fst] and [Pipeline.map snd] keep one component; joining
    zero chunks yields the pair of empty [\[...; bins; 0\]] spectrograms.
    Every partitioning of a spectrogram yields the bits
    {!hpss_of_spectrogram} computes on the whole of it.

    Only the harmonic median, along the frame axis, has memory. The window
    of frame [i] reaches [k_h - 1 - k_h / 2] frames past it — [15] at the
    default [k_h = 31], none at [k_h <= 2] — so the stage declares that many
    frames of latency and emits frame [i] with the push that delivers frame
    [i + k_h - 1 - k_h / 2]; a push that completes no frame emits no chunk.
    The windows of the last [k_h - 1 - k_h / 2] frames of a stream (all of
    a shorter one) cross its end and depend on its length through the
    reflection, so the flush emits them, in one chunk. The percussive median
    and the masks act within a frame and add no latency. Behind
    [Stft.power_stage c] the stage adds [k_h - 1 - k_h / 2] hops to the
    latency in source samples; lowering [k_h] is the one way to shorten it
    that keeps the separation librosa's.

    A push costs, per row, one window update for each frame it brings and,
    for each frame it completes, the percussive median and the masks; the
    state is two arrays of [k_h] floats per row, made at the first chunk.

    Raises [Invalid_argument] for the parameters as {!hpss_of_spectrogram}
    does, when the stage is built, and, at the chunk concerned, for a chunk
    of rank below two or of a dtype other than float32 and float64, one
    whose leading shape differs from the first chunk's, or one pushed after
    the flush — reset the stream before reusing it. *)

val hpss_masks_stage :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t * (float, 'a) Nx.t, 'k) Pipeline.t
(** [hpss_masks_stage ()] is {!hpss_masks} as a pipeline stage, emitting the
    mask pair of each frame exactly as {!hpss_of_spectrogram_stage} emits the
    components: the same latency, flush, joining and errors. Every
    partitioning of a spectrogram yields the bits {!hpss_masks} computes on
    the whole of it. *)

val hpss_of_stft_stage :
     ?kernel_size:int * int
  -> ?power:float
  -> ?margin:float * float
  -> unit
  -> ( (Complex.t, 'c) Nx.t
     , (Complex.t, 'c) Nx.t * (Complex.t, 'c) Nx.t
     , 'k )
     Pipeline.t
(** [hpss_of_stft_stage ()] is {!hpss_of_stft} as a pipeline stage, emitting
    the complex components of each frame, phase preserved, exactly as
    {!hpss_of_spectrogram_stage} emits the spectrogram's: the same latency,
    flush, joining and errors, over complex [\[...; bins; frames\]] chunks —
    {!Stft.stage} output. The threaded stream format passes through
    unchanged, as {!Stft.stage} leaves it. Every partitioning of a spectrum
    yields the bits {!hpss_of_stft} computes on the whole of it.

    Between an analysis and a synthesis stage it separates a signal as it
    streams:

    {[
      Pipeline.(
        Stft.stage cdtype c
        >> map fst (hpss_of_stft_stage ())
        >> Stft.synthesis_stage dtype c )
    ]}

    streams the harmonic part, [map snd] the percussive one — each computing
    both components and dropping one, as {!harmonic} does — and
    [Pipeline.fanout] over [stateless fst >> Stft.synthesis_stage dtype c]
    and [stateless snd >> Stft.synthesis_stage dtype c] both, from one
    analysis and one separation. The stream totals to {!Stft.invert} at its
    default length of the components {!hpss_of_stft} finds in
    {!Stft.transform}, bit for bit, and therefore to {!harmonic} and
    {!percussive} on every sample both carry: all of [x] when centred frames
    of an even [fft_size] tile it exactly, its length a multiple of [hop];
    otherwise the stream ends at the synthesis's default length, as every
    synthesis stream does. The chain declares
    [latency (Stft.stage cdtype c) + (k_h - 1 - k_h / 2) * hop
     + Config.synthesis_latency c] source samples of latency: 9728 at
    [fft_size = 2048], [hop = 512], centred, and the default kernel —
    441 ms at 22050 Hz.

    The real and imaginary parts travel with the magnitude in rings of their
    own, so the state is four arrays of [k_h] floats per row. *)

(** {2:synthesis Synthetic signals}

    Test signals: pure tones, sine sweeps and click tracks. The generators
    take the dtype of their result first, as {!Window.make} does, and compute
    every sample in double precision in librosa's order of operations. A tone
    or chirp sample is a function of its own index alone, never of an
    accumulated phase, rounded into the dtype once. A generator consumes no
    stream, so each is an offline function with no {!Pipeline} stage. *)

val tone :
     (float, 'a) Nx.dtype
  -> ?length:int
  -> ?duration:float
  -> ?phase:float
  -> sample_rate:int
  -> float
  -> (float, 'a) Nx.t
(** [tone dtype ~sample_rate ~length f] is the pure tone at [f] Hz sampled at
    [sample_rate], a rank-one tensor of [length] samples, as [librosa.tone]
    computes it: sample [k] is [cos (2 pi f k / sample_rate + phase)], its
    argument rounded in that order. [phase] is in radians and defaults to
    [-pi / 2], which makes the cosine a sine starting at zero.

    The length is [length] when given — it wins over [duration], as in
    librosa — and otherwise [duration] seconds of samples truncated toward
    zero, [Float.to_int (duration *. float sample_rate)]: [0.7] seconds at
    44.1 kHz are [30869] samples, [0.7 *. 44100.] rounding to
    [30869.999999999996]. A length of zero gives an empty tensor.

    Since every sample is computed from its index, [tone ~length:m] is bit
    for bit the first [m] samples of [tone ~length:n] for [m <= n], at every
    dtype. The error of a long tone is therefore not an accumulation but the
    rounding of each sample's argument, about [2 ** -52] times the argument:
    against the exact sinusoid, [5e-12] after ten seconds of a 440 Hz tone at
    22.05 kHz, [2e-9] after an hour, [3e-8] after an hour at 7040 Hz. librosa
    forms the same argument with the same operations, and the C library
    reduces it exactly before taking the cosine, so the two share that error
    rather than adding to it: bit for bit on macOS arm64, where numpy calls
    the same cosine, and within an ulp per sample under another C library, at
    any length.

    Raises [Invalid_argument] if [sample_rate] is below [1], if neither
    [length] nor [duration] is given, if [length] is negative, if [duration]
    is not finite and non-negative or makes [2 ** 62] samples or more, or if [f]
    or [phase] is not finite — librosa returns an empty tone for a negative
    length or duration. *)

val chirp :
     (float, 'a) Nx.dtype
  -> ?length:int
  -> ?duration:float
  -> ?linear:bool
  -> ?phase:float
  -> sample_rate:int
  -> fmin:float
  -> fmax:float
  -> unit
  -> (float, 'a) Nx.t
(** [chirp dtype ~sample_rate ~fmin ~fmax ~length ()] is the sine sweep from
    [fmin] Hz at the first sample towards [fmax] Hz, a rank-one tensor of
    [length] samples, as [librosa.chirp] computes it through
    [scipy.signal.chirp]: sample [k] is [cos (p t + phase)] at
    [t = k / sample_rate], where [p] integrates [2 pi] times the
    instantaneous frequency over a sweep of [T] seconds,

    {[
      linear       p t = 2 pi (fmin t + (fmax - fmin) t^2 / (2 T))
      exponential  p t = 2 pi fmin T ((fmax / fmin) ** (t / T) - 1) / ln (fmax / fmin)
    ]}

    and [2 pi fmin t] for an exponential sweep with [fmin = fmax]. [linear]
    defaults to [false], the exponential (geometric) sweep, which spends the
    same time on every octave; a descending sweep takes [fmin > fmax], and a
    linear one may start or end at [0] Hz. [phase] is in radians and defaults
    to [-pi / 2], a sine. librosa hands the phase to scipy in degrees,
    converted there and back, which moves some phases by an ulp ([0.7]
    becomes [0.7000000000000001]); the conversion is reproduced. The trailing
    [unit] lets every other argument be optional.

    With [length], [T] is [(1 / sample_rate) * length] — the sweep would
    reach [fmax] one sample past the end — and [length] wins over [duration].
    With [duration] alone, [T] is [duration] and the length is [duration]
    seconds of samples truncated toward zero, as for {!tone}. [T] depends on
    the length, so a shorter chirp is not a prefix of a longer one.

    One named deviation: the result holds exactly [length] samples. librosa
    recounts them as [int ((1 / sr) * length * sr)], which falls below
    [length] for about one length in a hundred at 22.05 and 44.1 kHz and one
    in six at 48 kHz and returns a sample fewer — [357] samples at 22.05 kHz
    give [356], [7] at 48 kHz give [6]. Every sample librosa does return is
    reproduced, and the last one here is scipy's sweep, on the same [T], at
    [(length - 1) / sample_rate].

    Parity is bit for bit on macOS arm64. Under another C library the linear
    sweep agrees within an ulp per sample, while the exponential one
    multiplies an ulp of its power by the phase swept: it agrees within about
    [2 ** -52] times the total phase in radians, [4e-12] over a second from
    20 Hz to 20 kHz.

    Raises [Invalid_argument] if [sample_rate] is below [1], if neither
    [length] nor [duration] is given, if [length] is negative, if [duration]
    is not finite and non-negative or makes [2 ** 62] samples or more, if
    [fmin], [fmax] or [phase] is not finite, or, for an exponential sweep, if
    [fmin] and [fmax] are not nonzero and of the same sign (scipy's rule,
    [fmin *. fmax > 0.]). *)

(** Where {!clicks} sounds its clicks, in the order given, duplicates
    included:
    - [`Times t] at the times [t], in seconds, each at sample
      [Float.to_int (t *. float sample_rate)], truncated toward zero as
      librosa converts them;
    - [`Frames f] at the frames [f] of a grid advancing [hop] samples, each
      at sample [f * hop];
    - [`Samples s] at the sample indices [s], such as the interval bounds
      {!Effects.split} returns: [`Frames] at a hop of one, which librosa
      spells that way.

    [`Times] is the one place the library takes seconds where it otherwise
    deals in frame and sample indices: [librosa.clicks] is most often handed
    beat times, and the truncation above is the conversion it performs. *)
type click_positions =
  [`Frames of int array | `Samples of int array | `Times of float array]

val clicks :
     (float, 'a) Nx.dtype
  -> ?hop:int
  -> ?click:(float, 'a) Nx.t
  -> ?click_frequency:float
  -> ?click_duration:float
  -> ?length:int
  -> sample_rate:int
  -> click_positions
  -> (float, 'a) Nx.t
(** [clicks dtype ~sample_rate positions] is the click track sounding a click
    at each of the [positions], as [librosa.clicks] computes it: silence,
    with the click added at every position in the order given. [hop]
    defaults to [512] and only [`Frames] reads it.

    The click is [click] when given, its time axis last, and otherwise
    librosa's default:
    [m = Float.to_int (float sample_rate *. click_duration)] samples, sample
    [k] being
    [2 ** (-10 k / (m - 1)) * sin (2 pi click_frequency k / sample_rate)], a
    sine at [click_frequency] Hz under an envelope decaying exponentially
    from [1] to [2 ** -10]. [click_frequency] defaults to [1000.] and
    [click_duration] to [0.1], 2205 samples at 22.05 kHz.

    The track holds [length] samples when [length] is given: a click running
    past the end is cut there, and the positions at or past it are dropped.
    Otherwise it ends where the latest click ends, at the largest position
    plus the click's length. The result is shaped like the click with its
    last axis that length: [[length]] for the default click,
    [[channels; length]] for a [[channels; m]] one.

    The track starts at zero and each click is added in turn, every sum
    rounded to [dtype] as it is formed, so overlapping clicks round in the
    order of [positions]. At float32 this is librosa's track, which is
    always float32, bit for bit wherever the C libraries agree on the default
    click; at float64 the sum stays in double (a track librosa does not
    offer), and the narrower dtypes sum in double and round once. The cost
    is the track's length plus the click's length per position; the samples
    split across domains, each range adding its clicks in order, which
    changes no bit.

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], if
    [click_frequency] or [click_duration] is not finite and positive —
    checked even with [click], which librosa does not — or the click would
    hold [2 ** 62] samples or more, if [length] is below [1], if there is
    neither a position nor a [length], if a time is not finite, if a
    position falls before the first sample (a time at or before
    [-1 / sample_rate], a negative frame or sample index), or if [click] has
    rank zero or a sample that is not finite. *)

(** {2:companding Mu-law companding}

    The mu-law maps a sample [x] in [[-1, 1]] to
    [sign x * ln (1 + mu |x|) / ln (1 + mu)], spending its resolution on
    quiet samples, and can quantize the result uniformly into
    [int (1 + mu)] levels: the continuous law, quantized as WaveNet codes
    its 8-bit sample classes, not the segmented code of ITU-T G.711. Both
    directions are elementwise and memoryless — any rank, in the shape and
    dtype of the input, every element computed in double precision and
    rounded to that dtype once — and each has a stateless {!Pipeline}
    stage. *)

val mu_compress :
  ?mu:float -> ?quantize:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [mu_compress x] is the mu-law compression of the samples [x], each in
    [[-1, 1]], as [librosa.mu_compress] computes it. [mu] defaults to [255.]
    and [quantize] to [true].

    Without quantization an element is
    [sign x * log1p (mu |x|) / log1p mu]: [1] and [-1] map to themselves
    exactly, and both zeros to [0.]. With it, that value [c] becomes the
    code [count - int (mu + 1) / 2], [count] being the number of points of
    numpy's [linspace (-1, 1, int (1 + mu))] strictly below [c] — librosa's
    [digitize (c, linspace (...), right=True)], computed without the table.
    The codes run from [-(int (mu + 1) / 2)] at [-1] to
    [int (1 + mu) - 1 - int (mu + 1) / 2] at [1]: [-128] to [127] at
    [mu = 255.], [-8] to [7] at [mu = 15.]. They are whole numbers in the
    dtype of [x], exact while it holds them (float32 for [mu] below
    [2 ** 25]), where librosa returns int64.

    Parity: bit for bit at float64 on macOS arm64, where numpy and the C
    library compute the same logarithms, and within an ulp or two of each
    value under another C library; codes agree exactly wherever the
    compressed value is not within such an error of a level boundary. A
    float32 input is widened exactly — a named deviation from librosa, whose
    float32 path takes the logarithm in float32 and divides it by a float64
    one, compressing [1] to [1.0000000027] and coding it [128], a level past
    the last.

    Raises [Invalid_argument] if [mu] is not finite and positive, if
    [quantize] is [true] and [mu] does not lie in [[1, 2 ** 53)] — librosa
    quantizes a smaller [mu] onto a one-point grid whose codes its own
    expander rejects — or if an element of [x] does not lie in [[-1, 1]],
    nan included (librosa lets nan through). *)

val mu_compress_stage :
     ?mu:float
  -> ?quantize:bool
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mu_compress_stage ()] is {!mu_compress} as a memoryless pipeline stage:
    rate 1:1, latency zero, format-preserving. Every partitioning of a
    signal yields the bits {!mu_compress} computes on the whole of it.

    Raises [Invalid_argument] for [mu] and [quantize] as {!mu_compress} does
    when the stage is built, and at the chunk concerned for an element
    outside [[-1, 1]]. *)

val mu_expand :
  ?mu:float -> ?quantize:bool -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [mu_expand y] is the mu-law expansion of [y], the inverse of the
    continuous {!mu_compress}, as [librosa.mu_expand] computes it: an element
    [v] in [[-1, 1]] maps to [sign v / mu * ((1 + mu) ** |v| - 1)]. [mu]
    defaults to [255.] and [quantize] to [true]: each element is then a code,
    first scaled to [v = y * 2 / (1 + mu)], so the codes in
    [[-(1 + mu) / 2, (1 + mu) / 2]] are accepted, non-integers included.
    The scaling reads code [c] as the compressed value [2 c / (1 + mu)],
    librosa's rule, rather than as the middle of the level {!mu_compress}
    coded: code [127] expands to [0.957] at [mu = 255.], and the quantized
    round trip [mu_expand (mu_compress x)] errs by up to
    [0.043] at [mu = 255.] and [0.31] at [mu = 15.], both at full scale. The
    continuous round trip agrees within [1e-15] at [mu = 255.].

    Parity: bit for bit at float64 on macOS arm64, where numpy and the C
    library compute the same powers, and within an ulp or two under another
    C library. A float32 input is widened exactly and expanded in double — a
    named deviation from librosa, which expands float32 input in float32.

    Raises [Invalid_argument] if [mu] is not finite and positive, if
    [quantize] is [true] and [mu] does not lie in [[1, 2 ** 53)], or if an
    element, scaled under [quantize], does not lie in [[-1, 1]], nan
    included. *)

val mu_expand_stage :
     ?mu:float
  -> ?quantize:bool
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mu_expand_stage ()] is {!mu_expand} as a memoryless pipeline stage:
    rate 1:1, latency zero, format-preserving. Every partitioning of a
    stream yields the bits {!mu_expand} computes on the whole of it.

    Raises [Invalid_argument] for [mu] and [quantize] as {!mu_expand} does
    when the stage is built, and at the chunk concerned for an element
    {!mu_expand} would refuse. *)

val version : string
(** [version] is the version of the SoundML distribution this library was built
    from. *)
