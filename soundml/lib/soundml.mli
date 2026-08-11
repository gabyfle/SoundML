(** Digital signal processing on Raven tensors.

    {1:conventions Conventions}

    The time axis is always the last axis. Audio is [[channels; frames]]; a
    rank-one tensor is mono audio. Spectral data is
    [[channels; bins; frames]]. Leading axes broadcast: every function
    accepts [[...; channels; frames]] and maps over the leading axes, so a
    batch of clips is one call.

    Functions return fresh tensors unless documented {e view}. Nothing copies
    defensively; nothing mutates its input unless the function takes [~out].

    Sample rates are explicit [~sample_rate] arguments on the functions whose
    result depends on one; rate-agnostic functions do not take it.

    Numerical defaults follow librosa 0.11; every deviation is a named
    option. Preconditions raise [Invalid_argument]; no [result] values in
    numeric paths.

    Incremental processing: every stateful algorithm is a Mealy kernel
    ([prepare]/[step]/[flush]/[reset]) composable as a {!Pipeline} stage.
    Offline functions are the one-chunk instance of the same kernels; the two
    cannot disagree. See {!Pipeline}.

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
    {!Chroma.cqt_projection} for constant-Q bins, and the per-frame
    normalisation both share. *)
module Chroma = Chroma

(** Sample-rate conversion: one exact-rational polyphase resampler behind the
    offline {!Resample.apply}, the incremental {!Resample.Kernel} and the
    {!Resample.stage} pipeline stage — bit-identical on every partitioning. *)
module Resample = Resample

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
    framing pass, one batched FFT, one magnitude power, one matrix product.
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
    [1.], floor [1e-10], and the 80 dB dynamic-range clamp under the global
    maximum.

    The DCT is orthonormal: the raw type-II transform scaled by
    [1 / sqrt (4 * n_mels)] on coefficient zero and [1 / sqrt (2 * n_mels)]
    elsewhere. [lifter], when positive, applies sinusoidal liftering —
    coefficient [k], counted from zero, is multiplied by
    [1 + (lifter / 2) * sin (pi * (k + 1) / lifter)] — and [0.]
    or absent applies none. The log-mel scaling, the DCT and the liftering
    are computed in double precision and rounded to the dtype of [x] once,
    at the boundary; the mel spectrogram beneath carries its own boundary
    roundings, one per operation, exactly as {!mel_spectrogram}.

    Raises [Invalid_argument] if the configurations disagree on [fft_size],
    if [n_mfcc] does not lie in [\[1, n_mels\]], if [lifter] is not finite
    and non-negative, or if [x] has rank zero. *)

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
    [Resample.(apply (Config.create ~sample_rate ~target ()) x)]. For repeated
    conversions at one ratio, build the config once. *)

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
    [1e-10] and the 80 dB dynamic-range clamp under each tensor's {e global}
    maximum, a whole-signal reduction.
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
    dynamic-range clamp under the global maximum, a whole-signal
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

val version : string
(** [version] is the version of the SoundML distribution this library was built
    from. *)
