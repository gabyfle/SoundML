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

(** Linear recursive filtering: scipy's transposed direct form
    ({!Filter.lfilter}) and second-order-section cascade ({!Filter.sosfilt}),
    each with initial and final states, steady states
    ({!Filter.lfilter_zi}, {!Filter.sosfilt_zi}) and an offline zero-phase
    form ({!Filter.filtfilt}, {!Filter.sosfiltfilt}); Butterworth design as a
    transfer function or as sections ({!Filter.butter},
    {!Filter.butter_sos}); and librosa's {!Filter.preemphasis} and
    {!Filter.deemphasis}. The filters stream as pipeline stages
    ({!Filter.lfilter_stage}, {!Filter.sosfilt_stage},
    {!Filter.preemphasis_stage}, {!Filter.deemphasis_stage}) that carry
    their float64 state across chunks and reproduce the offline functions
    bit for bit. *)
module Filter = Filter

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

(** {2:features_tonal Tonal features}

    Two descriptors of harmony over the constant-Q chromagram of
    {!chroma_cqt}: chroma energy normalised statistics (CENS), the
    dynamics- and timbre-robust chroma of audio matching, and the tonal
    centroid, the six-dimensional pitch-space embedding of harmonic-change
    detection. Each takes audio through a constant-Q configuration, or an
    already-computed chromagram through its [_of_chroma] form, and each has
    a {!Pipeline} stage over chromagram chunks.

    librosa analyses seven octaves at 36 bins per octave from C1, at hop
    512, by default; the matching configuration is
    [Cqt.Config.create ~bins_per_octave:36 ~n_bins:252 ~pad:(`Constant 0.)
    ~sample_rate ()], whose tuning is explicit and whose filter basis is
    exact where librosa estimates the one and sparsifies the other (see the
    parity sections of {!Chroma} and {!Cqt}).

    Parity with librosa 1.0: given the same chromagram, the features match
    [librosa.feature.chroma_cens] and [librosa.feature.tonnetz] to
    double-precision rounding — the quantisation decisions are exact, and
    the smoothing window is {!Window.make}'s exactly symmetric evaluation, a
    rounding away from scipy's. The named deviations:

    - {b The interior is double precision, rounded once.} librosa run
      natively on float32 rounds its L1-normalised chroma to float32,
      compares it against thresholds rounded to float32 and rounds the
      smoothed sequence before its final normalisation; here every step
      stays in double, which is librosa's float64 path on the same values.
      A quantisation level can differ from librosa's own float32 output
      only where a share lies within one float32 rounding of a threshold.
    - {b The tonal centroid keeps the dtype of its input}, where librosa
      promotes float32 chroma to float64.
    - {b Non-finite chroma raise} [Invalid_argument], as librosa's
      normalisation raises its own error; propagating them instead would
      quantise a NaN frame to silence. CENS also refuses negative chroma,
      which librosa's constant-Q chromagram clips to zero before quantising.
    - {b The quantiser amplifies the chromagram's own departures.} From
      audio, the chromagram carries the constant-Q departures named under
      {!Cqt}'s parity, the octave decimation's resampler above all: on the
      test suite's music and noise, the L1 shares differ from librosa's by up
      to [4.4e-5] at hop 512 and [3.3e-8] at an odd hop, where neither side
      resamples (measured). A share that close to [0.05], [0.1], [0.2] or
      [0.4] can land on the other level, which moves every CENS frame whose
      smoothing window covers it, by up to about [1e-2] at the defaults. The
      tonal centroid has no threshold and follows the chromagram
      proportionally: within [9.3e-5] and [4e-8] of librosa's on the same
      signals.

    librosa's [cq_to_chroma] window, its hybrid constant-Q mode and its
    chroma threshold are not offered, as for {!chroma_cqt}. CENS
    downsampling — the literature's [CENS(w, d)] keeps every [d]-th frame —
    is implemented by neither: slice the result. *)

val chroma_cens :
     Cqt.Config.t
  -> ?n_chroma:int
  -> ?norm:Chroma.norm
  -> ?smoothing:int option
  -> ?smoothing_window:Window.t
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [chroma_cens cqt x] is the CENS chromagram of the audio [x], shaped
    [[...; n_chroma; frames]] (Müller, Kurth & Clausen, {e Audio matching via
    chroma-based statistical features}, ISMIR 2005): {!chroma_cens_of_chroma}
    of the unnormalised constant-Q chromagram
    [chroma_cqt cqt ?n_chroma ~norm:`None x], bit for bit — the chromagram
    librosa quantises. The time axis is the last axis of [x]; leading axes
    broadcast, so a batch of clips is one call. [n_chroma] defaults to
    [12]; [norm], [smoothing] and [smoothing_window] are those of
    {!chroma_cens_of_chroma}.

    The constant-Q transform has no pipeline stage, so this function is
    offline over audio; {!chroma_cens_stage} streams the chromagram's side.

    Raises [Invalid_argument] on the parameters as {!chroma_cens_of_chroma}
    does, as {!chroma_cqt} does on [n_chroma] and [x], or if [x] holds an
    infinite or NaN value. *)

val chroma_cens_of_chroma :
     ?norm:Chroma.norm
  -> ?smoothing:int option
  -> ?smoothing_window:Window.t
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [chroma_cens_of_chroma c] is the CENS chromagram of the non-negative
    chromagram [c] — any [[...; n_chroma; frames]] tensor, raw or normalised
    — in the shape and dtype of [c], as [librosa.feature.chroma_cens]
    computes it from its chromagram:

    + Each frame is divided by its L1 length, giving each pitch class its
      share of the frame's energy. A frame whose length lies below the
      smallest normal of the dtype is left as it is, so a silent frame stays
      silent.
    + Each share [v] is quantised to [0.25] times the number of the
      thresholds [0.4], [0.2], [0.1] and [0.05] it strictly exceeds: five
      levels from [0] to [1], a logarithmic compression of the energy
      distribution.
    + With [smoothing = Some l], the quantised sequence [q] is smoothed along
      the frames by the symmetric [l + 2]-point [smoothing_window]
      ({!Window.make}[ ~periodic:false]) divided by its sum: output frame [t]
      is the sum over the window's points [m] of [w m * q (t + n / 2 - m)],
      [n = l + 2], with [q] zero beyond either end of the sequence and the
      points weighing at most the double-precision epsilon — Hann's zero
      endpoints — left out. An odd [l] centres the window, the default
      reading the [20] frames on either side; an even [l] reads one frame
      more ahead than behind. This is scipy's [ndimage.convolve] with
      [mode="constant"], librosa's own smoothing.
    + Each frame is divided by its length in [norm], with the guard of the
      first step: a frame of zeros — one whose whole window is silent —
      stays zero.

    [norm] defaults to [`P 2.], a unit vector per frame; [smoothing] to
    [Some 41], librosa's [win_len_smooth], and [~smoothing:None] skips the
    third step; [smoothing_window] to {!Window.Hann}. Pass an unnormalised
    chromagram ([Chroma.of_cqt ~norm:`None]) to reproduce librosa's bits:
    normalising it first changes the shares by a rounding, which a
    threshold can amplify.

    Every step runs in double precision — the lengths added in ascending
    order over the chroma axis, each smoothed value from [0.] in ascending
    frame order — and the result is rounded to the dtype of [c] once; any
    float dtype is accepted. The smoothing costs one multiply-add per window
    point, frame and band.

    Raises [Invalid_argument] if [c] has rank below two or holds a negative,
    infinite or NaN value, if [smoothing] is [Some l] with [l < 1], if
    [norm] is [`P p] with [p] not finite and positive, if the shape
    parameter of [smoothing_window] is invalid as documented for
    {!Window.make}, or if its [l + 2] points do not have a positive sum. *)

val chroma_cens_stage :
     ?norm:Chroma.norm
  -> ?smoothing:int option
  -> ?smoothing_window:Window.t
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [chroma_cens_stage ()] is {!chroma_cens_of_chroma} as a pipeline stage:
    rate 1:1 in frames and format-preserving, consuming and emitting
    [[...; n_chroma; frames]] chunks — compose
    [Chroma.of_cqt_stage ~norm:`None cqt >> chroma_cens_stage ()] over
    magnitude constant-Q chunks, or
    [Stft.power_stage s >> Chroma.stage ~norm:`None c >> chroma_cens_stage ()]
    for CENS of the linear-frequency chromagram. Every partitioning of a
    chromagram yields the bits {!chroma_cens_of_chroma} computes on the
    whole of it.

    An output frame is emitted once every frame its window reads has
    arrived, so the stage withholds the window's reach ahead and declares it
    as its latency: [l / 2] frames for a window whose endpoints are zero,
    such as Hann — [20] at the default [l = 41] (0.46 s at 22050 Hz and hop
    512), and [20] at [l = 40] too — and [(l + 2) / 2] for one whose
    endpoints are not, such as Hamming. The flush emits the last frames
    against the zero extension past the end. With [~smoothing:None] the
    stage is memoryless, with latency zero.

    Streaming from audio passes through {!Cqt.Kernel}, whose columns agree
    with {!Cqt.transform} to [1e-13] of the frame peak in float64 rather than
    bit for bit, so a share that close to a threshold can quantise
    differently from {!chroma_cens} on the same audio.

    Raises [Invalid_argument] on the parameters as {!chroma_cens_of_chroma}
    does, and, at the chunk concerned, for a chunk {!chroma_cens_of_chroma}
    would refuse or one whose leading shape — the chroma axis included —
    differs from the first chunk's. *)

val tonnetz :
  Cqt.Config.t -> ?n_chroma:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [tonnetz cqt x] is the tonal centroid of the audio [x], shaped
    [[...; 6; frames]]: {!tonnetz_of_chroma} of the constant-Q chromagram
    [chroma_cqt cqt ?n_chroma x] at its default [`Inf] normalisation, bit for
    bit — librosa's default chromagram. The time axis is the last axis of
    [x]; leading axes broadcast. [n_chroma] defaults to [12].

    Raises [Invalid_argument] as {!chroma_cqt} does, or if [x] holds an
    infinite or NaN value. *)

val tonnetz_of_chroma : (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [tonnetz_of_chroma c] is the tonal centroid of the chromagram [c]
    (Harte, Sandler & Gasser, {e Detecting harmonic change in musical
    audio}, AMCMM 2006), as [librosa.feature.tonnetz] computes it, mapping
    [[...; n_chroma; frames]] to [[...; 6; frames]]. Each frame, divided by
    its L1 length, is projected onto three circles of pitch space — fifths,
    minor thirds and major thirds, of radii [1], [1] and [0.5] — as a sine
    and a cosine coordinate each: rows [2i] and [2i + 1] weigh band [c] by
    [r_i sin (pi a_i d_c)] and [r_i cos (pi a_i d_c)], with
    [a = 7/6, 3/2, 2/3] and [d_c = 12 c / n_chroma] the band's position in
    semitones above the chromagram's first band — so a band count other
    than 12 is spread over the octave, as librosa spreads it. The sines are
    evaluated as librosa evaluates them, as cosines a quarter turn back.

    The L1 length and the projection add their terms in ascending order over
    the chroma axis, in double precision, and the result is rounded to the
    dtype of [c] once — a frame's centroid is the same bits alone or among
    others. A frame whose L1 length lies below the smallest normal of the
    dtype is left unnormalised, so a silent frame maps to zero; signed input
    is accepted, its length counting magnitudes. The projection costs six
    multiply-adds per band and frame.

    Raises [Invalid_argument] if [c] has rank below two or holds an infinite
    or NaN value. *)

val tonnetz_stage : unit -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [tonnetz_stage ()] is {!tonnetz_of_chroma} as a memoryless pipeline
    stage: rate 1:1 in frames, latency zero and format-preserving, mapping
    [[...; n_chroma; frames]] chunks to [[...; 6; frames]] — compose
    [Chroma.of_cqt_stage cqt >> tonnetz_stage ()] over magnitude constant-Q
    chunks for the streaming form of {!tonnetz}'s chain. Each frame is
    projected alone, so every partitioning yields {!tonnetz_of_chroma}'s
    bits, and an empty chunk maps to [[...; 6; 0]]. The twelve-band
    projection is prepared once, when the stage is built; another band
    count is prepared at each chunk.

    Raises [Invalid_argument], at the chunk concerned, for a chunk
    {!tonnetz_of_chroma} would refuse. *)

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

(** {2:features_pitch Fundamental frequency}

    YIN and probabilistic YIN: the fundamental frequency of raw audio, frame
    by frame from each frame's cumulative mean normalized difference — YIN's
    estimate with its {!Pipeline} stage, and pYIN's voiced pitch track,
    decoded over the whole signal. *)

val yin :
     ?frame_length:int
  -> ?hop:int
  -> ?trough_threshold:float
  -> ?alignment:[`Centered | `Left]
  -> ?pad:[`Constant of float | `Edge | `Reflect]
  -> fmin:float
  -> fmax:float
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [yin ~fmin ~fmax ~sample_rate x] is the fundamental frequency, in hertz,
    of each frame of the audio [x], shaped [[...; frames]] — the time axis
    replaced by the frame axis, as [librosa.yin] returns it — by the YIN
    estimator (de Cheveigné and Kawahara, {e YIN, a fundamental frequency
    estimator for speech and music}, JASA 2002). Leading axes broadcast, so
    a batch of clips is one call. {!Convert.frames_to_time} gives each
    frame's time: its centre's when frames are centered, its first sample's
    otherwise.

    Each frame of [frame_length] samples [y 0 .. y (W - 1)] yields its
    difference function over the candidate periods, from
    [floor (sample_rate / fmax)] to [ceil (sample_rate / fmin)] samples,

    {[
      d t = 2 (r 0 - r t) - (y 0 ^ 2 + ... + y (t - 1) ^ 2)
      d' t = d t / ((d 1 + ... + d t) / t)
    ]}

    [r] being the frame's autocorrelation and [d'] the cumulative mean
    normalized difference. The estimate is the shortest period at a trough
    of [d'] below [trough_threshold] — or, when no trough reaches it, the
    global minimum of [d'] — refined by the vertex of the parabola through
    [d'] there and at its two neighbours (none at the ends of the range, or
    where the vertex lies a full period away), and returned as
    [sample_rate / period]. A trough is a value below its left neighbour and
    not above its right one; the two ends compare with their one neighbour.

    [frame_length] defaults to [2048], [hop] to [frame_length / 4],
    [trough_threshold] to [0.1], [alignment] to [`Centered] and [pad] to
    [`Constant 0.], librosa's constant padding; [`Centered] and [`Left]
    are librosa's [center=True] and [center=False], as for
    {!Stft.Config.create}. Centered frame [p] is
    centered at sample [p * hop], the signal extended by [frame_length / 2]
    positions on each side: the constant, copies of the end sample
    ([`Edge]), or the signal mirrored about its end sample ([`Reflect]), as
    many times as a short signal needs. Uncentered frame [p] starts at sample
    [p * hop]. librosa accepts any of numpy's pad modes, with zero as its
    only constant; these three are offered, the constant at any finite
    value. The pitch range must satisfy
    [0 < fmin < fmax <= sample_rate / 2], and a period of [fmin] must fit in
    a frame: [sample_rate / fmin < frame_length - 1]. librosa warns when
    fewer than two periods fit; this function does not.

    Parity: librosa's difference function is reproduced as it is computed,
    including its lag-1 term, which lacks [y 0 ^ 2] — librosa zeroes the
    first energy term before reading it, and every cumulative mean carries
    the error — and its interpolation through [d'], where the paper
    interpolates [d]. The autocorrelation is a zero-padded real FFT of every
    frame at the length scipy picks, the least 5-smooth integer at least
    [2 * frame_length - 1], and the estimates agree with librosa's within a
    few ulps ([4e-15] relative over the test suite's signals). The trough,
    threshold and minimum tests and the parabola's branch are decided on
    these values, so a frame whose decision lies within that much of a tie
    can resolve it the other way.

    The interior runs in double precision and the estimates are rounded to
    the dtype of [x] once — librosa computes float32 input in single
    precision and returns float64. An empty signal, or an uncentered one
    shorter than a frame, produces no frames, [[...; 0]] — a deviation:
    librosa pads an all-zero frame for an empty signal at even frame lengths
    and rejects the other cases. A frame of zeros has no trough; its
    estimate is [sample_rate / floor (sample_rate / fmax)].

    Each frame costs two real FFTs of that 5-smooth length and one pass over
    the candidate periods.

    Raises [Invalid_argument] if [sample_rate], [frame_length] or [hop] is
    below [1], if [fmin] or [fmax] is not finite or the pitch range breaks
    the conditions above (naming an [fmin] and a [frame_length] that would
    satisfy them), if [trough_threshold] or a [`Constant] padding value is
    not finite, if [x] has rank zero, or if a sample of [x] is not finite
    (librosa's audio check). *)

val yin_stage :
     ?frame_length:int
  -> ?hop:int
  -> ?trough_threshold:float
  -> ?alignment:[`Centered | `Left]
  -> ?pad:[`Constant of float | `Edge | `Reflect]
  -> fmin:float
  -> fmax:float
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [yin_stage ~fmin ~fmax ~sample_rate ()] is {!yin} as a causal pipeline
    stage: audio chunks [[...; samples]] in, estimates [[...; frames]] out,
    rate [1/hop] and dtype-preserving. Each estimate depends on its own frame
    alone, so the stage emits it once the frame's last sample has arrived,
    and every partitioning of a signal yields the bits {!yin} computes on the
    whole of it.

    A centered frame reaches [frame_length / 2] samples past its centre, so
    the stage declares that latency; [`Reflect] holds its first frames until
    [frame_length / 2 + 1] samples have arrived, which adds nothing, and a
    stream shorter than that is framed at the flush, reflected as {!yin}
    reflects it. An uncentered frame reaches [frame_length - 1] samples past
    its start, the latency the stage then declares. The flush emits the
    frames the right border completes. The state is the unconsumed part of
    the last frame and the last [frame_length / 2 + 1] samples.

    Raises [Invalid_argument] for the parameters as {!yin} does, when the
    stage is built; at prepare time if the stream does not flow at
    [sample_rate] items per second; and, at the chunk concerned, for a chunk
    of rank zero, with a zero-size leading axis, of a leading shape other
    than the first chunk's or holding a non-finite sample, or pushed after
    the flush — reset the stream before reusing it. *)

(** The result of {!pyin}, three tensors shaped [[...; frames]]. *)
type 'a pyin =
  { f0: (float, 'a) Nx.t
        (** The decoded fundamental frequency of each frame, in hertz, on
            the pitch grid; the fill value where the frame is unvoiced. *)
  ; voiced: (bool, Nx.bool_elt) Nx.t
        (** Whether the decoded state of each frame is voiced. *)
  ; voiced_probability: (float, 'a) Nx.t
        (** The probability, in [[0, 1]], that each frame is voiced: the sum
            of its candidates' probabilities. *)
  }

val pyin :
     ?frame_length:int
  -> ?hop:int
  -> ?n_thresholds:int
  -> ?beta_parameters:float * float
  -> ?boltzmann_parameter:float
  -> ?resolution:float
  -> ?max_transition_rate:float
  -> ?switch_prob:float
  -> ?no_trough_prob:float
  -> ?fill:[`Value of float | `Decoded]
  -> ?alignment:[`Centered | `Left]
  -> ?pad:[`Constant of float | `Edge | `Reflect]
  -> ?transition_min_prob:float
  -> fmin:float
  -> fmax:float
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> 'a pyin
(** [pyin ~fmin ~fmax ~sample_rate x] is the fundamental frequency of the
    audio [x] with its voicing, by probabilistic YIN (Mauch and Dixon,
    {e pYIN: a fundamental frequency estimator using probabilistic threshold
    distributions}, ICASSP 2014) as [librosa.pyin] computes it: each tensor
    of the result is shaped [[...; frames]], over {!yin}'s frames. Leading
    axes broadcast; each signal is decoded on its own.

    Each frame's cumulative mean normalized difference [d'] is {!yin}'s. In
    place of YIN's single threshold, [n_thresholds] thresholds
    [1 / n_thresholds, 2 / n_thresholds, .., 1] carry the mass of a beta
    distribution of shapes [beta_parameters] between consecutive thresholds.
    Each threshold spreads its mass over the [m] troughs of [d'] below it,
    the trough of the [k]-th shortest period (counted from zero) weighted by
    the truncated Boltzmann law [(1 - e^-l) e^(-l k) / (1 - e^(-l m))], [l]
    the [boltzmann_parameter]; the mass of the thresholds no trough falls below,
    times [no_trough_prob], goes to the global minimum. Every trough left
    with a probability is a candidate at its frequency, refined as {!yin}
    refines it, on a grid of [1 / resolution] bins per semitone (rounded up)
    from [fmin] to [fmax]: candidates are rounded to the nearest bin, ties to
    even, a later (longer-period) candidate replacing an earlier one in its
    bin and a candidate past the top bin dropped, as librosa does. A frame's
    voiced probability is the sum of its candidates', clamped to [[0, 1]].

    A hidden Markov model over the pitch bins, each voiced or unvoiced, then
    decodes the most likely path by Viterbi: voiced states observe their
    bin's candidate probability, unvoiced states share one minus the voiced
    probability equally, the initial distribution is uniform over every
    state, a frame switches voicing with probability [switch_prob], and the
    pitch moves by a triangular law over at most
    [max_transition_rate * 12 * hop / sample_rate] semitones per frame
    (rounded half to even). Transitions below [transition_min_prob] are
    pruned from the search, as librosa prunes them; [0.] searches every
    transition (librosa's [None]). This follows the official pYIN software
    rather than the paper, as librosa does: the Boltzmann prior replaces the
    first trough below each threshold, and the observations are not halved.

    [frame_length] defaults to [2048], [hop] to [frame_length / 4],
    [n_thresholds] to [100], [beta_parameters] to [(2., 18.)],
    [boltzmann_parameter] to [2.], [resolution] to [0.1],
    [max_transition_rate] to [35.92] octaves per second, [switch_prob] and
    [no_trough_prob] to [0.01], [transition_min_prob] to [1e-4], [alignment]
    and [pad] to {!yin}'s, and [fill] to [`Value nan]: the frequency of an
    unvoiced frame is [nan], while [`Decoded] keeps the pitch of the decoded
    unvoiced state (librosa's [fill_na=None]). The voiced flag is exact; the
    frequencies are grid values [fmin * 2 ** (k / (12 * bins per
    semitone))].

    pyin has no stage: the Viterbi path of a frame depends on every later
    frame. The candidates agree with librosa's as {!yin}'s estimates do, and
    the voiced probabilities to about [1e-15]; the Viterbi decisions follow
    from them, with ties broken toward the first state as librosa breaks
    them. Every sum runs in a fixed order — the threshold masses in ascending
    order, the voiced probability in ascending bin order, the transition
    rows and the no-trough mass in numpy's pairwise order — so a frame's
    values do not depend on the frames around it; librosa sums the voiced
    probability of a one-frame signal pairwise instead, which can differ in
    the last place. The interior runs in double precision, and the
    frequencies and probabilities are rounded to the dtype of [x] once. An
    empty signal, or an uncentered one shorter than a frame, produces no
    frames. librosa rejects a candidate probability that rounds above [1];
    this function decodes it.

    Decoding costs, per frame and state, one step per predecessor: about
    twice the transition band's width, or every state of the grid when
    [transition_min_prob] is [0.], which makes the exact search quadratic in
    the grid. Back-pointers take two bytes per state and frame while the
    states fit in sixteen bits (librosa's own storage, which overflows
    beyond them), four bytes beyond; the transitions are kept as their band,
    never as the dense matrix librosa builds.

    Raises [Invalid_argument] for the parameters {!yin} shares, as {!yin}
    does, if [n_thresholds] is below [1], if a beta parameter,
    [boltzmann_parameter] or [resolution] is not finite and positive, if
    [max_transition_rate] or [transition_min_prob] is not finite and
    non-negative, if [switch_prob] or [no_trough_prob] does not lie in
    [[0, 1]], if the grid holds fewer than two bins, if the transition band
    is wider than the grid (librosa's own limit), if [transition_min_prob]
    prunes every transition into some state, if [x] has rank zero, or if a
    sample of [x] is not finite. *)

(** {2:features_tuning Pitch tracking and tuning estimation}

    The spectral peaks of a magnitude spectrogram, tracked frame by frame
    with parabolic interpolation, and the tuning of a recording estimated
    from them: its offset from the equal-tempered ladder through A440, in
    fractions of a bin — the [tuning] that {!Chroma.Config.create} and
    {!Cqt.Config.create} take, which they never estimate themselves. *)

(** The magnitude a bin of {!piptrack} must exceed to be detected.
    - [`Maximum] (librosa's default, [ref=None] or [ref=np.max]) is
      [threshold] times the largest magnitude of the frame, over every bin
      and not the band alone: a reference per frame, unlike the whole-signal
      [`Maximum] of {!Effects.reference}.
    - [`Value v] (a scalar [ref]) is the fixed magnitude [v] itself; the
      [threshold] plays no part, as in librosa, which does not scale a
      scalar [ref].

    librosa's other callables ([ref=np.mean], say) are not offered: a mean's
    bits depend on the order of its additions, which numpy takes from the
    memory layout of the spectrogram. *)
type piptrack_reference = [`Maximum | `Value of float]

val piptrack :
     ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:piptrack_reference
  -> ?fft_size:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t * (float, 'a) Nx.t
(** [piptrack ~sample_rate s] is [(pitches, magnitudes)], the frequencies in
    hertz and the magnitudes of the spectral peaks of the magnitude
    spectrogram [s] — {!Stft.power_spectrum}[ ~power:1.] output, or any
    non-negative [[...; bins; frames]] tensor — as [librosa.piptrack]
    computes them: each shaped and typed like [s], zero everywhere but at the
    peaks. Leading axes broadcast, so a batch of spectrograms is one call.

    Each frame is analysed on its own. Its bins at or below the frame's
    reference (see {!piptrack_reference}) are zeroed, and bin [k] is a peak
    when it is then a local maximum along the bin axis — above bin [k - 1]
    and at least bin [k + 1], so a plateau counts at its lower end — and its
    frequency lies in [\[fmin, fmax)]. Bin [0] is never a peak, and the top
    bin is one when it rises above its lower neighbour. At a peak the
    parabola through the magnitudes [xm], [x0] and [xp] of bins [k - 1], [k]
    and [k + 1], unthresholded, has its vertex [shift = -b / a] bins away,
    with [a = (xp + xm) - 2 * x0] and [b = (xp - xm) / 2] — no shift when
    [|b| >= |a|], nor at the top bin: the pitch is
    [(k + shift) * sample_rate / fft_size] and the magnitude the vertex's
    height, [x0 + (b / 2) * shift] (quadratic interpolation of spectral
    peaks, J. O. Smith III, {e Spectral Audio Signal Processing}).

    [fmin] defaults to [150.] Hz and [fmax] to [4000.] Hz; [fmin] is raised
    to [0] and [fmax] lowered to [sample_rate / 2], and [fmax <= fmin] leaves
    the band empty. The band reads bin [k] at [k] times the reciprocal of
    [fft_size / sample_rate], numpy's [rfftfreq] arithmetic, while the pitch
    is evaluated as written above: both are librosa's arithmetic, so a bin on
    an edge of the band is decided as librosa decides it. The Nyquist bin of
    an even transform, under an [fmax] at or above [sample_rate / 2], is in
    the band exactly when its rounded frequency falls below
    [sample_rate / 2], as it does at some rates (12345 Hz) but not at the
    usual ones with power-of-two transforms. [threshold]
    defaults to [0.1] and [reference] to [`Maximum]. [fft_size] defaults to
    the even size [2 * (bins - 1)] the bin count implies; pass it for the
    spectrogram of an odd-size transform, whose top bin lies below the
    Nyquist frequency and can be a peak. Values are computed in double
    precision and rounded to the dtype of [s] once. A spectrogram with no
    frames, or a zero-size leading axis, maps to a pair of empty tensors.

    The interpolation is biased wherever the peak falls between a bin and a
    half bin, because the transform of the window is not a parabola: an
    unpadded Hann analysis of a steady tone places it up to [0.053] bin from
    its frequency on linear magnitudes, and up to [0.115] bin on a power
    spectrogram (Smith, {e Bias of Parabolic Peak Interpolation}).

    Documented deviations from the reference implementation: a negative or
    NaN magnitude raises, where librosa takes absolute values and lets NaN
    through; an [fft_size] that does not imply the bin count raises, where
    librosa silently replaces it by [2 * (bins - 1)]; [`Value v] needs [v]
    finite and non-negative, where librosa takes [|v|]; a NaN [fmin] or
    [fmax] raises, where librosa detects nothing; a spectrogram without
    frames is accepted, where librosa raises; and librosa run
    natively on float32 data interpolates in mixed precision, so its float32
    results can differ in the last place from the double results rounded
    once here.

    Raises [Invalid_argument] if [s] has rank below two, fewer than two bins,
    or a negative or NaN value, if [sample_rate < 1], if [fmin] or [fmax] is
    NaN, if [threshold] is not finite and non-negative, if [reference] is
    [`Value v] with [v] not finite and non-negative, or if [fft_size] is
    below [2] or [fft_size / 2 + 1] is not the bin count of [s]. *)

val piptrack_stage :
     ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:piptrack_reference
  -> ?fft_size:int
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t * (float, 'a) Nx.t, 'k) Pipeline.t
(** [piptrack_stage ~sample_rate ()] is {!piptrack} as a memoryless pipeline
    stage: rate 1:1 in frames, latency zero and format-preserving, consuming
    {!Stft.power_stage}[ ~power:1.] output ([[...; bins; frames]] magnitude
    chunks) and emitting each chunk's [(pitches, magnitudes)] pair; an empty
    chunk maps to a pair of empty chunks. librosa's references are per-frame
    quantities — the frame's own maximum, or a fixed magnitude — so every
    partitioning of a spectrogram yields the bits {!piptrack} computes on the
    whole of it:
    [Stft.power_stage ~power:1. c >> piptrack_stage ~sample_rate ()] streams
    [piptrack ~sample_rate (Stft.power_spectrum ~power:1. c x)], and
    [Pipeline.map fst] or [Pipeline.map snd] keeps one of the two tensors.

    Parameters are validated when the stage is built, raising
    [Invalid_argument] exactly as {!piptrack} does; the checks that need a
    chunk — its rank and bin count, their agreement with [fft_size], and the
    non-negativity of the data — wait for the chunk concerned, since the
    stream {!Pipeline.Format} carries no bin count. *)

val pitch_tuning :
  ?resolution:float -> ?bins_per_octave:int -> (float, 'a) Nx.t -> float
(** [pitch_tuning f] is the tuning of the frequencies [f] — a tensor of any
    shape and float dtype, read as one collection of hertz values — as
    [librosa.pitch_tuning] estimates it: their common offset from the
    equal-tempered ladder of [bins_per_octave] steps per octave through
    A440, in fractions of a step, in [\[-0.5, 0.5)].

    Each positive frequency [f] has the residual
    [bins_per_octave * log2 (f / 27.5)] modulo [1], less one when it is at
    least [0.5]: its signed distance to the nearest step of the ladder. The
    residuals are counted in the [ceil (1 / resolution)] bins between numpy's
    [linspace (-0.5, 0.5, ceil (1 / resolution) + 1)] edges, each bin closed
    below, and the result is the left edge of the fullest bin, the lowest one
    on a tie — the left edge, not the centre, so the estimate sits up to one
    bin width below the residuals it counts. NaN and non-positive
    frequencies are skipped, and when none remains the tuning is [0.],
    without librosa's warning; an infinite frequency has no residual and
    counts in no bin, as in librosa.

    [resolution] defaults to [0.01] — cents at the default
    [bins_per_octave] of [12] — and need not divide [1]: [0.03] makes 34
    bins of width [1 / 34]. Counting sorts the residuals, so the cost does
    not grow with the bin count. The residuals are computed in double
    precision whatever the dtype of [f], where librosa run natively on
    float32 frequencies computes them in float32.

    The mode is a discrete decision: frequencies tuned exactly onto an edge
    (a ladder at +25 cents, say) have residuals within rounding of it, and
    the last bit of their arithmetic picks the side, a whole bin apart.

    Raises [Invalid_argument] if [resolution] does not lie in [\[1e-9, 1)] —
    librosa documents [(0, 1)], and a histogram finer than [1e-9] would need
    gigabytes there — or if [bins_per_octave < 1]. *)

val estimate_tuning :
     Stft.Config.t
  -> ?resolution:float
  -> ?bins_per_octave:int
  -> ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:piptrack_reference
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> float
(** [estimate_tuning c ~sample_rate x] is the tuning of the audio [x] as
    [librosa.estimate_tuning] estimates it, in fractions of a step of the
    [bins_per_octave]-tone ladder through A440, in [\[-0.5, 0.5)]:
    {!piptrack} over the magnitude spectrogram of [x] on the analysis
    geometry [c], then {!pitch_tuning} over the detected pitches whose
    magnitude reaches the median magnitude of all the detections — the
    louder half, numpy's median halving the two middle values of an even
    count. A signal with no detection has tuning [0.]. The result is the
    [tuning] that {!Chroma.Config.create} and {!Cqt.Config.create} take when
    [bins_per_octave] is the configuration's own — its [n_chroma] for a
    chroma filterbank.

    [Stft.Config.create ~fft_size:2048 ~pad:(`Constant 0.) ()] is librosa's
    default analysis, the one its constant-Q transforms run when their
    [tuning] is [None] — with [~bins_per_octave] set to the transform's, and
    to [36] for [librosa.feature.chroma_cqt]. [resolution] and
    [bins_per_octave] are {!pitch_tuning}'s, and [fmin], [fmax], [threshold]
    and [reference] {!piptrack}'s, with the same defaults; the FFT size is
    that of [c].

    A whole-signal reduction, offered without a stage: the median spans every
    frame, and the mode of a histogram does not decompose over chunks.
    Leading axes do not broadcast: every channel and clip of [x] pools into
    the one estimate, as in librosa, so a batch of clips takes one call per
    clip. The analysis runs in double precision whatever the dtype of [x] —
    the audio is widened, exactly, before the transform — so a float32 clip
    has the estimate of the float64 clip of the same samples, where librosa
    analyses it in single precision; and the detections are never rounded
    to the dtype. Cost: the transform, one pass over its frames, and a sort
    of the detections; only the detections are kept, never a
    spectrogram-sized result.

    Every step of the estimate is a discrete decision — the thresholds, the
    local maxima, the median and the bin of each residual — so an input
    within rounding of a decision can be estimated a whole [resolution] away
    from the reference's estimate; the analysis of a pure tone also carries
    the interpolation bias {!piptrack} documents, a few hundredths of a step.

    Raises [Invalid_argument] as {!piptrack} and {!pitch_tuning} do on
    their parameters, or if [x] has rank zero. *)

val estimate_tuning_of_spectrogram :
     ?resolution:float
  -> ?bins_per_octave:int
  -> ?fmin:float
  -> ?fmax:float
  -> ?threshold:float
  -> ?reference:piptrack_reference
  -> ?fft_size:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> float
(** [estimate_tuning_of_spectrogram ~sample_rate s] is {!estimate_tuning}
    over an already-computed magnitude spectrogram [s] —
    {!Stft.power_spectrum}[ ~power:1.] output, or any non-negative
    [[...; bins; frames]] tensor — librosa's [S=] mode: {!piptrack} with its
    [fft_size] rule, then the median selection and {!pitch_tuning}. For
    float64 audio [x],
    [estimate_tuning_of_spectrogram ~fft_size:(Stft.Config.fft_size c)
    ~sample_rate (Stft.power_spectrum ~power:1. c x)] is
    [estimate_tuning c ~sample_rate x].

    [librosa.feature.chroma_stft] estimates its tuning this way when its
    [tuning] is [None]: over the {e power} spectrogram it projects, with
    [~bins_per_octave] set to its [n_chroma] and [fft_size] left to the bin
    count. The magnitudes are read in double precision whatever their dtype;
    the pooling over leading axes and the whole-signal reduction are
    {!estimate_tuning}'s.

    Raises [Invalid_argument] as {!piptrack} and {!pitch_tuning} do. *)

(** {2:features_analysis Autocorrelation, linear prediction and polynomial
    features}

    Two whole-signal analyses along the last axis, {!autocorrelate} and
    {!lpc}, offline only like {!Effects.trim}; and the per-frame
    least-squares polynomial fit of a spectrogram, {!poly_features}, with its
    {!Pipeline} stage. *)

val autocorrelate : ?max_size:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [autocorrelate x] is the autocorrelation of every row of the last axis of
    [x], as [librosa.autocorrelate] computes it, shaped [[...; m]] in the
    dtype of [x]: lag [k] of a row of [n] samples is the unnormalised sum
    [Σₜ x t * x (t + k)] over [t] in [\[0, n - k)], for [k] in [\[0, m)] with
    [m = min max_size n] — lag [0] is the row's energy. [max_size] defaults
    to [n], every lag, and a larger value keeps every lag too. Leading axes
    broadcast, so a batch of signals is one call.

    The lags are the inverse real FFT of the power spectrum of the row
    zero-padded to the smallest [2^a 3^b 5^c] points at or above [2n - 1] —
    the length scipy's [next_fast_len] gives the reference, long enough that
    no lag wraps around (Wiener–Khinchin). The padding depends on [n] alone,
    so a smaller [max_size] keeps exactly the first lags of the full result,
    bit for bit; each row is transformed alone, so its lags are the same bits
    alone or among others. The transforms run in double precision whatever
    the dtype, and the lags round to the dtype of [x] once, at the boundary.
    An FFT rounds relative to the whole row, not to each lag: every lag is
    within a few ulps of lag [0], so a lag far smaller than lag [0] carries a
    correspondingly larger relative error, here as in the reference. On
    float64 input the two agree within [6e-16] times lag [0] on the test
    signals.

    Deviations from the reference: float32 input is transformed in double
    precision, where librosa transforms it in single precision, within about
    [2e-7] of lag [0] of these lags; an empty row gives [[...; 0]], where the
    reference raises; [max_size < 1] raises, where the reference returns no
    lags or garbage ones. Complex input and axes other than the last are not
    offered.

    Every lag reads the whole row, so there is no stage. Cost: two real FFTs
    of about [2n] points per row, through complex128 spectra and a float64
    result of that length — some 50 bytes per input sample at the peak.

    Raises [Invalid_argument] if [x] has rank zero or [max_size < 1]. *)

val lpc : order:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [lpc ~order x] is the linear-prediction coefficients of every row of the
    last axis of [x] by Burg's method, as [librosa.lpc] computes them, shaped
    [[...; order + 1]] in the dtype of [x]: the prediction-error filter
    [A(z) = a 0 + a 1 z^-1 + ... + a order z^-order] with [a 0 = 1], under
    which [-(a 1 * x (t - 1) + ... + a order * x (t - order))] predicts
    [x t] — the denominator of the all-pole model of [x]. Leading axes
    broadcast, so a batch of signals is one call.

    Burg's method (Burg 1975; Marple 1980, section III, which the reference
    follows) picks each order's reflection coefficient to minimise the sum of
    the forward and backward prediction-error energies, and extends the
    filter by the Levinson recursion. The coefficient's denominator is summed
    once, from the signal's energy, then updated order by order by Andersen's
    recursion [(1 - k^2) den - b^2 - f^2] over the end samples of the updated
    errors, with the smallest positive normal double added before each
    division. The arithmetic is librosa's kernel, operation for operation —
    every sum in index order, no fused multiply-add — in double precision: on
    float64 input the coefficients are librosa's bits, on every platform.
    Each row is computed alone, so a channel's coefficients are the same bits
    alone or among others.

    Ill-conditioned input: a signal of zeros gives [[1; 0; ...]] and a
    constant one [[1; -1; 0; ...]], as in the reference. An order far beyond
    what a signal determines — a noiseless sinusoid fitted at many times its
    count of partials, say — lets the recursive denominator fall below the
    true error energy, and a reflection coefficient can then exceed one in
    magnitude: the filter is no longer minimum-phase and its values are
    decided by rounding, exactly as the reference returns them while they
    stay finite. Where a coefficient overflows instead — samples beyond about
    [1e154], whose energy overflows double precision, or a divergence at such
    an order — this function raises, where the reference returns NaN or
    infinite coefficients (its documented [FloatingPointError] for
    ill-conditioned input is no longer raised by librosa 1.0).

    Further deviations: float32 input is widened exactly, analysed in double
    precision and rounded once, where librosa runs its kernel in single
    precision — on the same float32 samples, the coefficients are librosa's
    float64 ones rounded to float32, which its float32 kernel misses by up to
    [5e-4] on the autoregressive processes of the tests and by [3e-2] on a
    tone under weak noise; the bias before each division is the double's for
    every dtype, where the reference takes the input dtype's, a difference
    visible only on float32 signals of energy below about [1e-30]. A row of
    [n <= order] samples runs out of prediction errors after order [n - 1],
    so its remaining coefficients are zeros: the reference reads past the end
    of its buffers there (and returned zeros on every run measured).

    Every coefficient reads the whole row, so there is no stage. Cost: about
    [3 * n * order] multiply-adds per row of [n] samples.

    Raises [Invalid_argument] if [order < 1], if [x] has rank zero, if a
    sample is NaN or infinite, or if a coefficient overflows (above). *)

val poly_features :
     Stft.Config.t
  -> ?order:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [poly_features c ~sample_rate x] is the coefficients of the least-squares
    polynomial of degree [order] in frequency (Hz) fitted to every frame of
    the magnitude spectrogram {!Stft.power_spectrum}[ ~power:1. c x], as
    [librosa.feature.poly_features] computes them, shaped
    [[...; order + 1; frames]] with the highest degree first: row [0] holds
    the coefficient of [f ^ order] and row [order] the constant term.
    [order] defaults to [1], the spectral slope and intercept; [0] fits each
    frame's mean. The time axis is the last axis of [x]; leading axes
    broadcast, so a batch of clips is one call. This face consumes raw audio;
    {!poly_features_of_spectrogram} consumes an already-computed spectrogram.
    librosa's defaults are the geometry
    [Stft.Config.create ~fft_size:2048 ~hop:512 ~pad:(`Constant 0.) ()].

    Bin [k] sits at [k * sample_rate / fft_size] on the grid of [c],
    evaluated as one reciprocal and one multiply per bin as numpy's
    [rfftfreq] does, so an odd [fft_size] fits on its own grid, as the
    reference's audio path does.

    The fit is [numpy.polyfit]'s problem, solved the same way: the
    Vandermonde matrix of the bin frequencies, its columns scaled to unit
    norm — which brings the condition number of an order-3 fit over an
    audio-rate grid from about [1e12] to under [1e2] — and the least-squares
    solution through the scaled matrix's singular value decomposition, where
    singular values at or below [bins * eps] times the largest count as zero,
    so a rank-deficient fit (fewer distinct frequencies than [order + 1])
    yields numpy's minimum-norm solution. The decomposition is a one-sided
    Jacobi SVD (Hestenes 1958; Demmel and Veselić 1992), computed once per
    grid and order into the one matrix that maps a frame to its
    coefficients, so each coefficient is a fixed-order sum over its own
    frame's bins: the same bits alone or among others, computed in double
    precision and rounded to the dtype of [x] once. Every sum over the bins,
    the decomposition's and each frame's, is compensated (Neumaier 1974;
    Ogita, Rump and Oishi 2005), so its rounding does not grow with the bin
    count: an order-0 fit, the frame's mean, is within [3e-16] of the exact
    mean at every bin count measured, up to 8939.

    numpy solves through LAPACK's [gelsd] instead. On float64 input over FFT
    grids of 3 to 8939 bins, each coefficient agrees within [4.1e-15] times
    the sum of the magnitudes of the terms it adds, [Σⱼ |P i j| * |S j t|]
    for the fit's map [P], at orders 0 to 3 — mostly the reference's own
    rounding, as its order-0 fits miss the exact mean by up to [4e-15]. The
    scaled problem's condition number grows about fivefold per order (3.7 at
    order 1, 86 at order 3, [2.5e3] at order 5, [1.4e7] at order 10), and the
    agreement degrades with it — [2e-14] of that scale at order 5 and
    [1e-10] at order 10, over 257 to 2049 bins — so none is claimed beyond
    order 12 or so; past order 17 the rank rule itself decides on
    rounding-level ties. Over a custom grid
    ({!poly_features_of_spectrogram}'s [freqs]) the agreement follows that
    grid's conditioning instead: up to [3e-13] at order 3 on random grids of
    four bins.

    Deviations from the reference: the coefficients keep the dtype of [x],
    where librosa returns float64; an infinite magnitude makes only its own
    frame's coefficients non-finite, where the reference's solver spreads it
    over every frame (a NaN stays within its frame in both).

    Cost: the STFT, then [order + 1] compensated sums of [bins] products per
    frame, some eight flops per bin and coefficient; the fit's matrix is
    built once per call, in [O(bins * order^2)] flops per Jacobi sweep.

    Raises [Invalid_argument] if [order < 0], if [sample_rate < 1], if [x]
    has rank zero, or if the squares of a power of the bin frequencies up to
    [order] overflow double precision — from order 33 at 96 kHz to order 43
    at 8 kHz, order 38 at 22050 Hz, for an [fft_size] of 2048 — where the
    reference's scaling overflows, zeroes the highest-degree coefficients
    and returns a rank-reduced fit with a warning. *)

val poly_features_of_spectrogram :
     ?order:int
  -> ?freqs:(float, 'a) Nx.t
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [poly_features_of_spectrogram ~sample_rate s] is {!poly_features} over an
    already-computed spectrogram [s] — {!Stft.power_spectrum}[ ~power:1.]
    output, or any real [[...; bins; frames]] tensor, decibels and other
    signed values included — mapping [[...; bins; frames]] to
    [[...; order + 1; frames]]: the reference's [S=] path. [order] and the
    fit are those of {!poly_features}.

    The bin frequencies default to the FFT grid the bin count implies — bin
    [k] at [k * sample_rate / (2 * (bins - 1))], as for {!spectral_centroid}
    and the reference's [S=] path — so applying this function to
    [Stft.power_spectrum ~power:1. c x] is [poly_features c ~sample_rate x]
    bit for bit when [fft_size] is even; for an odd one, pass the grid of [c]
    as [freqs]. [freqs], a rank-one tensor of [bins] finite frequencies in
    any order, repeats allowed, replaces that grid; when it is given,
    [sample_rate] is validated but not otherwise consulted. It shares the
    dtype of [s], so a float32 spectrogram fits over float32 frequencies,
    where the reference also accepts a float64 grid; the fit widens them
    exactly to double precision. The per-frame frequency {e matrix} of a
    reassigned spectrogram is not supported. A spectrogram with no frames,
    or a zero-size leading axis, maps to zeros of the result shape; one with
    no bins raises.

    Deviation from the reference on a float32 [freqs]: [numpy.polyfit] also
    builds its matrix from those values in double precision, but takes its
    rank threshold from the float32 epsilon — [bins] times [1.2e-7], about
    [1.2e-4] over 1025 bins, instead of [2.3e-13] — so from order 6 or so
    over a thousand bins it discards singular values that the double
    threshold keeps and returns a rank-reduced fit, where this function
    returns the full-rank one.

    Raises [Invalid_argument] as {!poly_features} does on [order] and
    [sample_rate], if [s] has rank below two or no bins, if [freqs] is
    absent and [s] has a single bin (a one-bin spectrogram implies no FFT
    grid; the reference divides by zero), if [freqs] is present and is not
    rank-one with one finite frequency per bin, or if a power of the
    frequencies up to [order] vanishes at every bin or its squares overflow
    double precision, where the reference's scaling divides by zero or
    overflows. *)

val poly_features_stage :
     Stft.Config.t
  -> ?order:int
  -> sample_rate:int
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [poly_features_stage c ~sample_rate ()] is {!poly_features} as a
    memoryless pipeline stage: rate 1:1 in frames, latency zero and
    format-preserving. It consumes {e magnitude}-spectrum chunks — compose
    [Stft.power_stage ~power:1. c >> poly_features_stage c ~sample_rate ()]
    for the streaming form of [poly_features c ~sample_rate], bit for bit at
    every [fft_size] — mapping [[...; bins; frames]] to
    [[...; order + 1; frames]]; an empty chunk maps to an empty chunk. The
    fit is built on the grid of [c] when the stage is built, and every
    partitioning of a spectrogram yields the same bits. For a custom grid,
    wrap [poly_features_of_spectrogram ~freqs] in {!Pipeline.stateless}: it is
    per-frame too, and refits its matrix on every chunk.

    Raises [Invalid_argument] as {!poly_features} does on [order],
    [sample_rate] and the powers of the bin frequencies, when the stage is
    built. The stream {!Pipeline.Format} does not carry a bin count, so a
    chunk whose bin axis is not [Stft.Config.bins c] long, or a chunk of
    rank below two, is reported at that chunk, not at prepare time. *)

(** {2:features_inverse Feature inversion}

    Approximate inverses of the mel and cepstral features, as
    [librosa.feature.inverse] computes them: a cepstrum back to its mel
    power spectrogram, a mel spectrogram back to a linear magnitude
    spectrogram, and either back to audio through {!Stft.griffin_lim}. *)

val mfcc_to_mel :
     Mel.Config.t
  -> ?lifter:float
  -> ?reference:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mfcc_to_mel mel c] is the mel power spectrogram the cepstrum [c]
    encodes, mapping [[...; n_mfcc; frames]] to [[...; n_mels; frames]] with
    [n_mels = Mel.Config.n_mels mel] and [n_mfcc] read from [c]: the inverse
    of {!mfcc} and {!mfcc_stage} on the same [mel], as
    [librosa.feature.inverse.mfcc_to_mel] computes it. Each coefficient is
    divided by its lifter weight, the orthonormal type-II DCT is inverted —
    its transpose, the coefficients past [n_mfcc] taken as zero — and the
    log-mel is raised back to power, [reference * 10 ** (db / 10)].
    Leading axes broadcast, so a batch of cepstra is one call. [lifter]
    defaults to [0.] (none) and must be the forward call's; [reference]
    defaults to [1.], the forward log-mel's reference.

    Only the first [n_mfcc] coefficients survive the forward transform, so
    with [n_mfcc < n_mels] the result is the spectral envelope they
    describe — the log-mel whose higher coefficients are zero — and with
    [n_mfcc = n_mels] the exact inverse. Nothing undoes the forward
    log-mel's [1e-10] floor or {!mfcc}'s 80 dB clamp. The inverse DCT and
    the lifter division are one matrix, each frame folded from its own
    coefficients in a fixed order, so a frame's bits do not depend on its
    batch; everything is computed in double precision and rounded to the
    dtype of [c] once, where librosa computes in the input dtype. Dividing
    by a lifter weight near zero amplifies its coefficient accordingly; a
    weight that vanishes, as lifter [2.]'s does at coefficient [2], is
    refused.

    Documented deviations: only the orthonormal type-II DCT {!mfcc}
    computes is inverted (no [dct_type] or [norm]); [n_mfcc > n_mels]
    raises where librosa's inverse DCT silently truncates; and a vanishing
    lifter weight raises where librosa warns and divides by the smallest
    positive normal.

    Raises [Invalid_argument] if [c] has rank below two, if [n_mfcc] does
    not lie in [\[1, n_mels\]], if [lifter] is not finite and non-negative,
    if the lifter weight [1 + (lifter / 2) * sin (pi * (k + 1) / lifter)]
    of a kept coefficient [k] is below [epsilon_float] in magnitude, or if
    [reference] is not finite and positive. *)

val mfcc_to_mel_stage :
     Mel.Config.t
  -> ?n_mfcc:int
  -> ?lifter:float
  -> ?reference:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mfcc_to_mel_stage mel ()] is {!mfcc_to_mel} as a memoryless pipeline
    stage: rate 1:1 in frames, latency zero and format-preserving, mapping
    [[...; n_mfcc; frames]] chunks to [[...; n_mels; frames]]; an empty
    chunk maps to an empty chunk. [n_mfcc] defaults to [20], as for
    {!mfcc_stage}, so [mfcc_stage mel ~n_mfcc () >> mfcc_to_mel_stage mel
    ~n_mfcc ()] streams a cepstral round trip. Every partitioning of a
    cepstrum yields the bits {!mfcc_to_mel} computes on the whole of it.

    Raises [Invalid_argument] for the parameters as {!mfcc_to_mel} does,
    with the stage's [n_mfcc], when the stage is built. The stream
    {!Pipeline.Format} does not carry a coefficient count, so a chunk of
    rank below two or whose coefficient axis is not [n_mfcc] long is
    reported at that chunk. *)

val mel_to_stft :
  Mel.Config.t -> ?power:float -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [mel_to_stft mel m] is a non-negative linear spectrogram whose mel
    projection through [mel] best fits the mel spectrogram [m], mapping
    [[...; n_mels; frames]] to [[...; bins; frames]] with
    [bins = Mel.Config.bins mel]: the counterpart of
    [librosa.feature.inverse.mel_to_stft]. [m] is a power mel spectrogram
    when [power] is [2.], the default, and the result is then a magnitude
    spectrogram — the fitted spectrum raised to [1 / power] — ready for
    {!Stft.griffin_lim}. Leading axes broadcast, so a batch of spectrograms
    is one call.

    There are fewer bands than bins, so many non-negative spectra fit a
    frame equally well: each frame is solved on its own for one close to
    librosa's starting point. With [A] the filterbank ({!Mel.filterbank}),
    [b] a frame of [m] and [x0 = max 0 (A+ b)] the clipped pseudo-inverse
    librosa starts from (singular values up to [1e-15] of the largest
    dropped, as librosa's [scipy.linalg.pinv] call drops them), the fitted
    spectrum is the third step of the proximal-point method from [x0]:

    {[
      x_(k+1) = argmin_(x >= 0) 1/2 |A x - b|^2 + lambda/2 |x - x_k|^2
      lambda = 1e-9 * max_j |column j of A|^2
    ]}

    The first step is within [O (lambda)] of the non-negative
    least-squares solution nearest [x0]; the next two remove what is left
    of [lambda]'s pull on the fit. Each step is solved exactly, up to
    rounding, by a generalised Newton method on its [n_mels]-dimensional
    dual, whose Newton systems are banded since a mel filter overlaps only
    its neighbours. Wherever no filter reaches a bin (DC and Nyquist at the
    default bounds, and every bin outside [\[f_min, f_max\]]), the result
    is exactly [0.]. Negative or zero entries of [m] are fitted like any
    other; a zero frame yields the zero spectrum.

    Named deviation: librosa runs scipy's L-BFGS-B from [x0] over blocks of
    frames, each block one problem across its frames and every leading axis
    — as many frames as fit 256 KiB of [m] (256 frames of 128 float64
    bands, fewer in a batch) — on an objective it divides by the block's
    size, and stops on absolute tolerances. Whether it leaves [x0] at all,
    and where it stops, therefore depends on the absolute level of [m] and
    on the size and content of the block: the same frames can come back as
    [x0] at one level and be iterated on at another. The parity contract is
    therefore:
    - where [A+ b] is non-negative on a filterbank of full rank, the result
      is librosa's own output, [x0], to rounding (measured, [2e-14]
      relative). The envelopes of broadband signals tend to be such frames;
      tonal envelopes, even the smooth ones {!mfcc_to_mel} produces from 13
      or 20 coefficients, often reach a negative start, and the next case
      applies. So does every frame on a rank-deficient filterbank (more
      bands than the bins resolve, such as 128 on a 512-point FFT), where
      the sign of [A+ b] is decided by the rounding of the smallest
      singular values kept;
    - elsewhere a frame's fit [1/2 |A x - b|^2] exceeds librosa's by at
      most a relative [1e-9] (measured: [5e-11] at most on filterbanks of
      full rank, [8e-10] on a rank-deficient one, both where L-BFGS-B
      reaches the optimum), and is lower, often by orders of magnitude,
      wherever L-BFGS-B stops short of it;
    - the mel projection of the result reproduces any mel spectrogram of a
      signal to rounding (measured, [2e-14] relative per frame). Where no
      non-negative spectrum fits the bands, the dual grows as
      [1 / lambda] and its image through [A^T] cancels, losing digits to
      rounding: the optimality conditions of the fit then hold to [4e-6]
      of the frame's scale at worst, the scale being the largest entry of
      [|A^T b|], and the fit is within a relative [1e-9] of the optimum
      (measured against [scipy.optimize.nnls]);
    - the result is scale-equivariant, [mel_to_stft (c * m)] being
      [c ** (1 / power) * mel_to_stft m] to rounding where some spectrum
      fits the frame, and to [1e-6] of the frame's largest power where
      none does (measured); the fitted spectrum scales bit for bit when [c]
      is a power of two, and so does the result when [c ** (1 / power)] is
      one too.
    The spectrum itself is not compared outside the first case: the
    solutions form a polytope and librosa's point in it is an artefact of
    its stopping rule. Bins no filter reaches are exactly [0.] where
    librosa's pseudo-inverse leaves a residue of about [1e-15] (about
    [3e-8] after the square root).

    A frame's spectrum is a function of that frame alone, computed in a
    fixed order, so it is the same bits alone or among others, at every
    chunking and on any number of domains; everything is computed in double
    precision and rounded to the dtype of [m] once, where librosa builds
    its filterbank and pseudo-inverse in the input dtype. A frame costs one
    dense product with the pseudo-inverse, [O (bins * n_mels)], and three
    proximal steps. A step whose start already fits stops at its first
    gradient; any other runs some tens of Newton steps (at most about sixty
    measured), each a banded Cholesky factorisation, [O (n_mels * w ** 2)]
    for filters overlapping [w] neighbours, and a line search over sorted
    breakpoints, [O (bins * log bins)]. The pseudo-inverse, a singular value
    decomposition of the filterbank, is computed once per call.

    Raises [Invalid_argument] if [power] is not finite and positive, if [m]
    has rank below two or its mel axis is not [Mel.Config.n_mels mel] long
    (librosa reads the band count from [m]; here it comes from [mel]), or
    if [m] holds a NaN or infinite value. *)

val mel_to_stft_stage :
     Mel.Config.t
  -> ?power:float
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [mel_to_stft_stage mel ()] is {!mel_to_stft} as a memoryless pipeline
    stage: rate 1:1 in frames, latency zero and format-preserving, mapping
    [[...; n_mels; frames]] chunks to [[...; bins; frames]]; an empty chunk
    maps to an empty chunk. Each frame is solved on its own, with no state
    carried between frames, so every partitioning yields the bits
    {!mel_to_stft} computes on the whole spectrogram. The pseudo-inverse is
    computed once, when the stage is built.
    [mfcc_to_mel_stage mel ~n_mfcc () >> mel_to_stft_stage mel ()] streams
    the spectral half of {!mfcc_to_audio}.

    Raises [Invalid_argument] if [power] is not finite and positive, when
    the stage is built, and, at the chunk concerned, for a chunk
    {!mel_to_stft} would refuse. *)

val mel_to_audio :
     Stft.Config.t
  -> Mel.Config.t
  -> ?power:float
  -> ?n_iter:int
  -> ?length:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mel_to_audio stft mel m] is a signal whose mel spectrogram on
    [stft] and [mel] approximates [m], [[...; n_mels; frames]] to
    [[...; samples]]: [Stft.griffin_lim ?n_iter ?length stft
    (mel_to_stft mel ?power m)], the composition
    [librosa.feature.inverse.mel_to_audio] computes. [power] is
    {!mel_to_stft}'s; [n_iter] (default [32]) and [length] are
    {!Stft.griffin_lim}'s, with its other defaults — momentum [0.99] and a
    zero initial phase. Callers wanting another momentum or initial phase
    compose the two functions themselves. Griffin-Lim iterates over the
    whole signal, so there is no stage: stream the spectral half with
    {!mel_to_stft_stage}.

    Documented deviations, inherited from {!Stft.griffin_lim}: the initial
    phase is zero and deterministic, where librosa's [mel_to_audio] starts
    from a random phase it cannot be told to seed; the iteration runs in
    double precision and returns the dtype of [m], where librosa rounds its
    signal to float32 at every iteration and returns float32 whatever the
    input; and [length] applies to the final synthesis only, where librosa
    applies it at every iteration.

    Raises [Invalid_argument] if the two configurations disagree on
    [fft_size], naming both sizes, and for what {!mel_to_stft} and
    {!Stft.griffin_lim} refuse — among them a non-invertible [stft],
    [n_iter < 1] and a negative [length]. Every condition is checked
    before the solve and reported under this function's name. *)

val mfcc_to_audio :
     Stft.Config.t
  -> Mel.Config.t
  -> ?lifter:float
  -> ?reference:float
  -> ?power:float
  -> ?n_iter:int
  -> ?length:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [mfcc_to_audio stft mel c] is a signal whose cepstrum approximates [c]:
    [mel_to_audio stft mel ?power ?n_iter ?length
    (mfcc_to_mel mel ?lifter ?reference c)], the composition
    [librosa.feature.inverse.mfcc_to_audio] computes, with the parameters
    and deviations of {!mfcc_to_mel} and {!mel_to_audio}.

    Raises [Invalid_argument] as {!mfcc_to_mel} and {!mel_to_audio} do,
    every condition checked before any work but one: a cepstrum whose mel
    spectrogram holds a NaN or infinite value — a cepstrum holding a NaN,
    or large enough for the power to overflow — is refused once
    {!mfcc_to_mel} has computed that spectrogram. *)

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

(** {2:features_onsets Onset detection}

    Onsets picked from an onset strength envelope: the peak picker of
    librosa's onset detector with its {!Pipeline} stage, the detector itself
    on audio or on an envelope, and the backtracking that moves each onset
    back to the energy minimum before it.

    The rule is Dixon's ({e Onset detection revisited}, DAFx 2006, section
    2.6) in the parameterisation of Böck, Krebs and Schedl ({e Evaluating
    the online capabilities of onset detection methods}, ISMIR 2012,
    section 2.3.4): a frame is an onset when it is a local maximum, stands
    far enough above a local mean, and comes far enough after the previous
    onset.

    Results are frame indices, ascending, as [int] arrays. Frame [f] starts
    at sample [f * hop], and {!Convert.frames_to_time} gives seconds:
    [Convert.frames_to_time ~sample_rate ~hop
       (Nx.create Nx.float64 [|k|] (Array.map Float.of_int onsets))].
    A list of frames belongs to one signal, so each function reads one
    signal along its last axis: a rank-one [[frames]], or any shape whose
    leading axes all have size [1], such as the [[1; frames]] envelope of
    [[1; samples]] audio. A batch is picked signal by signal, as in librosa,
    whose index results refuse every envelope of rank above one.

    Only the greedy picker streams. The detector's normalisation divides by
    the maximum of the whole envelope, which Böck et al. (section 2.3.1)
    rule out online; the dynamic-programming selections run a backward pass
    over the whole envelope; and backtracking defaults to the normalised
    envelope. Those are offline, with no stage. *)

(** How {!peak_pick} chooses among the candidates — the frames that are
    local maxima clearing the threshold — so that no two peaks lie within
    [wait] frames of each other:
    - [`Greedy] (librosa's ['greedy'], the default) scans forward and keeps
      every candidate more than [wait] frames after the last one kept;
    - [`Dp_count] (['dp_count']) keeps, by dynamic programming, a set of
      candidates spaced more than [wait] frames apart with the most peaks;
    - [`Dp_value] (['dp_value']) keeps one with the largest sum of values.

    The dynamic-programming selections resolve ties as librosa does: a
    candidate is taken only when taking it strictly improves on skipping
    it, so equal totals resolve towards later frames. *)
type peak_selection = [`Greedy | `Dp_count | `Dp_value]

val peak_pick :
     ?selection:peak_selection
  -> pre_max:int
  -> post_max:int
  -> pre_avg:int
  -> post_avg:int
  -> delta:float
  -> wait:int
  -> (float, 'a) Nx.t
  -> int array
(** [peak_pick ~pre_max ~post_max ~pre_avg ~post_avg ~delta ~wait x] is the
    peaks of the envelope [x], as [librosa.util.peak_pick] picks them. Frame
    [n] is a candidate when
    - it equals the maximum of frames [\[n - pre_max, n + post_max)], and
    - it is at least [mean + delta], [mean] the mean of frames
      [\[n - pre_avg, n + post_avg)].

    Both windows are half-open and hold frame [n], so [post_max = 1] looks
    at no later frame. Both are cut short at either end of the signal, never
    padded, and the mean divides by the frames its window holds. Every frame
    of a plateau at the maximum is a candidate. The candidates are then
    selected as [selection] says, [`Greedy] by default: a peak at [p]
    excludes frames [p + 1] to [p + wait]. The windows and [delta] have no
    defaults, as in librosa; {!onset_detect_of_envelope} derives librosa's
    from the frame rate.

    The maximum is compared exactly and the mean is zero plus the window's
    frames in ascending order, divided by their count, in double precision:
    the reference's float64 arithmetic term for term, the
    dynamic-programming selections taking their means from prefix sums as
    it does. A float64 envelope is therefore picked exactly as librosa picks
    it, and a float32 one as librosa picks its float64 image: librosa sums a
    float32 window in single precision, so a frame within that rounding of
    its threshold can be decided the other way. NaN keeps the reference's
    semantics: it fails the greedy test of every frame whose maximum's
    window holds it, passes the dynamic-programming selections' maximum
    test there, and fails every comparison it reaches; infinities compare
    as IEEE numbers.

    Named deviations: the windows are integers, where librosa rounds
    fractional windows up; [delta] must be finite, where librosa accepts NaN
    and infinity and then finds no peak; a signal of no frames has no peak,
    where librosa raises.

    The greedy selection scans the maximum's window of each frame it reaches
    up to the first frame above it, and sums the mean's window at local
    maxima only: at worst [frames * (pre_max + post_max + pre_avg + post_avg)]
    operations. The dynamic-programming selections cost the same plus three
    arrays of [frames + 1] entries.

    Raises [Invalid_argument] if [pre_max], [pre_avg] or [wait] is negative,
    if [post_max] or [post_avg] is below [1], if [delta] is not finite and
    non-negative, or if [x] has rank zero, a leading axis of size other than
    [1] or a dtype other than float32 and float64. *)

val peak_pick_stage :
     pre_max:int
  -> post_max:int
  -> pre_avg:int
  -> post_avg:int
  -> delta:float
  -> wait:int
  -> ((float, 'a) Nx.t, int array, 'k) Pipeline.t
(** [peak_pick_stage ~pre_max ~post_max ~pre_avg ~post_avg ~delta ~wait] is
    the greedy {!peak_pick} as a causal pipeline stage: envelope chunks in,
    each holding one signal as {!peak_pick} reads it, and out the peaks each
    chunk decides, as frame indices of the stage's own input stream counted
    from its first frame since [prepare] or [reset]. Concatenated, they are
    {!peak_pick} of the envelope the stream carried, on every partitioning:
    the offline greedy picker is the stage's kernel on one chunk.

    Latency. A decision reads [max post_max post_avg - 1] frames past its own
    frame, and the stage declares that many frames of latency: frame [n] is
    decided by the push that brings the frames seen past
    [n + max post_max post_avg - 1], and the flush decides the last ones with
    the windows cut short at the end of the stream, as {!peak_pick} cuts
    them at the end of the signal. After [k] frames, every peak below
    [k - max post_max post_avg + 1] has been emitted; a push that decides no
    peak emits nothing. With [post_max = post_avg = 1], Böck et al.'s online
    setting, each frame is decided as it arrives. Behind the first undecided
    frame the stage keeps [max pre_max pre_avg] frames.

    The stage streams the picker, not {!onset_detect}, whose normalisation
    needs the whole envelope. An online detector feeds the unnormalised
    envelope straight in:
    {[
      Pipeline.(
        Stft.power_stage c >> Mel.stage m >> Db.stage (Db.Value 1.)
        >> onset_strength_stage ()
        >> peak_pick_stage ~pre_max:3 ~post_max:1 ~pre_avg:10 ~post_avg:1
             ~delta ~wait:3)
    ]}
    picks with no lookahead, [delta] then in decibels of mean flux rather
    than on the [\[0, 1\]] scale of the detector's [0.07]. That chain
    carries neither the centered shift nor the 80 dB clamp of
    {!onset_strength}, as {!onset_strength_stage} documents: add
    [fft_size / (2 * hop)] to its frames for the offline numbering.

    Raises [Invalid_argument] for the parameters as {!peak_pick} does, when
    the stage is built; when it is prepared, if its latency overflows the
    pipeline's exact rate arithmetic (windows near [max_int] frames); and, at
    the chunk concerned, for a chunk of rank zero, of a dtype other than
    float32 and float64, or with a leading axis of size other than [1]. *)

val onset_detect :
     Stft.Config.t
  -> Mel.Config.t
  -> ?normalize:bool
  -> ?backtrack:[`Envelope | `Energy of (float, 'e) Nx.t]
  -> ?selection:peak_selection
  -> ?pre_max:int
  -> ?post_max:int
  -> ?pre_avg:int
  -> ?post_avg:int
  -> ?delta:float
  -> ?wait:int
  -> (float, 'a) Nx.t
  -> int array
(** [onset_detect stft mel x] is the onset frames of the audio [x],
    ascending, as [librosa.onset.onset_detect] finds them from audio:
    {!onset_detect_of_envelope} over [onset_strength stft mel x] at lag [1],
    at the sample rate of [mel] and the hop of [stft], with the same
    options. [x] holds one signal, as {!peak_pick} reads it.

    librosa analyses with
    [Stft.Config.create ~fft_size:2048 ~hop ~pad:(`Constant 0.) ()] and
    [Mel.Config.create ~n_mels:128 ~sample_rate ~fft_size:2048 ()], [hop]
    being its [hop_length] (512 by default), and with these configurations
    the frames are its own. The envelope carries {!onset_strength}'s
    documented deviations, none of which applies at these settings, and one
    more: librosa projects through a float32 mel filterbank where SoundML's
    is double, so a frame within that rounding of a threshold could be
    decided the other way.

    Offline only, as {!onset_detect_of_envelope}; the cost is one
    {!onset_strength} and the picking.

    Raises [Invalid_argument] for the options as {!onset_detect_of_envelope}
    does, its [sample_rate] the rate of [mel] and its [hop] the hop of
    [stft]; if the configurations disagree on [fft_size]; or if [x] has rank
    zero, a leading axis of size other than [1] or a dtype other than
    float32 and float64 — all before any analysis. *)

val onset_detect_of_envelope :
     ?hop:int
  -> ?normalize:bool
  -> ?backtrack:[`Envelope | `Energy of (float, 'e) Nx.t]
  -> ?selection:peak_selection
  -> ?pre_max:int
  -> ?post_max:int
  -> ?pre_avg:int
  -> ?post_avg:int
  -> ?delta:float
  -> ?wait:int
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> int array
(** [onset_detect_of_envelope ~sample_rate e] is the onset frames of the
    onset strength envelope [e], ascending, as [librosa.onset.onset_detect]
    finds them given [onset_envelope=e]: [e] is {!onset_strength} output, or
    any envelope of one signal whose frames lie [hop] samples apart at
    [sample_rate].

    + With [normalize] (default [true]), [e] is shifted to a minimum of [0]
      and divided by its maximum plus the smallest positive normal double,
      so that it spans [\[0, 1\]]: a whole-signal step.
    + An envelope with no nonzero value (a constant one, once normalised) or
      with a value that is not finite has no onset, normalised or not.
    + The onsets are {!peak_pick}'s, at [selection] ([`Greedy] by default).
      The windows default to librosa's durations in whole frames, in integer
      division at the rate [r = sample_rate] and the hop [h]: [pre_max] and
      [wait] to [3 * r / 100 / h] (30 ms), [pre_avg] to [r / 10 / h]
      (100 ms), [post_avg] to [r / 10 / h + 1] (100 ms after the frame, the
      frame included) and [post_max] to [1] (no lookahead); [delta] defaults
      to [0.07], on the normalised scale:
      {v
      sample_rate  hop  pre_max  post_max  pre_avg  post_avg  wait
      22050        512        1         1        4         5     1
      16000        160        3         1       10        11     3
      44100        441        3         1       10        11     3
      8000         128        1         1        6         7     1
      v}
      A geometry whose windows round down to no frame — 8000 Hz at a hop of
      2400, say — compares each frame with itself alone and finds no onset.
      Without normalisation, [delta] is in the envelope's own units.
    + [backtrack] (default: none) then moves each onset back as
      {!onset_backtrack} does: [`Envelope] to the latest minimum at or before
      it of the envelope the picker read, normalised when normalising, and
      [`Energy c] to that of the curve [c] — librosa's [energy=] — such as
      [rms ~frame_length ~hop x] on the same frame grid.

    [hop] defaults to [512]. The arithmetic is librosa's float64 arithmetic,
    so a float64 envelope gives librosa's frames exactly and a float32 one
    the frames librosa finds in its float64 image (see {!peak_pick}).
    Offline only: the normalisation needs the whole envelope. With
    [~normalize:false], [`Greedy] and no backtracking, the result is
    {!peak_pick} at the same windows, which {!peak_pick_stage} streams.

    Named deviations, beyond {!peak_pick}'s:
    - empty results where librosa raises: an envelope of no frames when
      normalising, and backtracking when no onset was found;
    - every parameter is validated, even for an envelope with no onset,
      where librosa validates them only when it picks;
    - frames are the only unit, librosa's [units='samples'] being
      [f * hop] and [units='time'] {!Convert.frames_to_time}, and there is
      no dense [sparse=False] output.

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], for the
    windows and [delta], given or derived, as {!peak_pick} does, or if [e] or
    an [`Energy] curve has rank zero, a leading axis of size other than [1]
    or a dtype other than float32 and float64. *)

val onset_backtrack : int array -> (float, 'a) Nx.t -> int array
(** [onset_backtrack events energy] is each frame of [events] moved back to
    the latest minimum of [energy] at or before it, as
    [librosa.onset.onset_backtrack] moves it, so that a segment cut at the
    onset keeps its whole attack (Jehan, {e Creating music by listening},
    MIT, 2005). The minima are frame [0] and every interior frame [i] with
    [energy i <= energy (i - 1)] and [energy i < energy (i + 1)]: the last
    frame of a flat valley before a rise. An event on a minimum stays, an
    event past the end of [energy] moves to its last minimum, and the result
    keeps the order and the duplicates of [events]. The comparisons are
    exact at either dtype, so the frames are librosa's; a NaN is never a
    minimum.

    [energy] holds one signal along its last axis, so {!rms} output
    [[1; frames]] is read along its frames, where librosa reads such an input
    along its first axis, finds no interior frame and moves every event to
    frame [0]. No events give no frames, where librosa raises.

    Raises [Invalid_argument] if an event is negative, or if [energy] has
    rank zero, a leading axis of size other than [1] or a dtype other than
    float32 and float64. *)

(** {2:features_rhythm Tempo and beat}

    The rhythm of an onset envelope — {!onset_strength} output, [[...; frames]]
    at one value per hop: its local periodicities as the autocorrelation and
    Fourier tempograms, the tempo they imply, the beats of Ellis'
    dynamic-programming tracker and the predominant local pulse. Every function
    consumes the envelope rather than audio, so the analysis of a signal is a
    composition —
    [beat_track ~sample_rate ~hop:(Stft.Config.hop s) (onset_strength s m x)]
    — and an envelope from elsewhere is analysed the same way. librosa's
    [beat_track] and [plp] aggregate the envelope they compute from audio by
    the median over mel bands, which {!onset_strength} does not offer: the
    composition is librosa given the mean-aggregated envelope
    ([onset_envelope=onset_strength(y=y, ...)]), not given [y=y] itself, and a
    beat can move between the two. The tempograms and {!tempo}, whose audio
    path in librosa takes the mean, compose exactly.

    Values are computed in double precision from the envelope's values,
    whatever its dtype, and rounded to it once: a float32 envelope gives the
    float64 result on its own values, rounded. The tempograms stream; the
    tempo, the beats and the pulse read the whole envelope and have no
    stage. *)

val tempo_frequencies :
  (float, 'a) Nx.dtype -> ?hop:int -> sample_rate:int -> int -> (float, 'a) Nx.t
(** [tempo_frequencies dtype ~sample_rate n] is the tempo, in beats per
    minute, of each of the [n] lag bins of a {!tempogram}, as a rank-one
    tensor: lag [k] of an envelope at [sample_rate / hop] values per second
    repeats every [k] values, [60 * sample_rate / (hop * k)] beats per minute,
    and the zero lag is [Float.infinity]. [hop] defaults to [512]. The values
    are those of [librosa.tempo_frequencies], computed in its order and
    rounded to [dtype].

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], or if
    [n < 1]. *)

val fourier_tempo_frequencies :
  (float, 'a) Nx.dtype -> ?hop:int -> sample_rate:int -> int -> (float, 'a) Nx.t
(** [fourier_tempo_frequencies dtype ~sample_rate win] is the tempo, in beats
    per minute, of each of the [win / 2 + 1] bins of a [win]-frame
    {!fourier_tempogram}, as a rank-one tensor: bin [k] completes [k] cycles
    over [win] envelope values, [k * 60 * sample_rate / (hop * win)] beats per
    minute, from [0] up. [hop] defaults to [512]. The values are those of
    [librosa.fourier_tempo_frequencies], numpy's [rfftfreq] at the envelope's
    rate in values per minute, rounded to [dtype].

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], or if
    [win < 1]. *)

val tempogram :
     ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> ?norm:Chroma.norm
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [tempogram x] is the autocorrelation tempogram of the onset envelope [x],
    shaped [[...; win_length; frames]], as [librosa.feature.tempogram]
    computes it (Grosche, Müller and Kurth, {e Cyclic tempogram — a mid-level
    tempo representation for music signals}, ICASSP 2010): each frame is the
    autocorrelation, at lags [0] to [win_length - 1], of [win_length] envelope
    values under the periodic [window], normalised. Lag [k] measures how
    strongly the envelope repeats every [k] values; {!tempo_frequencies}
    names its tempo. Leading axes broadcast, so a batch of envelopes is one
    call; the function is rate-agnostic.

    [win_length] defaults to [384] values — 8.9 s at 22050 Hz and a hop of
    512 — [window] to {!Window.Hann} and [norm] to [`Inf], which divides each
    frame by its largest magnitude, the zero lag's. [`P p] divides by the
    frame's [p]-norm and [`None] keeps the raw autocorrelation; a frame whose
    length falls below the smallest normal double is left alone, so a silent
    frame stays zero.

    [alignment] defaults to [`Centered]: frame [t] starts [win_length / 2]
    values before value [t], one frame per envelope value, and the envelope
    is extended at each end by a linear ramp to zero over [win_length / 2]
    values, as librosa pads it with numpy's [linear_ramp]. [`Left] starts
    frame [t] at value [t], without padding, for the [n - win_length + 1]
    frames of an [n]-value envelope.

    The autocorrelation is the direct sum over the frame, each lag adding its
    products in ascending order, where librosa transforms: the two agree to a
    few units in the last place of each frame's zero lag. A frame costs
    [win_length * (win_length + 1) / 2] multiply-adds — some 74 000 at the
    default length — and frames are spread over domains. An empty envelope
    has no frames, [[...; win_length; 0]], where librosa's padding rejects
    it.

    Raises [Invalid_argument] if [win_length < 1], if [norm] is [`P p] with [p]
    not finite and positive, if [x] has rank zero, a dtype other than float32
    and float64 or a non-finite value, or if [alignment] is [`Left] and [x]
    holds fewer than [win_length] values. *)

val tempogram_stage :
     ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> ?norm:Chroma.norm
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [tempogram_stage ()] is {!tempogram} as a pipeline stage: rate 1:1 and
    format-preserving, consuming envelope chunks [[...; frames]] and emitting
    tempogram chunks [[...; win_length; frames]] — compose it after
    {!onset_strength_stage} to stream the tempogram of a signal's envelope
    (without the centred shift of {!onset_strength}, which that stage leaves
    to the caller). Every partitioning of an envelope yields the bits
    {!tempogram} computes on the whole of it.

    A frame is emitted once every envelope value it reads has arrived, and
    the left ramp needs only the first. Under [`Centered] the stage declares
    [win_length - 1 - win_length / 2] frames of latency — 191 at the default
    length, 4.4 s at 22050 Hz and a hop of 512 — and the flush emits the last
    frames of the stream, whose windows read the ramp towards its last value.
    Under [`Left] it declares [win_length - 1] frames, and the flush emits
    nothing but raises [Invalid_argument] if the stream held fewer than
    [win_length] values, as {!tempogram} raises on such an envelope. A push
    that completes no frame emits no chunk. The state is the first value and
    the last [win_length - 1] values of each row.

    Raises [Invalid_argument] for the parameters as {!tempogram} does, when
    the stage is built, and, at the chunk concerned, for a chunk {!tempogram}
    would refuse, one whose leading shape differs from the first chunk's, or
    one pushed after the flush — reset the stream before reusing it. *)

val fourier_tempogram :
     (Complex.t, 'c) Nx.dtype
  -> ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> (float, 'a) Nx.t
  -> (Complex.t, 'c) Nx.t
(** [fourier_tempogram cdtype x] is the Fourier tempogram of the onset
    envelope [x], shaped [[...; win_length / 2 + 1; frames]], as
    [librosa.feature.fourier_tempogram] computes it: the short-time Fourier
    transform of the envelope over [win_length]-value windows advancing one
    value at a time, {!Stft.transform}[ cdtype] on the configuration of
    [fft_size = win_length], [hop = 1], [window] and [alignment] over zero
    padding. Bin [k] measures the envelope's periodicity at the tempo
    {!fourier_tempo_frequencies} names, and its phase where the beats fall.
    [win_length] defaults to [384], [alignment] to [`Centered] and [window] to
    {!Window.Hann}. Centred frames number [n + 1] for an even [win_length]
    and [n] for an odd one, as librosa's do; the transform is computed in
    double precision and rounded into [cdtype]. An envelope shorter than a
    [`Left] frame has no frames, the convention of {!Stft}, where librosa
    rejects it.

    Raises [Invalid_argument] if [win_length < 1], or if [x] has rank zero, a
    dtype other than float32 and float64 or a non-finite value. *)

val fourier_tempogram_stage :
     (Complex.t, 'c) Nx.dtype
  -> ?win_length:int
  -> ?alignment:[`Centered | `Left]
  -> ?window:Window.t
  -> unit
  -> ((float, 'a) Nx.t, (Complex.t, 'c) Nx.t, 'k) Pipeline.t
(** [fourier_tempogram_stage cdtype ()] is {!fourier_tempogram} as a pipeline
    stage: {!Stft.stage}[ cdtype] on the same configuration, which streams the
    transform bit for bit, with [win_length / 2] frames of latency under
    [`Centered] and none under [`Left], and joins chunks as {!Stft.stage}
    does. Chunks are not scanned for non-finite values.

    Raises [Invalid_argument] if [win_length < 1]. *)

val tempo :
     ?hop:int
  -> ?start_bpm:float
  -> ?std_bpm:float
  -> ?ac_size:float
  -> ?tempo_max:float
  -> ?aggregate:[`Mean | `Median | `None]
  -> ?prior:(float -> float)
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [tempo ~sample_rate x] is the tempo of the onset envelope [x], in beats
    per minute, as [librosa.feature.tempo] estimates it. The {!tempogram} of
    [x] over [ac_size] seconds — [ac_size * sample_rate] truncated to whole
    samples, then floor-divided by [hop]: [344] values at the defaults —
    centred, Hann-windowed and normalised to its zero lag, is aggregated over
    frames into [a], and the tempo is that of the lag [k] maximising
    [log1p (1e6 * a k) + log_prior k], read off {!tempo_frequencies}. The
    result is shaped [[...; 1]], one tempo per envelope; under
    [aggregate = `None] every frame is estimated alone and the result is
    [[...; frames]]. Leading axes broadcast.

    [hop] defaults to [512], [start_bpm] to [120.], [std_bpm] to [1.],
    [ac_size] to [8.], [tempo_max] (librosa's [max_tempo]) to [320.] and [aggregate] to [`Mean];
    [`Median] takes numpy's median over frames instead. The prior is
    log-normal in tempo,
    [log_prior k = -0.5 * ((log2 bpm k - log2 start_bpm) / std_bpm) ^ 2], a
    preference for [start_bpm] with [std_bpm] octaves of spread. [prior], a
    log-density over beats per minute, replaces it, as scipy's
    [prior.logpdf] does in librosa: it is evaluated at every bin's tempo, the
    zero lag's [Float.infinity] included, and [start_bpm] and [std_bpm] are
    then validated but not used. Bins faster than [tempo_max] are excluded,
    up to the first slower one; when no bin is slower nothing is excluded, as
    in librosa, and [Float.infinity] excludes nothing at all, librosa's
    [None]. Equal scores go to the faster tempo, and a NaN score wins, as
    numpy's [argmax] decides.

    The result is in the dtype of [x], where librosa's is float64 whatever
    the input; a precomputed tempogram ([tg=]) is not accepted in place of
    the envelope, and a zero [std_bpm], which librosa takes, is refused.

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], if
    [start_bpm], [std_bpm] or [ac_size] is not finite and positive, if
    [ac_size] seconds span less than one hop, if [tempo_max] is not positive,
    if [x] has rank zero, a dtype other than float32 and float64 or a
    non-finite value, or if [x] has no frames and [aggregate] is not
    [`None]. *)

val beat_track :
     ?hop:int
  -> ?start_bpm:float
  -> ?tightness:float
  -> ?trim:bool
  -> ?bpm:float
  -> ?prior:(float -> float)
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> float * int array
(** [beat_track ~sample_rate x] is [(bpm, beats)]: the tempo of the onset
    envelope [x], in beats per minute, and the frames of its beats in
    ascending order, as [librosa.beat.beat_track] tracks them (Ellis,
    {e Beat tracking by dynamic programming}, J. New Music Research 36(1),
    2007). [x] is one envelope along its last axis, every leading axis of
    size [1] (as {!onset_detect_of_envelope} reads it), as librosa's default
    sparse output requires — a batch is tracked row by row, its beats being
    of no common count — and {!Convert.frames_to_time} turns the frames into
    seconds.

    The tempo is [bpm] when given, otherwise {!tempo} of [x] at [hop],
    [start_bpm] and [prior], its other parameters at their defaults; it fixes
    the beat period [p], [sample_rate / hop * 60 / bpm] frames rounded half to
    even. The envelope, divided by its standard deviation, is smoothed into a
    local score by a Gaussian of standard deviation [p / 32] over [p] frames
    each side. The dynamic program then scores, at each frame, the best beat
    sequence ending there: the frame's local score plus the best cumulative
    score of a predecessor between [p / 2] (rounded half to even) and [2 p]
    frames back, less [tightness * (log gap - log p) ^ 2], the latest
    predecessor winning among equal scores; frames before the first whose
    local score reaches a hundredth of the maximum start no sequence. The
    last beat is the last local maximum of the cumulative score that reaches
    half the median of its local maxima, the beats are its backtrace, and
    [trim] drops the beats of the leading and trailing frames whose local
    score does not exceed half the root mean square of the beats' local
    scores smoothed by the 5-point Hann window — without [trim], of those
    whose local score does not exceed zero.

    [hop] defaults to [512], [start_bpm] to [120.], [tightness] to [100.] and
    [trim] to [true]. An envelope of zeros has no beats and tempo [0.], even
    with [bpm] given, as in librosa.

    librosa's arithmetic is followed comparison for comparison, its quirks
    included: the envelope's first value never enters the local score (an
    off-by-one bound in librosa's loop), the tightness is rounded to single
    precision, and the trim threshold averages the smoothed scores over two
    more terms than there are beats where the frames allow. Its trim scans
    run past the ends of the envelope when every local score sits at or below
    the threshold; here they stop there, and such an envelope — a negative
    one, say — keeps no beat. The standard deviation and the means are sums in
    ascending order where numpy's are pairwise, so a decision can differ only
    where two scores agree to their last few bits.

    Deviations, each absent rather than renamed: no dense output
    ([sparse=False]), multichannel or time-varying tempo, and no [units].
    Parameters are validated before an envelope of zeros returns, and
    [start_bpm] even when [bpm] is given, where librosa skips the checks
    then.

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], if
    [start_bpm] or [tightness] is not finite and positive (the tightness in
    single precision too), if [bpm] is given and not finite and positive, if
    [x] has rank zero, a leading axis of size other than [1] or fewer than
    two frames, a dtype other than
    float32 and float64 or a non-finite value, or if the tempo is too fast
    for a beat to span one frame. *)

val plp :
     ?hop:int
  -> ?win_length:int
  -> ?tempo_min:float
  -> ?tempo_max:float
  -> ?prior:(float -> float)
  -> ?kernels:[`Reference | `Unit]
  -> sample_rate:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [plp ~sample_rate x] is the predominant local pulse of the onset envelope
    [x], in its shape and dtype, as [librosa.beat.plp] computes it (Grosche
    and Müller, {e Extracting predominant local pulse information from music
    recordings}, IEEE TASLP 19(6), 2011). Each frame of the
    {!fourier_tempogram} of [x] keeps only its strongest tempo between
    [tempo_min] and [tempo_max] — the coefficients whose [log1p (1e6 * |F|)],
    plus [prior] at their tempo, is the frame's greatest — and is divided by
    a modulus; the least-squares inverse ({!Stft.invert}) of the result,
    clipped at zero and divided by its maximum, is the pulse: one windowed
    sinusoid per frame at that frame's tempo and phase, overlap-added. Its
    peaks are beat candidates that follow a changing tempo. Leading axes
    broadcast.

    [hop] defaults to [512], [win_length] to [384], [tempo_min] to [30.],
    [tempo_max] to [300.] and [kernels] to [`Reference]; [tempo_min = 0.] and
    [tempo_max = Float.infinity] lift the bounds as librosa's [None] does.
    [prior] is a log-density over beats per minute, evaluated at every bin's
    tempo, zero included.

    [kernels] chooses the modulus. [`Reference] is librosa's: the modulus of
    the frame's greatest coefficient in numpy's lexicographic order of complex
    values, real part first. A frame whose strongest coefficient has a
    negative real part therefore finds a zeroed coefficient greater and is
    divided by [sqrt 2^-1022] alone, outweighing every other frame by some
    150 orders of magnitude — the pulse follows those frames. [`Unit]
    divides every frame by its own strongest modulus, kernels of unit
    amplitude: the method as Grosche and Müller publish it, a named deviation
    from librosa.

    An empty envelope gives an empty pulse, where librosa rejects it.

    Raises [Invalid_argument] if [sample_rate] or [hop] is below [1], if
    [win_length < 1], if [tempo_min] is not finite and non-negative, if
    [tempo_max] does not exceed [tempo_min], or if [x] has rank zero, a dtype
    other than float32 and float64 or a non-finite value. *)

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
