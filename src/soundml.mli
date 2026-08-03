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

    Audio-file I/O and Rubber Band effects ship separately as [soundml-io] and
    [soundml-rubberband]. *)

(** Streaming and offline chunk pipelines: one pipeline value drives both the
    offline driver {!Pipeline.run} and the online driver {!Pipeline.Stream}. *)
module Pipeline = Pipeline

(** Window-function specifications and their instantiation. *)
module Window = Window

(** The short-time Fourier transform: offline {!Stft.transform}, the
    incremental {!Stft.Kernel} and the {!Stft.stage}/{!Stft.power_stage}
    pipeline stages, all driving one frame grid. *)
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
    {!Convert.power_to_db} of [mel_spectrogram stft mel x] with librosa's
    [mfcc] settings: reference [1.], floor [1e-10], and the 80 dB
    dynamic-range clamp under the global maximum that [librosa.feature.mfcc]
    inherits from its [power_to_db] call.

    The DCT is orthonormal exactly as scipy's [dct norm='ortho']: the raw
    type-II transform scaled by [1 / sqrt (4 * n_mels)] on coefficient zero
    and [1 / sqrt (2 * n_mels)] elsewhere. [lifter], when positive, applies
    sinusoidal liftering — coefficient [k], counted from zero, is multiplied
    by [1 + (lifter / 2) * sin (pi * (k + 1) / lifter)] (librosa) — and [0.]
    or absent applies none. The log-mel scaling, the DCT and the liftering
    are computed in double precision and rounded to the dtype of [x] once,
    at the boundary; the mel spectrogram beneath carries its own boundary
    roundings, one per operation, exactly as {!mel_spectrogram}.

    Raises [Invalid_argument] if the configurations disagree on [fft_size],
    if [n_mfcc] does not lie in [\[1, n_mels\]], if [lifter] is not finite
    and non-negative, or if [x] has rank zero. *)

val version : string
(** [version] is the version of the SoundML distribution this library was built
    from. *)
