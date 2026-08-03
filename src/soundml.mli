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

val version : string
(** [version] is the version of the SoundML distribution this library was built
    from. *)
