(** Digital signal processing on Raven tensors.

    {1:conventions Conventions}

    The time axis is always the last axis. Audio is [[channels; frames]]; a
    rank-one tensor is mono audio.

    Functions return fresh tensors unless documented {e view}. Nothing copies
    defensively; nothing mutates its input.

    Sample rates are explicit arguments on the functions whose result depends
    on one; rate-agnostic functions do not take it.

    Preconditions raise [Invalid_argument]; no [result] values in numeric
    paths.

    Incremental processing: every stateful algorithm is a Mealy kernel
    ([prepare]/[step]/[flush]/[reset]) composable as a {!Pipeline} stage.
    Offline functions are the one-chunk instance of the same kernels; the two
    cannot disagree. See {!Pipeline}.

    Audio-file I/O and Rubber Band effects ship separately as [soundml-io] and
    [soundml-rubberband]. *)

(** Streaming and offline chunk pipelines: one pipeline value drives both the
    offline driver {!Pipeline.run} and the online driver {!Pipeline.Stream}. *)
module Pipeline = Pipeline

val version : string
(** [version] is the version of the SoundML distribution this library was built
    from. *)
