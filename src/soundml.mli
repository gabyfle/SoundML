(** SoundML — sound processing on Raven tensors.

    The library is being rebuilt: this module currently exposes nothing but the
    distribution version. Audio-file I/O and Rubber Band effects ship separately
    as [soundml-io] and [soundml-rubberband]. *)

val version : string
(** [version] is the version of the SoundML distribution this library was built
    from. *)
