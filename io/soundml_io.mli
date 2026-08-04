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

(** Audio-file input and output.

    [Soundml_io] decodes and encodes audio through libsndfile. Sample-rate
    conversion is {!Soundml.Resample}: decode at the file's native rate, then
    [Resample.apply] — the decoder adds no numeric of its own.

    Defaults never guess: {!read} decodes at the file's native rate and
    native channel count, absent [?format] on {!write} means the format the
    path's extension names. Preconditions raise [Invalid_argument]; runtime
    I/O failures return [(_, error) result] — the result channel carries
    only genuine I/O outcomes. *)

(** The type for decoded audio: the ecosystem currency plus the one fact a
    file adds. [data] is channel-first [[channels; frames]], C-contiguous,
    fresh and caller-owned; a mono file decodes to [[1; frames]]
    ([Nx.squeeze] gives the rank-one form the core equally accepts). *)
type 'a audio = {data: (float, 'a) Nx.t; sample_rate: int}

(** {1 Errors} *)

(** The type for I/O failures. Every constructor is reachable from an actual
    filesystem or file condition and is exercised by the test suite. *)
type error =
  | Not_found of {path: string}
      (** The path does not exist (or, on {!write}, its directory). *)
  | Permission_denied of {path: string}
  | Unrecognized_format of {path: string; details: string}
      (** libsndfile cannot identify the container: empty files, garbage,
          unknown extensions on {!Format.of_path}. [details] is libsndfile's
          message. *)
  | Unsupported of {path: string; details: string}
      (** The container/encoding/channel-count combination is valid to ask
          for but not encodable ([sf_format_check] refusal — e.g. nine
          channels into FLAC). *)
  | Truncated of {path: string; expected_frames: int; read_frames: int}
      (** The header advertised [expected_frames] but decode delivered only
          [read_frames]. Data is never silently zero-filled or partially
          returned by {!read}. *)
  | Io of {path: string; op: string; details: string}
      (** Any other system or libsndfile failure; [op] names the operation
          (["open"], ["read"], ["write"], ["close"]). *)

val pp_error : Stdlib.Format.formatter -> error -> unit
(** [pp_error fmt e] prints [e] in the house message format: lowercase,
    actionable, naming paths and numbers — e.g.
    [soundml-io: "clip.flac" is truncated (header advertises 22050 frames,
    decode delivered 0)]. *)

(** {1 Formats} *)

module Format : sig
  (** The type for containers on the write surface. Reading is not limited
      to this list — libsndfile decodes whatever it was built with — but
      writing and {!Info.format} round-trip through it. *)
  type container = [`Wav | `Aiff | `Flac | `Ogg | `Caf]

  (** The type for sample encodings. [`Vorbis] is valid only in [`Ogg]. *)
  type encoding = [`Pcm_16 | `Pcm_24 | `Pcm_32 | `Float32 | `Float64 | `Vorbis]

  (** The type for validated container/encoding pairs. *)
  type t

  val create : ?encoding:encoding -> container -> t
  (** [create c] is the format writing container [c] with its conventional
      encoding: [`Pcm_16] for [`Wav]/[`Aiff]/[`Flac]/[`Caf]
      (python-soundfile's defaults), [`Vorbis] for [`Ogg].

      Raises [Invalid_argument] if the pair is statically invalid: [`Vorbis]
      outside [`Ogg]; anything but [`Vorbis] inside [`Ogg]; [`Pcm_32],
      [`Float32] or [`Float64] inside [`Flac] (FLAC is 8/16/24-bit integer).
      Dynamic refusals (channel counts, rates) surface as {!Unsupported} at
      {!write}, where they are knowable. *)

  val of_path : string -> (t, error) result
  (** [of_path p] is the format [p]'s extension names ([.wav],
      [.aiff]/[.aif], [.flac], [.ogg]/[.oga], [.caf], case-insensitive),
      with the default encoding — {!Unrecognized_format} for anything
      else. *)

  val container : t -> container

  val encoding : t -> encoding

  val pp : Stdlib.Format.formatter -> t -> unit

  val equal : t -> t -> bool
end

module Info : sig
  (** The type for header probes. Transparent, like {!audio}. [frames] is
      [0] when the container cannot say (observed: a truncated Ogg stream);
      {!read} handles that case by decoding to EOF rather than trusting it.
      [format] is the write-surface description when the file's actual
      major/subtype pair is expressible there, [None] otherwise (an MP3, an
      ALAC CAF, …); [format_name] is libsndfile's human-readable name,
      always present. *)
  type t =
    { frames: int
    ; channels: int
    ; sample_rate: int
    ; format: Format.t option
    ; format_name: string }

  val pp : Stdlib.Format.formatter -> t -> unit
end

val info : string -> (Info.t, error) result
(** [info path] is the header-only probe: frames, channels, native rate,
    encoding — no decode. Measured cost class: one open/close, ~20 µs for
    WAV/FLAC, ~170–250 µs for Ogg. *)

(** {1 Reading} *)

val read :
  ?mono:bool -> (float, 'a) Nx.dtype -> string -> ('a audio, error) result
(** [read dt path] decodes the whole file at its native sample rate and
    native channel count — never a silent default rate.

    [?mono] (default [false]) downmixes to [[1; frames]]: sample [i] is the
    channel mean, summed in channel order in the element dtype and
    multiplied by [1/channels] — numpy's [mean] over the channel axis, bit
    for bit at these channel counts. Downmix is per-sample, so it commutes
    with decode chunking bit-identically.

    [Invalid_argument] on preconditions: [dt] neither float32 nor float64.
    Everything the filesystem or the file does wrong is a typed [error]: a
    header that advertises more frames than decode delivers is {!Truncated}
    (never zero-filled, never partially returned); an unknown-length stream
    decodes chunked to EOF. *)

(** {1 Writing} *)

val write : ?format:Format.t -> string -> 'a audio -> (unit, error) result
(** [write path a] encodes [a] to [path]. [format] defaults to
    [Format.of_path path]. [a.data] is [[channels; frames]] or rank-one
    (mono); any strided layout is accepted (one compaction copy when not
    contiguous, none when it is).

    Clipping policy, pinned: when encoding float data to a PCM subtype the
    handle is put in libsndfile's clipping mode ([SFC_SET_CLIPPING]), so
    out-of-range samples saturate at full scale — python-soundfile's
    behavior, and the only sane one; libsndfile's default would wrap.
    Float subtypes store values verbatim, including NaN and infinities.

    The writer hands libsndfile at most 65536 frames per [sf_writef] call,
    unconditionally: a single-call multi-megaframe Ogg/Vorbis write
    segfaults libsndfile 1.2.2 (measured threshold 2.2 M frames), and a
    user handing us a five-minute buffer must not be able to crash the
    process.

    Durability: [write] returns after [sf_close] (libsndfile buffers
    flushed to the OS); it does not fsync — the same close-to-close
    contract python-soundfile has.

    Raises [Invalid_argument] if [a.sample_rate < 1], [a.data] has rank 0
    or rank > 2, or its dtype is neither float32 nor float64. Everything
    else is typed: {!Unsupported} for encoder refusals ([sf_format_check]),
    {!Not_found} for a missing directory, {!Io} for the rest. *)
