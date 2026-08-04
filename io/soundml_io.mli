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
    conversion — including load-time conversion — is {!Soundml.Resample}, fed
    at decode time: {!read}[ ~sample_rate] drives the same kernel every other
    face of the library drives, and is bit-identical to decoding at the
    native rate and applying {!Soundml.Resample.apply} afterwards, without
    ever materializing the native-rate signal.

    Defaults never guess: absent [?sample_rate] means the file's native rate,
    absent [?mono] means the file's native channels, absent [?format] on
    {!write} means the format the path's extension names. Preconditions raise
    [Invalid_argument]; runtime I/O failures return [(_, error) result] —
    the result channel carries only genuine I/O outcomes. *)

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
          returned by {!read}; {!Reader.read} surfaces the same condition at
          the failing chunk. *)
  | Io of {path: string; op: string; details: string}
      (** Any other system or libsndfile failure; [op] names the operation
          (["open"], ["read"], ["write"], ["seek"], ["close"]). *)

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
     ?sample_rate:int
  -> ?quality:Soundml.Resample.quality
  -> ?mono:bool
  -> (float, 'a) Nx.dtype
  -> string
  -> ('a audio, error) result
(** [read dt path] decodes the whole file at its native sample rate and
    native channel count — never a silent default rate.

    [?sample_rate] resamples at decode time through
    {!Soundml.Resample.Kernel}: decoded blocks feed the kernel and the
    native-rate signal is never materialized. The result is bit-identical to
    [Resample.apply] over the native read (the delegation law, held by the
    test suite at byte equality), so everything the resampler documents —
    exact output length [ceil (frames * L / M)], compensated group delay,
    quality tiers — holds verbatim here. [?quality] defaults to [`High] and
    is forwarded to [Resample.Config.create]; passing it without
    [?sample_rate] raises [Invalid_argument] — a quality that changes
    nothing is a lie refused at the call site. When [sample_rate] equals the
    native rate the configuration is the identity and the decode path is the
    native one.

    [?mono] (default [false]) downmixes to [[1; frames]] {e before} any
    resampling: sample [i] is the channel mean, summed in channel order in
    the element dtype and multiplied by [1/channels] — numpy's [mean] over
    the channel axis, bit for bit at these channel counts. Downmix is
    per-sample, so it commutes with decode chunking bit-identically.

    [Invalid_argument] on preconditions: [dt] neither float32 nor float64
    ([Resample.Kernel]'s own restriction), [sample_rate < 1], [quality]
    without [sample_rate], an invalid [`Custom] quality spec. Everything the
    filesystem or the file does wrong is a typed [error]: a header that
    advertises more frames than decode delivers is {!Truncated} (never
    zero-filled, never partially returned); an unknown-length stream decodes
    chunked to EOF; a header whose native rate the resampler cannot plan
    against [sample_rate] (a hostile fmt chunk's 2 GHz, say) is {!Io} at
    open — the caller's ask was valid, the file's claim was not. *)

val fold :
     ?block:int
  -> ?sample_rate:int
  -> ?quality:Soundml.Resample.quality
  -> ?mono:bool
  -> (float, 'a) Nx.dtype
  -> string
  -> init:'acc
  -> f:('acc -> (float, 'a) Nx.t -> 'acc)
  -> ('acc, error) result
(** [fold dt path ~init ~f] decodes in blocks of [block] frames (default
    65536; the final block is short) and folds [f] over the chunks — each
    [[channels; n]], borrowed for the duration of the call ([f] must copy
    what it retains; the buffer is reused). [?sample_rate]/[?quality]/[?mono]
    behave exactly as in {!read}; with [sample_rate] the chunks are
    target-rate and the block size counts target frames.

    The property the tests pin: concatenating the chunks equals the
    corresponding {!read} bit-identically, for every block size — so
    [fold ~f:(Stream.push s)] followed by [Stream.flush] is the
    file-to-features streaming story, and it is the same pipeline value the
    offline path ran.

    Raises [Invalid_argument] if [block < 1] (plus {!read}'s
    preconditions). *)

module Reader : sig
  (** Persistent decode handles: open once, read in chunks, seek, close.
      Mutable; single-owner; not domain-safe (the pipeline-kernel
      contract). *)

  (** The type for open readers. *)
  type 'a t

  val open_ :
       ?sample_rate:int
    -> ?quality:Soundml.Resample.quality
    -> ?mono:bool
    -> (float, 'a) Nx.dtype
    -> string
    -> ('a t, error) result
  (** [open_ dt path] opens a reader delivering [[channels; n]] chunks with
      the exact option semantics of {!read} — including the delegation law:
      the concatenation of every chunk of a reader equals the corresponding
      {!read} bit-identically. All buffers (staging, resampler state, output
      carry) are allocated here. *)

  val info : 'a t -> Info.t
  (** [info r] is the {e native} header probe — the file's own rate and
      channels, before [?sample_rate]/[?mono]. *)

  val format : 'a t -> Soundml.Pipeline.Format.t
  (** [format r] is the reader's output as a pipeline source description:
      [Pipeline.Format.audio dt ~sample_rate ~channels] with the {e target}
      rate and post-downmix channel count — it plugs straight into
      [Pipeline.run ~source] and [Pipeline.Stream.prepare ~source]. Upstream
      latency is zero: the reader's output is already delay-compensated
      (the resampler's group delay is compensated by contract). *)

  val read :
       ?out:(float, 'a) Nx.t
    -> 'a t
    -> frames:int
    -> ((float, 'a) Nx.t option, error) result
  (** [read r ~frames] is the next chunk: [Ok (Some c)] with [c] shaped
      [[channels; n]], [n = frames] except for the short final chunk;
      [Ok None] at EOF, and at EOF forever after. For resampled readers,
      [frames] counts target-rate frames and the reader's internal carry
      absorbs the kernel's emission cadence (including the block bursts of
      FFT-executed resampler plans), so chunk sizes are exact regardless of
      the plan.

      [?out] lends a destination: a C-contiguous [[channels; frames]] tensor
      of the reader's dtype. The call writes into it and returns a view of
      its first [n] frames; [out] is borrowed for the duration of the call
      only — the reader retains no reference — and the natively-rated mono
      path decodes {e directly} into it (zero copies). [?out] precedes the
      positional argument so it is erasable. Without [out] each call
      allocates a fresh chunk.

      Decode failures are typed: a truncated file yields
      [Error (Truncated _)] at the failing chunk, after which the reader is
      positioned at EOF.

      Raises [Invalid_argument] if [frames < 1], if [out] has the wrong
      shape or is not C-contiguous, or if [r] is closed. *)

  val seek : 'a t -> frame:int -> (unit, error) result
  (** [seek r ~frame] positions [r] so the next {!read} delivers frame
      [frame] of the reader's {e own} grid — target-rate, post-resample.
      Reading after a seek is bit-identical to the same frames of an
      uninterrupted read: a native reader re-positions the decoder directly;
      a resampled reader re-primes by replaying the source through the
      kernel from the stream origin — forward seeks advance incrementally
      from the current position, backward seeks re-decode from frame zero —
      so the delivered samples are {e the} uninterrupted stream's samples,
      not a re-priming approximation. Seeking past the end of a stream of
      known length, or on an unseekable stream, yields [Error (Io _)].

      Raises [Invalid_argument] if [frame < 0] or [r] is closed. *)

  val close : 'a t -> unit
  (** [close r] releases the handle; idempotent. Reading or seeking a closed
      reader raises [Invalid_argument]. A GC finalizer closes leaked handles
      as a backstop; rely on it never — file descriptors are a bounded
      resource. *)
end

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

    One upstream asymmetry, pinned rather than papered over: a 0-frame FLAC
    write succeeds but produces a 0-byte file that no decoder recognizes —
    libsndfile 1.2.2 emits nothing for an empty FLAC stream, and
    python-soundfile writes the identical 0-byte file. Every other
    container round-trips empty data exactly.

    Raises [Invalid_argument] if [a.sample_rate < 1], [a.data] has rank 0
    or rank > 2, or its dtype is neither float32 nor float64. Everything
    else is typed: {!Unsupported} for encoder refusals ([sf_format_check]),
    {!Not_found} for a missing directory, {!Io} for the rest. *)
