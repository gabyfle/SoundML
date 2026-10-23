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

    [Soundml_io] decodes and encodes audio through libsndfile, from and to a
    path or bytes held in memory ({!Memory}: the same faces). Sample-rate
    conversion — including load-time conversion — is {!Soundml.Resample}, fed
    at decode time: {!read}[ ~sample_rate] drives the same kernel every other
    face of the library drives, and is bit-identical to decoding at the
    native rate and applying {!Soundml.Resample.apply} afterwards, without
    ever materializing the native-rate signal. Headerless PCM byte streams
    ({!Raw}) decode and encode in OCaml, bit-identical to {!read} and
    {!write} on the same encodings.

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
    filesystem or file condition and is exercised by the test suite. [path]
    is the path the face was given, or ["<memory>"] for the {!Memory}
    faces. *)
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
      encoding: [`Pcm_16] for [`Wav]/[`Aiff]/[`Flac]/[`Caf], [`Vorbis] for
      [`Ogg].

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
    quality tiers — holds verbatim here. The conversion's configuration is
    kept for later reads at the same rates and quality, as
    [Soundml.resample] keeps its own, so a corpus read at one rate designs it
    once. [?quality] defaults to [`High] and is forwarded to
    [Resample.Config.create]; passing it without
    [?sample_rate] raises [Invalid_argument] — a quality that changes
    nothing is a lie refused at the call site. When [sample_rate] equals the
    native rate the configuration is the identity and the decode path is the
    native one.

    [?mono] (default [false]) downmixes to [[1; frames]] {e before} any
    resampling: sample [i] is the channel mean, summed in channel order in
    the element dtype and multiplied by [1/channels] — one fixed summation
    order, so the downmix is reproducible bit for bit. Downmix is
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

  val read : 'a t -> frames:int -> ((float, 'a) Nx.t option, error) result
  (** [read r ~frames] is the next chunk: [Ok (Some c)] with [c] shaped
      [[channels; n]], [n = frames] except for the short final chunk;
      [Ok None] at EOF, and at EOF forever after. For resampled readers,
      [frames] counts target-rate frames and the reader's internal carry
      absorbs the kernel's emission cadence (including the block bursts of
      FFT-executed resampler plans), so chunk sizes are exact regardless of
      the plan.

      Each call allocates a fresh chunk, caller-owned: the reader retains no
      reference to it, and the natively-rated mono path decodes {e directly}
      into it (zero copies).

      Decode failures are typed: a truncated file yields
      [Error (Truncated _)] at the failing chunk, after which the reader is
      positioned at EOF.

      Raises [Invalid_argument] if [frames < 1] or if [r] is closed. *)

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
    flushed to the OS); it does not fsync — durability past the OS cache is
    the caller's concern.

    One upstream asymmetry, pinned rather than papered over: a 0-frame FLAC
    write succeeds but produces a 0-byte file that no decoder recognizes —
    libsndfile 1.2.2 emits nothing for an empty FLAC stream. Every other
    container round-trips empty data exactly.

    Raises [Invalid_argument] if [a.sample_rate < 1], [a.data] has rank 0
    or rank > 2, or its dtype is neither float32 nor float64. Everything
    else is typed: {!Unsupported} for encoder refusals ([sf_format_check],
    and the Vorbis bounds it misses: more than 255 channels or a rate above
    200 kHz), {!Not_found} for a missing directory, {!Io} for the rest. *)

(** {1 Headerless PCM} *)

module Raw : sig
  (** Headerless interleaved PCM — the byte streams of [sox -t raw] and
      [ffmpeg -f s16le] — decoded and encoded in OCaml, without libsndfile.

      Nothing in the bytes says how to read them, so nothing here guesses:
      the caller names the encoding and the channel count. The sample rate,
      which no conversion needs, stays the caller's: the faces take and
      return bare [[channels; frames]] tensors, and
      [Pipeline.Format.audio dt ~sample_rate ~channels] describes them to a
      pipeline.

      Samples convert exactly as {!Soundml_io.read} and {!Soundml_io.write}
      convert the same encoding inside a container, bit for bit: decoding a
      WAV file's data chunk is {!Soundml_io.read} of the file, and encoding
      a tensor yields the bytes {!Soundml_io.write} puts in the data chunk.
      {ul
      {- Integers read normalised, [v * 2^-(bits - 1)] ([`U8] centred on
         128), rounded once into float32.}
      {- Integers write through libsndfile's clipping writer: a sample at or
         past full scale saturates; any other is [x * 2^31] rounded half to
         even, then shifted down to the encoding's width — a floor, so 0.75
         LSB writes 0 and -0.5 LSB writes -1. NaN writes 0 (128 in [`U8]).
         The 8-bit encodings, which {!Soundml_io.write} does not offer,
         follow the same rule — which departs from libsndfile once: its
         [`S8] writer from double rounds [x * 2^31] through a float first,
         and so writes [1 - 2^-26] as -128, where [`S8] from float64 here
         rounds the double, as libsndfile's [`U8] and wider writers do.}
      {- A float of the tensor's own width moves as bits, NaN payloads and
         signalling NaNs included; a width change is C's conversion, round
         to nearest.}}

      Preconditions raise [Invalid_argument]; what a channel does is a typed
      {!Io}, whose [path] is the face's [?name] (default ["<channel>"]) —
      a channel failure, and a stream that ends inside a frame. No face
      guesses what a partial frame held: where libsndfile's RAW reader, sox
      and ffmpeg drop it silently, here it is an error, after every complete
      frame ({!Reader}) or instead of the data ({!val-read}). *)

  (** The type for sample encodings, named after ffmpeg's raw formats
      ([-f s16le]): integers are signed but for [`U8], 24-bit ones packed in
      three bytes; [le]/[be] is the byte order; floats are IEEE 754 binary32
      and binary64. *)
  type encoding =
    [ `S8
    | `U8
    | `S16le
    | `S16be
    | `S24le
    | `S24be
    | `S32le
    | `S32be
    | `F32le
    | `F32be
    | `F64le
    | `F64be ]

  val encoding_of_name : string -> encoding option
  (** [encoding_of_name s] is the encoding ffmpeg's format name [s] denotes
      — ["s16le"] is [`S16le] — and [None] for anything else, other
      spellings included. *)

  val encoding_name : encoding -> string
  (** [encoding_name e] is [e]'s ffmpeg format name. *)

  val sample_bytes : encoding -> int
  (** [sample_bytes e] is the size of one sample of [e] in bytes: 1, 2, 3, 4
      or 8. A frame of [channels] samples is [channels * sample_bytes e]
      bytes. *)

  val decode :
       encoding
    -> channels:int
    -> (float, 'a) Nx.dtype
    -> string
    -> (float, 'a) Nx.t
  (** [decode e ~channels dt s] is the [[channels; frames]] tensor of the
      interleaved frames [s] holds — fresh, C-contiguous, [frames] the
      length of [s] over the frame size. The empty string decodes to
      [[channels; 0]]. A [bytes] buffer nothing writes any more decodes
      through [Bytes.unsafe_to_string]: [decode] keeps no reference to its
      argument.

      Raises [Invalid_argument] if [channels < 1], or so many that the size
      of a frame overflows, if [dt] is neither float32 nor float64, or if
      [s] does not hold a whole number of frames — the length of a string in
      hand is the caller's to check, where the end of a stream is a runtime
      outcome ({!val-read}). *)

  val read :
       ?name:string
    -> encoding
    -> channels:int
    -> (float, 'a) Nx.dtype
    -> In_channel.t
    -> ((float, 'a) Nx.t, error) result
  (** [read e ~channels dt ic] decodes all that [ic] holds up to end of
      file, as {!decode} decodes a string. [ic] is put in binary mode and
      left open, at end of file.

      A stream that ends inside a frame is [Error (Io {op = "read"; _})]
      naming the byte counts, and no sample is returned — as
      {!Soundml_io.read} never returns part of a truncated file; {!Reader}
      delivers the complete frames first. A failure of the channel
      ([Sys_error], or [Sys_blocked_io] from a non-blocking descriptor) is
      [Error (Io {op = "read"; _})] as well.

      Raises [Invalid_argument] if [channels < 1], or so many that the size
      of a frame overflows, or if [dt] is neither float32 nor float64. *)

  module Reader : sig
    (** Chunked decoding of a channel — a microphone pipe, a socket, a file
        too large to hold. Mutable; single-owner; not domain-safe (the
        pipeline-kernel contract). The caller owns the channel and closes
        it.

        It slots straight into {!Soundml.Pipeline.Stream}, chunks of at most
        [max_chunk] frames being exactly what {!read} delivers. Mel frames
        off a microphone, captured by either of
        {v
sox -d -t raw -r 16000 -c 1 -e floating-point -b 32 -L - | ./kws
ffmpeg -f avfoundation -i :0 -ac 1 -ar 16000 -f f32le - | ./kws
        v}
        ([-L] forces sox's little-endian output; it defaults to the host's
        order), where [consume] takes each [[1; 40; n]] chunk of mel
        frames:
        {[
          let () =
            let open Soundml in
            let sample_rate = 16000 and max_chunk = 1024 in
            let stft = Stft.Config.create ~hop:160 ~fft_size:512 () in
            let mel = Mel.Config.create ~n_mels:40 ~sample_rate ~fft_size:512 () in
            let s =
              Pipeline.Stream.prepare
                Pipeline.(Stft.power_stage stft >> Mel.stage mel)
                ~source:(Pipeline.Format.audio Nx.float32 ~sample_rate ~channels:1)
                ~max_chunk
            in
            let r = Soundml_io.Raw.Reader.create `F32le ~channels:1 Nx.float32 stdin in
            let rec loop () =
              match Soundml_io.Raw.Reader.read r ~frames:max_chunk with
              | Ok (Some chunk) ->
                  Option.iter consume (Pipeline.Stream.push s chunk) ;
                  loop ()
              | Ok None ->
                  List.iter consume (Pipeline.Stream.flush s)
              | Error e ->
                  Stdlib.Format.eprintf "%a@." Soundml_io.pp_error e
            in
            loop ()
        ]}
        The test suite holds this loop, over a file, equal bit for bit to
        [Pipeline.run] on the whole signal. *)

    (** The type for readers. *)
    type 'a t

    val create :
         ?name:string
      -> encoding
      -> channels:int
      -> (float, 'a) Nx.dtype
      -> In_channel.t
      -> 'a t
    (** [create e ~channels dt ic] is a reader of frames of [e] from [ic],
        which it puts in binary mode. It allocates its one staging block —
        the whole frames that fit in 64 KiB, or one frame when a frame is
        larger — and reads nothing.

        Raises [Invalid_argument] if [channels < 1], or so many that the
        size of a frame overflows, or if [dt] is neither float32 nor
        float64. *)

    val read : 'a t -> frames:int -> ((float, 'a) Nx.t option, error) result
    (** [read r ~frames] is the next chunk: [Ok (Some c)] with [c] fresh,
        C-contiguous and caller-owned, shaped [[channels; n]] with
        [1 <= n <= frames]. It blocks until [frames] frames have arrived or
        the stream ends, so [n = frames] but for the last chunk before end
        of file or an error, and the chunks depend on the bytes alone, never
        on how the system splits the reads: a pipe delivering a byte at a
        time yields the chunks a file does.

        [r] never reads past the frames it owes: after a chunk of [frames]
        frames, [ic] stands at the frame boundary that ends it and [r] holds
        no byte. [Ok None] is end of file, and is end of file forever after,
        without another read of [ic].

        Failures come after data. A stream that ends inside a frame, or a
        failure of the channel — [Sys_error], or [Sys_blocked_io] from a
        non-blocking descriptor with nothing ready, which a reader that
        blocks cannot wait out — is [Error (Io {op = "read"; _})], reported
        once, after the chunk of every complete frame before it; [r] is at
        end of file from then on. So is it after an exception that escapes
        [read] — one a signal handler raises during a blocking read, such as
        [Sys.Break] — which propagates: the bytes that call had consumed are
        lost, and as [ic] may then stand inside a frame, where no later read
        could find the next frame's first byte, [r] reads no more.

        Each call allocates its whole chunk of [frames] frames before it
        reads, and copies a short last chunk to a buffer of its own size:
        [frames] sizes an allocation, and one too large for the host raises
        rig's [Out_of_memory].

        Raises [Invalid_argument] if [frames < 1], or if the size of a chunk
        of [frames] frames overflows. *)
  end

  val encode : encoding -> (float, 'a) Nx.t -> string
  (** [encode e x] is the interleaved frames of [x], [[channels; frames]] or
      rank-one (mono) in any strided layout (one compaction copy when not
      contiguous, none when it is): [frames * channels * sample_bytes e]
      bytes.

      Raises [Invalid_argument] if [x] has rank 0 or rank > 2, no channel or
      so many that the size of a frame overflows, or a dtype neither float32
      nor float64. *)

  val write :
       ?name:string
    -> encoding
    -> Out_channel.t
    -> (float, 'a) Nx.t
    -> (unit, error) result
  (** [write e oc x] writes [encode e x] to [oc], which it puts in binary
      mode, a staging block at a time — the encoded stream never exists
      whole — and then flushes [oc], so that every failure of the channel
      ([Sys_error], [Sys_blocked_io]) lands in the result:
      [Error (Io {op = "write"; _})]. [oc] stays open.
      Writing to a pipe whose reader has exited raises [SIGPIPE], which
      kills a program that does not ignore it; one that does
      ([Sys.set_signal Sys.sigpipe Sys.Signal_ignore]) gets the error.

      Raises [Invalid_argument] as {!encode} does. *)
end

(** {1 In memory} *)

(** The faces above over encoded bytes instead of a path — an HTTP body, an
    archive member, the result of {!Memory.write}.

    Each function is its path counterpart, option for option and law for
    law: decoding a file's bytes from memory is bit-identical to decoding
    the file from its path, through every face, and {!Memory.write}
    produces the bytes {!write} puts in a file. Preconditions raise the same
    [Invalid_argument]s, named after the memory face.

    One exception, libsndfile's: bytes have no name. When a stream's header
    names no container, libsndfile falls back on the file — its extension
    names headerless μ-law ([.au], [.snd]), VOX ADPCM ([.vox], [.vox8],
    [.vox6]), GSM 6.10 ([.gsm]) and an MP3 stream that does not open on a
    frame sync after any ID3v2 tag ([.mp3]), and a resource fork makes it
    SD2 — and from memory there is no file: these streams are
    {!Unrecognized_format}. The resource-fork probe still runs, and from
    memory it looks in the process's current directory, for [._] and
    [.AppleDouble/]: where either exists, such bytes are {!Io} (["open"])
    instead, and so is an MP3 without an ID3v2 tag, whose frame sync
    libsndfile checks only after the probe. A stream whose header names its
    container — every format {!write} produces among them — meets neither
    fallback.

    The input is a [string], copied once at open into memory the decoder
    owns and frees. The copy is not optional: the decoder runs with the
    runtime lock released, when the GC is free to move an OCaml string.
    Nothing reads the caller's string after the call returns, so it may die
    or change from then on, and a [bytes] buffer is passed safely as
    [Bytes.unsafe_to_string b]. The cost is the copy — measured ~17 µs per
    MB on warm memory, {!Memory.info} included, where the path probe reads
    a header only — and one encoded size of memory per open source, held
    by a reader until it is closed. A reader's copy counts toward the GC's
    pace, so one left to the finalizer is reclaimed under the pressure it
    exerts; the faces that close before returning charge the GC nothing.

    Errors keep their constructors and carry the path ["<memory>"].
    {!Not_found} and {!Permission_denied} name filesystem conditions and are
    unreachable; every other outcome is the path face's verdict on the same
    bytes, but for the nameless fallbacks above. *)
module Memory : sig
  val info : string -> (Info.t, error) result
  (** [info bytes] is {!Soundml_io.info} over [bytes], short of the
      fallbacks only a file has (above): bytes libsndfile identifies by a
      file's extension or resource fork are {!Unrecognized_format} here. *)

  val read :
       ?sample_rate:int
    -> ?quality:Soundml.Resample.quality
    -> ?mono:bool
    -> (float, 'a) Nx.dtype
    -> string
    -> ('a audio, error) result
  (** [read dt bytes] is {!Soundml_io.read} over [bytes]: the same decode,
      resampling and downmix, and the same parallel decode of large
      uncompressed streams. The allocation guard weighs a header's claim
      against [String.length bytes]. As for {!info}, bytes libsndfile
      identifies by a file's extension or resource fork (above) are
      {!Unrecognized_format} here — a headerless [.vox] decodes from its
      path and not from memory. *)

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
  (** [fold dt bytes ~init ~f] is {!Soundml_io.fold} over [bytes]. *)

  val open_reader :
       ?sample_rate:int
    -> ?quality:Soundml.Resample.quality
    -> ?mono:bool
    -> (float, 'a) Nx.dtype
    -> string
    -> ('a Reader.t, error) result
  (** [open_reader dt bytes] is {!Reader.open_} over [bytes]: an ordinary
      reader — {!Reader.read}, {!Reader.seek}, {!Reader.close} and the rest
      as documented there — decoding the copy taken at open, never the
      caller's string. *)

  val write : format:Format.t -> 'a audio -> (string, error) result
  (** [write ~format a] is the stream {!Soundml_io.write} encodes, as a
      string. [format] is required: there is no extension to default from.
      The input forms, clipping policy, call cap and refusals are
      {!Soundml_io.write}'s own, and so is its upstream asymmetry: a 0-frame
      FLAC encodes to [""], which {!read} reports as
      {!Unrecognized_format}.

      The bytes are those of the file {!Soundml_io.write} produces, with
      one exception: libsndfile draws an Ogg stream's serial number at
      random, so two Ogg/Vorbis encodes of the same audio differ in those
      bytes — and decode identically.

      The stream grows in memory while it is encoded — to at most twice its
      final size, or 64 KB if that is larger — and is copied once into the
      result. A stream the allocator cannot grow is {!Io} (["write"] or
      ["close"]), never a truncated [Ok]. *)
end
