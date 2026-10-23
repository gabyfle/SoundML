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

(** Interleaved PCM sample kernels: the conversions between headerless sample
    encodings and planar float32/float64 storage behind
    {!Soundml_io.Raw}, in pure OCaml.

    Every conversion is libsndfile's own, bit for bit, but one: integers
    read normalised ([v * 2^-(bits-1)], unsigned 8-bit centred on 128),
    floats of the storage width move as bits, and writing integers is
    libsndfile's clipping writer ([SFC_SET_CLIPPING]): saturate at full
    scale, otherwise round [x * 2^31] half to even and shift it down to the
    encoding's width (a floor), NaN to zero. The exception is [`S8] from
    float64, which rounds the double as libsndfile's [`U8] and wider writers
    do, where libsndfile's signed 8-bit writer from double rounds through a
    float first and wraps [1 - 2^-26] to -128. *)

(** The type for sample encodings: ffmpeg's raw format names. Integers are
    signed but for [`U8]; [le]/[be] is the byte order; floats are IEEE
    754. *)
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

val name : encoding -> string
(** [name e] is [e]'s ffmpeg format name: ["s16le"] for [`S16le]. *)

val of_name : string -> encoding option
(** [of_name s] is the encoding [s] names, exactly as {!name} spells it. *)

val sample_bytes : encoding -> int
(** [sample_bytes e] is the size of one sample of [e] in bytes. *)

(** The type for staging blocks: a whole number of interleaved frames of
    bytes, as a bigarray that channels read into and write from, and as the
    rig buffer over the same bytes that the kernels read and write through
    typed host-order views. Byte 0 is the bigarray's own first byte, aligned
    for every sample width. *)
type staging = private
  { bytes: (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  ; buffer: Rig.Buffer.t }

val staging : ?frames:int -> frame_bytes:int -> unit -> staging
(** [staging ~frame_bytes ()] is a fresh staging block of {!capacity} bytes:
    the whole [frame_bytes]-byte frames that fit in 64 KiB — the size of an
    OCaml channel's own buffer, so one channel refill fills it — and one
    frame when none fits. [frames] caps it at that many frames (at least
    one), for a conversion that never needs more.

    Raises [Invalid_argument] if [frame_bytes < 1]. *)

val capacity : staging -> int
(** [capacity s] is the size of [s] in bytes, a positive multiple of the
    frame size it was made for. *)

val decode :
     encoding
  -> (float, 'a) Nx.dtype
  -> staging
  -> Rig.Buffer.t
  -> channels:int
  -> frames:int
  -> total:int
  -> off:int
  -> unit
(** [decode e dt src dst ~channels ~frames ~total ~off] converts the [frames]
    interleaved frames of [channels] samples of [e] at byte 0 of [src] into
    frames [off] to [off + frames - 1] of [dst], planar [[channels; total]]
    storage of [dt] elements.

    Raises [Invalid_argument] if [dt] is neither float32 nor float64, or the
    frames lie outside [src] or [dst]. *)

val encode :
     encoding
  -> (float, 'a) Nx.dtype
  -> Rig.Buffer.t
  -> staging
  -> channels:int
  -> frames:int
  -> total:int
  -> off:int
  -> unit
(** [encode e dt src dst ~channels ~frames ~total ~off] converts frames [off]
    to [off + frames - 1] of [src], planar [[channels; total]] storage of
    [dt] elements, into [frames] interleaved frames of [e] at byte 0 of
    [dst]. [src] is only read.

    Raises [Invalid_argument] as {!decode} does. *)
