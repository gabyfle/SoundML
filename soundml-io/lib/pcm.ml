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

(* The kernels are written once per pair of source and destination storage, so
   that every element access compiles to a plain load or store and every float
   and int32/int64 stays unboxed: without flambda, an element function passed to
   a shared loop is a call that boxes its result (soundml/lib/loops.ml); the
   per-sample conversions are [@inline] functions instead. Each loop runs
   channel-outer: it writes one plane of the planar side sequentially while its
   accesses to the interleaved side stride by [channels] inside a staging block
   small enough to stay in cache — measured against frame-outer order, faster
   from two channels up and within a few percent on mono. Each plane's loop is
   unrolled by four, which OCaml does not do on its own: measured, 1.3x (24-bit)
   to 1.8x (16-bit) the plain loop, decoding and encoding alike.

   The staging bytes are read and written through rig's typed views, which are
   in the host's byte order. An encoding in the other order runs the same view
   plus a byte swap of each element, so the swapped kernels a little-endian host
   runs for big-endian encodings are the very code a big-endian host runs for
   little-endian ones. 24-bit samples are assembled byte by byte, in either
   order.

   Integer samples go through OCaml's 63-bit [int] (every OCaml 5 native target
   is 64-bit).

   Bounds: [check] establishes, once per call, that [channels >= 1], [off >= 0],
   [off + frames <= total], that the interleaved side holds [frames * channels]
   samples and the planar side [channels * total] elements. A kernel's running
   index [j] is the interleaved sample [f * channels + c] of frame [f] (its
   first byte [3 * (f * channels + c)] for 24-bit, read or written up to [+ 2]),
   below [frames * channels] for [f < frames] and [c < channels]; the planar
   element is [c * total + off + f], below [channels * total]. The unrolled step
   touches frames [f] to [f + 3] only while [f + 3 < frames], and the tail loop
   the rest: the unsafe accesses stay inside the views. *)

module A1 = Bigarray.Array1

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

let name = function
  | `S8 ->
      "s8"
  | `U8 ->
      "u8"
  | `S16le ->
      "s16le"
  | `S16be ->
      "s16be"
  | `S24le ->
      "s24le"
  | `S24be ->
      "s24be"
  | `S32le ->
      "s32le"
  | `S32be ->
      "s32be"
  | `F32le ->
      "f32le"
  | `F32be ->
      "f32be"
  | `F64le ->
      "f64le"
  | `F64be ->
      "f64be"

let all : encoding list =
  [ `S8
  ; `U8
  ; `S16le
  ; `S16be
  ; `S24le
  ; `S24be
  ; `S32le
  ; `S32be
  ; `F32le
  ; `F32be
  ; `F64le
  ; `F64be ]

let of_name s = List.find_opt (fun e -> String.equal (name e) s) all

let sample_bytes = function
  | `S8 | `U8 ->
      1
  | `S16le | `S16be ->
      2
  | `S24le | `S24be ->
      3
  | `S32le | `S32be | `F32le | `F32be ->
      4
  | `F64le | `F64be ->
      8

let big_endian = function
  | `S16be | `S24be | `S32be | `F32be | `F64be ->
      true
  | `S8 | `U8 | `S16le | `S24le | `S32le | `F32le | `F64le ->
      false

(* The encoding's byte order is the host's: typed views read it as it lies. *)
let host_order e = big_endian e = Sys.big_endian

(* {1 Staging}

   A plain bigarray, so that [In_channel.input_bigarray] and
   [Out_channel.output_bigarray] move bytes in and out of it without a claim,
   wrapped once for rig's typed views. Its first byte is the malloc'd block's,
   aligned for every sample width, and kernels always start there. *)

type staging =
  { bytes: (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A1.t
  ; buffer: Rig.Buffer.t }

(* An OCaml channel's buffer (IO_BUFFER_SIZE): one refill fills a block. *)
let staging_bytes = 65536

let staging ?frames ~frame_bytes () =
  if frame_bytes < 1 then
    invalid_arg
      (Printf.sprintf "Pcm.staging: %d-byte frames (a frame holds a sample)"
         frame_bytes ) ;
  let fit = Int.max 1 (staging_bytes / frame_bytes) in
  let fit =
    match frames with Some n -> Int.max 1 (Int.min n fit) | None -> fit
  in
  let capacity = fit * frame_bytes in
  let bytes = A1.create Bigarray.int8_unsigned Bigarray.c_layout capacity in
  {bytes; buffer= Rig.Buffer.of_bigarray bytes}

let capacity s = A1.dim s.bytes

(* {1 Element helpers} *)

type f32 = (float, Bigarray.float32_elt, Bigarray.c_layout) A1.t

type f64 = (float, Bigarray.float64_elt, Bigarray.c_layout) A1.t

type i8 = (int, Bigarray.int8_signed_elt, Bigarray.c_layout) A1.t

type u8 = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A1.t

type i16 = (int, Bigarray.int16_signed_elt, Bigarray.c_layout) A1.t

type u16 = (int, Bigarray.int16_unsigned_elt, Bigarray.c_layout) A1.t

type i32 = (int32, Bigarray.int32_elt, Bigarray.c_layout) A1.t

type i64 = (int64, Bigarray.int64_elt, Bigarray.c_layout) A1.t

let[@inline] sext16 v = (v lxor 0x8000) - 0x8000

let[@inline] sext24 v = (v lxor 0x800000) - 0x800000

(* [u] is a 16-bit pattern, [0 <= u < 2^16]. *)
let[@inline] swap16 u = ((u land 0xFF) lsl 8) lor (u lsr 8)

let[@inline] swap32 x =
  Int32.logor
    (Int32.logor (Int32.shift_left x 24)
       (Int32.logand (Int32.shift_left x 8) 0xFF0000l) )
    (Int32.logor
       (Int32.logand (Int32.shift_right_logical x 8) 0xFF00l)
       (Int32.shift_right_logical x 24) )

let[@inline] swap64 x =
  Int64.logor
    (Int64.shift_left (Int64.of_int32 (swap32 (Int64.to_int32 x))) 32)
    (Int64.logand
       (Int64.of_int32
          (swap32 (Int64.to_int32 (Int64.shift_right_logical x 32))) )
       0xFFFFFFFFL )

(* libsndfile 1.2.2's clipping writer (pcm.c, f2*_clip_array and d2*_clip_array,
   which [Soundml_io.write] turns on for every integer encoding): [s = x *
   2^31], exact; full scale at or past [2^31 - 1] and [-2^31]; otherwise [lrint
   s], round half to even, shifted down to the encoding's width — an arithmetic
   shift, so a floor, not a rounding at that width (0.75 LSB writes 0, -0.5 LSB
   writes -1). NaN fails both comparisons and reaches lrint, whose out-of-range
   result truncates to 0 in libsndfile's [int]. [s + 1.5 * 2^52 - 1.5 * 2^52] is
   round-half-to-even for [|s| < 2^51], which the comparisons guarantee.

   One writer departs: d2sc_clip_array, signed 8-bit from double, passes the
   double [s] to psf_lrintf (pcm.c:2374), which rounds it to a float, 24
   significant bits, before rounding to an integer: [1 - 2^-26] becomes 2^31,
   which overflows libsndfile's [int] and writes -128. [quantize] rounds the
   double, as d2uc_clip_array (pcm.c:2433) and every wider d2*_clip_array writer
   do; f2sc_clip_array, from float, rounds a float that holds [s] exactly, and
   agrees. [Soundml_io.write] offers no 8-bit encoding, so no parity with it is
   at stake. *)
let[@inline] quantize x ~shift ~hi ~lo =
  let s = x *. 0x1p31 in
  if s >= 2147483647. then hi
  else if s <= -2147483648. then lo
  else if Float.is_nan s then 0
  else Float.to_int (s +. 0x1.8p52 -. 0x1.8p52) asr shift

(* {1 Decoding}

   [Decode.<encoding>_<width>] reads the interleaved staging into planar storage
   of that width. libsndfile reads integers normalised, [v * 2^-(bits - 1)]
   (pcm.c, the normfacts of the *2f and *2d readers): an exact power-of-two
   scaling, so computing in double and rounding once at a float32 store is
   libsndfile's [(float) v * normfact]. A float of the storage's own width moves
   as bits (float32.c and double64.c copy the bytes); a float32 widened to an
   OCaml float would quiet a signalling NaN, so it travels as an int32. Width
   changes go through C's conversions, as libsndfile's do. A suffix [x] marks
   the byte-swapped kernels. *)

module Decode = struct
  let[@inline] s8 (a : i8) j = Float.of_int (A1.unsafe_get a j) *. 0x1p-7

  let[@inline] u8 (a : u8) j = Float.of_int (A1.unsafe_get a j - 128) *. 0x1p-7

  let[@inline] s16 (a : i16) j = Float.of_int (A1.unsafe_get a j) *. 0x1p-15

  let[@inline] s16x (a : u16) j =
    Float.of_int (sext16 (swap16 (A1.unsafe_get a j))) *. 0x1p-15

  let[@inline] s24 (a : u8) j ~lo ~hi =
    let v =
      A1.unsafe_get a (j + lo)
      lor (A1.unsafe_get a (j + 1) lsl 8)
      lor (A1.unsafe_get a (j + hi) lsl 16)
    in
    Float.of_int (sext24 v) *. 0x1p-23

  let[@inline] s32 (a : i32) j =
    Float.of_int (Int32.to_int (A1.unsafe_get a j)) *. 0x1p-31

  let[@inline] s32x (a : i32) j =
    Float.of_int (Int32.to_int (swap32 (A1.unsafe_get a j))) *. 0x1p-31

  let s8_32 (a : i8) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s8 a j0
        and x1 = s8 a (j0 + channels)
        and x2 = s8 a (j0 + (2 * channels))
        and x3 = s8 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s8 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s8_64 (a : i8) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s8 a j0
        and x1 = s8 a (j0 + channels)
        and x2 = s8 a (j0 + (2 * channels))
        and x3 = s8 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s8 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let u8_32 (a : u8) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = u8 a j0
        and x1 = u8 a (j0 + channels)
        and x2 = u8 a (j0 + (2 * channels))
        and x3 = u8 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (u8 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let u8_64 (a : u8) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = u8 a j0
        and x1 = u8 a (j0 + channels)
        and x2 = u8 a (j0 + (2 * channels))
        and x3 = u8 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (u8 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16_32 (a : i16) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s16 a j0
        and x1 = s16 a (j0 + channels)
        and x2 = s16 a (j0 + (2 * channels))
        and x3 = s16 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s16 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16_64 (a : i16) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s16 a j0
        and x1 = s16 a (j0 + channels)
        and x2 = s16 a (j0 + (2 * channels))
        and x3 = s16 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s16 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16x_32 (a : u16) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s16x a j0
        and x1 = s16x a (j0 + channels)
        and x2 = s16x a (j0 + (2 * channels))
        and x3 = s16x a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s16x a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16x_64 (a : u16) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s16x a j0
        and x1 = s16x a (j0 + channels)
        and x2 = s16x a (j0 + (2 * channels))
        and x3 = s16x a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s16x a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  (* [lo]/[hi]: the offsets of the least and most significant bytes, [0]/[2]
     little-endian and [2]/[0] big-endian; [j] steps over bytes. *)
  let s24_32 (a : u8) (d : f32) ~lo ~hi ~channels ~frames ~total ~off =
    let step = 3 * channels in
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref (3 * c) and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s24 a j0 ~lo ~hi
        and x1 = s24 a (j0 + step) ~lo ~hi
        and x2 = s24 a (j0 + (2 * step)) ~lo ~hi
        and x3 = s24 a (j0 + (3 * step)) ~lo ~hi in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * step) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s24 a !j ~lo ~hi) ;
        j := !j + step ;
        incr f
      done
    done

  let s24_64 (a : u8) (d : f64) ~lo ~hi ~channels ~frames ~total ~off =
    let step = 3 * channels in
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref (3 * c) and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s24 a j0 ~lo ~hi
        and x1 = s24 a (j0 + step) ~lo ~hi
        and x2 = s24 a (j0 + (2 * step)) ~lo ~hi
        and x3 = s24 a (j0 + (3 * step)) ~lo ~hi in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * step) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s24 a !j ~lo ~hi) ;
        j := !j + step ;
        incr f
      done
    done

  let s32_32 (a : i32) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s32 a j0
        and x1 = s32 a (j0 + channels)
        and x2 = s32 a (j0 + (2 * channels))
        and x3 = s32 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s32 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s32_64 (a : i32) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s32 a j0
        and x1 = s32 a (j0 + channels)
        and x2 = s32 a (j0 + (2 * channels))
        and x3 = s32 a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s32 a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s32x_32 (a : i32) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s32x a j0
        and x1 = s32x a (j0 + channels)
        and x2 = s32x a (j0 + (2 * channels))
        and x3 = s32x a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s32x a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let s32x_64 (a : i32) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = s32x a j0
        and x1 = s32x a (j0 + channels)
        and x2 = s32x a (j0 + (2 * channels))
        and x3 = s32x a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (s32x a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  (* float32 into float32 storage: bits, int32 to int32 *)
  let f32_32 (a : i32) (d : i32) ~channels ~frames ~total ~off =
    if channels = 1 then A1.blit (A1.sub a 0 frames) (A1.sub d off frames)
    else
      for c = 0 to channels - 1 do
        let o = (c * total) + off and j = ref c and f = ref 0 in
        while !f + 3 < frames do
          let j0 = !j and o0 = o + !f in
          let x0 = A1.unsafe_get a j0
          and x1 = A1.unsafe_get a (j0 + channels)
          and x2 = A1.unsafe_get a (j0 + (2 * channels))
          and x3 = A1.unsafe_get a (j0 + (3 * channels)) in
          A1.unsafe_set d o0 x0 ;
          A1.unsafe_set d (o0 + 1) x1 ;
          A1.unsafe_set d (o0 + 2) x2 ;
          A1.unsafe_set d (o0 + 3) x3 ;
          j := j0 + (4 * channels) ;
          f := !f + 4
        done ;
        while !f < frames do
          A1.unsafe_set d (o + !f) (A1.unsafe_get a !j) ;
          j := !j + channels ;
          incr f
        done
      done

  let f32x_32 (a : i32) (d : i32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = swap32 (A1.unsafe_get a j0)
        and x1 = swap32 (A1.unsafe_get a (j0 + channels))
        and x2 = swap32 (A1.unsafe_get a (j0 + (2 * channels)))
        and x3 = swap32 (A1.unsafe_get a (j0 + (3 * channels))) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (swap32 (A1.unsafe_get a !j)) ;
        j := !j + channels ;
        incr f
      done
    done

  (* the float32 load widens, as libsndfile's f2d does *)
  let f32_64 (a : f32) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get a j0
        and x1 = A1.unsafe_get a (j0 + channels)
        and x2 = A1.unsafe_get a (j0 + (2 * channels))
        and x3 = A1.unsafe_get a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (A1.unsafe_get a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let f32x_64 (a : i32) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = Int32.float_of_bits (swap32 (A1.unsafe_get a j0))
        and x1 = Int32.float_of_bits (swap32 (A1.unsafe_get a (j0 + channels)))
        and x2 =
          Int32.float_of_bits (swap32 (A1.unsafe_get a (j0 + (2 * channels))))
        and x3 =
          Int32.float_of_bits (swap32 (A1.unsafe_get a (j0 + (3 * channels))))
        in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f)
          (Int32.float_of_bits (swap32 (A1.unsafe_get a !j))) ;
        j := !j + channels ;
        incr f
      done
    done

  (* the float32 store narrows, as libsndfile's d2f does *)
  let f64_32 (a : f64) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get a j0
        and x1 = A1.unsafe_get a (j0 + channels)
        and x2 = A1.unsafe_get a (j0 + (2 * channels))
        and x3 = A1.unsafe_get a (j0 + (3 * channels)) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (A1.unsafe_get a !j) ;
        j := !j + channels ;
        incr f
      done
    done

  let f64x_32 (a : i64) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = Int64.float_of_bits (swap64 (A1.unsafe_get a j0))
        and x1 = Int64.float_of_bits (swap64 (A1.unsafe_get a (j0 + channels)))
        and x2 =
          Int64.float_of_bits (swap64 (A1.unsafe_get a (j0 + (2 * channels))))
        and x3 =
          Int64.float_of_bits (swap64 (A1.unsafe_get a (j0 + (3 * channels))))
        in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f)
          (Int64.float_of_bits (swap64 (A1.unsafe_get a !j))) ;
        j := !j + channels ;
        incr f
      done
    done

  (* a float64 load and store convert nothing: every bit survives *)
  let f64_64 (a : f64) (d : f64) ~channels ~frames ~total ~off =
    if channels = 1 then A1.blit (A1.sub a 0 frames) (A1.sub d off frames)
    else
      for c = 0 to channels - 1 do
        let o = (c * total) + off and j = ref c and f = ref 0 in
        while !f + 3 < frames do
          let j0 = !j and o0 = o + !f in
          let x0 = A1.unsafe_get a j0
          and x1 = A1.unsafe_get a (j0 + channels)
          and x2 = A1.unsafe_get a (j0 + (2 * channels))
          and x3 = A1.unsafe_get a (j0 + (3 * channels)) in
          A1.unsafe_set d o0 x0 ;
          A1.unsafe_set d (o0 + 1) x1 ;
          A1.unsafe_set d (o0 + 2) x2 ;
          A1.unsafe_set d (o0 + 3) x3 ;
          j := j0 + (4 * channels) ;
          f := !f + 4
        done ;
        while !f < frames do
          A1.unsafe_set d (o + !f) (A1.unsafe_get a !j) ;
          j := !j + channels ;
          incr f
        done
      done

  let f64x_64 (a : i64) (d : i64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = swap64 (A1.unsafe_get a j0)
        and x1 = swap64 (A1.unsafe_get a (j0 + channels))
        and x2 = swap64 (A1.unsafe_get a (j0 + (2 * channels)))
        and x3 = swap64 (A1.unsafe_get a (j0 + (3 * channels))) in
        A1.unsafe_set d o0 x0 ;
        A1.unsafe_set d (o0 + 1) x1 ;
        A1.unsafe_set d (o0 + 2) x2 ;
        A1.unsafe_set d (o0 + 3) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d (o + !f) (swap64 (A1.unsafe_get a !j)) ;
        j := !j + channels ;
        incr f
      done
    done
end

(* {1 Encoding}

   [Encode.<encoding>_<width>] writes planar storage of that width into the
   interleaved staging: integers through [quantize], floats as libsndfile writes
   them — a float of the storage's width as bits, a float32 tensor into a double
   encoding by C's widening, a float64 one into a float encoding by C's
   [(float)] cast (a float32 store, or [Int32.bits_of_float]). *)

module Encode = struct
  let[@inline] put_s8 (d : i8) j x =
    A1.unsafe_set d j (quantize x ~shift:24 ~hi:127 ~lo:(-128))

  let[@inline] put_u8 (d : u8) j x =
    A1.unsafe_set d j (quantize x ~shift:24 ~hi:127 ~lo:(-128) + 128)

  let[@inline] put_s16 (d : i16) j x =
    A1.unsafe_set d j (quantize x ~shift:16 ~hi:0x7FFF ~lo:(-0x8000))

  let[@inline] put_s16x (d : u16) j x =
    let q = quantize x ~shift:16 ~hi:0x7FFF ~lo:(-0x8000) in
    A1.unsafe_set d j (swap16 (q land 0xFFFF))

  let[@inline] put_s24 (d : u8) j x ~lo ~hi =
    let q = quantize x ~shift:8 ~hi:0x7FFFFF ~lo:(-0x800000) in
    A1.unsafe_set d (j + lo) (q land 0xFF) ;
    A1.unsafe_set d (j + 1) ((q lsr 8) land 0xFF) ;
    A1.unsafe_set d (j + hi) ((q lsr 16) land 0xFF)

  let[@inline] put_s32 (d : i32) j x =
    let q = quantize x ~shift:0 ~hi:0x7FFFFFFF ~lo:(-0x80000000) in
    A1.unsafe_set d j (Int32.of_int q)

  let[@inline] put_s32x (d : i32) j x =
    let q = quantize x ~shift:0 ~hi:0x7FFFFFFF ~lo:(-0x80000000) in
    A1.unsafe_set d j (swap32 (Int32.of_int q))

  let[@inline] put_bits32x (d : i32) j w = A1.unsafe_set d j (swap32 w)

  let[@inline] put_f32x (d : i32) j x =
    A1.unsafe_set d j (swap32 (Int32.bits_of_float x))

  let[@inline] put_f64x (d : i64) j x =
    A1.unsafe_set d j (swap64 (Int64.bits_of_float x))

  let[@inline] put_bits64x (d : i64) j w = A1.unsafe_set d j (swap64 w)

  let s8_32 (s : f32) (d : i8) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s8 d j0 x0 ;
        put_s8 d (j0 + channels) x1 ;
        put_s8 d (j0 + (2 * channels)) x2 ;
        put_s8 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s8 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s8_64 (s : f64) (d : i8) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s8 d j0 x0 ;
        put_s8 d (j0 + channels) x1 ;
        put_s8 d (j0 + (2 * channels)) x2 ;
        put_s8 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s8 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let u8_32 (s : f32) (d : u8) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_u8 d j0 x0 ;
        put_u8 d (j0 + channels) x1 ;
        put_u8 d (j0 + (2 * channels)) x2 ;
        put_u8 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_u8 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let u8_64 (s : f64) (d : u8) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_u8 d j0 x0 ;
        put_u8 d (j0 + channels) x1 ;
        put_u8 d (j0 + (2 * channels)) x2 ;
        put_u8 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_u8 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16_32 (s : f32) (d : i16) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s16 d j0 x0 ;
        put_s16 d (j0 + channels) x1 ;
        put_s16 d (j0 + (2 * channels)) x2 ;
        put_s16 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s16 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16_64 (s : f64) (d : i16) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s16 d j0 x0 ;
        put_s16 d (j0 + channels) x1 ;
        put_s16 d (j0 + (2 * channels)) x2 ;
        put_s16 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s16 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16x_32 (s : f32) (d : u16) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s16x d j0 x0 ;
        put_s16x d (j0 + channels) x1 ;
        put_s16x d (j0 + (2 * channels)) x2 ;
        put_s16x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s16x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s16x_64 (s : f64) (d : u16) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s16x d j0 x0 ;
        put_s16x d (j0 + channels) x1 ;
        put_s16x d (j0 + (2 * channels)) x2 ;
        put_s16x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s16x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  (* [lo]/[hi] as for [Decode.s24_32] *)
  let s24_32 (s : f32) (d : u8) ~lo ~hi ~channels ~frames ~total ~off =
    let step = 3 * channels in
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref (3 * c) and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s24 d j0 x0 ~lo ~hi ;
        put_s24 d (j0 + step) x1 ~lo ~hi ;
        put_s24 d (j0 + (2 * step)) x2 ~lo ~hi ;
        put_s24 d (j0 + (3 * step)) x3 ~lo ~hi ;
        j := j0 + (4 * step) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s24 d !j (A1.unsafe_get s (o + !f)) ~lo ~hi ;
        j := !j + step ;
        incr f
      done
    done

  let s24_64 (s : f64) (d : u8) ~lo ~hi ~channels ~frames ~total ~off =
    let step = 3 * channels in
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref (3 * c) and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s24 d j0 x0 ~lo ~hi ;
        put_s24 d (j0 + step) x1 ~lo ~hi ;
        put_s24 d (j0 + (2 * step)) x2 ~lo ~hi ;
        put_s24 d (j0 + (3 * step)) x3 ~lo ~hi ;
        j := j0 + (4 * step) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s24 d !j (A1.unsafe_get s (o + !f)) ~lo ~hi ;
        j := !j + step ;
        incr f
      done
    done

  let s32_32 (s : f32) (d : i32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s32 d j0 x0 ;
        put_s32 d (j0 + channels) x1 ;
        put_s32 d (j0 + (2 * channels)) x2 ;
        put_s32 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s32 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s32_64 (s : f64) (d : i32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s32 d j0 x0 ;
        put_s32 d (j0 + channels) x1 ;
        put_s32 d (j0 + (2 * channels)) x2 ;
        put_s32 d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s32 d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s32x_32 (s : f32) (d : i32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s32x d j0 x0 ;
        put_s32x d (j0 + channels) x1 ;
        put_s32x d (j0 + (2 * channels)) x2 ;
        put_s32x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s32x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let s32x_64 (s : f64) (d : i32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_s32x d j0 x0 ;
        put_s32x d (j0 + channels) x1 ;
        put_s32x d (j0 + (2 * channels)) x2 ;
        put_s32x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_s32x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  (* float32 storage into a float encoding: bits, int32 to int32 *)
  let f32_32 (s : i32) (d : i32) ~channels ~frames ~total ~off =
    if channels = 1 then A1.blit (A1.sub s off frames) (A1.sub d 0 frames)
    else
      for c = 0 to channels - 1 do
        let o = (c * total) + off and j = ref c and f = ref 0 in
        while !f + 3 < frames do
          let j0 = !j and o0 = o + !f in
          let x0 = A1.unsafe_get s o0
          and x1 = A1.unsafe_get s (o0 + 1)
          and x2 = A1.unsafe_get s (o0 + 2)
          and x3 = A1.unsafe_get s (o0 + 3) in
          A1.unsafe_set d j0 x0 ;
          A1.unsafe_set d (j0 + channels) x1 ;
          A1.unsafe_set d (j0 + (2 * channels)) x2 ;
          A1.unsafe_set d (j0 + (3 * channels)) x3 ;
          j := j0 + (4 * channels) ;
          f := !f + 4
        done ;
        while !f < frames do
          A1.unsafe_set d !j (A1.unsafe_get s (o + !f)) ;
          j := !j + channels ;
          incr f
        done
      done

  let f32x_32 (s : i32) (d : i32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_bits32x d j0 x0 ;
        put_bits32x d (j0 + channels) x1 ;
        put_bits32x d (j0 + (2 * channels)) x2 ;
        put_bits32x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_bits32x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  (* the float32 store narrows: C's [(float)], as libsndfile's d2f *)
  let f32_64 (s : f64) (d : f32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        A1.unsafe_set d j0 x0 ;
        A1.unsafe_set d (j0 + channels) x1 ;
        A1.unsafe_set d (j0 + (2 * channels)) x2 ;
        A1.unsafe_set d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let f32x_64 (s : f64) (d : i32) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_f32x d j0 x0 ;
        put_f32x d (j0 + channels) x1 ;
        put_f32x d (j0 + (2 * channels)) x2 ;
        put_f32x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_f32x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  (* the float32 load widens, as libsndfile's f2d does *)
  let f64_32 (s : f32) (d : f64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        A1.unsafe_set d j0 x0 ;
        A1.unsafe_set d (j0 + channels) x1 ;
        A1.unsafe_set d (j0 + (2 * channels)) x2 ;
        A1.unsafe_set d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        A1.unsafe_set d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  let f64x_32 (s : f32) (d : i64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_f64x d j0 x0 ;
        put_f64x d (j0 + channels) x1 ;
        put_f64x d (j0 + (2 * channels)) x2 ;
        put_f64x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_f64x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done

  (* a float64 load and store convert nothing: every bit survives *)
  let f64_64 (s : f64) (d : f64) ~channels ~frames ~total ~off =
    if channels = 1 then A1.blit (A1.sub s off frames) (A1.sub d 0 frames)
    else
      for c = 0 to channels - 1 do
        let o = (c * total) + off and j = ref c and f = ref 0 in
        while !f + 3 < frames do
          let j0 = !j and o0 = o + !f in
          let x0 = A1.unsafe_get s o0
          and x1 = A1.unsafe_get s (o0 + 1)
          and x2 = A1.unsafe_get s (o0 + 2)
          and x3 = A1.unsafe_get s (o0 + 3) in
          A1.unsafe_set d j0 x0 ;
          A1.unsafe_set d (j0 + channels) x1 ;
          A1.unsafe_set d (j0 + (2 * channels)) x2 ;
          A1.unsafe_set d (j0 + (3 * channels)) x3 ;
          j := j0 + (4 * channels) ;
          f := !f + 4
        done ;
        while !f < frames do
          A1.unsafe_set d !j (A1.unsafe_get s (o + !f)) ;
          j := !j + channels ;
          incr f
        done
      done

  let f64x_64 (s : i64) (d : i64) ~channels ~frames ~total ~off =
    for c = 0 to channels - 1 do
      let o = (c * total) + off and j = ref c and f = ref 0 in
      while !f + 3 < frames do
        let j0 = !j and o0 = o + !f in
        let x0 = A1.unsafe_get s o0
        and x1 = A1.unsafe_get s (o0 + 1)
        and x2 = A1.unsafe_get s (o0 + 2)
        and x3 = A1.unsafe_get s (o0 + 3) in
        put_bits64x d j0 x0 ;
        put_bits64x d (j0 + channels) x1 ;
        put_bits64x d (j0 + (2 * channels)) x2 ;
        put_bits64x d (j0 + (3 * channels)) x3 ;
        j := j0 + (4 * channels) ;
        f := !f + 4
      done ;
      while !f < frames do
        put_bits64x d !j (A1.unsafe_get s (o + !f)) ;
        j := !j + channels ;
        incr f
      done
    done
end

(* {1 Dispatch} *)

let width : type a. string -> (float, a) Nx.dtype -> int =
 fun op -> function
  | Nx.Float32 ->
      4
  | Nx.Float64 ->
      8
  | dt ->
      invalid_arg
        (Stdlib.Format.asprintf "%s: %a storage (float32 and float64 only)" op
           Nx.pp_dtype dt )

(* The precondition the kernels' unsafe accesses rest on (header comment),
   stated with divisions so that no product can overflow. *)
let check op ~sample ~interleaved ~elt ~planar ~channels ~frames ~total ~off =
  if
    not
      ( channels >= 1 && frames >= 0 && off >= 0 && total >= off
      && frames <= total - off
      && frames <= interleaved / sample / channels
      && total <= planar / elt / channels )
  then
    invalid_arg
      (Printf.sprintf
         "%s: %d frames of %d channels at frame %d of %d lie outside the \
          buffers"
         op frames channels off total )

let lo_hi e = if big_endian e then (2, 0) else (0, 2)

let decode (type a) e (dt : (float, a) Nx.dtype) src dst ~channels ~frames
    ~total ~off =
  check "Pcm.decode" ~sample:(sample_bytes e)
    ~interleaved:(Rig.Buffer.length src.buffer)
    ~elt:(width "Pcm.decode" dt) ~planar:(Rig.Buffer.length dst) ~channels
    ~frames ~total ~off ;
  let run sk dk k =
    Rig.Buffer.read src.buffer sk (fun a -> Rig.Buffer.write dst dk (k a))
  in
  let open Bigarray in
  let swap = not (host_order e) in
  match (dt, e) with
  | Nx.Float32, `S8 ->
      run int8_signed float32 (Decode.s8_32 ~channels ~frames ~total ~off)
  | Nx.Float64, `S8 ->
      run int8_signed float64 (Decode.s8_64 ~channels ~frames ~total ~off)
  | Nx.Float32, `U8 ->
      run int8_unsigned float32 (Decode.u8_32 ~channels ~frames ~total ~off)
  | Nx.Float64, `U8 ->
      run int8_unsigned float64 (Decode.u8_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S16le | `S16be) when swap ->
      run int16_unsigned float32 (Decode.s16x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S16le | `S16be) when swap ->
      run int16_unsigned float64 (Decode.s16x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S16le | `S16be) ->
      run int16_signed float32 (Decode.s16_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S16le | `S16be) ->
      run int16_signed float64 (Decode.s16_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S24le | `S24be) ->
      let lo, hi = lo_hi e in
      run int8_unsigned float32
        (Decode.s24_32 ~lo ~hi ~channels ~frames ~total ~off)
  | Nx.Float64, (`S24le | `S24be) ->
      let lo, hi = lo_hi e in
      run int8_unsigned float64
        (Decode.s24_64 ~lo ~hi ~channels ~frames ~total ~off)
  | Nx.Float32, (`S32le | `S32be) when swap ->
      run int32 float32 (Decode.s32x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S32le | `S32be) when swap ->
      run int32 float64 (Decode.s32x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S32le | `S32be) ->
      run int32 float32 (Decode.s32_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S32le | `S32be) ->
      run int32 float64 (Decode.s32_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F32le | `F32be) when swap ->
      run int32 int32 (Decode.f32x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F32le | `F32be) when swap ->
      run int32 float64 (Decode.f32x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F32le | `F32be) ->
      run int32 int32 (Decode.f32_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F32le | `F32be) ->
      run float32 float64 (Decode.f32_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F64le | `F64be) when swap ->
      run int64 float32 (Decode.f64x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F64le | `F64be) when swap ->
      run int64 int64 (Decode.f64x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F64le | `F64be) ->
      run float64 float32 (Decode.f64_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F64le | `F64be) ->
      run float64 float64 (Decode.f64_64 ~channels ~frames ~total ~off)
  | _ ->
      assert false (* [width] admits float32 and float64 only *)

let encode (type a) e (dt : (float, a) Nx.dtype) src dst ~channels ~frames
    ~total ~off =
  check "Pcm.encode" ~sample:(sample_bytes e)
    ~interleaved:(Rig.Buffer.length dst.buffer)
    ~elt:(width "Pcm.encode" dt) ~planar:(Rig.Buffer.length src) ~channels
    ~frames ~total ~off ;
  let run sk dk k =
    Rig.Buffer.read src sk (fun s -> Rig.Buffer.write dst.buffer dk (k s))
  in
  let open Bigarray in
  let swap = not (host_order e) in
  match (dt, e) with
  | Nx.Float32, `S8 ->
      run float32 int8_signed (Encode.s8_32 ~channels ~frames ~total ~off)
  | Nx.Float64, `S8 ->
      run float64 int8_signed (Encode.s8_64 ~channels ~frames ~total ~off)
  | Nx.Float32, `U8 ->
      run float32 int8_unsigned (Encode.u8_32 ~channels ~frames ~total ~off)
  | Nx.Float64, `U8 ->
      run float64 int8_unsigned (Encode.u8_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S16le | `S16be) when swap ->
      run float32 int16_unsigned (Encode.s16x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S16le | `S16be) when swap ->
      run float64 int16_unsigned (Encode.s16x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S16le | `S16be) ->
      run float32 int16_signed (Encode.s16_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S16le | `S16be) ->
      run float64 int16_signed (Encode.s16_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S24le | `S24be) ->
      let lo, hi = lo_hi e in
      run float32 int8_unsigned
        (Encode.s24_32 ~lo ~hi ~channels ~frames ~total ~off)
  | Nx.Float64, (`S24le | `S24be) ->
      let lo, hi = lo_hi e in
      run float64 int8_unsigned
        (Encode.s24_64 ~lo ~hi ~channels ~frames ~total ~off)
  | Nx.Float32, (`S32le | `S32be) when swap ->
      run float32 int32 (Encode.s32x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S32le | `S32be) when swap ->
      run float64 int32 (Encode.s32x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`S32le | `S32be) ->
      run float32 int32 (Encode.s32_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`S32le | `S32be) ->
      run float64 int32 (Encode.s32_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F32le | `F32be) when swap ->
      run int32 int32 (Encode.f32x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F32le | `F32be) when swap ->
      run float64 int32 (Encode.f32x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F32le | `F32be) ->
      run int32 int32 (Encode.f32_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F32le | `F32be) ->
      run float64 float32 (Encode.f32_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F64le | `F64be) when swap ->
      run float32 int64 (Encode.f64x_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F64le | `F64be) when swap ->
      run int64 int64 (Encode.f64x_64 ~channels ~frames ~total ~off)
  | Nx.Float32, (`F64le | `F64be) ->
      run float32 float64 (Encode.f64_32 ~channels ~frames ~total ~off)
  | Nx.Float64, (`F64le | `F64be) ->
      run float64 float64 (Encode.f64_64 ~channels ~frames ~total ~off)
  | _ ->
      assert false (* [width] admits float32 and float64 only *)
