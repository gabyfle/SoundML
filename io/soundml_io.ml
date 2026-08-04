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

(* Decode and encode through libsndfile, on the single-copy architecture: the
   destination is an [Nx_buffer] (off-heap, address-stable), the C stub writes
   through its bigarray view with the runtime lock released, and the result
   tensor is a zero-copy [Nx.of_buffer] view. The stubs own every sf_* call;
   this side owns format mapping, error typing and buffer bookkeeping. *)

type 'a audio = {data: (float, 'a) Nx.t; sample_rate: int}

type error =
  | Not_found of {path: string}
  | Permission_denied of {path: string}
  | Unrecognized_format of {path: string; details: string}
  | Unsupported of {path: string; details: string}
  | Truncated of {path: string; expected_frames: int; read_frames: int}
  | Io of {path: string; op: string; details: string}

let pp_error ppf = function
  | Not_found {path} ->
      Stdlib.Format.fprintf ppf "soundml-io: %S does not exist" path
  | Permission_denied {path} ->
      Stdlib.Format.fprintf ppf "soundml-io: %S: permission denied" path
  | Unrecognized_format {path; details} ->
      Stdlib.Format.fprintf ppf
        "soundml-io: %S is not a recognized audio format (%s)" path details
  | Unsupported {path; details} ->
      Stdlib.Format.fprintf ppf "soundml-io: %S cannot be encoded (%s)" path
        details
  | Truncated {path; expected_frames; read_frames} ->
      Stdlib.Format.fprintf ppf
        "soundml-io: %S is truncated (header advertises %d frames, decode \
         delivered %d)"
        path expected_frames read_frames
  | Io {path; op; details} ->
      Stdlib.Format.fprintf ppf "soundml-io: %S: %s failed (%s)" path op details

(* {1 libsndfile constants}

   Mirrors of the public, ABI-frozen sndfile.h enums — the container and subtype
   codes SF_INFO carries and the sf_error classes. The stubs keep every sf_*
   call in C; these codes are the data the seam exchanges. *)

let sf_format_wav = 0x010000

let sf_format_aiff = 0x020000

let sf_format_flac = 0x170000

let sf_format_caf = 0x180000

let sf_format_ogg = 0x200000

let sf_format_pcm_16 = 0x0002

let sf_format_pcm_24 = 0x0003

let sf_format_pcm_32 = 0x0004

let sf_format_float = 0x0006

let sf_format_double = 0x0007

let sf_format_vorbis = 0x0060

let sf_format_typemask = 0x0FFF0000

let sf_format_submask = 0x0000FFFF

let sf_err_unrecognised_format = 1

let sf_err_malformed_file = 3

let sf_err_unsupported_encoding = 4

(* Filesystem classification of a failed open, probed by the stub with open(2).
   Kept in sync with soundml_io_stubs.c. *)

let fs_noent = 1

let fs_acces = 2

let fs_notdir = 3

let fs_refused = 5

(* Decode/encode layout modes; kept in sync with soundml_io_stubs.c. *)

let mode_direct = 0

let mode_planar = 1

let mode_downmix = 2

(* Stub-level failure reported in the error slot of a read/write outcome when
   the failure is the stub's own (staging allocation), not libsndfile's. *)

let soundml_io_err_staging = -1

(* {1 Formats} *)

module Format = struct
  type container = [`Wav | `Aiff | `Flac | `Ogg | `Caf]

  type encoding = [`Pcm_16 | `Pcm_24 | `Pcm_32 | `Float32 | `Float64 | `Vorbis]

  type t = {container: container; encoding: encoding}

  let container_name = function
    | `Wav ->
        "wav"
    | `Aiff ->
        "aiff"
    | `Flac ->
        "flac"
    | `Ogg ->
        "ogg"
    | `Caf ->
        "caf"

  let encoding_name = function
    | `Pcm_16 ->
        "pcm_16"
    | `Pcm_24 ->
        "pcm_24"
    | `Pcm_32 ->
        "pcm_32"
    | `Float32 ->
        "float32"
    | `Float64 ->
        "float64"
    | `Vorbis ->
        "vorbis"

  let valid container encoding =
    match (container, encoding) with
    | `Ogg, `Vorbis ->
        true
    | `Ogg, _ | _, `Vorbis ->
        false
    | `Flac, (`Pcm_16 | `Pcm_24) ->
        true
    | `Flac, _ ->
        false
    | (`Wav | `Aiff | `Caf), _ ->
        true

  let default_encoding = function `Ogg -> `Vorbis | _ -> `Pcm_16

  let create ?encoding container =
    let encoding =
      match encoding with Some e -> e | None -> default_encoding container
    in
    if not (valid container encoding) then
      invalid_arg
        (Printf.sprintf "Soundml_io.Format.create: %s cannot carry %s"
           (container_name container) (encoding_name encoding) ) ;
    {container; encoding}

  let of_path path =
    let ext = String.lowercase_ascii (Filename.extension path) in
    match ext with
    | ".wav" ->
        Ok (create `Wav)
    | ".aiff" | ".aif" ->
        Ok (create `Aiff)
    | ".flac" ->
        Ok (create `Flac)
    | ".ogg" | ".oga" ->
        Ok (create `Ogg)
    | ".caf" ->
        Ok (create `Caf)
    | _ ->
        Error
          (Unrecognized_format
             { path
             ; details=
                 Printf.sprintf "no container matches the extension %S" ext } )

  let container t = t.container

  let encoding t = t.encoding

  let pp ppf t =
    Stdlib.Format.fprintf ppf "%s/%s"
      (container_name t.container)
      (encoding_name t.encoding)

  let equal a b = a.container = b.container && a.encoding = b.encoding

  (* The SF_INFO format word of a validated pair. *)
  let code t =
    let major =
      match t.container with
      | `Wav ->
          sf_format_wav
      | `Aiff ->
          sf_format_aiff
      | `Flac ->
          sf_format_flac
      | `Ogg ->
          sf_format_ogg
      | `Caf ->
          sf_format_caf
    in
    let sub =
      match t.encoding with
      | `Pcm_16 ->
          sf_format_pcm_16
      | `Pcm_24 ->
          sf_format_pcm_24
      | `Pcm_32 ->
          sf_format_pcm_32
      | `Float32 ->
          sf_format_float
      | `Float64 ->
          sf_format_double
      | `Vorbis ->
          sf_format_vorbis
    in
    major lor sub

  (* The write-surface description of a file's format word, when its
     major/subtype pair is expressible there. *)
  let of_code code =
    let container =
      let major = code land sf_format_typemask in
      if major = sf_format_wav then Some `Wav
      else if major = sf_format_aiff then Some `Aiff
      else if major = sf_format_flac then Some `Flac
      else if major = sf_format_ogg then Some `Ogg
      else if major = sf_format_caf then Some `Caf
      else None
    in
    let encoding =
      let sub = code land sf_format_submask in
      if sub = sf_format_pcm_16 then Some `Pcm_16
      else if sub = sf_format_pcm_24 then Some `Pcm_24
      else if sub = sf_format_pcm_32 then Some `Pcm_32
      else if sub = sf_format_float then Some `Float32
      else if sub = sf_format_double then Some `Float64
      else if sub = sf_format_vorbis then Some `Vorbis
      else None
    in
    match (container, encoding) with
    | Some container, Some encoding when valid container encoding ->
        Some {container; encoding}
    | _ ->
        None
end

module Info = struct
  type t =
    { frames: int
    ; channels: int
    ; sample_rate: int
    ; format: Format.t option
    ; format_name: string }

  let pp ppf i =
    Stdlib.Format.fprintf ppf "%d Hz, %d channel%s, %d frames, %s" i.sample_rate
      i.channels
      (if i.channels = 1 then "" else "s")
      i.frames i.format_name
end

(* {1 The C seam} *)

type handle

external stub_open_read :
  string -> (handle * int * int * int * int * bool, int * int * string) result
  = "soundml_io_open_read"

external stub_open_write :
  string -> int -> int -> int -> (handle, int * int * string) result
  = "soundml_io_open_write"

external stub_readf :
     handle
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> int (* mode *)
  -> int (* destination frame offset *)
  -> int (* destination frames per channel *)
  -> int (* frames to transfer *)
  -> int (* file channels *)
  -> int * int * string = "soundml_io_readf_bc" "soundml_io_readf"

external stub_writef :
     handle
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int * int * string = "soundml_io_writef_bc" "soundml_io_writef"

external stub_seek : handle -> int -> int * string = "soundml_io_seek"

external stub_close : handle -> int * string = "soundml_io_close"

external stub_format_check : int -> int -> int -> bool
  = "soundml_io_format_check"

external stub_format_name : int -> string = "soundml_io_format_name"

(* {1 Error mapping} *)

(* A failed read open: filesystem conditions first (probed with open(2) by the
   stub), then libsndfile's own verdict on the bytes. *)
let read_open_error path (fs, sf_err, details) =
  if fs = fs_noent then Not_found {path}
  else if fs = fs_acces then Permission_denied {path}
  else if fs = fs_notdir then
    Io {path; op= "open"; details= "a path component is not a directory"}
  else if sf_err = sf_err_unrecognised_format || sf_err = sf_err_malformed_file
  then Unrecognized_format {path; details}
  else if sf_err = sf_err_unsupported_encoding then Unsupported {path; details}
  else Io {path; op= "open"; details}

let write_open_error path ~unsupported (fs, _sf_err, details) =
  if fs = fs_refused then Unsupported {path; details= unsupported}
  else if fs = fs_noent then Not_found {path}
  else if fs = fs_acces then Permission_denied {path}
  else if fs = fs_notdir then
    Io {path; op= "open"; details= "a path component is not a directory"}
  else Io {path; op= "open"; details}

(* {1 Preconditions} *)

let check_dtype : type a. string -> (float, a) Nx.dtype -> unit =
 fun op dt ->
  match dt with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | dt ->
      invalid_arg
        (Stdlib.Format.asprintf
           "%s: cannot carry %a audio (the decoder delivers float32 and \
            float64)"
           op Nx.pp_dtype dt )

let check_sample_rate op = function
  | Some rate when rate < 1 ->
      invalid_arg
        (Printf.sprintf
           "%s: sample_rate %d is not positive (a signal has a rate)" op rate )
  | _ ->
      ()

let check_quality op quality sample_rate =
  match (quality, sample_rate) with
  | Some _, None ->
      invalid_arg
        ( op
        ^ ": ?quality without ?sample_rate changes nothing (a resampling \
           quality needs a resampling)" )
  | _ ->
      ()

let kind_of_dtype : type a. (float, a) Nx.dtype -> (float, a) Nx_buffer.kind =
  function
  | Nx.Float32 ->
      Nx_buffer.float32
  | Nx.Float64 ->
      Nx_buffer.float64
  | _ ->
      assert false (* guarded by [check_dtype] *)

let elt_size : type a. (float, a) Nx.dtype -> int = function
  | Nx.Float32 ->
      4
  | Nx.Float64 ->
      8
  | _ ->
      assert false

(* {1 Probing} *)

let info_of_header ~frames ~channels ~sample_rate ~code =
  { Info.frames
  ; channels
  ; sample_rate
  ; format= Format.of_code code
  ; format_name= stub_format_name code }

let info path =
  match stub_open_read path with
  | Error e ->
      Error (read_open_error path e)
  | Ok (h, frames, channels, sample_rate, code, _seekable) ->
      (* A close error on a read-only probe puts no data at risk. *)
      let (_ : int * string) = stub_close h in
      Ok (info_of_header ~frames ~channels ~sample_rate ~code)

(* {1 Reading} *)

(* Allocation guard for header-advertised frame counts: the destination is sized
   from the header, so a lying header must be refused as a typed error before it
   can turn into an unbounded allocation. No decoder expands a file's bytes by
   anything near 4096x (PCM is at most 8x, FLAC and Vorbis far below); a header
   whose destination would exceed that bound against the file's byte size — or
   overflow the size arithmetic — is malformed by construction. The floor keeps
   small legitimate files unaffected. *)
let alloc_guard_floor_bytes = 1 lsl 26 (* 64 MB *)

let alloc_guard_ratio = 4096

let mul_no_overflow a b = a = 0 || b = 0 || a <= max_int / b

let destination_admissible ~path ~frames ~channels ~elt =
  if
    (not (mul_no_overflow frames channels))
    || not (mul_no_overflow (frames * channels) elt)
  then false
  else
    let need = frames * channels * elt in
    need <= alloc_guard_floor_bytes
    ||
    let bytes =
      match In_channel.with_open_bin path In_channel.length with
      | n ->
          Int64.to_int n
      | exception Sys_error _ ->
          0
    in
    mul_no_overflow bytes alloc_guard_ratio && need <= bytes * alloc_guard_ratio

let read_mode ~mono ~channels =
  if channels = 1 then (mode_direct, 1)
  else if mono then (mode_downmix, 1)
  else (mode_planar, channels)

(* Decode a stream whose header cannot say its length: chunked to EOF,
   collecting per-block planar pieces (channel stride [block]) and concatenating
   once — the documented transient for this degenerate case only. *)
let read_to_eof kind h ~mode ~channels ~channels_out =
  let block = 65536 in
  let rec grow acc total =
    let piece = Nx_buffer.create kind (channels_out * block) in
    let ba = Nx_buffer.to_bigarray1 piece in
    let delivered, _err, _details =
      stub_readf h ba mode 0 block block channels
    in
    let acc = if delivered > 0 then (piece, delivered) :: acc else acc in
    if delivered < block then (List.rev acc, total + delivered)
    else grow acc (total + delivered)
  in
  let pieces, total = grow [] 0 in
  let dst = Nx_buffer.create kind (channels_out * total) in
  let dst_ba = Nx_buffer.to_bigarray1 dst in
  let pos = ref 0 in
  List.iter
    (fun (piece, n) ->
      let piece_ba = Nx_buffer.to_bigarray1 piece in
      for c = 0 to channels_out - 1 do
        Bigarray.Array1.blit
          (Bigarray.Array1.sub piece_ba (c * block) n)
          (Bigarray.Array1.sub dst_ba ((c * total) + !pos) n)
      done ;
      pos := !pos + n )
    pieces ;
  (dst, total)

(* {1 The reader core}

   One state record serves every reading face: [Reader] wraps it directly,
   [read] drives it once over the whole file, [fold] loops it over a lent chunk.
   A native state decodes straight into each call's destination (carry-less); a
   resampled state feeds decoded blocks to a [Resample.Kernel] and buffers the
   kernel's emission cadence — per-step pieces from direct plans, block bursts
   from FFT-executed ones — in an output carry so chunk sizes are exact
   regardless of the plan. *)

module Resample = Soundml.Resample

(* The flat storage of a C-contiguous tensor, shared (never copied): the
   bigarray view starts at the tensor's own offset, so fresh buffers and
   contiguous views into larger storage take the same path. *)
let planar_array1 t =
  Bigarray.Array1.sub
    (Nx_buffer.to_bigarray1 (Nx.data t))
    (Nx.offset t) (Nx.numel t)

(* The decode block feeding the resampler, in frames. Unlike the native staging
   (stub-local, L2-resident: the deinterleave is a pure layout pass), the kernel
   amortizes per-step dispatch and — on FFT-executed plans — stacks more
   transform lines the longer its chunks are, so the fused path stages a few
   megabytes rather than a quarter of one: measured on the reference host, 30 s
   stereo float32 through the near-unity 44.1 -> 48 kHz plan runs 1.5x its
   offline decomposition at 32768-frame blocks and within 1-3% at this size (the
   44.1 -> 16 kHz cascade likewise). Clamped to the advertised length so short
   files never over-stage; still O(1) in file length — the native-rate signal
   never exists as a whole buffer. *)
let decode_block_frames ~channels ~elt ~advertised =
  let budget = 4194304 / (channels * elt) in
  let block = Stdlib.min 1048576 (Stdlib.max 4096 budget) in
  if advertised > 0 then Stdlib.min block (Stdlib.max 4096 advertised)
  else block

(* A kernel output not yet delivered: [plen] frames, planar with channel stride
   [plen], the leading [poff] already served. [pt] keeps the storage alive and
   lets the unknown-length path concatenate tensors directly. *)
type 'a piece =
  { pt: (float, 'a) Nx.t
  ; pba: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  ; plen: int
  ; mutable poff: int }

type 'a resampler =
  { config: Resample.Config.t
  ; kernel: 'a Resample.Kernel.t
  ; stage_ba: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
        (* [channels_out * decode_block] planar staging the decoder fills *)
  ; stage_t: (float, 'a) Nx.t (* the [channels_out; decode_block] view *)
  ; decode_block: int }

type 'a state =
  { handle: handle
  ; path: string
  ; dt: (float, 'a) Nx.dtype
  ; kind: (float, 'a) Nx_buffer.kind
  ; native: Info.t
  ; mode: int
  ; channels: int (* the file's channels, what the decoder delivers *)
  ; channels_out: int (* post-downmix, what the reader delivers *)
  ; target_rate: int
  ; resampler: 'a resampler option
  ; carry: 'a piece Queue.t
  ; mutable carry_len: int
  ; mutable src_pos: int (* source frames decoded so far, absolute *)
  ; mutable out_served: int (* output-grid frames delivered or dropped *)
  ; mutable closed: bool
  ; mutable decoder_eof: bool
  ; mutable flushed: bool (* the kernel's one flush has happened *)
  ; mutable finished: bool (* [Ok None] was reached; forever after *) }

let close_state st =
  if not st.closed then begin
    st.closed <- true ;
    (* a close error on a read-only handle puts no data at risk *)
    let (_ : int * string) = stub_close st.handle in
    ()
  end

let ensure_open st op =
  if st.closed then invalid_arg (Printf.sprintf "%s: the reader is closed" op)

(* Truncation is judged against the header: the destination of {!read} is sized
   from the advertised count, so a shortfall is an error, never a silently
   partial result. Streams advertising [0] have no claim to break and decode to
   EOF. *)
let truncated_at_eof st =
  st.native.Info.frames > 0 && st.src_pos < st.native.Info.frames

let make_state (type a) (dt : (float, a) Nx.dtype) path h ~frames ~channels
    ~sample_rate:native_rate ~code ~mono ~target ~quality : a state =
  let mode, channels_out = read_mode ~mono ~channels in
  let kind = kind_of_dtype dt in
  let native =
    info_of_header ~frames ~channels ~sample_rate:native_rate ~code
  in
  let resampler, target_rate =
    match target with
    | Some target when target <> native_rate ->
        (* [Config.create] validates the pair and may raise; the handle must not
           leak with the exception *)
        let config =
          try
            Resample.Config.create ?quality ~sample_rate:native_rate ~target ()
          with e ->
            let (_ : int * string) = stub_close h in
            raise e
        in
        let decode_block =
          decode_block_frames ~channels:channels_out ~elt:(elt_size dt)
            ~advertised:frames
        in
        let kernel =
          Resample.Kernel.prepare config dt ~channels:channels_out
            ~max_block:decode_block
        in
        let stage = Nx_buffer.create kind (channels_out * decode_block) in
        ( Some
            { config
            ; kernel
            ; stage_ba= Nx_buffer.to_bigarray1 stage
            ; stage_t= Nx.of_buffer stage ~shape:[|channels_out; decode_block|]
            ; decode_block }
        , target )
    | _ ->
        (None, native_rate)
  in
  { handle= h
  ; path
  ; dt
  ; kind
  ; native
  ; mode
  ; channels
  ; channels_out
  ; target_rate
  ; resampler
  ; carry= Queue.create ()
  ; carry_len= 0
  ; src_pos= 0
  ; out_served= 0
  ; closed= false
  ; decoder_eof= false
  ; flushed= false
  ; finished= false }

let open_state op ?sample_rate ?quality ?(mono = false) dt path =
  check_dtype op dt ;
  check_sample_rate op sample_rate ;
  check_quality op quality sample_rate ;
  match stub_open_read path with
  | Error e ->
      Error (read_open_error path e)
  | Ok (h, frames, channels, native_rate, code, _seekable) ->
      Ok
        (make_state dt path h ~frames ~channels ~sample_rate:native_rate ~code
           ~mono ~target:sample_rate ~quality )

(* {2 The carry} *)

let enqueue_piece st t =
  let t = if Nx.is_c_contiguous t then t else Nx.contiguous t in
  let plen = (Nx.shape t).(Nx.ndim t - 1) in
  if plen > 0 then begin
    Queue.push {pt= t; pba= planar_array1 t; plen; poff= 0} st.carry ;
    st.carry_len <- st.carry_len + plen
  end

(* [serve_from_carry st ~dst_ba ~dst_total ~dst_off ~n] copies the next [n]
   carried frames into the planar destination (channel stride [dst_total], frame
   origin [dst_off]) — the kernel-output blit of the fused read path, and the
   only copy between kernel output and caller. *)
let serve_from_carry st ~dst_ba ~dst_total ~dst_off ~n =
  let written = ref 0 in
  while !written < n do
    let p = Queue.peek st.carry in
    let k = Stdlib.min (p.plen - p.poff) (n - !written) in
    for c = 0 to st.channels_out - 1 do
      Bigarray.Array1.blit
        (Bigarray.Array1.sub p.pba ((c * p.plen) + p.poff) k)
        (Bigarray.Array1.sub dst_ba ((c * dst_total) + dst_off + !written) k)
    done ;
    p.poff <- p.poff + k ;
    if p.poff = p.plen then ignore (Queue.pop st.carry) ;
    written := !written + k
  done ;
  st.carry_len <- st.carry_len - n ;
  st.out_served <- st.out_served + n

let drop_from_carry st n =
  let dropped = ref 0 in
  while !dropped < n do
    let p = Queue.peek st.carry in
    let k = Stdlib.min (p.plen - p.poff) (n - !dropped) in
    p.poff <- p.poff + k ;
    if p.poff = p.plen then ignore (Queue.pop st.carry) ;
    dropped := !dropped + k
  done ;
  st.carry_len <- st.carry_len - n ;
  st.out_served <- st.out_served + n

(* {2 Decoding into the kernel} *)

(* One decode block: stage, feed the kernel, enqueue what it emits. Respects the
   header's frame count as the length authority (the same authority that sizes
   {!read}'s destination); a short read below it is decoder EOF. *)
let decode_step st rs =
  let advertised = st.native.Info.frames in
  let want =
    if advertised > 0 then Stdlib.min rs.decode_block (advertised - st.src_pos)
    else rs.decode_block
  in
  if want = 0 then st.decoder_eof <- true
  else begin
    let got, _err, _details =
      stub_readf st.handle rs.stage_ba st.mode 0 rs.decode_block want
        st.channels
    in
    if got > 0 then begin
      let chunk =
        if got = rs.decode_block then rs.stage_t
        else Nx.shrink [|(0, st.channels_out); (0, got)|] rs.stage_t
      in
      ( match Resample.Kernel.step rs.kernel chunk with
      | Some piece ->
          enqueue_piece st piece
      | None ->
          () ) ;
      st.src_pos <- st.src_pos + got
    end ;
    if got < want then st.decoder_eof <- true
  end

(* Decoder EOF, resolved exactly once: a broken header claim wins over the flush
   (no flush output is delivered on a [Truncated] outcome — both sides of the
   delegation law fail identically); a clean EOF drains the kernel's one flush
   into the carry. *)
let resolve_eof st rs =
  if st.flushed then Ok ()
  else if truncated_at_eof st then begin
    st.flushed <- true ;
    st.finished <- true ;
    Queue.clear st.carry ;
    st.carry_len <- 0 ;
    Error
      (Truncated
         { path= st.path
         ; expected_frames= st.native.Info.frames
         ; read_frames= st.src_pos } )
  end
  else begin
    ( match Resample.Kernel.flush rs.kernel with
    | Some piece ->
        enqueue_piece st piece
    | None ->
        () ) ;
    st.flushed <- true ;
    Ok ()
  end

(* {2 Reading one chunk} *)

let shrink_frames t ~channels ~frames ~n =
  if n = frames then t else Nx.shrink [|(0, channels); (0, n)|] t

let validate_out st ~frames out =
  let shape = Nx.shape out in
  if shape <> [|st.channels_out; frames|] then
    invalid_arg
      (Printf.sprintf
         "Soundml_io.Reader.read: out is %s; this call needs [%d; %d]"
         ( "["
         ^ String.concat "; " (Array.to_list (Array.map string_of_int shape))
         ^ "]" )
         st.channels_out frames ) ;
  if not (Nx.is_c_contiguous out) then
    invalid_arg "Soundml_io.Reader.read: out is not C-contiguous"

(* The native chunk: decode straight into the destination — the caller's [out]
   (zero copies on mono; one staged pass otherwise) or a fresh chunk buffer. *)
let read_chunk_native (type a) (st : a state) ~frames out :
    ((float, a) Nx.t option, error) result =
  let advertised = st.native.Info.frames in
  let want =
    if advertised > 0 then Stdlib.min frames (advertised - st.src_pos)
    else frames
  in
  if st.decoder_eof || want = 0 then begin
    (* [want = 0]: the advertised extent is exhausted — EOF without another
       decoder call *)
    if want = 0 then st.decoder_eof <- true ;
    st.finished <- true ;
    Ok None
  end
  else begin
    let dst_ba, view =
      match out with
      | Some o ->
          ( planar_array1 o
          , fun n -> shrink_frames o ~channels:st.channels_out ~frames ~n )
      | None ->
          let buf = Nx_buffer.create st.kind (st.channels_out * frames) in
          let t = Nx.of_buffer buf ~shape:[|st.channels_out; frames|] in
          ( Nx_buffer.to_bigarray1 buf
          , fun n -> shrink_frames t ~channels:st.channels_out ~frames ~n )
    in
    let got, err, details =
      stub_readf st.handle dst_ba st.mode 0 frames want st.channels
    in
    if err = soundml_io_err_staging then begin
      st.finished <- true ;
      Error (Io {path= st.path; op= "read"; details})
    end
    else begin
      st.src_pos <- st.src_pos + got ;
      st.out_served <- st.out_served + got ;
      if got < want then begin
        st.decoder_eof <- true ;
        if truncated_at_eof st then begin
          st.finished <- true ;
          Error
            (Truncated
               { path= st.path
               ; expected_frames= advertised
               ; read_frames= st.src_pos } )
        end
        else if got = 0 then begin
          st.finished <- true ;
          Ok None
        end
        else Ok (Some (view got))
      end
      else Ok (Some (view got))
    end
  end

(* The resampled chunk: decode blocks through the kernel until the carry covers
   the request or EOF, then serve exactly [min frames carry]. *)
let read_chunk_resampled (type a) (st : a state) rs ~frames out :
    ((float, a) Nx.t option, error) result =
  let rec fill () =
    if st.carry_len >= frames then Ok ()
    else if st.decoder_eof then resolve_eof st rs
    else begin
      decode_step st rs ; fill ()
    end
  in
  match fill () with
  | Error e ->
      Error e
  | Ok () ->
      let n = Stdlib.min frames st.carry_len in
      if n = 0 then begin
        st.finished <- true ;
        Ok None
      end
      else begin
        match out with
        | Some o ->
            serve_from_carry st ~dst_ba:(planar_array1 o) ~dst_total:frames
              ~dst_off:0 ~n ;
            Ok (Some (shrink_frames o ~channels:st.channels_out ~frames ~n))
        | None ->
            let buf = Nx_buffer.create st.kind (st.channels_out * n) in
            serve_from_carry st
              ~dst_ba:(Nx_buffer.to_bigarray1 buf)
              ~dst_total:n ~dst_off:0 ~n ;
            Ok (Some (Nx.of_buffer buf ~shape:[|st.channels_out; n|]))
      end

let read_chunk st ?out ~frames () =
  ensure_open st "Soundml_io.Reader.read" ;
  if frames < 1 then
    invalid_arg
      (Printf.sprintf
         "Soundml_io.Reader.read: cannot read %d frames (frames must be \
          positive)"
         frames ) ;
  Option.iter (validate_out st ~frames) out ;
  if st.finished then Ok None
  else
    match st.resampler with
    | None ->
        read_chunk_native st ~frames out
    | Some rs ->
        read_chunk_resampled st rs ~frames out

(* {2 Seeking} *)

let seek_state st ~frame =
  ensure_open st "Soundml_io.Reader.seek" ;
  if frame < 0 then
    invalid_arg
      (Printf.sprintf
         "Soundml_io.Reader.seek: cannot seek to frame %d (frames are \
          non-negative)"
         frame ) ;
  match st.resampler with
  | None ->
      let pos, details = stub_seek st.handle frame in
      if pos < 0 then
        Error
          (Io
             { path= st.path
             ; op= "seek"
             ; details= (if details = "" then "seek failed" else details) } )
      else begin
        st.src_pos <- pos ;
        st.out_served <- pos ;
        st.decoder_eof <- false ;
        st.finished <- false ;
        Ok ()
      end
  | Some rs -> (
      let advertised = st.native.Info.frames in
      let bound =
        if advertised > 0 then
          Some (Resample.Config.output_frames rs.config ~n:advertised)
        else None
      in
      match bound with
      | Some total when frame > total ->
          Error
            (Io
               { path= st.path
               ; op= "seek"
               ; details=
                   Printf.sprintf "cannot seek to frame %d of a %d-frame stream"
                     frame total } )
      | _ -> (
          (* Bit-identity is the contract, so the kernel is never fed an
             approximated suffix: backward targets restart the absolute stream
             at the origin, forward targets advance it — the delivered samples
             are the uninterrupted stream's own. *)
          let restart () =
            let pos, details = stub_seek st.handle 0 in
            if pos < 0 then
              Error
                (Io
                   { path= st.path
                   ; op= "seek"
                   ; details= (if details = "" then "seek failed" else details)
                   } )
            else begin
              Resample.Kernel.reset rs.kernel ;
              Queue.clear st.carry ;
              st.carry_len <- 0 ;
              st.src_pos <- 0 ;
              st.out_served <- 0 ;
              st.decoder_eof <- false ;
              st.flushed <- false ;
              st.finished <- false ;
              Ok ()
            end
          in
          let rec advance () =
            let need = frame - st.out_served in
            if need = 0 then Ok ()
            else if st.carry_len > 0 then begin
              drop_from_carry st (Stdlib.min need st.carry_len) ;
              advance ()
            end
            else if st.decoder_eof then
              if st.flushed then
                (* only reachable on unknown-length streams: the stream ended
                   before [frame]; the position is EOF *)
                Ok ()
              else
                match resolve_eof st rs with
                | Error e ->
                    Error e
                | Ok () ->
                    advance ()
            else begin
              decode_step st rs ; advance ()
            end
          in
          let start = if frame < st.out_served then restart () else Ok () in
          match start with
          | Error e ->
              Error e
          | Ok () ->
              st.finished <- false ;
              advance () ) )

(* {1 Reading} *)

let read (type a) ?sample_rate ?quality ?(mono = false)
    (dt : (float, a) Nx.dtype) path : (a audio, error) result =
  match open_state "Soundml_io.read" ?sample_rate ?quality ~mono dt path with
  | Error e ->
      Error e
  | Ok st -> (
      let elt = elt_size dt in
      let advertised = st.native.Info.frames in
      let liar () =
        Error
          (Io
             { path
             ; op= "read"
             ; details=
                 Printf.sprintf
                   "the header advertises %d frames x %d channels, more than \
                    the file could hold"
                   advertised st.channels } )
      in
      let finish result = close_state st ; result in
      match st.resampler with
      | None -> (
          if advertised = 0 then
            (* unknown length (or genuinely empty): decode to EOF *)
            let dst, total =
              read_to_eof st.kind st.handle ~mode:st.mode ~channels:st.channels
                ~channels_out:st.channels_out
            in
            finish
              (Ok
                 { data= Nx.of_buffer dst ~shape:[|st.channels_out; total|]
                 ; sample_rate= st.target_rate } )
          else if
            not
              (destination_admissible ~path ~frames:advertised
                 ~channels:st.channels_out ~elt )
          then finish (liar ())
          else
            match read_chunk st ~frames:advertised () with
            | Ok (Some data) ->
                finish (Ok {data; sample_rate= st.target_rate})
            | Ok None ->
                (* unreachable for a positive advertised count: a shortfall is
                   [Truncated] above; defensive completeness *)
                finish
                  (Ok
                     { data= Nx.zeros dt [|st.channels_out; 0|]
                     ; sample_rate= st.target_rate } )
            | Error e ->
                finish (Error e) )
      | Some rs ->
          if advertised = 0 then begin
            (* unknown length: chunked to EOF, pieces concatenated once — the
               documented transient for this degenerate case only *)
            let rec pump () =
              if st.decoder_eof then resolve_eof st rs
              else begin
                decode_step st rs ; pump ()
              end
            in
            match pump () with
            | Error e ->
                finish (Error e)
            | Ok () ->
                let pieces =
                  Queue.fold (fun acc p -> p.pt :: acc) [] st.carry
                in
                let data =
                  match List.rev pieces with
                  | [] ->
                      Nx.zeros dt [|st.channels_out; 0|]
                  | pieces ->
                      Nx.concatenate ~axis:1 pieces
                in
                finish (Ok {data; sample_rate= st.target_rate})
          end
          else if
            not
              (destination_admissible ~path ~frames:advertised
                 ~channels:st.channels_out ~elt )
          then finish (liar ())
          else begin
            let l =
              (Resample.Config.rate rs.config).Soundml.Pipeline.Rate.num
            in
            if not (mul_no_overflow advertised l) then finish (liar ())
            else begin
              let total_out =
                Resample.Config.output_frames rs.config ~n:advertised
              in
              if
                not
                  (destination_admissible ~path ~frames:total_out
                     ~channels:st.channels_out ~elt )
              then finish (liar ())
              else begin
                (* the preallocated fused destination: kernel pieces land here
                   as they are emitted; the native-rate signal never exists as a
                   whole buffer *)
                let dst =
                  Nx_buffer.create st.kind (st.channels_out * total_out)
                in
                let dst_ba = Nx_buffer.to_bigarray1 dst in
                let rec pump () =
                  if st.carry_len > 0 then begin
                    serve_from_carry st ~dst_ba ~dst_total:total_out
                      ~dst_off:st.out_served ~n:st.carry_len ;
                    pump ()
                  end
                  else if st.decoder_eof then
                    if st.flushed then Ok ()
                    else
                      match resolve_eof st rs with
                      | Error e ->
                          Error e
                      | Ok () ->
                          pump ()
                  else begin
                    decode_step st rs ; pump ()
                  end
                in
                match pump () with
                | Error e ->
                    finish (Error e)
                | Ok () ->
                    assert (st.out_served = total_out) ;
                    finish
                      (Ok
                         { data=
                             Nx.of_buffer dst
                               ~shape:[|st.channels_out; total_out|]
                         ; sample_rate= st.target_rate } )
              end
            end
          end )

let fold ?(block = 65536) ?sample_rate ?quality ?(mono = false) dt path ~init ~f
    =
  if block < 1 then
    invalid_arg
      (Printf.sprintf
         "Soundml_io.fold: cannot fold %d-frame blocks (block must be positive)"
         block ) ;
  match open_state "Soundml_io.fold" ?sample_rate ?quality ~mono dt path with
  | Error e ->
      Error e
  | Ok st ->
      Fun.protect
        ~finally:(fun () -> close_state st)
        (fun () ->
          (* the one lent chunk buffer, reused for every block *)
          let out =
            Nx.of_buffer
              (Nx_buffer.create st.kind (st.channels_out * block))
              ~shape:[|st.channels_out; block|]
          in
          let rec loop acc =
            match read_chunk st ~out ~frames:block () with
            | Ok None ->
                Ok acc
            | Ok (Some chunk) ->
                loop (f acc chunk)
            | Error e ->
                Error e
          in
          loop init )

module Reader = struct
  type 'a t = 'a state

  let open_ ?sample_rate ?quality ?mono dt path =
    open_state "Soundml_io.Reader.open_" ?sample_rate ?quality ?mono dt path

  let info st =
    ensure_open st "Soundml_io.Reader.info" ;
    st.native

  let format st =
    ensure_open st "Soundml_io.Reader.format" ;
    Soundml.Pipeline.Format.audio st.dt ~sample_rate:st.target_rate
      ~channels:st.channels_out

  let read ?out st ~frames = read_chunk st ?out ~frames ()

  let seek st ~frame = seek_state st ~frame

  let close = close_state
end

(* {1 Writing} *)

let write ?format path (a : 'a audio) =
  check_dtype "Soundml_io.write" (Nx.dtype a.data) ;
  if a.sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "Soundml_io.write: sample_rate %d is not positive (a signal has a \
          rate)"
         a.sample_rate ) ;
  let channels, frames =
    match Nx.shape a.data with
    | [|frames|] ->
        (1, frames)
    | [|channels; frames|] ->
        (channels, frames)
    | shape ->
        invalid_arg
          (Printf.sprintf
             "Soundml_io.write: rank-%d audio (writable audio is [channels; \
              frames] or rank-one mono)"
             (Array.length shape) )
  in
  let format_result =
    match format with Some f -> Ok f | None -> Format.of_path path
  in
  match format_result with
  | Error e ->
      Error e
  | Ok fmt -> (
      let code = Format.code fmt in
      let unsupported =
        Stdlib.Format.asprintf "%a refuses %d channel%s at %d Hz" Format.pp fmt
          channels
          (if channels = 1 then "" else "s")
          a.sample_rate
      in
      (* Vorbis carries at most 255 channels (an 8-bit header field); libsndfile
         1.2.2's sf_format_check does not enforce the bound and its Ogg writer
         segfaults past it (measured: 256 channels crash, 255 encode fine), so
         the refusal must happen here, before any sf_open. *)
      if Format.encoding fmt = `Vorbis && channels > 255 then
        Error (Unsupported {path; details= unsupported})
      else if not (stub_format_check code channels a.sample_rate) then
        Error (Unsupported {path; details= unsupported})
      else
        match stub_open_write path code channels a.sample_rate with
        | Error e ->
            Error (write_open_error path ~unsupported e)
        | Ok h -> (
            let buf = Nx.to_buffer a.data in
            let ba = Nx_buffer.to_bigarray1 buf in
            let mode = if channels = 1 then mode_direct else mode_planar in
            let written, _err, details =
              stub_writef h ba mode 0 frames frames channels
            in
            let close_code, close_details = stub_close h in
            if written < frames then
              Error
                (Io
                   { path
                   ; op= "write"
                   ; details=
                       ( if details = "" then
                           Printf.sprintf "wrote %d of %d frames" written frames
                         else details ) } )
            else
              match close_code with
              | 0 ->
                  Ok ()
              | _ ->
                  Error (Io {path; op= "close"; details= close_details}) ) )
