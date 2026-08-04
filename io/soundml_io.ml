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

let read (type a) ?(mono = false) (dt : (float, a) Nx.dtype) path :
    (a audio, error) result =
  check_dtype "Soundml_io.read" dt ;
  match stub_open_read path with
  | Error e ->
      Error (read_open_error path e)
  | Ok (h, frames, channels, sample_rate, _code, _seekable) ->
      let mode, channels_out = read_mode ~mono ~channels in
      let kind = kind_of_dtype dt in
      let finish buf frames_out =
        let (_ : int * string) = stub_close h in
        Ok
          { data= Nx.of_buffer buf ~shape:[|channels_out; frames_out|]
          ; sample_rate }
      in
      if frames = 0 then
        (* Unknown length (or genuinely empty): decode to EOF. *)
        let dst, total = read_to_eof kind h ~mode ~channels ~channels_out in
        finish dst total
      else if
        not
          (destination_admissible ~path ~frames ~channels:channels_out
             ~elt:(elt_size dt) )
      then
        let (_ : int * string) = stub_close h in
        Error
          (Io
             { path
             ; op= "read"
             ; details=
                 Printf.sprintf
                   "the header advertises %d frames x %d channels, more than \
                    the file could hold"
                   frames channels } )
      else
        let buf = Nx_buffer.create kind (channels_out * frames) in
        let ba = Nx_buffer.to_bigarray1 buf in
        let delivered, _err, _details =
          stub_readf h ba mode 0 frames frames channels
        in
        if delivered < frames then
          let (_ : int * string) = stub_close h in
          Error
            (Truncated {path; expected_frames= frames; read_frames= delivered})
        else finish buf frames

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
