(* Headerless PCM: [Raw] against libsndfile, bit for bit. Decoding a container's
   data payload equals [read] of the container — every lossless corpus file, and
   random bytes of every encoding in hand-built WAV, AIFF and CAF headers, at
   both dtypes — and encoding a tensor equals the payload [write] puts in the
   container, over full-scale, clipping, tie and sub-LSB signals. Then the
   identity laws of the lossless cells, the pinned NaN and floor behaviour, the
   chunked reader's protocol — a channel refill splitting a frame at every byte
   offset, a pipe fed one byte per write, end of file, partial frames, an
   exception escaping mid-frame — the error matrix, and the streaming loop the
   documentation shows.

   Comparisons are on storage bits ([Nx.bitcast]): [Nx.to_array] widens float32
   to double, which quiets signalling NaNs and would hide a decode that did the
   same. *)

open Windtrap
open Soundml_io

let tmp_dir = fixture (fun () -> Filename.temp_dir "soundml_io_raw" "")

let temp_path ext = Filename.temp_file ~temp_dir:(tmp_dir ()) "raw" ext

let ok ~msg = function Ok v -> v | Error e -> failf "%s: %a" msg pp_error e

let invalid = function Invalid_argument _ -> true | _ -> false

let encodings : Raw.encoding list =
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

let random_bytes n seed =
  let st = Random.State.make [|seed|] in
  String.init n (fun _ -> Char.chr (Random.State.int st 256))

(* {2 Storage bits} *)

let bits (type a) (t : (float, a) Nx.t) =
  match Nx.dtype t with
  | Nx.Float32 ->
      Array.map Int64.of_int32 (Nx.to_array (Nx.bitcast Nx.int32 t))
  | Nx.Float64 ->
      Nx.to_array (Nx.bitcast Nx.int64 t)
  | dt ->
      failf "bits: %a storage" Nx.pp_dtype dt

let shape_string s =
  String.concat "x" (Array.to_list (Array.map string_of_int s))

let check_bits ~msg expected actual =
  if Nx.shape expected <> Nx.shape actual then
    failf "%s: shape %s, expected %s" msg
      (shape_string (Nx.shape actual))
      (shape_string (Nx.shape expected)) ;
  let e = bits expected and a = bits actual in
  Array.iteri
    (fun i ev ->
      if not (Int64.equal ev a.(i)) then
        failf "%s: element %d holds bits %Lx, expected %Lx" msg i a.(i) ev )
    e

let check_string ~msg expected actual =
  if not (String.equal expected actual) then begin
    let n = Int.min (String.length expected) (String.length actual) in
    let i = ref 0 in
    while !i < n && Char.equal expected.[!i] actual.[!i] do
      incr i
    done ;
    failf "%s: %d bytes, expected %d; they first differ at byte %d" msg
      (String.length actual) (String.length expected) !i
  end

(* {2 Container payloads} *)

let u32le s i = Int32.to_int (String.get_int32_le s i) land 0xFFFFFFFF

let u32be s i = Int32.to_int (String.get_int32_be s i) land 0xFFFFFFFF

(* The body (offset, size) of the first chunk [id] from byte [pos] on, chunks
   being a 4-byte id and a size field read by [size], [header] bytes in all;
   [pad] rounds bodies up to even sizes (RIFF and IFF). *)
let rec find_chunk s ~pos ~header ~size ~pad id =
  if pos + header > String.length s then failf "no %S chunk" id
  else
    let n = size s (pos + 4) in
    if String.equal (String.sub s pos 4) id then (pos + header, n)
    else
      let next = pos + header + n + if pad then n land 1 else 0 in
      find_chunk s ~pos:next ~header ~size ~pad id

(* The data payload of a WAV, AIFF or CAF file, and whether it is big-endian. *)
let payload path =
  let s = In_channel.with_open_bin path In_channel.input_all in
  match String.sub s 0 4 with
  | "RIFF" ->
      let o, n = find_chunk s ~pos:12 ~header:8 ~size:u32le ~pad:true "data" in
      (String.sub s o n, false)
  | "FORM" ->
      let o, n = find_chunk s ~pos:12 ~header:8 ~size:u32be ~pad:true "SSND" in
      let offset = u32be s o in
      (String.sub s (o + 8 + offset) (n - 8 - offset), true)
  | "caff" ->
      let size s i = Int64.to_int (String.get_int64_be s i) in
      let o, n = find_chunk s ~pos:8 ~header:12 ~size ~pad:false "data" in
      (* a data chunk of unknown size (-1) runs to the end of the file; its body
         opens with a 4-byte edit count *)
      let n = if n < 0 then String.length s - o else n in
      (String.sub s (o + 4) (n - 4), true)
  | magic ->
      failf "%s: unknown container %S" path magic

let raw_encoding (encoding : Format.encoding) ~big : Raw.encoding =
  match (encoding, big) with
  | `Pcm_16, false ->
      `S16le
  | `Pcm_16, true ->
      `S16be
  | `Pcm_24, false ->
      `S24le
  | `Pcm_24, true ->
      `S24be
  | `Pcm_32, false ->
      `S32le
  | `Pcm_32, true ->
      `S32be
  | `Float32, false ->
      `F32le
  | `Float32, true ->
      `F32be
  | `Float64, false ->
      `F64le
  | `Float64, true ->
      `F64be
  | `Vorbis, _ ->
      fail "Vorbis has no raw form"

(* [data] behind a minimal header libsndfile reads as [e]: WAV (format tag 1, or
   3 for floats) for the little-endian and unsigned encodings, AIFF for the
   signed big-endian integers, CAF (lpcm, float, big-endian) for the big-endian
   floats; 8000 Hz. *)
let container (e : Raw.encoding) ~channels data =
  let bps = Raw.sample_bytes e in
  let fb = channels * bps and n = String.length data in
  let b = Buffer.create (n + 128) in
  let str = Buffer.add_string b in
  let le32 v = Buffer.add_int32_le b (Int32.of_int v)
  and be32 v = Buffer.add_int32_be b (Int32.of_int v) in
  ( match e with
  | `U8 | `S16le | `S24le | `S32le | `F32le | `F64le ->
      let tag = match e with `F32le | `F64le -> 3 | _ -> 1 in
      str "RIFF" ;
      le32 (36 + n) ;
      str "WAVE" ;
      str "fmt " ;
      le32 16 ;
      Buffer.add_uint16_le b tag ;
      Buffer.add_uint16_le b channels ;
      le32 8000 ;
      le32 (8000 * fb) ;
      Buffer.add_uint16_le b fb ;
      Buffer.add_uint16_le b (8 * bps) ;
      str "data" ;
      le32 n
  | `S8 | `S16be | `S24be | `S32be ->
      str "FORM" ;
      be32 (46 + n) ;
      str "AIFF" ;
      str "COMM" ;
      be32 18 ;
      Buffer.add_uint16_be b channels ;
      be32 (n / fb) ;
      Buffer.add_uint16_be b (8 * bps) ;
      (* 8000 as an 80-bit extended float *)
      str "\x40\x0B\xFA\x00\x00\x00\x00\x00\x00\x00" ;
      str "SSND" ;
      be32 (8 + n) ;
      be32 0 ;
      be32 0
  | `F32be | `F64be ->
      str "caff" ;
      Buffer.add_uint16_be b 1 ;
      Buffer.add_uint16_be b 0 ;
      str "desc" ;
      Buffer.add_int64_be b 32L ;
      Buffer.add_int64_be b (Int64.bits_of_float 8000.) ;
      str "lpcm" ;
      be32 1 (* float, big-endian *) ;
      be32 fb ;
      be32 1 ;
      be32 channels ;
      be32 (8 * bps) ;
      str "data" ;
      Buffer.add_int64_be b (Int64.of_int (4 + n)) ;
      be32 0 ) ;
  str data ; Buffer.contents b

(* {2 Decode parity with read} *)

let corpus = "corpus"

let decode_parity_tests =
  [ test "every lossless corpus file: Raw.decode of the payload = read"
      (fun () ->
        let checked = ref 0 in
        Sys.readdir corpus |> Array.to_list |> List.sort String.compare
        |> List.iter (fun file ->
            let path = Filename.concat corpus file in
            if not (Sys.is_directory path) then
              let probe = ok ~msg:file (info path) in
              match probe.Info.format with
              | Some f
                when ( match Format.container f with
                       | `Wav | `Aiff | `Caf ->
                           true
                       | `Flac | `Ogg ->
                           false )
                     && Format.encoding f <> `Vorbis ->
                  let data, big = payload path in
                  let e = raw_encoding (Format.encoding f) ~big in
                  let channels = probe.Info.channels in
                  let cell (type a) (dt : (float, a) Nx.dtype) =
                    let msg =
                      Stdlib.Format.asprintf "%s as %s at %a" file
                        (Raw.encoding_name e) Nx.pp_dtype dt
                    in
                    let a = ok ~msg (read dt path) in
                    check_bits ~msg a.data (Raw.decode e ~channels dt data)
                  in
                  cell Nx.float32 ; cell Nx.float64 ; incr checked
              | _ ->
                  () ) ;
        (* the eight WAV, AIFF and CAF files of the corpus, at least *)
        is_true ~msg:"lossless corpus files checked" (!checked >= 8) )
  ; test "random bytes of every encoding x 1-3 channels: Raw.decode = read"
      (fun () ->
        List.iter
          (fun e ->
            List.iter
              (fun channels ->
                let fb = channels * Raw.sample_bytes e in
                let data = random_bytes (fb * 3000) ((fb * 7) + channels) in
                let ext =
                  match e with
                  | `F32be | `F64be ->
                      ".caf"
                  | `S8 | `S16be | `S24be | `S32be ->
                      ".aiff"
                  | _ ->
                      ".wav"
                in
                let path = temp_path ext in
                Out_channel.with_open_bin path (fun oc ->
                    Out_channel.output_string oc (container e ~channels data) ) ;
                let cell (type a) (dt : (float, a) Nx.dtype) =
                  let msg =
                    Stdlib.Format.asprintf "%s x %d at %a" (Raw.encoding_name e)
                      channels Nx.pp_dtype dt
                  in
                  let a = ok ~msg (read dt path) in
                  check_bits ~msg a.data (Raw.decode e ~channels dt data)
                in
                cell Nx.float32 ; cell Nx.float64 )
              [1; 2; 3] )
          encodings ) ]

(* {2 Encode parity with write} *)

(* Full scale and past it, infinities, huge and denormal values, the edges of
   the 32- and 24-bit grids, the quarter-LSB ties of the 8-, 16-, 24- and 32-bit
   grids around small and large codes, then uniform noise over [-1.5, 1.5] and
   tiny noise around zero. NaN payloads only for the float encodings: what an
   integer encoding makes of NaN is the C library's lrint, pinned on the raw
   side alone. *)
let write_signal ~nan ~frames seed =
  let specials =
    [ 0.
    ; -0.
    ; 1.
    ; -1.
    ; 1.5
    ; -1.5
    ; infinity
    ; neg_infinity
    ; 1e300
    ; -1e300
    ; 5e-324
    ; -5e-324
    ; 1. -. 0x1p-31
    ; 1. -. 0x1p-32
    ; 1. -. 0x1p-24
    ; -1. +. 0x1p-31
    ; 0x1p-16
    ; -0x1p-16
    ; 0x1p-17
    ; -0x1p-17
    ; 0x1p-24
    ; -0x1p-24 ]
  in
  let ties =
    List.concat_map
      (fun bits ->
        let lsb = Float.ldexp 1. (-(bits - 1)) in
        List.concat_map
          (fun k ->
            List.map
              (fun fr -> (Float.of_int k +. fr) *. lsb)
              [0.; 0.25; 0.5; 0.75; -0.25; -0.5; -0.75] )
          [-3; -2; -1; 0; 1; 2; 3; 99; -99] )
      [8; 16; 24; 32]
  in
  let nans =
    if nan then
      [ Int64.float_of_bits 0x7FF0000000000001L (* signalling *)
      ; Int64.float_of_bits 0xFFF8000000000123L ]
    else []
  in
  let fixed = Array.of_list (specials @ ties @ nans) in
  let st = Random.State.make [|seed|] in
  Array.init frames (fun i ->
      if i < Array.length fixed then fixed.(i)
      else if i mod 7 = 0 then Random.State.float st 1e-6 -. 5e-7
      else Random.State.float st 3. -. 1.5 )

let write_cells =
  [ (`Wav, `Pcm_16)
  ; (`Wav, `Pcm_24)
  ; (`Wav, `Pcm_32)
  ; (`Wav, `Float32)
  ; (`Wav, `Float64)
  ; (`Aiff, `Pcm_16)
  ; (`Aiff, `Pcm_24)
  ; (`Aiff, `Pcm_32)
  ; (`Caf, `Pcm_16)
  ; (`Caf, `Float32)
  ; (`Caf, `Float64) ]

let encode_parity_tests =
  [ test "Raw.encode = the payload write stores; decoding it = read" (fun () ->
        List.iter
          (fun (container, encoding) ->
            List.iter
              (fun channels ->
                let frames = 5000 in
                let nan =
                  match encoding with `Float32 | `Float64 -> true | _ -> false
                in
                let data =
                  Array.concat
                    (List.init channels (fun c ->
                         write_signal ~nan ~frames (17 + c) ) )
                in
                let cell (type a) (dt : (float, a) Nx.dtype) =
                  let x = Nx.create dt [|channels; frames|] data in
                  let fmt = Format.create ~encoding container in
                  let path =
                    temp_path
                      ( match container with
                      | `Wav ->
                          ".wav"
                      | `Aiff ->
                          ".aiff"
                      | _ ->
                          ".caf" )
                  in
                  ok ~msg:"write"
                    (write ~format:fmt path {data= x; sample_rate= 8000}) ;
                  let stored, big = payload path in
                  let e = raw_encoding encoding ~big in
                  let msg =
                    Stdlib.Format.asprintf "%a x %d from %a" Format.pp fmt
                      channels Nx.pp_dtype dt
                  in
                  let mine = Raw.encode e x in
                  check_string ~msg stored mine ;
                  let a = ok ~msg (read dt path) in
                  check_bits ~msg:(msg ^ ", read back") a.data
                    (Raw.decode e ~channels dt mine)
                in
                cell Nx.float32 ; cell Nx.float64 )
              [1; 3] )
          write_cells )
  ; test "a strided tensor encodes as its contiguous copy" (fun () ->
        let x =
          Nx.create Nx.float64 [|3; 64|]
            (Array.init 192 (fun i -> Float.of_int (i - 96) /. 97.))
        in
        let t = Nx.transpose (Nx.transpose x) in
        let s = Nx.slice [Nx.R (0, 3); Nx.R (8, 40)] x in
        List.iter
          (fun e ->
            check_string ~msg:"transposed back" (Raw.encode e x)
              (Raw.encode e t) ;
            check_string ~msg:"sliced"
              (Raw.encode e (Nx.contiguous s))
              (Raw.encode e s) )
          encodings )
  ; test "rank one is mono" (fun () ->
        let x = Nx.create Nx.float32 [|5|] [|0.; 0.5; -0.5; 1.; -1.|] in
        List.iter
          (fun e ->
            check_string ~msg:(Raw.encoding_name e)
              (Raw.encode e (Nx.reshape [|1; 5|] x))
              (Raw.encode e x) )
          encodings ) ]

(* {2 Pinned conversions} *)

let pin_tests =
  [ test "NaN writes the zero code to every integer encoding" (fun () ->
        List.iter
          (fun (e, zero) ->
            check_string
              ~msg:(Raw.encoding_name e ^ " from float64")
              zero
              (Raw.encode e (Nx.create Nx.float64 [|1; 1|] [|Float.nan|])) ;
            check_string
              ~msg:(Raw.encoding_name e ^ " from float32")
              zero
              (Raw.encode e (Nx.create Nx.float32 [|1; 1|] [|Float.nan|])) )
          [ (`S8, "\x00")
          ; (`U8, "\x80")
          ; (`S16le, "\x00\x00")
          ; (`S16be, "\x00\x00")
          ; (`S24le, "\x00\x00\x00")
          ; (`S24be, "\x00\x00\x00")
          ; (`S32le, "\x00\x00\x00\x00")
          ; (`S32be, "\x00\x00\x00\x00") ] )
  ; test "integers floor below the LSB, as libsndfile's clipping writer"
      (fun () ->
        let lsb = 0x1p-15 in
        let x =
          Nx.create Nx.float64 [|6|]
            [| 0.75 *. lsb
             ; -0.5 *. lsb
             ; 1.5 *. lsb
             ; -0.25 *. lsb
             ; -1e-10
             ; 1. -. 0x1p-20 |]
        in
        (* 0, -1, 1, -1, 0 (rounds to -0 at 2^-31), 32767 *)
        check_string ~msg:"s16le"
          "\x00\x00\xff\xff\x01\x00\xff\xff\x00\x00\xff\x7f"
          (Raw.encode `S16le x) ;
        check_string ~msg:"s16be"
          "\x00\x00\xff\xff\x00\x01\xff\xff\x00\x00\x7f\xff"
          (Raw.encode `S16be x) )
  ; test "8-bit from float64 rounds the double, s8 too" (fun () ->
        let x =
          Nx.create Nx.float64 [|1; 5|]
            [| 1. -. 0x1p-26
             ; 0.5 -. 0x1p-27
             ; -0.5 -. 0x1p-27
             ; 0.25 +. 0x1p-30
             ; (1. /. 128.) -. 0x1p-31 |]
        in
        (* 127, 63, -65, 32 and 0. libsndfile 1.2.2's u8 writer agrees (ff bf 3f
           a0 80), but its s8 writer from double, d2sc_clip_array, rounds
           through a float and writes 80 40 c0 20 00: the first sample wraps to
           -128. *)
        check_string ~msg:"s8" "\x7f\x3f\xbf\x20\x00" (Raw.encode `S8 x) ;
        check_string ~msg:"u8" "\xff\xbf\x3f\xa0\x80" (Raw.encode `U8 x) )
  ; test "full scale saturates; float32 keeps every bit, signalling NaN too"
      (fun () ->
        let x = Nx.create Nx.float64 [|4|] [|1.; -1.; 2.; -2.|] in
        check_string ~msg:"s24le"
          "\xff\xff\x7f\x00\x00\x80\xff\xff\x7f\x00\x00\x80"
          (Raw.encode `S24le x) ;
        let snan =
          Nx.bitcast Nx.float32 (Nx.create Nx.int32 [|1|] [|0x7F800001l|])
        in
        check_string ~msg:"f32le" "\x01\x00\x80\x7f" (Raw.encode `F32le snan) ;
        check_string ~msg:"f32be" "\x7f\x80\x00\x01" (Raw.encode `F32be snan) ;
        check_bits ~msg:"f32le decode" snan
          (Nx.reshape [|1|]
             (Raw.decode `F32le ~channels:1 Nx.float32 "\x01\x00\x80\x7f") ) ;
        check_bits ~msg:"f32be decode" snan
          (Nx.reshape [|1|]
             (Raw.decode `F32be ~channels:1 Nx.float32 "\x7f\x80\x00\x01") ) )
  ; test "integers read normalised: v * 2^-(bits - 1), u8 centred on 128"
      (fun () ->
        let cells =
          [ (`S8, "\x80\x7f\x01", [|-1.; 127. /. 128.; 1. /. 128.|])
          ; (`U8, "\x00\xff\x80", [|-1.; 127. /. 128.; 0.|])
          ; ( `S16le
            , "\x00\x80\xff\x7f\xff\xff"
            , [|-1.; 32767. /. 32768.; -0x1p-15|] )
          ; ( `S16be
            , "\x80\x00\x7f\xff\xff\xff"
            , [|-1.; 32767. /. 32768.; -0x1p-15|] )
          ; ( `S24le
            , "\x00\x00\x80\x01\x00\x00\xff\xff\xff"
            , [|-1.; 0x1p-23; -0x1p-23|] )
          ; ( `S24be
            , "\x80\x00\x00\x00\x00\x01\xff\xff\xff"
            , [|-1.; 0x1p-23; -0x1p-23|] )
          ; (`S32le, "\x00\x00\x00\x80\x01\x00\x00\x00", [|-1.; 0x1p-31|])
          ; (`S32be, "\x80\x00\x00\x00\x00\x00\x00\x01", [|-1.; 0x1p-31|]) ]
        in
        List.iter
          (fun (e, s, expected) ->
            let n = Array.length expected in
            check_bits ~msg:(Raw.encoding_name e)
              (Nx.create Nx.float64 [|1; n|] expected)
              (Raw.decode e ~channels:1 Nx.float64 s) )
          cells ) ]

(* {2 Identity laws} *)

let identity_tests =
  [ test "encode (decode s) = s wherever the dtype holds the encoding" (fun () ->
        List.iter
          (fun e ->
            List.iter
              (fun channels ->
                let fb = channels * Raw.sample_bytes e in
                let s = random_bytes (fb * 2000) ((fb * 11) + channels) in
                let cell (type a) (dt : (float, a) Nx.dtype) =
                  check_string
                    ~msg:
                      (Stdlib.Format.asprintf "%s x %d at %a"
                         (Raw.encoding_name e) channels Nx.pp_dtype dt )
                    s
                    (Raw.encode e (Raw.decode e ~channels dt s))
                in
                (* float32 holds integers of up to 24 bits and float32 bits;
                   float64 everything but float32's signalling NaNs *)
                ( match e with
                | `S32le | `S32be | `F64le | `F64be ->
                    ()
                | _ ->
                    cell Nx.float32 ) ;
                match e with `F32le | `F32be -> () | _ -> cell Nx.float64 )
              [1; 2; 3] )
          encodings ) ]

(* {2 The chunked reader} *)

(* Every chunk until the end, each checked for shape and layout. *)
let drain r ~channels ~frames =
  let rec go acc =
    match Raw.Reader.read r ~frames with
    | Ok (Some c) ->
        ( match Nx.shape c with
        | [|ch; n|] when ch = channels && n >= 1 && n <= frames ->
            ()
        | shape ->
            failf "a chunk of %s for %d channels, frames=%d"
              (shape_string shape) channels frames ) ;
        is_true ~msg:"chunks are C-contiguous" (Nx.is_c_contiguous c) ;
        go (c :: acc)
    | Ok None ->
        (List.rev acc, None)
    | Error e ->
        (List.rev acc, Some e)
  in
  go []

(* The chunks, in order, are [whole] cut every [frames] frames. Compared element
   by element on bits, without concatenating: an Nx copy of float32 data is not
   this test's subject. *)
let check_chunks ~msg ~frames whole chunks =
  let channels = (Nx.shape whole).(0) and total = (Nx.shape whole).(1) in
  let wb = bits whole in
  let served =
    List.fold_left
      (fun off c ->
        let n = (Nx.shape c).(1) in
        if off + n < total && n <> frames then
          failf "%s: a %d-frame chunk before the last (frames=%d)" msg n frames ;
        let cb = bits c in
        for ch = 0 to channels - 1 do
          for f = 0 to n - 1 do
            if not (Int64.equal cb.((ch * n) + f) wb.((ch * total) + off + f))
            then
              failf "%s: frame %d of channel %d differs (frames=%d)" msg
                (off + f) ch frames
          done
        done ;
        off + n )
      0 chunks
  in
  equal ~msg:(msg ^ ": frames served") int total served

(* A regular file of [junk] bytes then [s], opened with the junk consumed: the
   channel's next 65536-byte refill then falls [65536 - junk] bytes into [s], at
   that offset modulo the frame size, deterministically. *)
let with_prefixed_file s ~junk k =
  let path = temp_path ".raw" in
  Out_channel.with_open_bin path (fun oc ->
      Out_channel.output_string oc (String.make junk 'J') ;
      Out_channel.output_string oc s ) ;
  In_channel.with_open_bin path (fun ic ->
      if junk > 0 then
        ignore (In_channel.really_input_string ic junk : string option) ;
      k ic )

(* A pipe fed by a writer domain, one write(2) per piece, paced so that the
   reader's refills likely see them one at a time — likely only: the asserted
   equalities hold at any delivery granularity, so the pacing decides coverage,
   never the verdict. The payloads fit the pipe's buffer, so the writer always
   finishes and is joined before the read end closes. *)
let via_pipe ?(pace = 0.0003) pieces k =
  let rd, wr = Unix.pipe ~cloexec:true () in
  let writer =
    Domain.spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Unix.close wr)
          (fun () ->
            List.iter
              (fun p ->
                let n = String.length p in
                let rec put o =
                  if o < n then put (o + Unix.write_substring wr p o (n - o))
                in
                put 0 ;
                if pace > 0. then Unix.sleepf pace )
              pieces ) )
  in
  let ic = Unix.in_channel_of_descr rd in
  Fun.protect
    ~finally:(fun () -> Domain.join writer ; close_in_noerr ic)
    (fun () -> k ic)

let bytewise s = List.init (String.length s) (fun i -> String.make 1 s.[i])

let reader_cells =
  (* (encoding, channels, frames, dtype tag): payloads over 64 KiB *)
  [ (`S24le, 2, 12000, `F32)
  ; (`F64be, 3, 3000, `F64)
  ; (`S16le, 1, 36000, `F32)
  ; (`U8, 5, 14000, `F64)
  ; (`S32be, 2, 9000, `F32)
  ; (`F32le, 3, 6000, `F32) ]

let chunk_sizes = [1; 2; 3; 7; 64; 1000; 65536]

let reader_cell (type a) (dt : (float, a) Nx.dtype) (e, channels, frames) =
  let fb = channels * Raw.sample_bytes e in
  let s = random_bytes (frames * fb) (fb + channels) in
  let whole = Raw.decode e ~channels dt s in
  for junk = 0 to fb - 1 do
    List.iter
      (fun size ->
        let msg =
          Printf.sprintf "%s x %d, refill %d bytes into a frame"
            (Raw.encoding_name e) channels
            ((65536 - junk) mod fb)
        in
        let chunks, err =
          with_prefixed_file s ~junk (fun ic ->
              drain (Raw.Reader.create e ~channels dt ic) ~channels ~frames:size )
        in
        Option.iter (fun e -> failf "%s: %a" msg pp_error e) err ;
        check_chunks ~msg ~frames:size whole chunks )
      chunk_sizes
  done

let reader_tests =
  [ test "a refill splitting a frame at every byte offset, every chunk size"
      (fun () ->
        List.iter
          (fun (e, channels, frames, dt) ->
            match dt with
            | `F32 ->
                reader_cell Nx.float32 (e, channels, frames)
            | `F64 ->
                reader_cell Nx.float64 (e, channels, frames) )
          reader_cells )
  ; test "a pipe delivering one byte per write yields the file's chunks"
      (fun () ->
        List.iter
          (fun (e, channels, _, _) ->
            let fb = channels * Raw.sample_bytes e in
            (* about 240 bytes: one paced write each *)
            let s = random_bytes (fb * Int.max 8 (240 / fb)) (fb * 3) in
            let whole = Raw.decode e ~channels Nx.float64 s in
            List.iter
              (fun size ->
                let chunks, err =
                  via_pipe (bytewise s) (fun ic ->
                      drain
                        (Raw.Reader.create e ~channels Nx.float64 ic)
                        ~channels ~frames:size )
                in
                let msg =
                  Printf.sprintf "%s x %d over a pipe" (Raw.encoding_name e)
                    channels
                in
                Option.iter (fun e -> failf "%s: %a" msg pp_error e) err ;
                check_chunks ~msg ~frames:size whole chunks )
              [1; 3; 64] )
          reader_cells )
  ; test "a full chunk leaves the channel at its frame boundary" (fun () ->
        let e = `S24be and channels = 2 in
        let fb = 6 in
        let s = random_bytes (fb * 100) 5 in
        with_prefixed_file s ~junk:0 (fun ic ->
            let r = Raw.Reader.create e ~channels Nx.float32 ic in
            for _ = 1 to 2 do
              match Raw.Reader.read r ~frames:30 with
              | Ok (Some c) ->
                  equal ~msg:"full chunk" (array int) [|2; 30|] (Nx.shape c)
              | _ ->
                  fail "expected a full chunk"
            done ;
            check_string ~msg:"the bytes after two chunks"
              (String.sub s (60 * fb) (40 * fb))
              (In_channel.input_all ic) ) )
  ; test "EOF is Ok None after the short last chunk, and forever" (fun () ->
        let s = random_bytes (4 * 10) 1 in
        with_prefixed_file s ~junk:0 (fun ic ->
            let r = Raw.Reader.create `S16be ~channels:2 Nx.float32 ic in
            ( match Raw.Reader.read r ~frames:7 with
            | Ok (Some c) ->
                equal ~msg:"full" (array int) [|2; 7|] (Nx.shape c)
            | _ ->
                fail "expected a full chunk" ) ;
            ( match Raw.Reader.read r ~frames:7 with
            | Ok (Some c) ->
                equal ~msg:"short last" (array int) [|2; 3|] (Nx.shape c)
            | _ ->
                fail "expected the short last chunk" ) ;
            for _ = 1 to 3 do
              match Raw.Reader.read r ~frames:7 with
              | Ok None ->
                  ()
              | Ok (Some _) ->
                  fail "EOF delivered data"
              | Error e ->
                  failf "EOF errored: %a" pp_error e
            done ) )
  ; test "empty input: Ok None at once, [channels; 0] from read and decode"
      (fun () ->
        ( match
            via_pipe [] (fun ic ->
                drain
                  (Raw.Reader.create `F32le ~channels:2 Nx.float32 ic)
                  ~channels:2 ~frames:16 )
          with
        | [], None ->
            ()
        | _ ->
            fail "an empty stream delivered something" ) ;
        let t =
          ok ~msg:"read" (via_pipe [] (Raw.read `F32le ~channels:2 Nx.float32))
        in
        equal ~msg:"read" (array int) [|2; 0|] (Nx.shape t) ;
        equal ~msg:"decode" (array int) [|4; 0|]
          (Nx.shape (Raw.decode `S24be ~channels:4 Nx.float64 "")) ;
        equal ~msg:"encode" string ""
          (Raw.encode `S16le (Nx.zeros Nx.float32 [|3; 0|])) ) ]

(* {2 Partial frames} *)

exception Interrupted

(* [Interrupted] out of a [Reader.read] blocked in read(2) with a frame and a
   half staged: SIGUSR1, sent once the pipe is empty — once the reader has
   staged every byte — by a domain born with the signal blocked, as a domain
   takes its creator's mask. The runtime runs a handler only on a domain that
   does not block the signal, and the handler raises only on the test's own
   domain and only while [armed]. *)
let interrupted_mid_frame () =
  let frame = Bytes.create 4 in
  Bytes.set_int16_le frame 0 1000 ;
  Bytes.set_int16_le frame 2 (-1000) ;
  let frame = Bytes.to_string frame in
  let staged = frame ^ String.sub frame 0 3
  and rest = String.sub frame 3 1 ^ frame ^ frame in
  let rd, wr = Unix.pipe ~cloexec:true () in
  let put s =
    let n = String.length s in
    let rec go o = if o < n then go (o + Unix.write_substring wr s o (n - o)) in
    go 0
  in
  let closed = ref false in
  let close_wr () =
    if not !closed then begin
      closed := true ;
      Unix.close wr
    end
  in
  put staged ;
  let ic = Unix.in_channel_of_descr rd in
  let armed = Atomic.make true in
  let previous =
    Sys.signal Sys.sigusr1
      (Sys.Signal_handle
         (fun _ ->
           if Atomic.get armed && Domain.is_main_domain () then
             raise Interrupted ) )
  in
  let mask = Unix.sigprocmask Unix.SIG_BLOCK [Sys.sigusr1] in
  let sender =
    Fun.protect
      ~finally:(fun () ->
        ignore (Unix.sigprocmask Unix.SIG_SETMASK mask : int list) )
      (fun () ->
        Domain.spawn (fun () ->
            let deadline = Unix.gettimeofday () +. 30. in
            let rec wait () =
              match Unix.select [rd] [] [] 0. with
              | [], _, _ ->
                  Unix.kill (Unix.getpid ()) Sys.sigusr1
              | _ ->
                  if Unix.gettimeofday () < deadline then begin
                    Unix.sleepf 0.001 ; wait ()
                  end
            in
            wait () ) )
  in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set armed false ;
      Domain.join sender ;
      Sys.set_signal Sys.sigusr1 previous ;
      close_wr () ;
      close_in_noerr ic )
    (fun () ->
      let r = Raw.Reader.create `S16le ~channels:2 Nx.float32 ic in
      let interrupted =
        match Raw.Reader.read r ~frames:4 with
        | exception Interrupted ->
            true
        | _ ->
            false
      in
      Atomic.set armed false ;
      is_true ~msg:"the blocked read raised the handler's exception" interrupted ;
      put rest ;
      close_wr () ;
      for _ = 1 to 2 do
        match Raw.Reader.read r ~frames:4 with
        | Ok None ->
            ()
        | Ok (Some c) ->
            failf "a %s chunk after the exception, from mid-frame"
              (shape_string (Nx.shape c))
        | Error e ->
            failf "after the exception: %a" pp_error e
      done ;
      check_string ~msg:"the bytes the reader left unread" rest
        (In_channel.input_all ic) )

let partial_tests =
  [ test "the reader delivers every complete frame, then the error, then EOF"
      (fun () ->
        let e = `S24le and channels = 2 and fb = 6 in
        for extra = 1 to fb - 1 do
          let s = random_bytes ((10 * fb) + extra) extra in
          let complete =
            Raw.decode e ~channels Nx.float32 (String.sub s 0 (10 * fb))
          in
          let chunks, err, after =
            via_pipe ~pace:0. (bytewise s) (fun ic ->
                let r =
                  Raw.Reader.create ~name:"mic" e ~channels Nx.float32 ic
                in
                let chunks, err = drain r ~channels ~frames:4 in
                (chunks, err, Raw.Reader.read r ~frames:4) )
          in
          check_chunks ~msg:"complete frames" ~frames:4 complete chunks ;
          ( match err with
          | Some (Io {path= "mic"; op= "read"; details}) ->
              is_true ~msg:"the details name the byte counts"
                (String.equal details
                   (Printf.sprintf
                      "the stream ends %d byte%s into a 6-byte frame" extra
                      (if extra = 1 then "" else "s") ) )
          | Some e ->
              failf "unexpected %a" pp_error e
          | None ->
              failf "a %d-byte partial frame went unreported" extra ) ;
          match after with
          | Ok None ->
              ()
          | _ ->
              fail "the reader is not at EOF after the error"
        done )
  ; test "a partial frame alone is the error at once" (fun () ->
        match
          via_pipe ["\x01\x02\x03"] (fun ic ->
              let r = Raw.Reader.create `S16le ~channels:2 Nx.float32 ic in
              let first = Raw.Reader.read r ~frames:8 in
              (first, Raw.Reader.read r ~frames:8) )
        with
        | Error (Io {op= "read"; _}), Ok None ->
            ()
        | _ ->
            fail "expected the error, then EOF" )
  ; test "read refuses a partial frame; decode raises" (fun () ->
        ( match
            via_pipe ["abcde"]
              (Raw.read ~name:"pipe" `S16le ~channels:2 Nx.float32)
          with
        | Error (Io {path= "pipe"; op= "read"; _}) ->
            ()
        | Error e ->
            failf "unexpected %a" pp_error e
        | Ok _ ->
            fail "read returned data for a partial frame" ) ;
        raises_match invalid (fun () ->
            Raw.decode `S16le ~channels:2 Nx.float32 "abc" ) )
  ; test ~timeout:60.
      "an exception mid-frame ends the reader: nothing misaligned after"
      interrupted_mid_frame ]

(* {2 Channels} *)

let channel_tests =
  [ test "read decodes a whole channel" (fun () ->
        let s = random_bytes (8 * 1000) 3 in
        let t =
          with_prefixed_file s ~junk:0 (Raw.read `F64le ~channels:1 Nx.float64)
        in
        check_bits ~msg:"read"
          (Raw.decode `F64le ~channels:1 Nx.float64 s)
          (ok ~msg:"read" t) )
  ; test "write leaves the channel flushed and open" (fun () ->
        let x =
          Nx.create Nx.float32 [|2; 300|]
            (Array.init 600 (fun i -> Float.of_int (i - 300) /. 301.))
        in
        let path = temp_path ".raw" in
        let oc = Out_channel.open_bin path in
        ok ~msg:"write" (Raw.write `S24be oc x) ;
        let stored = In_channel.with_open_bin path In_channel.input_all in
        check_string ~msg:"the file before close" (Raw.encode `S24be x) stored ;
        Out_channel.close oc )
  ; test "a closed channel is Io, reading and writing" (fun () ->
        let path = temp_path ".raw" in
        Out_channel.with_open_bin path (fun oc ->
            Out_channel.output_string oc "abcd" ) ;
        let ic = In_channel.open_bin path in
        In_channel.close ic ;
        ( match Raw.read `S16le ~channels:1 Nx.float32 ic with
        | Error (Io {op= "read"; _}) ->
            ()
        | _ ->
            fail "read of a closed channel" ) ;
        ( match
            Raw.Reader.read
              (Raw.Reader.create `S16le ~channels:1 Nx.float32 ic)
              ~frames:4
          with
        | Error (Io {op= "read"; _}) ->
            ()
        | _ ->
            fail "Reader.read of a closed channel" ) ;
        let oc = Out_channel.open_bin path in
        Out_channel.close oc ;
        match Raw.write `S16le oc (Nx.zeros Nx.float32 [|1; 4|]) with
        | Error (Io {op= "write"; _}) ->
            ()
        | _ ->
            fail "write to a closed channel" )
  ; test "a non-blocking channel with nothing ready is Io, then EOF" (fun () ->
        let rd, wr = Unix.pipe ~cloexec:true () in
        Unix.set_nonblock rd ;
        let ic = Unix.in_channel_of_descr rd in
        Fun.protect
          ~finally:(fun () -> Unix.close wr ; close_in_noerr ic)
          (fun () ->
            let r = Raw.Reader.create `S16le ~channels:1 Nx.float32 ic in
            ( match Raw.Reader.read r ~frames:4 with
            | Error (Io {op= "read"; _}) ->
                ()
            | _ ->
                fail "Reader.read of a channel not ready" ) ;
            ( match Raw.Reader.read r ~frames:4 with
            | Ok None ->
                ()
            | _ ->
                fail "the reader is not at EOF after the error" ) ;
            match Raw.read `S16le ~channels:1 Nx.float32 ic with
            | Error (Io {op= "read"; _}) ->
                ()
            | _ ->
                fail "read of a channel not ready" ) ) ]

(* {2 Preconditions and names} *)

let precondition_tests =
  [ test "no channel, a non-float dtype, a bad rank: Invalid_argument" (fun () ->
        List.iter
          (fun channels ->
            raises_match invalid (fun () ->
                Raw.decode `S16le ~channels Nx.float32 "" ) ;
            raises_match invalid (fun () ->
                with_prefixed_file "" ~junk:0
                  (Raw.read `S16le ~channels Nx.float32) ) ;
            raises_match invalid (fun () ->
                with_prefixed_file "" ~junk:0
                  (Raw.Reader.create `S16le ~channels Nx.float32) ) )
          [0; -1] ;
        raises_match invalid (fun () ->
            Raw.decode `S16le ~channels:1 Nx.float16 "" ) ;
        raises_match invalid (fun () ->
            Raw.encode `S16le (Nx.zeros Nx.float16 [|1; 4|]) ) ;
        raises_match invalid (fun () ->
            Raw.encode `S16le (Nx.zeros Nx.float32 [|0; 4|]) ) ;
        raises_match invalid (fun () ->
            Raw.encode `S16le (Nx.scalar Nx.float32 0.) ) ;
        raises_match invalid (fun () ->
            Raw.encode `S16le (Nx.zeros Nx.float32 [|1; 2; 4|]) ) ;
        with_prefixed_file "abcd" ~junk:0 (fun ic ->
            let r = Raw.Reader.create `S16le ~channels:2 Nx.float32 ic in
            raises_match invalid (fun () -> Raw.Reader.read r ~frames:0) ;
            raises_match invalid (fun () -> Raw.Reader.read r ~frames:max_int) ) )
  ; test "encoding names: ffmpeg's twelve, exactly" (fun () ->
        List.iter
          (fun e ->
            equal ~msg:(Raw.encoding_name e) bool true
              (Raw.encoding_of_name (Raw.encoding_name e) = Some e) )
          encodings ;
        equal ~msg:"names" (list string)
          [ "s8"
          ; "u8"
          ; "s16le"
          ; "s16be"
          ; "s24le"
          ; "s24be"
          ; "s32le"
          ; "s32be"
          ; "f32le"
          ; "f32be"
          ; "f64le"
          ; "f64be" ]
          (List.map Raw.encoding_name encodings) ;
        List.iter
          (fun name ->
            is_true ~msg:name (Option.is_none (Raw.encoding_of_name name)) )
          ["s12le"; "S16LE"; "s16"; "pcm_s16le"; ""] ;
        equal ~msg:"sample bytes" (list int)
          [1; 1; 2; 2; 3; 3; 4; 4; 4; 4; 8; 8]
          (List.map Raw.sample_bytes encodings) ) ]

(* {2 The documented streaming loop} *)

let stream_tests =
  [ test "Reader into Stream (power >> mel) = Pipeline.run on the signal"
      (fun () ->
        let open Soundml in
        let sample_rate = 16000 and max_chunk = 1024 in
        let stft = Stft.Config.create ~hop:160 ~fft_size:512 () in
        let mel = Mel.Config.create ~n_mels:40 ~sample_rate ~fft_size:512 () in
        let p = Pipeline.(Stft.power_stage stft >> Mel.stage mel) in
        let source =
          Pipeline.Format.audio Nx.float32 ~sample_rate ~channels:1
        in
        let signal =
          Nx.create Nx.float64 [|1; 20000|]
            (Array.init 20000 (fun i ->
                 Float.sin (Float.of_int i *. 0.05) *. 0.5 ) )
        in
        let bytes = Raw.encode `F32le signal in
        let streamed =
          with_prefixed_file bytes ~junk:0 (fun stdin ->
              let s = Pipeline.Stream.prepare p ~source ~max_chunk in
              let r = Raw.Reader.create `F32le ~channels:1 Nx.float32 stdin in
              let out = ref [] in
              let consume m = out := m :: !out in
              let rec loop () =
                match Raw.Reader.read r ~frames:max_chunk with
                | Ok (Some chunk) ->
                    Option.iter consume (Pipeline.Stream.push s chunk) ;
                    loop ()
                | Ok None ->
                    List.iter consume (Pipeline.Stream.flush s)
                | Error e ->
                    failf "%a" pp_error e
              in
              loop () ; List.rev !out )
        in
        let offline =
          Pipeline.run ~source p
            (Raw.decode `F32le ~channels:1 Nx.float32 bytes)
        in
        ( match Nx.shape offline with
        | [|1; 40; n|] when n > 100 ->
            ()
        | shape ->
            failf "mel frames shaped %s" (shape_string shape) ) ;
        is_true ~msg:"several chunks streamed" (List.length streamed > 10) ;
        check_bits ~msg:"mel frames" offline
          (Nx.concatenate ~axis:(-1) streamed) ) ]

let suite =
  [ group "raw: decode parity with read" decode_parity_tests
  ; group "raw: encode parity with write" encode_parity_tests
  ; group "raw: pinned conversions" pin_tests
  ; group "raw: identity" identity_tests
  ; group "raw: chunked reader" reader_tests
  ; group "raw: partial frames" partial_tests
  ; group "raw: channels" channel_tests
  ; group "raw: preconditions" precondition_tests
  ; group "raw: streaming" stream_tests ]
