(* The memory faces. Decoding a file's bytes from memory is decoding the file
   from its path, bit for bit, through every face — read (native, downmixed,
   resampled), fold, a Reader's chunks and its seeks — over the whole corpus,
   and over fixtures large enough for the parallel whole-file decode to engage;
   the one exception, a stream libsndfile names by its file's extension, is
   pinned as decoding from its path alone. Encoding to memory is writing the
   file, byte for byte, in every writable cell; Ogg/Vorbis draws a random stream
   serial per encode, so its bytes are held to their length and its decode to
   the file's. The malformed corpus fed from memory lands in the path's outcome
   classes, named ["<memory>"], and never crashes. And the bytes a memory face
   is given are copied at open: a reader outlives and ignores the caller's
   buffer, and the GC moving that buffer mid-decode changes nothing. Comparisons
   are on the float bits (NaN payloads included) and on the encoded bytes, never
   a tolerance. *)

open Windtrap
open Soundml_io

let tmp_dir = fixture (fun () -> Filename.temp_dir "soundml_io_memory" "")

let slurp path = In_channel.with_open_bin path In_channel.input_all

let ok ~msg = function Ok v -> v | Error e -> failf "%s: %a" msg pp_error e

let check_bits = Io_law.check_bits

let check_bytes ~msg expected actual =
  if not (String.equal expected actual) then begin
    let n = Int.min (String.length expected) (String.length actual) in
    let rec first i =
      if i < n && Char.equal expected.[i] actual.[i] then first (i + 1) else i
    in
    failf
      "%s: %d bytes encoded, the file holds %d; they first differ at byte %d"
      msg (String.length actual) (String.length expected) (first 0)
  end

let check_chunks ~msg expected actual =
  equal ~msg:(msg ^ ": chunk count") int (List.length expected)
    (List.length actual) ;
  List.iteri
    (fun i (e, a) -> check_bits ~msg:(Printf.sprintf "%s: chunk %d" msg i) e a)
    (List.combine expected actual)

let check_info ~msg (expected : Info.t) (actual : Info.t) =
  equal ~msg:(msg ^ ": frames") int expected.frames actual.frames ;
  equal ~msg:(msg ^ ": channels") int expected.channels actual.channels ;
  equal ~msg:(msg ^ ": sample rate") int expected.sample_rate actual.sample_rate ;
  equal ~msg:(msg ^ ": format name") string expected.format_name
    actual.format_name ;
  is_true ~msg:(msg ^ ": format")
    (Option.equal Format.equal expected.format actual.format)

let check_audio ~msg (expected : _ audio) (actual : _ audio) =
  equal ~msg:(msg ^ ": sample rate") int expected.sample_rate actual.sample_rate ;
  check_bits ~msg expected.data actual.data

let error_path = function
  | Not_found {path}
  | Permission_denied {path}
  | Unrecognized_format {path; _}
  | Unsupported {path; _}
  | Truncated {path; _}
  | Io {path; _} ->
      path

let same_outcome ~msg ~same from_path from_memory =
  match (from_path, from_memory) with
  | Ok a, Ok b ->
      same a b
  | Error a, Error b -> (
      equal ~msg:(msg ^ ": error class") string (Io_errors.variant_name a)
        (Io_errors.variant_name b) ;
      equal ~msg:(msg ^ ": named source") string "<memory>" (error_path b) ;
      match (a, b) with
      | Truncated x, Truncated y ->
          equal
            ~msg:(msg ^ ": expected frames")
            int x.expected_frames y.expected_frames ;
          equal ~msg:(msg ^ ": read frames") int x.read_frames y.read_frames
      | Io x, Io y ->
          equal ~msg:(msg ^ ": op") string x.op y.op
      | _ ->
          () )
  | Ok _, Error e ->
      failf "%s: the path decodes, memory reports %a" msg pp_error e
  | Error e, Ok _ ->
      failf "%s: memory decodes, the path reports %a" msg pp_error e

let collect acc chunk = Nx.copy chunk :: acc

(* [drain r] reads [r] dry through [schedule]'s chunk sizes — copies of the
   chunks in order, or the first error. *)
let drain ?(schedule = fun () -> 4096) r =
  Fun.protect
    ~finally:(fun () -> Reader.close r)
    (fun () ->
      let rec go acc =
        match Reader.read r ~frames:(schedule ()) with
        | Ok None ->
            Ok (List.rev acc)
        | Ok (Some c) ->
            go (Nx.copy c :: acc)
        | Error e ->
            Error e
      in
      go [] )

let opened_drain = function Error e -> Error e | Ok r -> drain r

(* {2 The corpus, from memory} *)

let corpus =
  [ "aiff_pcm16_22050_mono.aiff"
  ; "caf_pcm16_22050_stereo.caf"
  ; "flac_pcm16_22050_mono.flac"
  ; "flac_pcm24_22050_stereo.flac"
  ; "ogg_vorbis_22050_mono.ogg"
  ; "ogg_vorbis_44100_stereo.ogg"
  ; "wav_float32_22050_stereo.wav"
  ; "wav_float64_22050_mono.wav"
  ; "wav_pcm16_22050_mono.wav"
  ; "wav_pcm16_44100_stereo.wav"
  ; "wav_pcm24_22050_stereo.wav"
  ; "wav_pcm32_22050_mono.wav" ]

let corpus_path file = Filename.concat "corpus" file

let info_case file =
  test ("info " ^ file) (fun () ->
      let path = corpus_path file in
      check_info ~msg:"info"
        (ok ~msg:"info" (info path))
        (ok ~msg:"Memory.info" (Memory.info (slurp path))) )

let read_case file =
  test ("read " ^ file) (fun () ->
      let path = corpus_path file in
      let bytes = slurp path in
      let same (type a) ~msg ?sample_rate ?quality ?mono
          (dt : (float, a) Nx.dtype) =
        check_audio ~msg
          (ok ~msg (read ?sample_rate ?quality ?mono dt path))
          (ok ~msg (Memory.read ?sample_rate ?quality ?mono dt bytes))
      in
      same ~msg:"f32" Nx.float32 ;
      same ~msg:"f64" Nx.float64 ;
      same ~msg:"f32 mono" ~mono:true Nx.float32 ;
      same ~msg:"f64 mono" ~mono:true Nx.float64 ;
      same ~msg:"f32 -> 16000 Hz" ~sample_rate:16000 Nx.float32 ;
      same ~msg:"f32 -> 48000 Hz" ~sample_rate:48000 Nx.float32 ;
      same ~msg:"f64 -> 16000 Hz fast" ~sample_rate:16000 ~quality:`Fast
        Nx.float64 ;
      same ~msg:"f32 -> 16000 Hz mono" ~sample_rate:16000 ~mono:true Nx.float32 )

let fold_case ?(blocks = [1000; 65536]) file =
  let name =
    Printf.sprintf "fold %s blocks %s" file
      (String.concat "," (List.map string_of_int blocks))
  in
  test name (fun () ->
      let path = corpus_path file in
      let bytes = slurp path in
      List.iter
        (fun (block, sample_rate) ->
          let msg =
            Printf.sprintf "block %d%s" block
              ( match sample_rate with
              | None ->
                  ""
              | Some r ->
                  Printf.sprintf " -> %d Hz" r )
          in
          let from_path =
            ok ~msg
              (fold ~block ?sample_rate Nx.float32 path ~init:[] ~f:collect)
          in
          let from_memory =
            ok ~msg
              (Memory.fold ~block ?sample_rate Nx.float32 bytes ~init:[]
                 ~f:collect )
          in
          check_chunks ~msg (List.rev from_path) (List.rev from_memory) ;
          let whole = ok ~msg (read ?sample_rate Nx.float32 path) in
          match List.rev from_memory with
          | [] ->
              equal ~msg:(msg ^ ": empty") int 0 (Nx.shape whole.data).(1)
          | pieces ->
              check_bits ~msg:(msg ^ ": fold == read") whole.data
                (Nx.concatenate ~axis:1 pieces) )
        (List.concat_map (fun b -> [(b, None); (b, Some 16000)]) blocks) )

let reader_case file =
  test ("reader " ^ file) (fun () ->
      let path = corpus_path file in
      let bytes = slurp path in
      List.iteri
        (fun seed sample_rate ->
          let msg =
            match sample_rate with
            | None ->
                "native"
            | Some r ->
                Printf.sprintf "-> %d Hz" r
          in
          let rp = ok ~msg (Reader.open_ ?sample_rate Nx.float64 path) in
          let rm = ok ~msg (Memory.open_reader ?sample_rate Nx.float64 bytes) in
          check_info ~msg (Reader.info rp) (Reader.info rm) ;
          let module PF = Soundml.Pipeline.Format in
          is_true ~msg:(msg ^ ": format rate")
            (Soundml.Pipeline.Rate.equal
               (PF.items_per_second (Reader.format rp))
               (PF.items_per_second (Reader.format rm)) ) ;
          equal
            ~msg:(msg ^ ": format channels")
            int
            (PF.channels (Reader.format rp))
            (PF.channels (Reader.format rm)) ;
          (* both readers in lockstep through one random schedule *)
          let draw = Io_law.lcg (100 + seed) in
          let rec step i =
            let frames = 1 + int_of_float (draw () *. 999.) in
            match (Reader.read rp ~frames, Reader.read rm ~frames) with
            | Ok None, Ok None ->
                ()
            | Ok (Some a), Ok (Some b) ->
                check_bits ~msg:(Printf.sprintf "%s: chunk %d" msg i) a b ;
                step (i + 1)
            | Error e, _ | _, Error e ->
                failf "%s: chunk %d: %a" msg i pp_error e
            | _ ->
                failf "%s: chunk %d: one reader is at EOF, the other is not" msg
                  i
          in
          step 0 ; Reader.close rp ; Reader.close rm )
        [None; Some 16000] )

(* The memory reader seeks where the path reader seeks, in every case — same
   chunks, same errors; and onto the uninterrupted read's samples wherever the
   path face's seek law holds: every reader but a native Ogg/Vorbis one, whose
   libsndfile seek is not sample-exact (observed on this corpus, path and memory
   alike, down to a [Truncated] after a seek near the end). *)
let seek_case file =
  test ("seek " ^ file) (fun () ->
      let path = corpus_path file in
      let bytes = slurp path in
      let ogg = Filename.check_suffix file ".ogg" in
      List.iter
        (fun sample_rate ->
          let whole = ok ~msg:"read" (read ?sample_rate Nx.float32 path) in
          let channels = (Nx.shape whole.data).(0) in
          let total = (Nx.shape whole.data).(1) in
          let law = not (ogg && Option.is_none sample_rate) in
          let rp =
            ok ~msg:"open_" (Reader.open_ ?sample_rate Nx.float32 path)
          in
          let r =
            ok ~msg:"open_reader"
              (Memory.open_reader ?sample_rate Nx.float32 bytes)
          in
          List.iter
            (fun frame ->
              let frame = Int.min frame (total - 1) in
              let msg = Printf.sprintf "seek %d" frame in
              ok ~msg (Reader.seek rp ~frame) ;
              ok ~msg (Reader.seek r ~frame) ;
              let n = Int.min 256 (total - frame) in
              same_outcome ~msg
                ~same:(fun expected chunk ->
                  match (expected, chunk) with
                  | Some expected, Some chunk ->
                      check_bits
                        ~msg:(msg ^ ": the path reader's chunk")
                        expected chunk ;
                      if law then
                        check_bits
                          ~msg:
                            (Printf.sprintf "span [%d; %d)" frame (frame + n))
                          (Nx.slice
                             [Nx.R (0, channels); Nx.R (frame, frame + n)]
                             whole.data )
                          chunk
                  | None, None when not law ->
                      ()
                  | _ ->
                      failf "%s: EOF where %d frames remain" msg (total - frame) )
                (Reader.read rp ~frames:n) (Reader.read r ~frames:n) )
            Io_law.seek_positions ;
          (* the end, and past it: EOF and a typed refusal where the law holds,
             the path reader's outcome everywhere *)
          let msg = "seek to the end" in
          same_outcome ~msg
            ~same:(fun () () -> ())
            (Reader.seek rp ~frame:total)
            (Reader.seek r ~frame:total) ;
          let at_end = Reader.read r ~frames:16 in
          same_outcome ~msg
            ~same:(fun expected chunk ->
              if Option.is_some expected || Option.is_some chunk then
                fail "a read at the end position delivered data" )
            (Reader.read rp ~frames:16)
            at_end ;
          ( match at_end with
          | Ok None ->
              ()
          | _ when not law ->
              ()
          | _ ->
              fail "a read at the end position is not EOF" ) ;
          let past = Reader.seek r ~frame:(total + 1) in
          same_outcome ~msg:"seek past the end"
            ~same:(fun () () -> fail "seeking past the end succeeded")
            (Reader.seek rp ~frame:(total + 1))
            past ;
          ( match past with
          | Error (Io {op= "seek"; _}) ->
              ()
          | _ ->
              fail "seeking past the end is not Io seek" ) ;
          Reader.close rp ; Reader.close r )
        [None; Some 16000] )

let corpus_tests =
  test "the list is every file of corpus/" (fun () ->
      let on_disk =
        Sys.readdir "corpus" |> Array.to_list
        |> List.filter (fun f -> not (Sys.is_directory (corpus_path f)))
        |> List.sort String.compare
      in
      equal ~msg:"corpus" (list string) on_disk
        (List.sort String.compare corpus) )
  :: List.concat_map
       (fun file ->
         [ info_case file
         ; read_case file
         ; fold_case file
         ; reader_case file
         ; seek_case file ] )
       corpus
  @ [fold_case ~blocks:[1] "wav_pcm16_22050_mono.wav"]

(* {2 The parallel whole-file decode, from memory}

   The corpus is far below the split's admission floor; these fixtures are above
   it on the reference host (an estimated 200-600 us of decode, two to four
   extents), so on arm64 macOS the duplicates decode the handle's own copy.
   Elsewhere the decode is sequential and the equality holds the same. *)

let split_fixture ~name ~format ~channels seed =
  let path = Filename.concat (tmp_dir ()) name in
  if not (Sys.file_exists path) then begin
    let data =
      Io_law.make_signal ~channels ~frames:441000 ~sample_rate:44100 seed
    in
    ok ~msg:name (write ~format path {data; sample_rate= 44100})
  end ;
  path

let split_case ~name ~format ~channels seed =
  test name (fun () ->
      let path = split_fixture ~name ~format ~channels seed in
      let bytes = slurp path in
      check_audio ~msg:"f32"
        (ok ~msg:"read" (read Nx.float32 path))
        (ok ~msg:"Memory.read" (Memory.read Nx.float32 bytes)) ;
      check_audio ~msg:"f64"
        (ok ~msg:"read" (read Nx.float64 path))
        (ok ~msg:"Memory.read" (Memory.read Nx.float64 bytes)) )

let split_tests =
  [ split_case ~name:"split_pcm16_stereo.wav"
      ~format:(Format.create ~encoding:`Pcm_16 `Wav)
      ~channels:2 41
  ; split_case ~name:"split_pcm24_stereo.aiff"
      ~format:(Format.create ~encoding:`Pcm_24 `Aiff)
      ~channels:2 42
  ; split_case ~name:"split_pcm16_mono.wav"
      ~format:(Format.create ~encoding:`Pcm_16 `Wav)
      ~channels:1 43 ]

(* {2 Encoding to memory is writing the file} *)

let container_ext = function
  | `Wav ->
      ".wav"
  | `Aiff ->
      ".aiff"
  | `Flac ->
      ".flac"
  | `Caf ->
      ".caf"
  | `Ogg ->
      ".ogg"

(* Noise across most of full scale, with one sample either side of it so the PCM
   cells exercise the clipping mode. *)
let signal ~channels ~frames seed =
  let draw = Io_law.lcg seed in
  Nx.create Nx.float64 [|channels; frames|]
    (Array.init (channels * frames) (fun i ->
         match i with 1 -> 1.5 | 2 -> -1.5 | _ -> (1.6 *. draw ()) -. 0.8 ) )

(* [unstamped s] is [s] with the timestamp of its PEAK chunk zeroed. libsndfile
   stamps the float WAV and AIFF PEAK chunk with the wall-clock second of the
   encode (its version field, then [time (NULL)]), so two encodes a second apart
   differ there and nowhere else. The chunk precedes the sample data, so its
   marker's first occurrence is the chunk. *)
let unstamped s =
  let n = String.length s in
  let rec find p =
    if p + 16 > n then None
    else if String.sub s p 4 = "PEAK" then Some p
    else find (p + 1)
  in
  match find 0 with
  | None ->
      s
  | Some p ->
      let b = Bytes.of_string s in
      Bytes.fill b (p + 12) 4 '\000' ;
      Bytes.to_string b

(* [write] then [read] against [Memory.write] then [Memory.read]: the bytes (or,
   for Ogg, their length) and both decodes. *)
let check_encode format (a : _ audio) =
  let path =
    Filename.temp_file ~temp_dir:(tmp_dir ()) "encode"
      (container_ext (Format.container format))
  in
  ok ~msg:"write" (write ~format path a) ;
  let file = slurp path in
  let bytes = ok ~msg:"Memory.write" (Memory.write ~format a) in
  ( match Format.container format with
  | `Ogg ->
      (* the stream serial differs per encode, never the page layout *)
      equal ~msg:"ogg stream length" int (String.length file)
        (String.length bytes)
  | _ ->
      check_bytes ~msg:"encoded bytes" (unstamped file) (unstamped bytes) ) ;
  check_audio ~msg:"f64"
    (ok ~msg:"read" (read Nx.float64 path))
    (ok ~msg:"Memory.read" (Memory.read Nx.float64 bytes)) ;
  check_audio ~msg:"f32"
    (ok ~msg:"read" (read Nx.float32 path))
    (ok ~msg:"Memory.read" (Memory.read Nx.float32 bytes)) ;
  Sys.remove path

let writable =
  List.concat_map
    (fun container ->
      List.map
        (fun encoding -> Format.create ~encoding container)
        [`Pcm_16; `Pcm_24; `Pcm_32; `Float32; `Float64] )
    [`Wav; `Aiff; `Caf]
  @ [ Format.create ~encoding:`Pcm_16 `Flac
    ; Format.create ~encoding:`Pcm_24 `Flac
    ; Format.create `Ogg ]

let encode_case format ~channels ~frames =
  let name =
    Stdlib.Format.asprintf "%a ch%d n%d" Format.pp format channels frames
  in
  test name (fun () ->
      let x = signal ~channels ~frames (frames + channels) in
      (* mono cells encode from float32, stereo from float64 *)
      if channels = 1 then
        check_encode format {data= Nx.cast Nx.float32 x; sample_rate= 22050}
      else check_encode format {data= x; sample_rate= 22050} )

let encode_matrix =
  List.concat_map
    (fun format ->
      List.concat_map
        (fun channels ->
          List.filter_map
            (fun frames ->
              (* the empty FLAC has its own pin below *)
              if frames = 0 && Format.container format = `Flac then None
              else Some (encode_case format ~channels ~frames) )
            [0; 1; 4096; 65537] )
        [1; 2] )
    writable

let small_audio ~channels ~frames =
  {data= Nx.zeros Nx.float32 [|channels; frames|]; sample_rate= 22050}

let refused ~msg = function
  | Error (Unsupported {path; _}) ->
      equal ~msg:(msg ^ ": named source") string "<memory>" path
  | Error e ->
      failf "%s: unexpected %a" msg pp_error e
  | Ok _ ->
      failf "%s: encoded" msg

let encode_tests =
  [ test "an empty flac encodes to no bytes, which read refuses" (fun () ->
        let bytes =
          ok ~msg:"Memory.write"
            (Memory.write ~format:(Format.create `Flac)
               (small_audio ~channels:2 ~frames:0) )
        in
        equal ~msg:"a 0-frame flac is empty" string "" bytes ;
        match Memory.read Nx.float32 bytes with
        | Error (Unrecognized_format {path; _}) ->
            equal ~msg:"named source" string "<memory>" path
        | Ok _ ->
            fail "no bytes decoded"
        | Error e ->
            failf "expected Unrecognized_format, got %a" pp_error e )
  ; test "the clipping golden, encoded to memory" (fun () ->
        List.iter
          (fun (case : Tutils.Golden.case) ->
            let lo = Tutils.Golden.int_param case "lo" in
            let hi = Tutils.Golden.int_param case "hi" in
            let denominator =
              Float.of_int (Tutils.Golden.int_param case "denominator")
            in
            let sample_rate = Tutils.Golden.int_param case "sample_rate" in
            let n = hi - lo + 1 in
            let ramp =
              Nx.create Nx.float64 [|1; n|]
                (Array.init n (fun i -> Float.of_int (lo + i) /. denominator))
            in
            let bytes =
              ok ~msg:"Memory.write"
                (Memory.write
                   ~format:(Format.create ~encoding:`Pcm_16 `Wav)
                   {data= ramp; sample_rate} )
            in
            let audio = ok ~msg:"Memory.read" (Memory.read Nx.float64 bytes) in
            equal ~msg:"shape" (array int) case.shape (Nx.shape audio.data) ;
            let decoded = Nx.to_array audio.data in
            Array.iteri
              (fun i ev ->
                if Float.compare ev decoded.(i) <> 0 then
                  failf "sample %d decoded as %.17g, python-soundfile got %.17g"
                    i decoded.(i) ev )
              case.values )
          (Tutils.Golden.load "vectors/clipping.json").cases )
  ; test "Unsupported: nine channels into FLAC" (fun () ->
        refused ~msg:"Memory.write"
          (Memory.write ~format:(Format.create `Flac)
             (small_audio ~channels:9 ~frames:16) ) )
  ; test "Unsupported: three hundred channels into Vorbis" (fun () ->
        refused ~msg:"Memory.write"
          (Memory.write ~format:(Format.create `Ogg)
             (small_audio ~channels:300 ~frames:16) ) )
  ; test "Unsupported: Vorbis above 200 kHz" (fun () ->
        refused ~msg:"Memory.write"
          (Memory.write ~format:(Format.create `Ogg)
             {(small_audio ~channels:1 ~frames:16) with sample_rate= 200_001} ) )
  ; test "the preconditions are write's, named after the memory face" (fun () ->
        let format = Format.create `Wav in
        let refuses a =
          raises_match (Exn.invalid_arg ~substring:"Soundml_io.Memory.write")
            (fun () -> Memory.write ~format a )
        in
        refuses {data= Nx.zeros Nx.float32 [||]; sample_rate= 22050} ;
        refuses {data= Nx.zeros Nx.float32 [|1; 1; 8|]; sample_rate= 22050} ;
        refuses {data= Nx.zeros Nx.float32 [|1; 8|]; sample_rate= 0} ;
        refuses {data= Nx.zeros Nx.float16 [|1; 8|]; sample_rate= 22050} )
  ; test "a 2.5 M-frame vorbis stream grows through the sink" (fun () ->
        (* the write cap's regression test (above libsndfile 1.2.2's 2.2 M-frame
           single-call segfault), here through the growable sink *)
        let frames = 2_500_000 in
        let x =
          Nx.create Nx.float32 [|1; frames|]
            (Array.init frames (fun i ->
                 0.45 *. sin (2. *. Float.pi *. 440. *. Float.of_int i /. 22050.) )
            )
        in
        let bytes =
          ok ~msg:"Memory.write"
            (Memory.write ~format:(Format.create `Ogg)
               {data= x; sample_rate= 22050} )
        in
        let i = ok ~msg:"Memory.info" (Memory.info bytes) in
        equal ~msg:"frames" int frames i.Info.frames ) ]

(* {2 The malformed corpus, from memory: the path's classes, never a crash} *)

let malformed_case file =
  test file (fun () ->
      let path = Filename.concat "corpus/malformed" file in
      let bytes = slurp path in
      same_outcome ~msg:"info" ~same:(check_info ~msg:"info") (info path)
        (Memory.info bytes) ;
      let audio msg = check_audio ~msg in
      same_outcome ~msg:"read f32" ~same:(audio "read f32")
        (read Nx.float32 path)
        (Memory.read Nx.float32 bytes) ;
      same_outcome ~msg:"read f64" ~same:(audio "read f64")
        (read Nx.float64 path)
        (Memory.read Nx.float64 bytes) ;
      same_outcome ~msg:"read -> 16000 Hz" ~same:(audio "read -> 16000 Hz")
        (read ~sample_rate:16000 Nx.float32 path)
        (Memory.read ~sample_rate:16000 Nx.float32 bytes) ;
      same_outcome ~msg:"drained reader"
        ~same:(check_chunks ~msg:"reader")
        (opened_drain (Reader.open_ Nx.float32 path))
        (opened_drain (Memory.open_reader Nx.float32 bytes)) ;
      same_outcome ~msg:"drained reader -> 16000 Hz"
        ~same:(check_chunks ~msg:"reader -> 16000 Hz")
        (opened_drain (Reader.open_ ~sample_rate:16000 Nx.float32 path))
        (opened_drain (Memory.open_reader ~sample_rate:16000 Nx.float32 bytes)) ;
      same_outcome ~msg:"fold" ~same:(check_chunks ~msg:"fold")
        (Result.map List.rev
           (fold ~block:4096 Nx.float64 path ~init:[] ~f:collect) )
        (Result.map List.rev
           (Memory.fold ~block:4096 Nx.float64 bytes ~init:[] ~f:collect) ) )

(* 8 KB that name no container: no magic, no MPEG frame sync. *)
let garbage =
  let draw = Io_law.lcg 97531 in
  String.init 8192 (fun _ -> Char.chr (int_of_float (draw () *. 256.)))

let malformed_tests =
  List.map malformed_case Io_errors.malformed_files
  @ [ test "no bytes, and garbage, are Unrecognized_format" (fun () ->
          List.iter
            (fun (msg, bytes) ->
              match Memory.read Nx.float32 bytes with
              | Error (Unrecognized_format {path; _}) ->
                  equal ~msg:(msg ^ ": named source") string "<memory>" path
              | Ok _ ->
                  failf "%s decoded" msg
              | Error e ->
                  failf "%s: unexpected %a" msg pp_error e )
            [("no bytes", ""); ("garbage", garbage)] )
    ; test "a stream named only by its extension decodes from its path only"
        (fun () ->
          (* Any bytes are VOX ADPCM, and libsndfile takes a headerless stream
             for one by its [.vox] extension (format_from_extension): 8000 Hz
             mono, two samples per byte. Memory has no extension to go by. *)
          let path = Filename.concat (tmp_dir ()) "headerless.vox" in
          Out_channel.with_open_bin path (fun oc ->
              Out_channel.output_string oc garbage ) ;
          let a = ok ~msg:"read from the path" (read Nx.float32 path) in
          equal ~msg:"sample rate" int 8000 a.sample_rate ;
          equal ~msg:"shape" (array int) [|1; 16384|] (Nx.shape a.data) ;
          match Memory.read Nx.float32 garbage with
          | Error (Unrecognized_format {path; _}) ->
              equal ~msg:"named source" string "<memory>" path
          | Ok _ ->
              fail "memory identified a nameless stream"
          | Error e ->
              failf "unexpected %a" pp_error e )
    ; test "the allocation guard weighs the claim against the bytes" (fun () ->
          (* liar_int64.flac advertises 68.7 G frames in 32949 bytes *)
          match
            Memory.read Nx.float64 (slurp "corpus/malformed/liar_int64.flac")
          with
          | Error (Io {op= "read"; path; _}) ->
              equal ~msg:"named source" string "<memory>" path
          | Ok _ ->
              fail "a lying header decoded"
          | Error e ->
              failf "unexpected %a" pp_error e ) ]

(* {2 The copy: ownership, the GC, concurrency} *)

let gc_fixture =
  fixture (fun () ->
      let path = Filename.concat (tmp_dir ()) "gc.wav" in
      let data =
        Io_law.make_signal ~channels:2 ~frames:70000 ~sample_rate:22050 9
      in
      ok ~msg:"fixture"
        (write
           ~format:(Format.create ~encoding:`Float32 `Wav)
           path {data; sample_rate= 22050} ) ;
      path )

let reference path = (ok ~msg:"read" (read Nx.float32 path)).data

let concat_chunks = function
  | [] ->
      fail "no chunk was delivered"
  | pieces ->
      Nx.concatenate ~axis:1 pieces

(* [spawn f] is a domain running [f], when the runtime has room for one. Not
   even attempted otherwise, as the core's pool does not: under
   OCAMLRUNPARAM=d=1 — the macOS ASan leg — a refused spawn still starts and
   ends a thread, which that ASan runtime traps on. *)
let spawn f =
  if Domain.recommended_domain_count () < 2 then None
  else try Some (Domain.spawn f) with Failure _ -> None

(* [with_hammer f] runs [f] while another domain forces minor collections and
   compactions back to back — the conditions under which an OCaml string moves
   while a stub runs with the runtime lock released. Without a domain to spare
   [f] runs alone. *)
let with_hammer f =
  let stop = Atomic.make false in
  match
    spawn (fun () ->
        while not (Atomic.get stop) do
          Gc.minor () ; Gc.compact ()
        done )
  with
  | None ->
      f ()
  | Some d ->
      Fun.protect ~finally:(fun () -> Atomic.set stop true ; Domain.join d) f

let ownership_tests =
  [ test "a reader decodes the bytes as they were at open" (fun () ->
        let path = gc_fixture () in
        let buffer = Bytes.of_string (slurp path) in
        let r =
          ok ~msg:"open_reader"
            (Memory.open_reader Nx.float32 (Bytes.unsafe_to_string buffer))
        in
        (* the documented bytes idiom: the caller's buffer is its own again *)
        Bytes.fill buffer 0 (Bytes.length buffer) '\x00' ;
        check_bits ~msg:"after the caller zeroed its buffer" (reference path)
          (concat_chunks (ok ~msg:"drain" (drain r))) )
  ; test "a reader outlives its input through full majors and compactions"
      (fun () ->
        let path = gc_fixture () in
        let r =
          ok ~msg:"open_reader" (Memory.open_reader Nx.float32 (slurp path))
        in
        Gc.full_major () ;
        Gc.compact () ;
        let schedule () = Gc.compact () ; 7000 in
        check_bits ~msg:"chunks" (reference path)
          (concat_chunks (ok ~msg:"drain" (drain ~schedule r))) )
  ; test "decodes and encodes are unmoved by a collecting domain" (fun () ->
        (* a small input starts in the minor heap and is then promoted to a
           compactable pool: the string that moves. Each round decodes a fresh
           one, and a large one, while the other domain collects *)
        let path = gc_fixture () in
        let expected = reference path in
        let source = slurp path in
        let format = Format.create ~encoding:`Pcm_16 `Wav in
        let small =
          { data= Io_law.make_signal ~channels:1 ~frames:100 ~sample_rate:22050 5
          ; sample_rate= 22050 }
        in
        let small_bytes = ok ~msg:"small" (Memory.write ~format small) in
        let small_expected =
          (ok ~msg:"small" (Memory.read Nx.float32 small_bytes)).data
        in
        let fresh s = Bytes.to_string (Bytes.of_string s) in
        with_hammer (fun () ->
            for i = 1 to 20 do
              let msg = Printf.sprintf "round %d" i in
              check_bits ~msg expected
                (ok ~msg (Memory.read Nx.float32 (fresh source))).data ;
              check_bytes ~msg:(msg ^ ": encode") small_bytes
                (ok ~msg (Memory.write ~format small)) ;
              check_bits ~msg:(msg ^ ": small") small_expected
                (ok ~msg (Memory.read Nx.float32 (fresh small_bytes))).data
            done ) )
  ; test "concurrent failed opens keep their own verdict" (fun () ->
        (* libsndfile's failed-open error is one process global: memory opens
           take the same lock as path opens, so a garbage open racing valid ones
           is never misattributed *)
        let valid = slurp (corpus_path "wav_pcm16_22050_mono.wav") in
        let garbage = slurp "corpus/malformed/garbage.wav" in
        let worker () =
          for _ = 1 to 200 do
            ( match Memory.info garbage with
            | Error (Unrecognized_format _) ->
                ()
            | Error e ->
                failf "garbage: misattributed %a" pp_error e
            | Ok _ ->
                fail "garbage probed fine" ) ;
            match Memory.info valid with
            | Ok _ ->
                ()
            | Error e ->
                failf "valid: %a" pp_error e
          done
        in
        let attempt f = match f () with () -> None | exception e -> Some e in
        let domains = List.filter_map (fun _ -> spawn worker) [1; 2; 3] in
        let here = attempt worker in
        let joined =
          List.map (fun d -> attempt (fun () -> Domain.join d)) domains
        in
        match List.filter_map Fun.id (here :: joined) with
        | [] ->
            ()
        | e :: _ ->
            raise e )
  ; test "a closed reader refuses reads and seeks; close is idempotent"
      (fun () ->
        let r =
          ok ~msg:"open_reader"
            (Memory.open_reader Nx.float32 (slurp (gc_fixture ())))
        in
        Reader.close r ;
        Reader.close r ;
        raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
            Reader.read r ~frames:16 ) ;
        raises_match (Exn.invalid_arg ~substring:"closed") (fun () ->
            Reader.seek r ~frame:0 ) )
  ; test "the reading preconditions are named after the memory face" (fun () ->
        let bytes = slurp (gc_fixture ()) in
        raises_match (Exn.invalid_arg ~substring:"Soundml_io.Memory.read")
          (fun () -> Memory.read ~quality:`Fast Nx.float32 bytes ) ;
        raises_match
          (Exn.invalid_arg ~substring:"Soundml_io.Memory.open_reader")
          (fun () -> Memory.open_reader ~sample_rate:0 Nx.float32 bytes ) ;
        raises_match (Exn.invalid_arg ~substring:"Soundml_io.Memory.fold")
          (fun () ->
            Memory.fold ~block:0 Nx.float32 bytes ~init:() ~f:(fun () _ -> ()) ) ;
        raises_match (Exn.invalid_arg ~substring:"Soundml_io.Memory.read")
          (fun () -> Memory.read Nx.float16 bytes ) )
  ; test "an exception from f closes the handle and escapes" (fun () ->
        let count = ref 0 in
        raises_match (Exn.failure ~substring:"boom") (fun () ->
            Memory.fold ~block:1024 Nx.float32
              (slurp (gc_fixture ()))
              ~init:()
              ~f:(fun () _ ->
                incr count ;
                if !count = 2 then failwith "boom" ) ) ;
        equal ~msg:"f ran until the raise" int 2 !count )
  ; test "pp_error names the memory source" (fun () ->
        match
          Memory.read Nx.float32 (slurp "corpus/malformed/trunc_hdr_past.flac")
        with
        | Error (Truncated _ as e) ->
            starts_with ~msg:"message"
              ~affix:
                {|soundml-io: "<memory>" is truncated (header advertises 22050 frames|}
              (Stdlib.Format.asprintf "%a" pp_error e)
        | Ok _ ->
            fail "a truncated stream decoded"
        | Error e ->
            failf "unexpected %a" pp_error e ) ]

let suite =
  [ group "memory: the corpus" corpus_tests
  ; group "memory: the parallel decode" split_tests
  ; group "memory: encode == write" encode_matrix
  ; group "memory: encoding" encode_tests
  ; group "memory: malformed corpus" malformed_tests
  ; group "memory: the copy" ownership_tests ]
