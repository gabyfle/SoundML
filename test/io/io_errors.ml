(* The error matrix: every [error] constructor reachable from an actual
   filesystem or file condition, the [Invalid_argument] preconditions, and the
   malformed-file corpus (test/io/corpus/malformed/, MANIFEST-pinned): each
   malformed file, through [info] and [read] at both dtypes, must produce a
   typed error or valid data — never a crash, never zero-fill. Assertions match
   error variants, never libsndfile message strings (the messages travel in
   [details] and vary across libsndfile builds). *)

open Windtrap
open Soundml_io

let tmp_dir = fixture (fun () -> Filename.temp_dir "soundml_io_errors" "")

let variant_name = function
  | Not_found _ ->
      "Not_found"
  | Permission_denied _ ->
      "Permission_denied"
  | Unrecognized_format _ ->
      "Unrecognized_format"
  | Unsupported _ ->
      "Unsupported"
  | Truncated _ ->
      "Truncated"
  | Io _ ->
      "Io"

let expect_error ~msg pred = function
  | Ok _ ->
      failf "%s: expected an error, decoded fine" msg
  | Error e ->
      if not (pred e) then
        failf "%s: unexpected %s: %a" msg (variant_name e) pp_error e

let small_audio ~channels ~frames =
  {data= Nx.zeros Nx.float32 [|channels; frames|]; sample_rate= 22050}

(* {2 The reachable-variant matrix} *)

let matrix_tests =
  [ test "Not_found: reading a missing path" (fun () ->
        let missing = Filename.concat (tmp_dir ()) "does_not_exist.wav" in
        expect_error ~msg:"read"
          (function Not_found {path} -> path = missing | _ -> false)
          (read Nx.float32 missing) ;
        expect_error ~msg:"info"
          (function Not_found _ -> true | _ -> false)
          (info missing) )
  ; test "Not_found: writing into a missing directory" (fun () ->
        let path =
          Filename.concat (tmp_dir ())
            (Filename.concat "no_such_dir" "clip.wav")
        in
        expect_error ~msg:"write"
          (function Not_found _ -> true | _ -> false)
          (write path (small_audio ~channels:1 ~frames:16)) )
  ; test "Permission_denied: a chmod-000 fixture" (fun () ->
        if Unix.geteuid () = 0 then
          skip ~reason:"root reads through permission bits" () ;
        let path = Filename.temp_file ~temp_dir:(tmp_dir ()) "locked" ".wav" in
        Unix.chmod path 0o000 ;
        Fun.protect
          ~finally:(fun () -> Unix.chmod path 0o600 ; Sys.remove path)
          (fun () ->
            expect_error ~msg:"read"
              (function Permission_denied {path= p} -> p = path | _ -> false)
              (read Nx.float64 path) ;
            expect_error ~msg:"info"
              (function Permission_denied _ -> true | _ -> false)
              (info path) ) )
  ; test "Unrecognized_format: an empty file" (fun () ->
        expect_error ~msg:"read"
          (function Unrecognized_format _ -> true | _ -> false)
          (read Nx.float32 "corpus/malformed/empty.wav") ;
        expect_error ~msg:"info"
          (function Unrecognized_format _ -> true | _ -> false)
          (info "corpus/malformed/empty.wav") )
  ; test "Unrecognized_format: 8 KB of seeded garbage" (fun () ->
        expect_error ~msg:"read"
          (function Unrecognized_format _ -> true | _ -> false)
          (read Nx.float64 "corpus/malformed/garbage.wav") )
  ; test "Unrecognized_format: Format.of_path on an unknown extension"
      (fun () ->
        expect_error ~msg:"of_path"
          (function
            | Unrecognized_format {path; _} -> path = "x.xyz" | _ -> false )
          (Format.of_path "x.xyz") )
  ; test "Unsupported: nine channels into FLAC" (fun () ->
        let path = Filename.concat (tmp_dir ()) "nine.flac" in
        expect_error ~msg:"write"
          (function Unsupported _ -> true | _ -> false)
          (write path (small_audio ~channels:9 ~frames:16)) )
  ; test "Unsupported: three hundred channels into Vorbis" (fun () ->
        let path = Filename.concat (tmp_dir ()) "wall.ogg" in
        expect_error ~msg:"write"
          (function Unsupported _ -> true | _ -> false)
          (write path (small_audio ~channels:300 ~frames:16)) )
  ; test "Truncated: the truncated-FLAC fixture" (fun () ->
        expect_error ~msg:"read"
          (function
            | Truncated {expected_frames= 22050; read_frames; _} ->
                read_frames < 22050
            | _ ->
                false )
          (read Nx.float64 "corpus/malformed/trunc.flac") )
  ; test "Truncated: a header past which no data survives" (fun () ->
        (* trunc_hdr_past.flac opens fine, advertises 22050 frames, decodes 0 —
           the recon's exact case *)
        expect_error ~msg:"read"
          (function
            | Truncated {expected_frames= 22050; read_frames= 0; _} ->
                true
            | _ ->
                false )
          (read Nx.float32 "corpus/malformed/trunc_hdr_past.flac") )
  ; test "Io: writing through a path whose parent is a regular file" (fun () ->
        let blocker = Filename.temp_file ~temp_dir:(tmp_dir ()) "flat" "" in
        let path = Filename.concat blocker "clip.wav" in
        expect_error ~msg:"write"
          (function Io {op= "open"; _} -> true | _ -> false)
          (write path (small_audio ~channels:1 ~frames:16)) )
  ; test "Io: a header advertising more than the file could hold" (fun () ->
        (* STREAMINFO total samples at the 36-bit maximum: 68.7 G frames, 550 GB
           of float64 — refused as a typed error before any allocation can be
           attempted *)
        expect_error ~msg:"read"
          (function Io {op= "read"; _} -> true | _ -> false)
          (read Nx.float64 "corpus/malformed/liar_int64.flac") ) ]

(* {2 The unknown-length stream} *)

let unknown_length_tests =
  [ test "an Ogg with unknown frames probes as 0 and decodes to EOF" (fun () ->
        ( match info "corpus/malformed/trunc.ogg" with
        | Error e ->
            failf "info: %a" pp_error e
        | Ok i ->
            equal ~msg:"frames are unknown" int 0 i.Info.frames ;
            equal ~msg:"channels" int 1 i.Info.channels ) ;
        match read Nx.float32 "corpus/malformed/trunc.ogg" with
        | Error e ->
            failf "read: %a" pp_error e
        | Ok audio ->
            (* decoded to EOF, never zero-filled: whatever count the decoder
               delivers is the tensor's exact extent *)
            let shape = Nx.shape audio.data in
            equal ~msg:"channel-first rank two" int 2 (Array.length shape) ;
            equal ~msg:"mono" int 1 shape.(0) ;
            equal ~msg:"rate" int 22050 audio.sample_rate ) ]

(* {2 The malformed matrix: typed error or valid data, never a crash} *)

let malformed_files =
  [ "empty.wav"
  ; "garbage.wav"
  ; "trunc.wav"
  ; "trunc.flac"
  ; "trunc.ogg"
  ; "trunc_hdr20.wav"
  ; "trunc_hdr20.flac"
  ; "trunc_hdr20.ogg"
  ; "trunc_hdr_past.wav"
  ; "trunc_hdr_past.flac"
  ; "trunc_hdr_past.ogg"
  ; "liar_datasize.wav"
  ; "liar_int32.wav"
  ; "liar_channels0.wav"
  ; "liar_channels65535.wav"
  ; "liar_frames.flac"
  ; "liar_int64.flac" ]

let outcome_name = function Ok _ -> "valid data" | Error e -> variant_name e

(* [drain_reader path] opens a reader and reads it dry — typed error or valid
   chunks, never a crash. The iteration cap bounds a hypothetically endless
   stream; the corpus never reaches it. *)
let drain_reader ?sample_rate path =
  match Reader.open_ ?sample_rate Nx.float32 path with
  | Error e ->
      Error e
  | Ok r ->
      Fun.protect
        ~finally:(fun () -> Reader.close r)
        (fun () ->
          let rec drain total steps =
            if steps > 4096 then fail "the drain cap was reached" ;
            match Reader.read r ~frames:4096 with
            | Ok None ->
                Ok total
            | Ok (Some c) ->
                drain (total + (Nx.shape c).(1)) (steps + 1)
            | Error e ->
                Error e
          in
          drain 0 0 )

let malformed_tests =
  List.map
    (fun file ->
      test file (fun () ->
          let path = Filename.concat "corpus/malformed" file in
          let probe = info path in
          let r32 = read Nx.float32 path in
          let r64 = read Nx.float64 path in
          (* the remaining matrix legs: a reader drained dry, and the resampled
             read — typed error or valid data, never a crash *)
          let drained = drain_reader path in
          let drained_rs = drain_reader ~sample_rate:16000 path in
          let resampled = read ~sample_rate:16000 Nx.float32 path in
          (* reaching this point is the assertion — every call returned a value;
             record the outcomes so a behavior change is visible *)
          let describe = function
            | Ok (audio : _ audio) ->
                let shape = Nx.shape audio.data in
                if Array.length shape <> 2 then fail "rank is not two" ;
                Printf.sprintf "ok [%d; %d]" shape.(0) shape.(1)
            | Error e ->
                variant_name e
          in
          let _summary =
            Printf.sprintf "info: %s, f32: %s, f64: %s, resampled: %s"
              (outcome_name probe) (describe r32) (describe r64)
              (describe resampled)
          in
          (* a drained reader agrees with the whole-file read on both the
             outcome class and the delivered extent *)
          ( match (r32, drained) with
          | Ok audio, Ok total ->
              equal ~msg:"drained extent" int (Nx.shape audio.data).(1) total
          | Error _, Error _ ->
              ()
          | Ok _, Error _ | Error _, Ok _ ->
              fail "read and the drained reader disagree on the outcome class"
          ) ;
          ( match (resampled, drained_rs) with
          | Ok audio, Ok total ->
              equal ~msg:"resampled drained extent" int
                (Nx.shape audio.data).(1)
                total
          | Error _, Error _ ->
              ()
          | Ok _, Error _ | Error _, Ok _ ->
              fail
                "resampled read and the drained reader disagree on the outcome \
                 class" ) ;
          (* both dtypes must land in the same outcome class (the exact error
             variant may differ: liar_frames.flac trips the allocation guard at
             float64 and decodes short — Truncated — at float32) *)
          match (r32, r64) with
          | Ok a, Ok b ->
              equal ~msg:"dtype-independent shape" (array int) (Nx.shape a.data)
                (Nx.shape b.data)
          | Error _, Error _ ->
              ()
          | _ ->
              fail "float32 and float64 reads disagree on the outcome class" ) )
    malformed_files

(* {2 Invalid_argument preconditions} *)

let precondition_tests =
  [ test "read refuses a non-decodable dtype" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> read Nx.float16 "corpus/wav_pcm16_22050_mono.wav") )
  ; test "write refuses rank zero" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () ->
            write "x.wav" {data= Nx.zeros Nx.float32 [||]; sample_rate= 22050} ) )
  ; test "write refuses rank three" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () ->
            write "x.wav"
              {data= Nx.zeros Nx.float32 [|1; 1; 8|]; sample_rate= 22050} ) )
  ; test "write refuses a non-positive sample rate" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () ->
            write "x.wav" {data= Nx.zeros Nx.float32 [|1; 8|]; sample_rate= 0} ) )
  ; test "Format.create refuses Vorbis outside Ogg" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> Format.create ~encoding:`Vorbis `Wav) )
  ; test "Format.create refuses PCM inside Ogg" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> Format.create ~encoding:`Pcm_16 `Ogg) )
  ; test "Format.create refuses float and 32-bit encodings inside FLAC"
      (fun () ->
        List.iter
          (fun encoding ->
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> Format.create ~encoding `Flac) )
          [`Pcm_32; `Float32; `Float64] ) ]

(* {2 pp_error} *)

let pp_tests =
  [ test "pp_error covers every variant in the house format" (fun () ->
        let cases =
          [ (Not_found {path= "a.wav"}, {|soundml-io: "a.wav" does not exist|})
          ; ( Permission_denied {path= "a.wav"}
            , {|soundml-io: "a.wav": permission denied|} )
          ; ( Unrecognized_format {path= "a.wav"; details= "why"}
            , {|soundml-io: "a.wav" is not a recognized audio format (why)|} )
          ; ( Unsupported {path= "a.flac"; details= "why"}
            , {|soundml-io: "a.flac" cannot be encoded (why)|} )
          ; ( Truncated
                {path= "clip.flac"; expected_frames= 22050; read_frames= 0}
            , {|soundml-io: "clip.flac" is truncated (header advertises 22050 frames, decode delivered 0)|}
            )
          ; ( Io {path= "a.wav"; op= "seek"; details= "why"}
            , {|soundml-io: "a.wav": seek failed (why)|} ) ]
        in
        List.iter
          (fun (e, expected) ->
            equal ~msg:"message" string expected
              (Stdlib.Format.asprintf "%a" pp_error e) )
          cases ) ]

let suite =
  [ group "errors: the reachable matrix" matrix_tests
  ; group "errors: unknown-length streams" unknown_length_tests
  ; group "errors: malformed corpus" malformed_tests
  ; group "errors: preconditions" precondition_tests
  ; group "errors: pp_error" pp_tests ]
