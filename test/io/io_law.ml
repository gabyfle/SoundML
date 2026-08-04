(* The laws that make the reading faces one decoder.

   The delegation law: [read ~sample_rate] equals [Resample.apply] over the
   native read, bit for bit — the property librosa's maintainers could only pose
   as an open question (librosa #1518). The comparison is on the float bits,
   never a tolerance, across containers, dtypes, qualities, downmix and plan
   shapes (direct, cascade, FFT-executed near-unity) — the law is executor-blind
   because it is the resampler's own partition law: io adds no numeric of its
   own, and these tests prove io fed the kernel honestly (no dropped tail, no
   double flush, no chunk mangling).

   The fold and Reader laws: concatenating the chunks of [fold] (every block
   size, including one-frame blocks) or of a [Reader] (random schedules, with
   and without [?out]) equals the corresponding [read] bit-identically, native
   and resampled.

   Fixture lengths are chosen to cross the decode staging block (32768 frames
   for stereo float32, 16384 for stereo float64), so the kernel is genuinely fed
   multiple chunks and the FFT-executed plans genuinely drain a partial block at
   EOF. *)

open Windtrap
open Soundml_io
module Resample = Soundml.Resample

let tmp_dir = fixture (fun () -> Filename.temp_dir "soundml_io_law" "")

(* {2 Deterministic fixtures} *)

let lcg seed =
  let state = ref seed in
  fun () ->
    state := ((1103515245 * !state) + 12345) land 0x3FFFFFFF ;
    Float.of_int !state /. Float.of_int (1 lsl 30)

(* Audio-like content: a per-channel chirp plus one-pole lowpassed noise,
   peak-bounded — neither trivially compressible nor incompressible. *)
let make_signal ~channels ~frames ~sample_rate seed =
  let data =
    Array.init channels (fun c ->
        let draw = lcg (seed + (100 * c)) in
        let lp = ref 0. in
        Array.init frames (fun i ->
            let t = Float.of_int i /. Float.of_int sample_rate in
            let f0 = 220. *. Float.of_int (c + 1) in
            let sweep =
              0.45 *. sin (2. *. Float.pi *. (f0 +. (400. *. t)) *. t)
            in
            lp := (0.85 *. !lp) +. (0.15 *. ((2. *. draw ()) -. 1.)) ;
            sweep +. (0.3 *. !lp) ) )
  in
  Nx.create Nx.float64 [|channels; frames|] (Array.concat (Array.to_list data))

let write_fixture ~name ~format ~channels ~frames ~sample_rate seed =
  let path = Filename.concat (tmp_dir ()) name in
  if not (Sys.file_exists path) then begin
    let data = make_signal ~channels ~frames ~sample_rate seed in
    match write ~format path {data; sample_rate} with
    | Ok () ->
        ()
    | Error e ->
        failf "fixture %s: %a" name pp_error e
  end ;
  path

(* One deterministic file per law axis; memoized across cases through the
   filesystem, so every case reads identical bytes. *)
let fixture_file = function
  | `Wav_pcm16 ->
      write_fixture ~name:"law_pcm16.wav"
        ~format:(Format.create ~encoding:`Pcm_16 `Wav)
        ~channels:2 ~frames:70000 ~sample_rate:22050 1
  | `Wav_float32 ->
      write_fixture ~name:"law_f32.wav"
        ~format:(Format.create ~encoding:`Float32 `Wav)
        ~channels:2 ~frames:70000 ~sample_rate:22050 2
  | `Flac ->
      write_fixture ~name:"law.flac"
        ~format:(Format.create ~encoding:`Pcm_16 `Flac)
        ~channels:2 ~frames:70000 ~sample_rate:22050 3
  | `Ogg ->
      write_fixture ~name:"law.ogg" ~format:(Format.create `Ogg) ~channels:2
        ~frames:70000 ~sample_rate:22050 4
  | `Wav_44k1 ->
      write_fixture ~name:"law_44k1.wav"
        ~format:(Format.create ~encoding:`Float32 `Wav)
        ~channels:2 ~frames:99371 ~sample_rate:44100 5
  | `Wav_11k025_mono ->
      write_fixture ~name:"law_11k025.wav"
        ~format:(Format.create ~encoding:`Pcm_16 `Wav)
        ~channels:1 ~frames:33000 ~sample_rate:11025 6
  | `Wav_small ->
      write_fixture ~name:"law_small.wav"
        ~format:(Format.create ~encoding:`Pcm_16 `Wav)
        ~channels:2 ~frames:4111 ~sample_rate:22050 7

let file_name = function
  | `Wav_pcm16 ->
      "wav-pcm16"
  | `Wav_float32 ->
      "wav-float32"
  | `Flac ->
      "flac"
  | `Ogg ->
      "ogg"
  | `Wav_44k1 ->
      "wav-44k1"
  | `Wav_11k025_mono ->
      "wav-11k025-mono"
  | `Wav_small ->
      "wav-small"

let dtype_name : type a. (float, a) Nx.dtype -> string = function
  | Nx.Float32 ->
      "f32"
  | Nx.Float64 ->
      "f64"
  | _ ->
      "other"

(* {2 Bit equality} *)

let check_bits ~msg expected actual =
  equal ~msg:(msg ^ ": shape") (array int) (Nx.shape expected) (Nx.shape actual) ;
  let e = Nx.to_array expected and a = Nx.to_array actual in
  Array.iteri
    (fun i ev ->
      if Int64.bits_of_float ev <> Int64.bits_of_float a.(i) then
        failf "%s: sample %d is %.17g, the law demands %.17g" msg i a.(i) ev )
    e

let ok ~msg = function Ok v -> v | Error e -> failf "%s: %a" msg pp_error e

(* {2 The delegation law} *)

let law_case (type a) file ~target ~quality ~quality_name
    (dt : (float, a) Nx.dtype) ~mono =
  let name =
    Printf.sprintf "%s -> %d Hz %s %s%s" (file_name file) target quality_name
      (dtype_name dt)
      (if mono then " mono" else "")
  in
  test name (fun () ->
      let path = fixture_file file in
      let fused =
        ok ~msg:"read ~sample_rate"
          (read ~sample_rate:target ?quality ~mono dt path)
      in
      let native = ok ~msg:"native read" (read ~mono dt path) in
      let config =
        Resample.Config.create ?quality ~sample_rate:native.sample_rate ~target
          ()
      in
      equal ~msg:"target rate" int target fused.sample_rate ;
      check_bits ~msg:"delegation"
        (Resample.apply config native.data)
        fused.data )

let delegation_tests =
  (* containers x rates x quality x dtype, stereo *)
  List.concat_map
    (fun file ->
      List.concat_map
        (fun target ->
          List.concat_map
            (fun (quality_name, quality) ->
              [ law_case file ~target ~quality ~quality_name Nx.float32
                  ~mono:false
              ; law_case file ~target ~quality ~quality_name Nx.float64
                  ~mono:false ] )
            [("high", None); ("fast", Some `Fast)] )
        [16000; 48000] )
    [`Wav_pcm16; `Wav_float32; `Flac; `Ogg]
  (* downmix before resample, on both a lossless and a lossy container *)
  @ [ law_case `Wav_pcm16 ~target:16000 ~quality:None ~quality_name:"high"
        Nx.float32 ~mono:true
    ; law_case `Wav_pcm16 ~target:48000 ~quality:(Some `Fast)
        ~quality_name:"fast" Nx.float64 ~mono:true
    ; law_case `Flac ~target:16000 ~quality:None ~quality_name:"high" Nx.float64
        ~mono:true
    ; law_case `Ogg ~target:48000 ~quality:None ~quality_name:"high" Nx.float32
        ~mono:true ]
  (* the FFT-executed near-unity plan, its EOF drain mid-block *)
  @ [ law_case `Wav_44k1 ~target:48000 ~quality:None ~quality_name:"high"
        Nx.float32 ~mono:false
    ; law_case `Wav_44k1 ~target:48000 ~quality:None ~quality_name:"high"
        Nx.float64 ~mono:false
    ; law_case `Wav_44k1 ~target:16000 ~quality:None ~quality_name:"high"
        Nx.float32 ~mono:false ]
  (* a direct dot-product plan *)
  @ [ law_case `Wav_11k025_mono ~target:8000 ~quality:None ~quality_name:"high"
        Nx.float32 ~mono:false
    ; law_case `Wav_11k025_mono ~target:8000 ~quality:None ~quality_name:"high"
        Nx.float64 ~mono:false ]

let identity_tests =
  [ test "read ~sample_rate at the native rate is the native read" (fun () ->
        let path = fixture_file `Wav_pcm16 in
        let a = ok ~msg:"identity" (read ~sample_rate:22050 Nx.float32 path) in
        let b = ok ~msg:"native" (read Nx.float32 path) in
        equal ~msg:"rate" int 22050 a.sample_rate ;
        check_bits ~msg:"identity" b.data a.data ) ]

(* {2 fold == read} *)

let fold_case (type a) file ~block ?sample_rate (dt : (float, a) Nx.dtype) ~mono
    =
  let name =
    Printf.sprintf "%s block %d%s %s%s" (file_name file) block
      ( match sample_rate with
      | None ->
          ""
      | Some r ->
          Printf.sprintf " -> %d Hz" r )
      (dtype_name dt)
      (if mono then " mono" else "")
  in
  test name (fun () ->
      let path = fixture_file file in
      let whole = ok ~msg:"read" (read ?sample_rate ~mono dt path) in
      let chunks =
        ok ~msg:"fold"
          (fold ~block ?sample_rate ~mono dt path ~init:[] ~f:(fun acc c ->
               (* the chunk is borrowed; copy what the test retains *)
               Nx.copy c :: acc ) )
      in
      List.iter
        (fun c ->
          let shape = Nx.shape c in
          if shape.(1) > block then
            failf "a %d-frame chunk exceeds the block size" shape.(1) )
        chunks ;
      match List.rev chunks with
      | [] ->
          equal ~msg:"an empty fold matches an empty read" (array int)
            [|(Nx.shape whole.data).(0); 0|]
            (Nx.shape whole.data)
      | pieces ->
          check_bits ~msg:"fold == read" whole.data
            (Nx.concatenate ~axis:1 pieces) )

let fold_tests =
  [ fold_case `Wav_small ~block:1 Nx.float32 ~mono:false
  ; fold_case `Wav_small ~block:3 Nx.float64 ~mono:false
  ; fold_case `Wav_small ~block:1 ~sample_rate:16000 Nx.float32 ~mono:false
  ; fold_case `Wav_pcm16 ~block:1000 Nx.float32 ~mono:false
  ; fold_case `Wav_pcm16 ~block:65536 Nx.float64 ~mono:false
  ; fold_case `Wav_pcm16 ~block:4097 ~sample_rate:16000 Nx.float32 ~mono:false
  ; fold_case `Wav_pcm16 ~block:1000 ~sample_rate:48000 Nx.float64 ~mono:true
  ; fold_case `Flac ~block:4096 Nx.float32 ~mono:false
  ; fold_case `Flac ~block:5000 ~sample_rate:48000 Nx.float32 ~mono:false
  ; fold_case `Ogg ~block:4096 ~sample_rate:16000 Nx.float64 ~mono:false
  ; fold_case `Wav_44k1 ~block:8192 ~sample_rate:48000 Nx.float32 ~mono:false
  ; fold_case `Wav_11k025_mono ~block:2048 ~sample_rate:8000 Nx.float64
      ~mono:false ]

(* {2 Reader == read, random schedules} *)

let reader_case (type a) file ?sample_rate (dt : (float, a) Nx.dtype) ~mono
    ~use_out seed =
  let name =
    Printf.sprintf "%s%s %s%s%s seed %d" (file_name file)
      ( match sample_rate with
      | None ->
          ""
      | Some r ->
          Printf.sprintf " -> %d Hz" r )
      (dtype_name dt)
      (if mono then " mono" else "")
      (if use_out then " ?out" else "")
      seed
  in
  test name (fun () ->
      let path = fixture_file file in
      let whole = ok ~msg:"read" (read ?sample_rate ~mono dt path) in
      let r = ok ~msg:"open_" (Reader.open_ ?sample_rate ~mono dt path) in
      let channels = (Nx.shape whole.data).(0) in
      let draw = lcg seed in
      let rec drain acc =
        let frames = 1 + int_of_float (draw () *. 4999.) in
        let out =
          if use_out then Some (Nx.zeros dt [|channels; frames|]) else None
        in
        match Reader.read ?out r ~frames with
        | Error e ->
            failf "Reader.read: %a" pp_error e
        | Ok None ->
            List.rev acc
        | Ok (Some chunk) ->
            let shape = Nx.shape chunk in
            if shape.(1) > frames then
              failf "a %d-frame chunk exceeds the request" shape.(1) ;
            drain (Nx.copy chunk :: acc)
      in
      let pieces = drain [] in
      Reader.close r ;
      match pieces with
      | [] ->
          equal ~msg:"empty" (array int) [|channels; 0|] (Nx.shape whole.data)
      | pieces ->
          check_bits ~msg:"Reader == read" whole.data
            (Nx.concatenate ~axis:1 pieces) )

let reader_tests =
  [ reader_case `Wav_pcm16 Nx.float32 ~mono:false ~use_out:false 11
  ; reader_case `Wav_pcm16 Nx.float32 ~mono:false ~use_out:true 12
  ; reader_case `Wav_pcm16 Nx.float64 ~mono:true ~use_out:false 13
  ; reader_case `Flac Nx.float64 ~mono:false ~use_out:true 14
  ; reader_case `Wav_pcm16 ~sample_rate:16000 Nx.float32 ~mono:false
      ~use_out:false 15
  ; reader_case `Wav_pcm16 ~sample_rate:16000 Nx.float32 ~mono:false
      ~use_out:true 16
  ; reader_case `Wav_44k1 ~sample_rate:48000 Nx.float32 ~mono:false
      ~use_out:false 17
  ; reader_case `Wav_44k1 ~sample_rate:48000 Nx.float64 ~mono:true ~use_out:true
      18
  ; reader_case `Ogg ~sample_rate:48000 Nx.float32 ~mono:false ~use_out:false 19
  ; reader_case `Wav_11k025_mono ~sample_rate:8000 Nx.float32 ~mono:false
      ~use_out:true 20 ]

(* {2 Seek == uninterrupted read} *)

let seek_case (type a) file ?sample_rate (dt : (float, a) Nx.dtype) ~frames
    ~positions =
  let name =
    Printf.sprintf "%s%s %s" (file_name file)
      ( match sample_rate with
      | None ->
          ""
      | Some r ->
          Printf.sprintf " -> %d Hz" r )
      (dtype_name dt)
  in
  test name (fun () ->
      let path = fixture_file file in
      let whole = ok ~msg:"read" (read ?sample_rate dt path) in
      let total = (Nx.shape whole.data).(1) in
      let channels = (Nx.shape whole.data).(0) in
      let r = ok ~msg:"open_" (Reader.open_ ?sample_rate dt path) in
      List.iter
        (fun frame ->
          let frame = Stdlib.min frame (total - 1) in
          ok ~msg:(Printf.sprintf "seek %d" frame) (Reader.seek r ~frame) ;
          let n = Stdlib.min frames (total - frame) in
          match Reader.read r ~frames:n with
          | Error e ->
              failf "read after seek %d: %a" frame pp_error e
          | Ok None ->
              failf "seek %d: EOF where %d frames remain" frame (total - frame)
          | Ok (Some chunk) ->
              check_bits
                ~msg:(Printf.sprintf "span [%d; %d)" frame (frame + n))
                (Nx.shrink [|(0, channels); (frame, frame + n)|] whole.data)
                chunk )
        positions ;
      (* seeking to the very end reads EOF; past it is a typed refusal *)
      ok ~msg:"seek to end" (Reader.seek r ~frame:total) ;
      ( match Reader.read r ~frames:16 with
      | Ok None ->
          ()
      | Ok (Some _) ->
          fail "a read at the end position delivered data"
      | Error e ->
          failf "read at end: %a" pp_error e ) ;
      ( match Reader.seek r ~frame:(total + 1) with
      | Error (Io {op= "seek"; _}) ->
          ()
      | Error e ->
          failf "seek past end: unexpected %a" pp_error e
      | Ok () ->
          fail "seeking past the end succeeded" ) ;
      Reader.close r )

let seek_positions =
  (* inside the first 2K of every plan; around plausible OLS block boundaries
     (+/-1 across 4096 and 8192); backward jumps; the edges *)
  [0; 5; 100; 1024; 4095; 4096; 4097; 8191; 8193; 63; 2048; 0; 8192; 1; 30000]

let seek_tests =
  [ seek_case `Wav_pcm16 Nx.float32 ~frames:512 ~positions:seek_positions
  ; seek_case `Flac Nx.float64 ~frames:512 ~positions:seek_positions
  ; seek_case `Wav_pcm16 ~sample_rate:16000 Nx.float32 ~frames:256
      ~positions:seek_positions
  ; seek_case `Wav_44k1 ~sample_rate:48000 Nx.float32 ~frames:256
      ~positions:seek_positions
  ; seek_case `Wav_44k1 ~sample_rate:48000 Nx.float64 ~frames:256
      ~positions:[4095; 4096; 4097; 0; 100000]
  ; seek_case `Wav_11k025_mono ~sample_rate:8000 Nx.float32 ~frames:256
      ~positions:seek_positions ]

let suite =
  [ group "law: delegation (read ~sample_rate == apply o read)" delegation_tests
  ; group "law: identity rate" identity_tests
  ; group "law: fold == read" fold_tests
  ; group "law: Reader == read" reader_tests
  ; group "law: seek == uninterrupted read" seek_tests ]
