(* Write-then-read round-trips, sample-exact per format x dtype x channels (the
   §8.1 grids): every asserted signal is grid-valued — exactly representable in
   the target encoding — so lossless round-trips are sample-exact rather than
   tolerance-based. "Sample-exact" is [Float.compare = 0] per sample, which
   distinguishes -0.0 from 0.0 and equates NaN with NaN — bit-equality up to NaN
   payloads.

   The 24-bit cells are mandatory: float32 holds every 24-bit integer value
   exactly (24-bit mantissa). The 32-bit cells respect what the dtype can carry:
   the full k/2^31 grid at float64, the 2^8 sub-grid (k/2^23) at float32. Float
   subtypes store verbatim, including denormals, signed zero, infinities and
   NaN. Ogg/Vorbis is lossy: shape and rate are exact, the payload is held to
   SNR >= 40 dB, and the 2.5 M-frame write pins the 65536-frame sf_writef cap
   (libsndfile 1.2.2 segfaults at 2.2 M frames in one call). *)

open Windtrap
open Soundml_io

let tmp_dir = fixture (fun () -> Filename.temp_dir "soundml_io_roundtrip" "")

let temp_path ext =
  let dir = tmp_dir () in
  Filename.temp_file ~temp_dir:dir "clip" ext

(* {2 Grid-valued signals} *)

let lcg seed =
  let state = ref seed in
  fun () ->
    state := ((1103515245 * !state) + 12345) land 0x3FFFFFFF ;
    !state

(* [pcm_values ~denom ~m ~step n seed] is [n] values [k * step / denom] with [k]
   uniform over the symmetric range (a 60-bit draw covers the 32-bit grid), full
   scale pinned in the leading samples. *)
let pcm_values ~denom ~m ~step n seed =
  let draw = lcg seed in
  let wide () = (draw () lsl 30) lxor draw () in
  Array.init n (fun i ->
      let k =
        match i with
        | 0 ->
            -m
        | 1 ->
            m - 1
        | 2 ->
            0
        | _ ->
            (wide () mod (2 * m)) - m
      in
      Float.of_int (k * step) /. denom )

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let float32_specials =
  [| 0.
   ; -0.
   ; 1.
   ; -1.
   ; f32 1e-40 (* denormal *)
   ; Int32.float_of_bits 1l (* smallest denormal *)
   ; Int32.float_of_bits 0x7F7FFFFFl (* largest finite *)
   ; infinity
   ; neg_infinity
   ; nan |]

let float64_specials =
  [| 0.
   ; -0.
   ; 1.
   ; -1.
   ; 1e-310 (* denormal *)
   ; Int64.float_of_bits 1L (* smallest denormal *)
   ; infinity
   ; neg_infinity
   ; nan |]

let float_values ~specials ~round n seed =
  let draw = lcg seed in
  Array.init n (fun i ->
      if i < Array.length specials then specials.(i)
      else round ((Float.of_int (draw ()) /. Float.of_int (1 lsl 29)) -. 1.) )

(* The grid for one encoding cell, at the dtype the cell declares. *)
let grid_values (type a) (encoding : Format.encoding) (dt : (float, a) Nx.dtype)
    n seed =
  match (encoding, dt) with
  | `Pcm_16, _ ->
      pcm_values ~denom:32768. ~m:32768 ~step:1 n seed
  | `Pcm_24, _ ->
      pcm_values ~denom:8388608. ~m:8388608 ~step:1 n seed
  | `Pcm_32, Nx.Float64 ->
      pcm_values ~denom:2147483648. ~m:(1 lsl 31) ~step:1 n seed
  | `Pcm_32, _ ->
      (* the 2^8 sub-grid float32 carries exactly *)
      pcm_values ~denom:2147483648. ~m:8388608 ~step:256 n seed
  | `Float32, _ ->
      float_values ~specials:float32_specials ~round:f32 n seed
  | `Float64, _ ->
      float_values ~specials:float64_specials ~round:Fun.id n seed
  | `Vorbis, _ ->
      (* not grid-comparable; the Ogg tests build their own signal *)
      assert false

let signal encoding dt ~channels ~frames seed =
  let data =
    Array.init channels (fun c -> grid_values encoding dt frames (seed + c))
  in
  Nx.create dt [|channels; frames|] (Array.concat (Array.to_list data))

(* {2 Sample-exact comparison} *)

let check_exact ~msg expected actual =
  let e = Nx.to_array expected and a = Nx.to_array actual in
  if Array.length e <> Array.length a then
    failf "%s: %d samples decoded, %d written" msg (Array.length a)
      (Array.length e) ;
  Array.iteri
    (fun i ev ->
      if Float.compare ev a.(i) <> 0 then
        failf "%s: sample %d decoded as %.17g, written %.17g" msg i a.(i) ev )
    e

let check_audio ~msg ~sample_rate ~shape expected (audio : _ audio) =
  equal ~msg:(msg ^ ": sample rate") int sample_rate audio.sample_rate ;
  equal ~msg:(msg ^ ": shape") (array int) shape (Nx.shape audio.data) ;
  check_exact ~msg expected audio.data

let roundtrip ?format ~ext ~sample_rate dt x =
  let path = temp_path ext in
  ( match write ?format path {data= x; sample_rate} with
  | Ok () ->
      ()
  | Error e ->
      failf "write: %a" pp_error e ) ;
  match read dt path with
  | Ok audio ->
      Sys.remove path ; audio
  | Error e ->
      failf "read: %a" pp_error e

(* {2 The lossless matrix} *)

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

let dtype_name : type a. (float, a) Nx.dtype -> string = function
  | Nx.Float32 ->
      "f32"
  | Nx.Float64 ->
      "f64"
  | _ ->
      "other"

let lossless_case (type a) container encoding (dt : (float, a) Nx.dtype)
    ~channels ~frames =
  let name =
    Stdlib.Format.asprintf "%a %s ch%d n%d" Format.pp
      (Format.create ~encoding container)
      (dtype_name dt) channels frames
  in
  test name (fun () ->
      let x = signal encoding dt ~channels ~frames 20250804 in
      let audio =
        roundtrip
          ~format:(Format.create ~encoding container)
          ~ext:(container_ext container) ~sample_rate:22050 dt x
      in
      check_audio ~msg:name ~sample_rate:22050 ~shape:[|channels; frames|] x
        audio )

let matrix_tests =
  (* the full length x channel grid on the headline cell *)
  List.concat_map
    (fun frames ->
      List.map
        (fun channels -> lossless_case `Wav `Pcm_16 Nx.float64 ~channels ~frames)
        [1; 2; 8] )
    [0; 1; 3; 4096]
  (* every encoding x dtype cell, mono and stereo *)
  @ List.concat_map
      (fun (container, encodings) ->
        List.concat_map
          (fun encoding ->
            let dtypes =
              match encoding with `Float64 -> [`F64] | _ -> [`F32; `F64]
            in
            List.concat_map
              (fun dt ->
                if (container, encoding, dt) = (`Wav, `Pcm_16, `F64) then []
                  (* the headline cell already runs the full grid *)
                else
                  List.map
                    (fun channels ->
                      match dt with
                      | `F32 ->
                          lossless_case container encoding Nx.float32 ~channels
                            ~frames:4096
                      | `F64 ->
                          lossless_case container encoding Nx.float64 ~channels
                            ~frames:4096 )
                    [1; 2] )
              dtypes )
          encodings )
      [ (`Wav, [`Pcm_16; `Pcm_24; `Pcm_32; `Float32; `Float64])
      ; (`Aiff, [`Pcm_16; `Pcm_24; `Pcm_32; `Float32; `Float64])
      ; (`Caf, [`Pcm_16; `Pcm_24; `Pcm_32; `Float32; `Float64])
      ; (`Flac, [`Pcm_16; `Pcm_24]) ]
  (* eight channels off the headline cell *)
  @ [ lossless_case `Flac `Pcm_16 Nx.float32 ~channels:8 ~frames:4096
    ; lossless_case `Caf `Float32 Nx.float32 ~channels:8 ~frames:4096
    ; lossless_case `Wav `Pcm_24 Nx.float64 ~channels:8 ~frames:4096 ]
  (* 65537 crosses the 65536-frame write-chunk boundary *)
  @ [ lossless_case `Wav `Pcm_16 Nx.float32 ~channels:2 ~frames:65537
    ; lossless_case `Aiff `Pcm_16 Nx.float64 ~channels:1 ~frames:65537
    ; lossless_case `Caf `Float32 Nx.float32 ~channels:2 ~frames:65537
    ; lossless_case `Flac `Pcm_16 Nx.float64 ~channels:1 ~frames:65537 ]
  (* the tiny lengths on every other lossless container (the headline grid runs
     them on WAV): empty and near-empty files exercise the header-only and
     single-frame paths. The FLAC empty cell is excluded — libsndfile writes a
     0-byte file no decoder recognizes; pinned below as documented
     python-soundfile parity. *)
  @ List.concat_map
      (fun frames ->
        [ lossless_case `Aiff `Pcm_16 Nx.float64 ~channels:2 ~frames
        ; lossless_case `Caf `Pcm_16 Nx.float64 ~channels:2 ~frames ] )
      [0; 1; 3]
  @ List.map
      (fun frames -> lossless_case `Flac `Pcm_16 Nx.float64 ~channels:2 ~frames)
      [1; 3]
  @ [ test "an empty flac writes 0 bytes and reads back Unrecognized_format"
        (fun () ->
          (* libsndfile 1.2.2 emits nothing for a 0-frame FLAC stream and the
             resulting empty file is unreadable — by libsndfile itself and by
             python-soundfile 0.14.0, which writes the identical 0-byte file.
             Pinned as upstream parity; documented on {!write}. *)
          let path = temp_path ".flac" in
          ( match
              write path {data= Nx.zeros Nx.float32 [|2; 0|]; sample_rate= 22050}
            with
          | Ok () ->
              ()
          | Error e ->
              failf "write: %a" pp_error e ) ;
          equal ~msg:"a 0-frame flac is a 0-byte file" int 0
            (In_channel.with_open_bin path In_channel.length |> Int64.to_int) ;
          ( match read Nx.float32 path with
          | Error (Unrecognized_format _) ->
              ()
          | Ok _ ->
              fail "a 0-byte flac decoded"
          | Error e ->
              failf "read: expected Unrecognized_format, got %a" pp_error e ) ;
          Sys.remove path ) ]

(* {2 Write-input forms} *)

let input_form_tests =
  [ test "rank-one mono writes the same file as [1; frames]" (fun () ->
        let values = grid_values `Pcm_16 Nx.float64 4096 7 in
        let rank1 = Nx.create Nx.float64 [|4096|] values in
        let rank2 = Nx.create Nx.float64 [|1; 4096|] values in
        let a = roundtrip ~ext:".wav" ~sample_rate:8000 Nx.float64 rank1 in
        let b = roundtrip ~ext:".wav" ~sample_rate:8000 Nx.float64 rank2 in
        equal ~msg:"shape" (array int) [|1; 4096|] (Nx.shape a.data) ;
        check_exact ~msg:"rank1 vs rank2" a.data b.data )
  ; test "a non-contiguous view writes the same bytes as its compaction"
      (fun () ->
        let values = grid_values `Pcm_16 Nx.float64 512 11 in
        let base = Nx.create Nx.float64 [|256; 2|] values in
        let view = Nx.transpose base in
        is_false ~msg:"the view is non-contiguous" (Nx.is_c_contiguous view) ;
        let a = roundtrip ~ext:".wav" ~sample_rate:8000 Nx.float64 view in
        let b =
          roundtrip ~ext:".wav" ~sample_rate:8000 Nx.float64
            (Nx.contiguous view)
        in
        check_exact ~msg:"view vs compaction" a.data b.data ) ]

(* {2 Downmix}

   [?mono] is the channel mean: summed in channel order in the element dtype,
   multiplied by 1/channels. The expectation is computed here with the same
   operation order — in float32 every intermediate is rounded through the
   float32 grid. *)

let downmix_expected (type a) (dt : (float, a) Nx.dtype) x =
  let shape = Nx.shape x in
  let channels = shape.(0) and frames = shape.(1) in
  let values = Nx.to_array x in
  let round : float -> float =
    match dt with Nx.Float32 -> f32 | _ -> Fun.id
  in
  let inv = 1. /. Float.of_int channels in
  Array.init frames (fun i ->
      let acc = ref 0. in
      for c = 0 to channels - 1 do
        acc := round (!acc +. values.((c * frames) + i))
      done ;
      round (!acc *. inv) )

let downmix_case (type a) (dt : (float, a) Nx.dtype) ~channels =
  let name =
    Printf.sprintf "read ~mono downmixes %d channels (%s)" channels
      (dtype_name dt)
  in
  test name (fun () ->
      let x = signal `Float32 dt ~channels ~frames:1024 31337 in
      (* NaN and infinities poison a mean; keep the payload finite *)
      let x = Nx.where (Nx.isfinite x) x (Nx.zeros_like x) in
      let path = temp_path ".wav" in
      let fmt =
        Format.create
          ~encoding:(match dt with Nx.Float64 -> `Float64 | _ -> `Float32)
          `Wav
      in
      ( match write ~format:fmt path {data= x; sample_rate= 22050} with
      | Ok () ->
          ()
      | Error e ->
          failf "write: %a" pp_error e ) ;
      match read ~mono:true dt path with
      | Error e ->
          failf "read ~mono: %a" pp_error e
      | Ok audio ->
          Sys.remove path ;
          equal ~msg:"shape" (array int) [|1; 1024|] (Nx.shape audio.data) ;
          let expected = downmix_expected dt x in
          let actual = Nx.to_array audio.data in
          Array.iteri
            (fun i ev ->
              if Float.compare ev actual.(i) <> 0 then
                failf "sample %d downmixed to %.17g, expected %.17g" i
                  actual.(i) ev )
            expected )

let downmix_tests =
  [ downmix_case Nx.float64 ~channels:2
  ; downmix_case Nx.float64 ~channels:8
  ; downmix_case Nx.float32 ~channels:2
  ; downmix_case Nx.float32 ~channels:8
  ; test "read ~mono of a mono file is the plain read" (fun () ->
        let x = signal `Float32 Nx.float32 ~channels:1 ~frames:512 5 in
        let path = temp_path ".wav" in
        ( match
            write
              ~format:(Format.create ~encoding:`Float32 `Wav)
              path
              {data= x; sample_rate= 22050}
          with
        | Ok () ->
            ()
        | Error e ->
            failf "write: %a" pp_error e ) ;
        let plain = read Nx.float32 path in
        let mono = read ~mono:true Nx.float32 path in
        Sys.remove path ;
        match (plain, mono) with
        | Ok a, Ok b ->
            check_exact ~msg:"mono vs plain" a.data b.data
        | _ ->
            fail "reads failed" ) ]

(* {2 Ogg/Vorbis} *)

let sine_signal ~channels ~frames ~sample_rate =
  let data =
    Array.init channels (fun c ->
        Array.init frames (fun i ->
            let t = Float.of_int i /. Float.of_int sample_rate in
            let f = 440. +. (Float.of_int c *. 110.) in
            0.45 *. sin (2. *. Float.pi *. f *. t) ) )
  in
  Nx.create Nx.float32 [|channels; frames|] (Array.concat (Array.to_list data))

let snr_db reference decoded =
  let r = Nx.to_array reference and d = Nx.to_array decoded in
  let signal = ref 0. and noise = ref 0. in
  Array.iteri
    (fun i rv ->
      signal := !signal +. (rv *. rv) ;
      let e = rv -. d.(i) in
      noise := !noise +. (e *. e) )
    r ;
  10. *. Float.log10 (!signal /. !noise)

let ogg_case ~channels ~frames =
  let name =
    Printf.sprintf "ogg/vorbis ch%d n%d: shape exact, SNR >= 40 dB" channels
      frames
  in
  test name (fun () ->
      let x = sine_signal ~channels ~frames ~sample_rate:44100 in
      let audio = roundtrip ~ext:".ogg" ~sample_rate:44100 Nx.float32 x in
      equal ~msg:"sample rate" int 44100 audio.sample_rate ;
      equal ~msg:"shape" (array int) [|channels; frames|] (Nx.shape audio.data) ;
      let snr = snr_db x audio.data in
      if snr < 40. then failf "SNR %.1f dB below the 40 dB floor" snr )

let ogg_tests =
  [ ogg_case ~channels:1 ~frames:4096
  ; (* a stereo clip this short is dominated by encoder warmup: at 4096 frames
       the measured SNR is 38.8 dB; the floor holds from ~0.4 s on *)
    ogg_case ~channels:2 ~frames:16384
  ; ogg_case ~channels:2 ~frames:65537
  ; test "an empty ogg round-trips to [1; 0]" (fun () ->
        let x = Nx.zeros Nx.float32 [|1; 0|] in
        let audio = roundtrip ~ext:".ogg" ~sample_rate:22050 Nx.float32 x in
        equal ~msg:"shape" (array int) [|1; 0|] (Nx.shape audio.data) )
  ; test "a 2.5 M-frame vorbis write survives the chunked write path" (fun () ->
        (* above the measured 2.2 M-frame single-call segfault threshold of
           libsndfile 1.2.2: the 65536-frame cap is structural, and this is its
           regression test (gate C4) *)
        let frames = 2_500_000 in
        let x = sine_signal ~channels:1 ~frames ~sample_rate:22050 in
        let path = temp_path ".ogg" in
        ( match write path {data= x; sample_rate= 22050} with
        | Ok () ->
            ()
        | Error e ->
            failf "write: %a" pp_error e ) ;
        ( match info path with
        | Ok i ->
            equal ~msg:"frames" int frames i.Info.frames
        | Error e ->
            failf "info: %a" pp_error e ) ;
        Sys.remove path ) ]

(* {2 Format defaulting} *)

let format_tests =
  [ test "write defaults the format to the path's extension" (fun () ->
        let x = signal `Pcm_16 Nx.float64 ~channels:1 ~frames:256 3 in
        let path = temp_path ".flac" in
        ( match write path {data= x; sample_rate= 22050} with
        | Ok () ->
            ()
        | Error e ->
            failf "write: %a" pp_error e ) ;
        ( match info path with
        | Ok i ->
            ( match i.Info.format with
            | Some f ->
                is_true ~msg:"container is flac"
                  (Format.equal f (Format.create `Flac))
            | None ->
                fail "expected a write-surface format" ) ;
            equal ~msg:"frames" int 256 i.Info.frames
        | Error e ->
            failf "info: %a" pp_error e ) ;
        Sys.remove path ) ]

let suite =
  [ group "roundtrip: lossless grids" matrix_tests
  ; group "roundtrip: write-input forms" input_form_tests
  ; group "roundtrip: downmix" downmix_tests
  ; group "roundtrip: ogg/vorbis" ogg_tests
  ; group "roundtrip: format defaulting" format_tests ]
