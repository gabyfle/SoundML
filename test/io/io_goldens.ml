(* Decode-parity goldens against python-soundfile, replayed from the committed
   vectors under test/vectors/io/ over the committed fixtures under
   test/io/corpus/ (both written by generate_vectors.py, which records the
   soundfile and bundled-libsndfile versions in each file).

   Lossless cells are sample-exact in both dtypes. The goldens store the float64
   decode; the float32 expectation is its correctly-rounded float32 cast — an
   equality the generator asserts against python-soundfile's own float32 decode
   for every fixture at generation time, so the cast is a recorded fact about
   libsndfile's conversions, not an assumption. Ogg/Vorbis decodes are held to
   the measured cross-stack noise ceiling (1.79e-7, gated at 2e-7): the codec
   stack itself does not promise bit-parity between builds, and neither do we.

   The write-clipping golden pins SFC_SET_CLIPPING: a +/-1.5 full-scale ramp on
   the exact k/64 grid encoded to PCM_16 decodes byte-for-byte as
   python-soundfile's clipped output — saturation at full scale, not
   wraparound. *)

open Windtrap
open Soundml_io

let vectors_dir = Filename.concat ".." (Filename.concat "vectors" "io")

let golden_file name = Tutils.Golden.load (Filename.concat vectors_dir name)

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let ogg_tolerance = 2e-7

let check_values ~msg ~tolerance expected (actual : (float, 'a) Nx.t) =
  let a = Nx.to_array actual in
  if Array.length expected <> Array.length a then
    failf "%s: decoded %d samples, golden has %d" msg (Array.length a)
      (Array.length expected) ;
  Array.iteri
    (fun i ev ->
      let ok =
        if tolerance = 0. then Float.compare ev a.(i) = 0
        else Float.abs (ev -. a.(i)) <= tolerance
      in
      if not ok then
        failf "%s: sample %d decoded as %.17g, python-soundfile got %.17g" msg i
          a.(i) ev )
    expected

let decode_case (case : Tutils.Golden.case) =
  let file = Tutils.Golden.string_param case "file" in
  let sample_rate = Tutils.Golden.int_param case "sample_rate" in
  let lossy = Tutils.Golden.string_param case "container" = "ogg" in
  let tolerance = if lossy then ogg_tolerance else 0. in
  [ test (case.name ^ " f64") (fun () ->
        match read Nx.float64 file with
        | Error e ->
            failf "read: %a" pp_error e
        | Ok audio ->
            equal ~msg:"sample rate" int sample_rate audio.sample_rate ;
            equal ~msg:"shape" (array int) case.shape (Nx.shape audio.data) ;
            check_values ~msg:"f64 decode" ~tolerance case.values audio.data )
  ; test (case.name ^ " f32") (fun () ->
        match read Nx.float32 file with
        | Error e ->
            failf "read: %a" pp_error e
        | Ok audio ->
            equal ~msg:"sample rate" int sample_rate audio.sample_rate ;
            equal ~msg:"shape" (array int) case.shape (Nx.shape audio.data) ;
            let expected = Array.map f32 case.values in
            check_values ~msg:"f32 decode" ~tolerance expected audio.data ) ]

let decode_tests = List.concat_map decode_case (golden_file "decode.json").cases

(* {2 Header probes over the same fixtures} *)

let info_tests =
  List.map
    (fun (case : Tutils.Golden.case) ->
      let file = Tutils.Golden.string_param case "file" in
      let sample_rate = Tutils.Golden.int_param case "sample_rate" in
      test ("info " ^ case.name) (fun () ->
          match info file with
          | Error e ->
              failf "info: %a" pp_error e
          | Ok i -> (
              equal ~msg:"frames" int case.shape.(1) i.Info.frames ;
              equal ~msg:"channels" int case.shape.(0) i.Info.channels ;
              equal ~msg:"sample rate" int sample_rate i.Info.sample_rate ;
              is_true ~msg:"format name is present"
                (String.length i.Info.format_name > 0) ;
              match i.Info.format with
              | None ->
                  fail "expected a write-surface format"
              | Some f ->
                  equal ~msg:"container" string
                    (Tutils.Golden.string_param case "container")
                    ( match Format.container f with
                    | `Wav ->
                        "wav"
                    | `Aiff ->
                        "aiff"
                    | `Flac ->
                        "flac"
                    | `Ogg ->
                        "ogg"
                    | `Caf ->
                        "caf" ) ) ) )
    (golden_file "decode.json").cases

(* {2 The write-clipping golden} *)

let clipping_tests =
  List.map
    (fun (case : Tutils.Golden.case) ->
      test ("write clipping: " ^ case.name) (fun () ->
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
          let dir = Filename.temp_dir "soundml_io_clip" "" in
          let path = Filename.temp_file ~temp_dir:dir "ramp" ".wav" in
          ( match
              write
                ~format:(Format.create ~encoding:`Pcm_16 `Wav)
                path {data= ramp; sample_rate}
            with
          | Ok () ->
              ()
          | Error e ->
              failf "write: %a" pp_error e ) ;
          ( match read Nx.float64 path with
          | Error e ->
              failf "read: %a" pp_error e
          | Ok audio ->
              equal ~msg:"shape" (array int) case.shape (Nx.shape audio.data) ;
              check_values ~msg:"clipped decode" ~tolerance:0. case.values
                audio.data ) ;
          Sys.remove path ) )
    (golden_file "clipping.json").cases

let suite =
  [ group "goldens: decode parity" decode_tests
  ; group "goldens: header probes" info_tests
  ; group "goldens: write clipping" clipping_tests ]
