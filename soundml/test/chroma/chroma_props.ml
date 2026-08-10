(* Contracts of the chroma configuration and both projections: creation
   validation, accessors and defaults, pp/equal, the fresh-copy semantics of
   [filterbank], the shape and broadcast behavior of the projections, the
   normalisation options including the sub-tiny frame that is left alone, the
   structure of the constant-Q assignment matrix, and the flat features' own
   preconditions. *)

open Windtrap
open Soundml

let sample_rate = 22050

let config ?n_chroma ?tuning ?ctroct ?octwidth ?base_c ?(fft_size = 512) () =
  Chroma.Config.create ?n_chroma ?tuning ?ctroct ?octwidth ?base_c ~sample_rate
    ~fft_size ()

(* {2 Creation validation} *)

let validation_tests =
  [ test "n_chroma must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero bands"
          "create: cannot build 0 chroma bands (n_chroma must be at least 1)"
          (fun () -> ignore (config ~n_chroma:0 ()) ) )
  ; test "sample_rate must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero rate"
          "create: cannot use a sample rate of 0 Hz (sample_rate must be at \
           least 1)" (fun () ->
            ignore (Chroma.Config.create ~sample_rate:0 ~fft_size:512 ()) ) )
  ; test "fft_size must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero fft"
          "create: cannot use an FFT of size 0 (fft_size must be at least 1)"
          (fun () -> ignore (config ~fft_size:0 ()) ) )
  ; test "tuning must be finite" (fun () ->
        raises_invalid_arg ~msg:"nan tuning"
          "create: cannot shift the scale by nan bins (tuning must be finite)"
          (fun () -> ignore (config ~tuning:Float.nan ()) ) )
  ; test "ctroct must be finite" (fun () ->
        raises_invalid_arg ~msg:"infinite centre"
          "create: cannot centre the octave envelope at inf (ctroct must be \
           finite)" (fun () -> ignore (config ~ctroct:Float.infinity ()) ) )
  ; test "octwidth must be finite and positive" (fun () ->
        raises_invalid_arg ~msg:"zero width"
          "create: cannot use an octave envelope of half-width 0 (octwidth \
           must be finite and positive)" (fun () ->
            ignore (config ~octwidth:(Some 0.) ()) ) )
  ; test "a p-norm exponent must be finite and positive" (fun () ->
        let c = config () in
        let s = Nx.ones Nx.float64 [|257; 4|] in
        raises_invalid_arg ~msg:"zero exponent"
          "apply: cannot normalise in the 0-norm (the exponent must be finite \
           and positive)" (fun () -> ignore (Chroma.apply ~norm:(`P 0.) c s) ) ;
        raises_invalid_arg ~msg:"negative exponent"
          "apply: cannot normalise in the -2-norm (the exponent must be finite \
           and positive)" (fun () ->
            ignore (Chroma.apply ~norm:(`P (-2.)) c s) ) ) ]

(* {2 Accessors, pp, equal} *)

let accessor_tests =
  [ test "accessors return the creation parameters" (fun () ->
        let c =
          config ~n_chroma:24 ~tuning:0.3 ~ctroct:4.5 ~octwidth:None
            ~base_c:false ~fft_size:1024 ()
        in
        equal ~msg:"n_chroma" int 24 (Chroma.Config.n_chroma c) ;
        equal ~msg:"sample_rate" int sample_rate (Chroma.Config.sample_rate c) ;
        equal ~msg:"fft_size" int 1024 (Chroma.Config.fft_size c) ;
        equal ~msg:"bins" int 513 (Chroma.Config.bins c) ;
        equal ~msg:"tuning" (float 0.) 0.3 (Chroma.Config.tuning c) ;
        equal ~msg:"ctroct" (float 0.) 4.5 (Chroma.Config.ctroct c) ;
        is_true ~msg:"octwidth" (Chroma.Config.octwidth c = None) ;
        is_true ~msg:"base_c" (not (Chroma.Config.base_c c)) )
  ; test "defaults are twelve bands centred five octaves up" (fun () ->
        let c = config () in
        equal ~msg:"n_chroma" int 12 (Chroma.Config.n_chroma c) ;
        equal ~msg:"tuning" (float 0.) 0. (Chroma.Config.tuning c) ;
        equal ~msg:"ctroct" (float 0.) 5. (Chroma.Config.ctroct c) ;
        is_true ~msg:"octwidth" (Chroma.Config.octwidth c = Some 2.) ;
        is_true ~msg:"base_c" (Chroma.Config.base_c c) )
  ; test "pp prints the compact single-line form" (fun () ->
        equal ~msg:"pp" string
          "chroma(n_chroma=12, sample_rate=22050, fft_size=512, tuning=0, \
           ctroct=5, octwidth=2, base_c=true)"
          (Format.asprintf "%a" Chroma.Config.pp (config ())) ;
        equal ~msg:"pp/flat" string
          "chroma(n_chroma=12, sample_rate=22050, fft_size=512, tuning=0, \
           ctroct=5, octwidth=none, base_c=true)"
          (Format.asprintf "%a" Chroma.Config.pp (config ~octwidth:None ())) )
  ; test "equal compares creation parameters" (fun () ->
        is_true ~msg:"same" (Chroma.Config.equal (config ()) (config ())) ;
        is_true ~msg:"different n_chroma"
          (not (Chroma.Config.equal (config ()) (config ~n_chroma:24 ()))) ;
        is_true ~msg:"different octwidth"
          (not (Chroma.Config.equal (config ()) (config ~octwidth:None ()))) ;
        is_true ~msg:"different base_c"
          (not (Chroma.Config.equal (config ()) (config ~base_c:false ()))) ) ]

(* {2 The filterbank} *)

let filterbank_tests =
  [ test "filterbank returns a fresh copy" (fun () ->
        let c = config () in
        let first = Chroma.filterbank Nx.float64 c in
        let before = Nx.to_array first in
        Nx.set_item [0; 0] 12345. first ;
        let second = Chroma.filterbank Nx.float64 c in
        equal ~msg:"config untouched" (float 0.) before.(0)
          (Nx.item [0; 0] second) )
  ; test "the filterbank has one row per band and one column per bin" (fun () ->
        let c = config ~n_chroma:24 ~fft_size:256 () in
        equal ~msg:"shape" (array int) [|24; 129|]
          (Nx.shape (Chroma.filterbank Nx.float64 c)) )
  ; test "a flat envelope leaves every column normalised" (fun () ->
        (* Without the octave envelope the columns are exactly the unit-length
           bump vectors, so every column norm is one. *)
        let c = config ~octwidth:None ~fft_size:256 () in
        let w = Chroma.filterbank Nx.float64 c in
        let norms = Nx.sqrt (Nx.sum ~axes:[-2] (Nx.square w)) in
        Array.iteri
          (fun j v ->
            if Float.abs (v -. 1.) > 1e-12 then
              failf "column %d has norm %.17g" j v )
          (Nx.to_array norms) ) ]

(* {2 Normalisation} *)

let signal n =
  Array.init n (fun i -> Float.abs (Float.sin (0.37 *. Float.of_int i)) +. 0.1)

let spectrum bins frames =
  Nx.create Nx.float64 [|bins; frames|] (signal (bins * frames))

let normalisation_tests =
  [ test "the infinity norm sets every frame's peak to one" (fun () ->
        let c = config ~fft_size:256 () in
        let chroma = Chroma.apply c (spectrum 129 6) in
        let peaks = Nx.to_array (Nx.max ~axes:[-2] chroma) in
        Array.iteri
          (fun t v ->
            if Float.abs (v -. 1.) > 1e-12 then
              failf "frame %d peaks at %.17g" t v )
          peaks )
  ; test "the 1-norm sets every frame's sum to one" (fun () ->
        let c = config ~fft_size:256 () in
        let chroma = Chroma.apply ~norm:(`P 1.) c (spectrum 129 6) in
        let sums = Nx.to_array (Nx.sum ~axes:[-2] chroma) in
        Array.iteri
          (fun t v ->
            if Float.abs (v -. 1.) > 1e-12 then
              failf "frame %d sums to %.17g" t v )
          sums )
  ; test "`None is the raw product" (fun () ->
        let c = config ~fft_size:256 () in
        let s = spectrum 129 6 in
        let raw = Chroma.apply ~norm:`None c s in
        let expected = Nx.matmul (Chroma.filterbank Nx.float64 c) s in
        Tutils.check_close ~msg:"raw" ~expected:(Nx.to_array expected) raw )
  ; test "a sub-tiny frame is left alone rather than amplified" (fun () ->
        let c = config ~fft_size:256 () in
        let s = Nx.zeros Nx.float64 [|129; 3|] in
        let chroma = Chroma.apply c s in
        Array.iteri
          (fun i v ->
            if not (Float.equal v 0.) then
              failf "silent frame entry %d normalised to %.17g" i v )
          (Nx.to_array chroma) ;
        (* A frame whose projection lands just above the smallest normal is
           normalised; one below is not. *)
        let scale = 1e-300 in
        let live = Nx.mul_s (spectrum 129 3) scale in
        let peaks = Nx.to_array (Nx.max ~axes:[-2] (Chroma.apply c live)) in
        Array.iteri
          (fun t v ->
            if Float.abs (v -. 1.) > 1e-12 then
              failf "scaled frame %d peaks at %.17g" t v )
          peaks ) ]

(* {2 The constant-Q projection} *)

let cqt_config ?(n_bins = 84) ?(bins_per_octave = 12) ?(hop = 512) () =
  Cqt.Config.create ~n_bins ~bins_per_octave ~hop ~sample_rate ()

let projection_tests =
  [ test "every constant-Q bin lands in exactly one band" (fun () ->
        List.iter
          (fun (n_bins, bins_per_octave, n_chroma) ->
            let m =
              Chroma.cqt_projection Nx.float64 ~n_chroma
                (cqt_config ~n_bins ~bins_per_octave ())
            in
            equal ~msg:"shape" (array int) [|n_chroma; n_bins|] (Nx.shape m) ;
            let sums = Nx.to_array (Nx.sum ~axes:[-2] m) in
            Array.iteri
              (fun k v ->
                if not (Float.equal v 1.) then
                  failf "bin %d lands in %.17g bands" k v )
              sums )
          [(84, 12, 12); (252, 36, 12); (120, 24, 24); (84, 12, 4)] )
  ; test "an incompatible resolution raises" (fun () ->
        raises_invalid_arg ~msg:"non-multiple"
          "cqt_projection: cannot fold 12 bins per octave onto 24 chroma bands \
           (bins_per_octave must be an integer multiple of n_chroma)" (fun () ->
            ignore
              (Chroma.cqt_projection Nx.float64 ~n_chroma:24 (cqt_config ())) ) ;
        raises_invalid_arg ~msg:"non-multiple/of_cqt"
          "of_cqt: cannot fold 12 bins per octave onto 5 chroma bands \
           (bins_per_octave must be an integer multiple of n_chroma)" (fun () ->
            ignore
              (Chroma.of_cqt ~n_chroma:5 (cqt_config ())
                 (Nx.ones Nx.float64 [|84; 3|]) ) ) )
  ; test "the projection is periodic in the ladder's octave" (fun () ->
        let m =
          Nx.to_array (Chroma.cqt_projection Nx.float64 (cqt_config ()))
        in
        for c = 0 to 11 do
          for k = 0 to 71 do
            let a = m.((c * 84) + k) and b = m.((c * 84) + k + 12) in
            if not (Float.equal a b) then
              failf "band %d: bin %d and bin %d differ" c k (k + 12)
          done
        done ) ]

(* {2 Shapes and batching} *)

let shape_tests =
  [ test "leading axes broadcast to per-spectrogram calls" (fun () ->
        let c = config ~fft_size:256 () in
        let first = spectrum 129 5 in
        let second = Nx.mul_s first 0.25 in
        let batch = Nx.stack ~axis:0 [first; second] in
        let stacked = Chroma.apply c batch in
        equal ~msg:"shape" (array int) [|2; 12; 5|] (Nx.shape stacked) ;
        List.iteri
          (fun row s ->
            let alone = Chroma.apply c s in
            let slice =
              Nx.reshape (Nx.shape alone)
                (Nx.shrink [|(row, row + 1); (0, 12); (0, 5)|] stacked)
            in
            Tutils.check_close
              ~msg:(Printf.sprintf "row %d" row)
              ~expected:(Nx.to_array alone) slice )
          [first; second] )
  ; test "a bins mismatch names both sides" (fun () ->
        let c = config ~fft_size:256 () in
        raises_invalid_arg ~msg:"wrong bins"
          "apply: cannot project 65 frequency bins through a matrix built for \
           an FFT of size 256 (129 bins)" (fun () ->
            ignore (Chroma.apply c (spectrum 65 3)) ) ;
        raises_invalid_arg ~msg:"wrong cqt bins"
          "of_cqt: cannot project 60 constant-Q bins through a configuration \
           of 84 bins" (fun () ->
            ignore (Chroma.of_cqt (cqt_config ()) (spectrum 60 3)) ) )
  ; test "a rank-one tensor is rejected" (fun () ->
        raises_invalid_arg ~msg:"rank one"
          "apply: cannot project a rank-1 tensor (the projection needs [...; \
           bins; frames])" (fun () ->
            ignore
              (Chroma.apply (config ~fft_size:256 ())
                 (Nx.ones Nx.float64 [|129|]) ) ) )
  ; test "an empty frame axis maps to an empty chromagram" (fun () ->
        let c = config ~fft_size:256 () in
        equal ~msg:"apply" (array int) [|12; 0|]
          (Nx.shape (Chroma.apply c (Nx.zeros Nx.float64 [|129; 0|]))) ;
        equal ~msg:"of_cqt" (array int) [|12; 0|]
          (Nx.shape
             (Chroma.of_cqt (cqt_config ()) (Nx.zeros Nx.float64 [|84; 0|])) ) )
  ]

(* {2 The flat features} *)

let flat_tests =
  [ test "chroma_stft rejects a fft_size disagreement" (fun () ->
        let stft = Stft.Config.create ~fft_size:512 () in
        let chroma = config ~fft_size:256 () in
        raises_invalid_arg ~msg:"mismatch"
          "chroma_stft: cannot project a 512-point STFT through a filterbank \
           built for an FFT of size 256 (the two configurations must agree on \
           fft_size)" (fun () ->
            ignore (chroma_stft stft chroma (Nx.ones Nx.float64 [|2048|])) ) )
  ; test "chroma_stft is apply after power_spectrum" (fun () ->
        let stft = Stft.Config.create ~fft_size:256 ~pad:(`Constant 0.) () in
        let chroma = config ~fft_size:256 () in
        let x = Nx.create Nx.float64 [|4096|] (signal 4096) in
        Tutils.check_close ~msg:"chroma_stft"
          ~expected:
            (Nx.to_array (Chroma.apply chroma (Stft.power_spectrum stft x)))
          (chroma_stft stft chroma x) )
  ; test "chroma_cqt is of_cqt after the magnitude transform" (fun () ->
        let c = cqt_config ~n_bins:36 () in
        let x = Nx.create Nx.float64 [|4096|] (signal 4096) in
        Tutils.check_close ~msg:"chroma_cqt"
          ~expected:
            (Nx.to_array (Chroma.of_cqt c (Cqt.power_spectrum ~power:1. c x)))
          (chroma_cqt c x) )
  ; test "the float32 features never leave float32" (fun () ->
        let stft = Stft.Config.create ~fft_size:256 ~pad:(`Constant 0.) () in
        let chroma = config ~fft_size:256 () in
        let c = cqt_config ~n_bins:36 () in
        let x =
          Nx.cast Nx.float32 (Nx.create Nx.float64 [|4096|] (signal 4096))
        in
        is_true ~msg:"stft dtype"
          (Nx.dtype (chroma_stft stft chroma x) = Nx.float32) ;
        is_true ~msg:"cqt dtype" (Nx.dtype (chroma_cqt c x) = Nx.float32) ) ]

let suite =
  [ group "validation" validation_tests
  ; group "accessors" accessor_tests
  ; group "filterbank" filterbank_tests
  ; group "normalisation" normalisation_tests
  ; group "projection" projection_tests
  ; group "shapes" shape_tests
  ; group "flat" flat_tests ]
