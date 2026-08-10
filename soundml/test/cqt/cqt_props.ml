(* Contracts of the constant-Q configuration and transform: creation validation
   (every precondition raises its documented message), accessors and defaults,
   pp/equal, the frame grid against the transform's own shape, the selectivity
   of the filter bank on pure tones, the exactness of the magnitude convenience,
   batch broadcasting and the degenerate inputs. *)

open Windtrap
open Soundml

let sample_rate = 22050

let c1 = 32.70319566257483

let config ?fmin ?bins_per_octave ?gamma ?tuning ?filter_scale ?norm ?window
    ?scale ?hop ?pad ?(n_bins = 84) ?(sample_rate = sample_rate) () =
  Cqt.Config.create ?fmin ?bins_per_octave ?gamma ?tuning ?filter_scale ?norm
    ?window ?scale ?hop ?pad ~n_bins ~sample_rate ()

(* {2 Creation validation} *)

let validation_tests =
  [ test "n_bins must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero bins"
          "create: cannot use n_bins = 0 (n_bins must be at least 1)" (fun () ->
            ignore (config ~n_bins:0 ()) ) )
  ; test "bins_per_octave must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero resolution"
          "create: cannot use bins_per_octave = 0 (bins_per_octave must be at \
           least 1)" (fun () -> ignore (config ~bins_per_octave:0 ()) ) )
  ; test "hop must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero hop"
          "create: cannot use hop = 0 (hop must be at least 1)" (fun () ->
            ignore (config ~hop:0 ()) ) )
  ; test "sample_rate must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero rate"
          "create: cannot use sample_rate = 0 (sample_rate must be at least 1)"
          (fun () -> ignore (config ~sample_rate:0 ()) ) )
  ; test "fmin must be finite and positive" (fun () ->
        raises_invalid_arg ~msg:"zero fmin"
          "create: cannot use fmin = 0 (fmin must be finite and positive)"
          (fun () -> ignore (config ~fmin:0. ()) ) ;
        raises_invalid_arg ~msg:"infinite fmin"
          "create: cannot use fmin = inf (fmin must be finite and positive)"
          (fun () -> ignore (config ~fmin:Float.infinity ()) ) )
  ; test "tuning must be finite" (fun () ->
        raises_invalid_arg ~msg:"nan tuning"
          "create: cannot use tuning = nan (tuning must be finite)" (fun () ->
            ignore (config ~tuning:Float.nan ()) ) )
  ; test "filter_scale must be finite and positive" (fun () ->
        raises_invalid_arg ~msg:"zero scale"
          "create: cannot use filter_scale = 0 (filter_scale must be finite \
           and positive)" (fun () -> ignore (config ~filter_scale:0. ()) ) )
  ; test "norm must be finite and positive" (fun () ->
        raises_invalid_arg ~msg:"negative norm"
          "create: cannot use norm = -1 (norm must be finite and positive)"
          (fun () -> ignore (config ~norm:(-1.) ()) ) )
  ; test "a fixed gamma must be finite and non-negative" (fun () ->
        raises_invalid_arg ~msg:"negative gamma"
          "create: cannot use gamma = -3 (gamma must be finite and \
           non-negative)" (fun () -> ignore (config ~gamma:(`Fixed (-3.)) ()) ) )
  ; test "the window shape parameter is validated under this entry point"
      (fun () ->
        raises_invalid_arg ~msg:"negative beta"
          "create: cannot use a kaiser window with beta -1 (beta must be \
           finite and non-negative)" (fun () ->
            ignore (config ~window:(Window.Kaiser (-1.)) ()) ) )
  ; test "filters may not reach past Nyquist" (fun () ->
        (* 101 bins from C1 at 22050 Hz reach 11004.6 Hz, just under the 11025
           Hz Nyquist; 102 bins reach 11659 Hz. *)
        ignore (config ~n_bins:101 ()) ;
        raises_invalid_arg ~msg:"past Nyquist"
          "create: cannot place 102 bins from 32.7032 Hz at a sample rate of \
           22050 Hz (bin 101, centred at 11175.3 Hz, reaches 11659 Hz, above \
           the Nyquist frequency 11025)" (fun () ->
            ignore (config ~n_bins:102 ()) ) ) ]

(* {2 Accessors, pp, equal} *)

let accessor_tests =
  [ test "accessors return the creation parameters" (fun () ->
        let c =
          config ~fmin:55. ~bins_per_octave:24 ~gamma:(`Fixed 12.) ~tuning:0.25
            ~filter_scale:0.75 ~norm:2. ~window:Window.Blackman ~scale:false
            ~hop:256 ~pad:`Reflect ~n_bins:48 ~sample_rate:44100 ()
        in
        equal ~msg:"n_bins" int 48 (Cqt.Config.n_bins c) ;
        equal ~msg:"bins_per_octave" int 24 (Cqt.Config.bins_per_octave c) ;
        equal ~msg:"sample_rate" int 44100 (Cqt.Config.sample_rate c) ;
        equal ~msg:"hop" int 256 (Cqt.Config.hop c) ;
        equal ~msg:"fmin" (float 0.) 55. (Cqt.Config.fmin c) ;
        equal ~msg:"tuning" (float 0.) 0.25 (Cqt.Config.tuning c) ;
        equal ~msg:"filter_scale" (float 0.) 0.75 (Cqt.Config.filter_scale c) ;
        equal ~msg:"norm" (float 0.) 2. (Cqt.Config.norm c) ;
        is_true ~msg:"gamma" (Cqt.Config.gamma c = `Fixed 12.) ;
        is_true ~msg:"window"
          (Window.equal (Cqt.Config.window c) Window.Blackman) ;
        is_true ~msg:"scale" (not (Cqt.Config.scale c)) ;
        is_true ~msg:"pad" (Cqt.Config.pad c = `Reflect) ;
        equal ~msg:"n_octaves" int 2 (Cqt.Config.n_octaves c) )
  ; test "defaults are C1, twelve bins per octave, constant Q" (fun () ->
        let c = config () in
        equal ~msg:"fmin" (float 0.) c1 (Cqt.Config.fmin c) ;
        equal ~msg:"bins_per_octave" int 12 (Cqt.Config.bins_per_octave c) ;
        equal ~msg:"hop" int 512 (Cqt.Config.hop c) ;
        equal ~msg:"tuning" (float 0.) 0. (Cqt.Config.tuning c) ;
        equal ~msg:"filter_scale" (float 0.) 1. (Cqt.Config.filter_scale c) ;
        equal ~msg:"norm" (float 0.) 1. (Cqt.Config.norm c) ;
        is_true ~msg:"gamma" (Cqt.Config.gamma c = `Constant_q) ;
        is_true ~msg:"window" (Window.equal (Cqt.Config.window c) Window.Hann) ;
        is_true ~msg:"scale" (Cqt.Config.scale c) ;
        is_true ~msg:"pad zeroes" (Cqt.Config.pad c = `Constant 0.) ;
        equal ~msg:"n_octaves" int 7 (Cqt.Config.n_octaves c) )
  ; test "a partial bottom octave still counts as an octave" (fun () ->
        equal ~msg:"n_octaves" int 8
          (Cqt.Config.n_octaves (config ~n_bins:90 ())) ;
        equal ~msg:"single bin" int 1
          (Cqt.Config.n_octaves (config ~n_bins:1 ())) )
  ; test "pp prints the compact single-line form" (fun () ->
        equal ~msg:"pp" string
          "cqt(n_bins=12, bins_per_octave=12, sample_rate=22050, fmin=1500, \
           hop=512, gamma=constant_q, tuning=0, filter_scale=1, norm=1, \
           window=hann, scale=true, pad=constant(0))"
          (Format.asprintf "%a" Cqt.Config.pp
             (config ~n_bins:12 ~fmin:1500. ()) ) ;
        equal ~msg:"pp/erb" string
          "cqt(n_bins=12, bins_per_octave=12, sample_rate=22050, fmin=1500, \
           hop=512, gamma=erb, tuning=0, filter_scale=1, norm=1, window=hann, \
           scale=true, pad=constant(0))"
          (Format.asprintf "%a" Cqt.Config.pp
             (config ~n_bins:12 ~fmin:1500. ~gamma:`Erb ()) ) )
  ; test "equal compares creation parameters" (fun () ->
        is_true ~msg:"same parameters"
          (Cqt.Config.equal (config ()) (config ())) ;
        is_true ~msg:"different fmin"
          (not (Cqt.Config.equal (config ()) (config ~fmin:55. ()))) ;
        is_true ~msg:"different gamma"
          (not (Cqt.Config.equal (config ()) (config ~gamma:`Erb ()))) ;
        is_true ~msg:"different fixed gamma"
          (not
             (Cqt.Config.equal
                (config ~gamma:(`Fixed 1.) ())
                (config ~gamma:(`Fixed 2.) ()) ) ) ;
        is_true ~msg:"different hop"
          (not (Cqt.Config.equal (config ()) (config ~hop:256 ()))) ;
        is_true ~msg:"different pad"
          (not (Cqt.Config.equal (config ()) (config ~pad:`Edge ()))) ) ]

(* {2 The frame grid} *)

let signal n = Array.init n (fun i -> Float.sin (0.37 *. Float.of_int i) *. 0.5)

let tensor n = Nx.create Nx.float64 [|n|] (signal n)

let frame_tests =
  [ test "frames matches the transform's own shape" (fun () ->
        List.iter
          (fun c ->
            List.iter
              (fun n ->
                let z = Cqt.transform Nx.complex128 c (tensor n) in
                equal ~msg:(Printf.sprintf "n=%d" n) int (Cqt.frames c ~n)
                  (Nx.dim 1 z) ;
                equal
                  ~msg:(Printf.sprintf "bins n=%d" n)
                  int (Cqt.Config.n_bins c) (Nx.dim 0 z) )
              [1; 7; 63; 64; 255; 512; 513; 1024; 4096] )
          [ config ~n_bins:24 ()
          ; config ~n_bins:24 ~hop:255 ()
          ; config ~n_bins:12 ~fmin:1500. ~hop:64 ()
          ; config ~n_bins:36 ~hop:128 () ] )
  ; test "a zero-length signal produces no frames" (fun () ->
        let c = config ~n_bins:24 () in
        equal ~msg:"frames" int 0 (Cqt.frames c ~n:0) ;
        let z = Cqt.transform Nx.complex128 c (Nx.zeros Nx.float64 [|0|]) in
        equal ~msg:"shape" (array int) [|24; 0|] (Nx.shape z) )
  ; test "frames rejects a negative length" (fun () ->
        raises_invalid_arg ~msg:"negative"
          "frames: cannot analyse a signal of length -1 (length must be \
           non-negative)" (fun () -> ignore (Cqt.frames (config ()) ~n:(-1)) ) )
  ]

(* {2 Selectivity}

   A pure tone at a bin's centre frequency must peak in that bin. The check
   walks several octaves of the ladder, including the ones the recursion reaches
   only after decimating. *)

let tone_tests =
  let tone_case name c dtype =
    test name (fun () ->
        let freqs = Nx.to_array (Cqt.frequencies Nx.float64 c) in
        let n = 4 * sample_rate / 10 in
        List.iter
          (fun k ->
            let f = freqs.(k) in
            let samples =
              Array.init n (fun i ->
                  Float.sin
                    ( 2. *. Float.pi *. f *. Float.of_int i
                    /. Float.of_int sample_rate ) )
            in
            let x = Nx.cast dtype (Nx.create Nx.float64 [|n|] samples) in
            let power = Cqt.power_spectrum c x in
            (* Read the middle frame, clear of both boundaries. *)
            let frames = Nx.dim 1 power in
            let column =
              Nx.to_array
                (Nx.shrink
                   [|(0, Nx.dim 0 power); (frames / 2, (frames / 2) + 1)|]
                   power )
            in
            let best = ref 0 in
            Array.iteri (fun i v -> if v > column.(!best) then best := i) column ;
            if !best <> k then
              failf "%s: a tone at bin %d (%.3f Hz) peaked at bin %d" name k f
                !best )
          [2; 15; 28; 41; 54; 67; 80] )
  in
  [ tone_case "pure tones peak in their own bin (float64)" (config ()) Nx.float64
  ; tone_case "pure tones peak in their own bin (float32)" (config ())
      Nx.float32
  ; tone_case "pure tones peak in their own bin (odd hop)" (config ~hop:511 ())
      Nx.float64 ]

(* {2 The magnitude convenience} *)

let magnitude_tests =
  [ test "power_spectrum ~power:1 is the magnitude of the transform" (fun () ->
        let c = config ~n_bins:36 () in
        let x = tensor 4096 in
        let expected =
          Nx.magnitude Nx.float64 (Cqt.transform Nx.complex128 c x)
        in
        let got = Cqt.power_spectrum ~power:1. c x in
        equal ~msg:"shape" (array int) (Nx.shape expected) (Nx.shape got) ;
        let e = Nx.to_array expected and a = Nx.to_array got in
        Array.iteri
          (fun i v ->
            if not (Float.equal v a.(i)) then
              failf "magnitude %d: %.17g vs %.17g" i a.(i) v )
          e )
  ; test "power_spectrum squares by default" (fun () ->
        let c = config ~n_bins:24 () in
        let x = tensor 2048 in
        let magnitude = Cqt.power_spectrum ~power:1. c x in
        Tutils.check_close ~rtol:1e-12 ~atol:1e-15 ~msg:"power"
          ~expected:(Nx.to_array (Nx.square magnitude))
          (Cqt.power_spectrum c x) )
  ; test "the float32 convenience never leaves float32" (fun () ->
        let c = config ~n_bins:24 () in
        let x = Nx.cast Nx.float32 (tensor 2048) in
        let power = Cqt.power_spectrum c x in
        is_true ~msg:"dtype" (Nx.dtype power = Nx.float32) ) ]

(* {2 Batching} *)

let batch_tests =
  [ test "leading axes broadcast to per-signal calls" (fun () ->
        let c = config ~n_bins:36 ~hop:256 () in
        let n = 3000 in
        let first = signal n in
        let second = Array.init n (fun i -> first.(n - 1 - i) *. 0.5) in
        let batch = Nx.create Nx.float64 [|2; n|] (Array.append first second) in
        let stacked = Cqt.transform Nx.complex128 c batch in
        equal ~msg:"shape" (array int)
          [|2; 36; Cqt.frames c ~n|]
          (Nx.shape stacked) ;
        List.iteri
          (fun row samples ->
            let alone =
              Cqt.transform Nx.complex128 c (Nx.create Nx.float64 [|n|] samples)
            in
            let slice =
              Nx.reshape (Nx.shape alone)
                (Nx.shrink
                   [| (row, row + 1)
                    ; (0, Nx.dim 1 stacked)
                    ; (0, Nx.dim 2 stacked) |]
                   stacked )
            in
            let e = Nx.to_array alone and a = Nx.to_array slice in
            Array.iteri
              (fun i (v : Complex.t) ->
                let w = a.(i) in
                if not (Float.equal v.re w.re && Float.equal v.im w.im) then
                  failf "row %d value %d: (%.17g, %.17g) vs (%.17g, %.17g)" row
                    i w.re w.im v.re v.im )
              e )
          [first; second] )
  ; test "a zero-size leading axis yields an empty transform" (fun () ->
        let c = config ~n_bins:24 () in
        let x = Nx.zeros Nx.float64 [|0; 2048|] in
        equal ~msg:"complex" (array int)
          [|0; 24; Cqt.frames c ~n:2048|]
          (Nx.shape (Cqt.transform Nx.complex128 c x)) ;
        equal ~msg:"power" (array int)
          [|0; 24; Cqt.frames c ~n:2048|]
          (Nx.shape (Cqt.power_spectrum c x)) )
  ; test "a rank-zero tensor is rejected" (fun () ->
        let c = config () in
        raises_invalid_arg ~msg:"transform"
          "transform: cannot analyse a rank-zero tensor (the time axis must \
           exist)" (fun () ->
            ignore (Cqt.transform Nx.complex128 c (Nx.scalar Nx.float64 1.)) ) ;
        raises_invalid_arg ~msg:"power_spectrum"
          "power_spectrum: cannot analyse a rank-zero tensor (the time axis \
           must exist)" (fun () ->
            ignore (Cqt.power_spectrum c (Nx.scalar Nx.float64 1.)) ) ) ]

(* {2 The frequency ladder} *)

let ladder_tests =
  [ test "octave transposition is exact" (fun () ->
        let c = config ~n_bins:84 () in
        let f = Nx.to_array (Cqt.frequencies Nx.float64 c) in
        for k = 0 to 71 do
          if not (Float.equal f.(k + 12) (2. *. f.(k))) then
            failf "bin %d: %.17g is not twice %.17g" (k + 12) f.(k + 12) f.(k)
        done )
  ; test "tuning shifts the whole ladder" (fun () ->
        let plain = config ~n_bins:24 () in
        let tuned = config ~n_bins:24 ~tuning:0.5 () in
        let a = Nx.to_array (Cqt.frequencies Nx.float64 plain)
        and b = Nx.to_array (Cqt.frequencies Nx.float64 tuned) in
        let ratio = Float.pow 2. (0.5 /. 12.) in
        Array.iteri
          (fun k v ->
            let expected = a.(k) *. ratio in
            if Float.abs (b.(k) -. expected) > 1e-9 *. expected then
              failf "bin %d: %.17g, expected %.17g" k b.(k) expected )
          b )
  ; test "filter lengths shorten as the ladder rises" (fun () ->
        let c = config ~n_bins:84 () in
        let l = Nx.to_array (Cqt.filter_lengths Nx.float64 c) in
        for k = 1 to 83 do
          if not (l.(k) < l.(k - 1)) then
            failf "bin %d: length %.6g is not below %.6g" k l.(k) l.(k - 1)
        done ;
        (* Scaling the support scales every length by the same factor. *)
        let half = config ~n_bins:84 ~filter_scale:0.5 () in
        let h = Nx.to_array (Cqt.filter_lengths Nx.float64 half) in
        Array.iteri
          (fun k v ->
            if Float.abs ((2. *. v) -. l.(k)) > 1e-9 *. l.(k) then
              failf "bin %d: half-scale length %.6g" k v )
          h ) ]

let suite =
  [ group "validation" validation_tests
  ; group "accessors" accessor_tests
  ; group "frames" frame_tests
  ; group "selectivity" tone_tests
  ; group "magnitude" magnitude_tests
  ; group "batching" batch_tests
  ; group "ladder" ladder_tests ]
