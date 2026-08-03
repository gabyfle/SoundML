(* Contracts of the Mel configuration and projection: creation validation (every
   precondition raises its documented message), accessors and defaults,
   pp/equal, the fresh-copy semantics of [filterbank], the shape and broadcast
   behavior of [apply], and the flat features' own preconditions — the fft_size
   agreement naming both sizes, the [n_mfcc] range, the lifter domain, and the
   zero-lifter equivalence. *)

open Windtrap
open Soundml

let farray = array (float 0.)

let mel_config ?f_min ?f_max ?scale ?norm ?(n_mels = 8) ?(sample_rate = 8000)
    ?(fft_size = 64) () =
  Mel.Config.create ?f_min ?f_max ?scale ?norm ~n_mels ~sample_rate ~fft_size ()

(* {2 Creation validation} *)

let validation_tests =
  [ test "n_mels must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero bands"
          "create: cannot build 0 mel bands (n_mels must be at least 1)"
          (fun () -> ignore (mel_config ~n_mels:0 ()) ) )
  ; test "sample_rate must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero rate"
          "create: cannot use a sample rate of 0 Hz (sample_rate must be at \
           least 1)" (fun () -> ignore (mel_config ~sample_rate:0 ()) ) )
  ; test "fft_size must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero fft"
          "create: cannot use an FFT of size 0 (fft_size must be at least 1)"
          (fun () -> ignore (mel_config ~fft_size:0 ()) ) )
  ; test "f_min must be finite and non-negative" (fun () ->
        raises_invalid_arg ~msg:"negative f_min"
          "create: cannot start the filterbank at -1 Hz (f_min must be finite \
           and non-negative)" (fun () -> ignore (mel_config ~f_min:(-1.) ()) ) ;
        raises_invalid_arg ~msg:"nan f_min"
          "create: cannot start the filterbank at nan Hz (f_min must be finite \
           and non-negative)" (fun () ->
            ignore (mel_config ~f_min:Float.nan ()) ) )
  ; test "f_max must exceed f_min" (fun () ->
        raises_invalid_arg ~msg:"inverted bounds"
          "create: cannot span [2000, 1000] Hz (f_max must be finite and \
           greater than f_min)" (fun () ->
            ignore (mel_config ~f_min:2000. ~f_max:1000. ()) ) )
  ; test "f_max must not exceed Nyquist" (fun () ->
        raises_invalid_arg ~msg:"beyond Nyquist"
          "create: cannot extend the filterbank to 4001 Hz at a sample rate of \
           8000 Hz (f_max must not exceed the Nyquist frequency 4000)"
          (fun () -> ignore (mel_config ~f_max:4001. ()) ) )
  ; test "empty filters raise instead of warning" (fun () ->
        (* 64 bands over 17 bins: most filters span no bin — librosa emits a
           warning and returns zero rows, this configuration is rejected *)
        raises_invalid_arg ~msg:"unsupported resolution"
          "create: cannot support 64 mel bands with an FFT of size 32 (at \
           least one filter spans no FFT bin; raise fft_size or lower n_mels)"
          (fun () -> ignore (mel_config ~n_mels:64 ~fft_size:32 ()) ) ) ]

(* {2 Accessors, pp, equal} *)

let accessor_tests =
  [ test "accessors return the creation parameters" (fun () ->
        let c =
          mel_config ~f_min:100. ~f_max:3500. ~scale:`Htk ~norm:`None ~n_mels:6
            ~sample_rate:8000 ~fft_size:128 ()
        in
        equal ~msg:"n_mels" int 6 (Mel.Config.n_mels c) ;
        equal ~msg:"sample_rate" int 8000 (Mel.Config.sample_rate c) ;
        equal ~msg:"fft_size" int 128 (Mel.Config.fft_size c) ;
        equal ~msg:"bins" int 65 (Mel.Config.bins c) ;
        equal ~msg:"f_min" (float 0.) 100. (Mel.Config.f_min c) ;
        equal ~msg:"f_max" (float 0.) 3500. (Mel.Config.f_max c) ;
        is_true ~msg:"scale" (Mel.Config.scale c = `Htk) ;
        is_true ~msg:"norm" (Mel.Config.norm c = `None) )
  ; test "defaults are 0 Hz to Nyquist on the Slaney scale" (fun () ->
        let c = mel_config () in
        equal ~msg:"f_min" (float 0.) 0. (Mel.Config.f_min c) ;
        equal ~msg:"f_max is Nyquist" (float 0.) 4000. (Mel.Config.f_max c) ;
        is_true ~msg:"scale" (Mel.Config.scale c = `Slaney) ;
        is_true ~msg:"norm" (Mel.Config.norm c = `Slaney) )
  ; test "pp prints the compact single-line form" (fun () ->
        equal ~msg:"pp" string
          "mel(n_mels=8, sample_rate=8000, fft_size=64, f_min=0, f_max=4000, \
           scale=slaney, norm=slaney)"
          (Format.asprintf "%a" Mel.Config.pp (mel_config ())) )
  ; test "equal compares creation parameters" (fun () ->
        is_true ~msg:"same parameters"
          (Mel.Config.equal (mel_config ()) (mel_config ())) ;
        is_true ~msg:"different n_mels"
          (not (Mel.Config.equal (mel_config ()) (mel_config ~n_mels:6 ()))) ;
        is_true ~msg:"different scale"
          (not (Mel.Config.equal (mel_config ()) (mel_config ~scale:`Htk ()))) ;
        is_true ~msg:"different norm"
          (not (Mel.Config.equal (mel_config ()) (mel_config ~norm:`None ()))) ;
        is_true ~msg:"different f_max"
          (not (Mel.Config.equal (mel_config ()) (mel_config ~f_max:3000. ()))) )
  ]

(* {2 The filterbank accessor copies} *)

let copy_tests =
  [ test "filterbank returns a fresh tensor every call" (fun () ->
        let c = mel_config () in
        let before = Nx.to_array (Mel.filterbank Nx.float64 c) in
        let stolen = Mel.filterbank Nx.float64 c in
        Nx.set_item [0; 1] 1e9 stolen ;
        equal ~msg:"the config-owned matrix is unaffected" farray before
          (Nx.to_array (Mel.filterbank Nx.float64 c)) ;
        (* two calls never alias each other either *)
        let a = Mel.filterbank Nx.float32 c in
        let b = Mel.filterbank Nx.float32 c in
        let expected = Nx.to_array b in
        Nx.set_item [0; 1] 1e9 a ;
        equal ~msg:"sibling copies are independent" farray expected
          (Nx.to_array b) )
  ; test "a float32 filterbank is the float64 matrix rounded" (fun () ->
        let c = mel_config ~n_mels:6 ~fft_size:128 () in
        let expected =
          Nx.to_array (Nx.cast Nx.float32 (Mel.filterbank Nx.float64 c))
        in
        equal ~msg:"cast at the boundary" farray expected
          (Nx.to_array (Mel.filterbank Nx.float32 c)) ) ]

(* {2 apply: shapes, broadcast, preconditions} *)

let apply_tests =
  [ test "apply is the filterbank matmul" (fun () ->
        let c = mel_config () in
        let bins = Mel.Config.bins c in
        let s =
          Nx.init Nx.float64 [|bins; 7|] (fun i ->
              Float.sin (0.31 *. Float.of_int ((i.(0) * 7) + i.(1))) )
        in
        equal ~msg:"one batched matmul" farray
          (Nx.to_array (Nx.matmul (Mel.filterbank Nx.float64 c) s))
          (Nx.to_array (Mel.apply c s)) )
  ; test "leading axes broadcast" (fun () ->
        let c = mel_config () in
        let bins = Mel.Config.bins c in
        let batch =
          Nx.init Nx.float64 [|2; 3; bins; 5|] (fun i ->
              Float.of_int
                ((i.(0) * 1000) + (i.(1) * 100) + (i.(2) * 10) + i.(3))
              /. 97. )
        in
        let out = Mel.apply c batch in
        equal ~msg:"batched shape" (array int) [|2; 3; 8; 5|] (Nx.shape out) ;
        for i = 0 to 1 do
          for j = 0 to 2 do
            equal
              ~msg:(Printf.sprintf "slice %d,%d" i j)
              farray
              (Nx.to_array (Mel.apply c (Nx.slice [I i; I j; A; A] batch)))
              (Nx.to_array (Nx.slice [I i; I j; A; A] out))
          done
        done )
  ; test "empty chunks stay empty" (fun () ->
        let c = mel_config () in
        let bins = Mel.Config.bins c in
        equal ~msg:"zero frames" (array int) [|8; 0|]
          (Nx.shape (Mel.apply c (Nx.zeros Nx.float32 [|bins; 0|]))) ;
        equal ~msg:"zero-size leading axis" (array int) [|0; 8; 4|]
          (Nx.shape (Mel.apply c (Nx.zeros Nx.float64 [|0; bins; 4|]))) )
  ; test "rank and bins preconditions" (fun () ->
        let c = mel_config () in
        raises_invalid_arg ~msg:"rank one"
          "apply: cannot project a rank-1 tensor (the mel projection needs \
           [...; bins; frames])" (fun () ->
            ignore (Mel.apply c (Nx.zeros Nx.float64 [|33|])) ) ;
        raises_invalid_arg ~msg:"bins mismatch"
          "apply: cannot project 32 frequency bins through a filterbank built \
           for an FFT of size 64 (33 bins)" (fun () ->
            ignore (Mel.apply c (Nx.zeros Nx.float64 [|32; 4|])) ) ) ]

(* {2 The flat features} *)

let stft8 () = Stft.Config.create ~fft_size:8 ~hop:2 ()

let mel8 () = mel_config ~n_mels:3 ~sample_rate:1000 ~fft_size:8 ()

let signal n = Array.init n (fun i -> Float.sin (0.4 *. Float.of_int i))

let feature_tests =
  [ test "mel_spectrogram is power_spectrum then apply" (fun () ->
        let stft_config = stft8 () and mel_config = mel8 () in
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        equal ~msg:"composition" farray
          (Nx.to_array
             (Mel.apply mel_config (Stft.power_spectrum stft_config x)) )
          (Nx.to_array (mel_spectrogram stft_config mel_config x)) ;
        equal ~msg:"magnitude power" farray
          (Nx.to_array
             (Mel.apply mel_config
                (Stft.power_spectrum ~power:1. stft_config x) ) )
          (Nx.to_array (mel_spectrogram stft_config mel_config ~power:1. x)) )
  ; test "mel_spectrogram names both fft sizes on mismatch" (fun () ->
        raises_invalid_arg ~msg:"disagreeing configs"
          "mel_spectrogram: cannot project a 16-point STFT through a \
           filterbank built for an FFT of size 8 (the two configurations must \
           agree on fft_size)" (fun () ->
            ignore
              (mel_spectrogram
                 (Stft.Config.create ~fft_size:16 ())
                 (mel8 ())
                 (Nx.zeros Nx.float64 [|32|]) ) ) )
  ; test "mfcc validates n_mfcc against n_mels" (fun () ->
        raises_invalid_arg ~msg:"too many coefficients"
          "mfcc: cannot keep 4 cepstral coefficients of 3 mel bands (n_mfcc \
           must lie in [1, n_mels])" (fun () ->
            ignore
              (mfcc (stft8 ()) (mel8 ()) ~n_mfcc:4
                 (Nx.zeros Nx.float64 [|32|]) ) ) ;
        raises_invalid_arg ~msg:"zero coefficients"
          "mfcc: cannot keep 0 cepstral coefficients of 3 mel bands (n_mfcc \
           must lie in [1, n_mels])" (fun () ->
            ignore
              (mfcc (stft8 ()) (mel8 ()) ~n_mfcc:0
                 (Nx.zeros Nx.float64 [|32|]) ) ) )
  ; test "mfcc validates the lifter domain" (fun () ->
        raises_invalid_arg ~msg:"negative lifter"
          "mfcc: cannot lifter with a coefficient of -1 (lifter must be finite \
           and non-negative)" (fun () ->
            ignore
              (mfcc (stft8 ()) (mel8 ()) ~n_mfcc:3 ~lifter:(-1.)
                 (Nx.zeros Nx.float64 [|32|]) ) ) )
  ; test "a zero lifter is no liftering" (fun () ->
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        equal ~msg:"lifter 0 = absent" farray
          (Nx.to_array (mfcc (stft8 ()) (mel8 ()) ~n_mfcc:3 x))
          (Nx.to_array (mfcc (stft8 ()) (mel8 ()) ~n_mfcc:3 ~lifter:0. x)) )
  ; test "mfcc shapes follow n_mfcc and the frame grid" (fun () ->
        let stft_config = stft8 () and mel_config = mel8 () in
        let x =
          Nx.create Nx.float32 [|2; 41|]
            (Array.init 82 (fun i -> Float.of_int i /. 82.))
        in
        let frames = Stft.frames stft_config ~n:41 in
        equal ~msg:"batched shape" (array int) [|2; 2; frames|]
          (Nx.shape (mfcc stft_config mel_config ~n_mfcc:2 x)) ;
        (* the empty signal maps to the empty cepstrum on the same grid *)
        equal ~msg:"empty signal" (array int) [|3; 0|]
          (Nx.shape
             (mfcc stft_config mel_config ~n_mfcc:3
                (Nx.zeros Nx.float64 [|0|]) ) ) ) ]

let suite =
  [ group "config-validation" validation_tests
  ; group "config-accessors" accessor_tests
  ; group "filterbank-copies" copy_tests
  ; group "apply" apply_tests
  ; group "features" feature_tests ]
