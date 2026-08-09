(* Contracts of the flat spectral-shape features: every documented precondition
   raises its exact message, the [...; 1; frames] result shape with broadcasting
   leading axes, the empty-input zeros, the exact closed-form values on
   degenerate spectra (a single-peak frame pins the grid arithmetic; a flat
   frame pins flatness at one), the centroid-reuse and default-parameter
   equivalences, and the dtype boundary (a float32 result is the float64 result
   rounded once). *)

open Windtrap
open Soundml

let farray = array (float 0.)

(* A positive [bins; frames] test spectrogram; [sample_rate 1024] with five bins
   implies an eight-point FFT whose grid arithmetic is exact: bin [k] sits at [k
   * 128] Hz. *)
let sr = 1024

let bins = 5

let spectrum frames =
  Nx.init Nx.float64 [|bins; frames|] (fun i ->
      Float.exp (Float.sin (0.7 *. Float.of_int ((i.(0) * frames) + i.(1)))) )

(* A frame holding a single nonzero bin: every feature has a closed form. *)
let peak ~bin ~frames =
  Nx.init Nx.float64 [|bins; frames|] (fun i ->
      if i.(0) = bin then 0.7 else 0. )

(* {2 Preconditions} *)

let precondition_tests =
  [ test "rank must be at least two" (fun () ->
        raises_invalid_arg ~msg:"rank one"
          "spectral_centroid: cannot analyse a rank-1 tensor (a spectrogram is \
           [...; bins; frames])" (fun () ->
            ignore
              (spectral_centroid ~sample_rate:sr (Nx.zeros Nx.float64 [|8|])) ) )
  ; test "sample_rate must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero rate"
          "spectral_rolloff: cannot use a sample rate of 0 Hz (sample_rate \
           must be at least 1)" (fun () ->
            ignore (spectral_rolloff ~sample_rate:0 (spectrum 3)) ) )
  ; test "magnitudes must be non-negative" (fun () ->
        let negative = Nx.create Nx.float64 [|2; 2|] [|1.; 2.; -3.; 4.|] in
        raises_invalid_arg ~msg:"negative entry"
          "spectral_bandwidth: cannot analyse a spectrogram with negative or \
           NaN values (a magnitude spectrogram is non-negative)" (fun () ->
            ignore (spectral_bandwidth ~sample_rate:sr negative) ) ;
        let poisoned =
          Nx.create Nx.float64 [|2; 2|] [|1.; Float.nan; 3.; 4.|]
        in
        raises_invalid_arg ~msg:"NaN entry"
          "spectral_flatness: cannot analyse a spectrogram with negative or \
           NaN values (a magnitude spectrogram is non-negative)" (fun () ->
            ignore (spectral_flatness poisoned) ) )
  ; test "one bin implies no FFT grid" (fun () ->
        raises_invalid_arg ~msg:"single bin"
          "spectral_centroid: cannot derive bin frequencies for a 1-bin \
           spectrogram (the implied FFT size is 0; pass freqs explicitly)"
          (fun () ->
            ignore
              (spectral_centroid ~sample_rate:sr (Nx.zeros Nx.float64 [|1; 4|])) ) ;
        (* the same spectrogram is analysable under an explicit grid *)
        equal ~msg:"explicit freqs rescue" (array int) [|1; 4|]
          (Nx.shape
             (spectral_centroid
                ~freqs:(Nx.create Nx.float64 [|1|] [|440.|])
                ~sample_rate:sr
                (Nx.zeros Nx.float64 [|1; 4|]) ) ) )
  ; test "freqs must be rank-one with one frequency per bin" (fun () ->
        raises_invalid_arg ~msg:"rank-two freqs"
          "spectral_centroid: cannot use a rank-2 freqs tensor (freqs is \
           rank-one, one frequency per bin)" (fun () ->
            ignore
              (spectral_centroid
                 ~freqs:(Nx.zeros Nx.float64 [|bins; 1|])
                 ~sample_rate:sr (spectrum 3) ) ) ;
        raises_invalid_arg ~msg:"short freqs"
          "spectral_centroid: cannot pair 4 bin frequencies with 5 bins (freqs \
           holds one frequency per bin)" (fun () ->
            ignore
              (spectral_centroid
                 ~freqs:(Nx.zeros Nx.float64 [|4|])
                 ~sample_rate:sr (spectrum 3) ) ) )
  ; test "bandwidth validates the deviation power" (fun () ->
        raises_invalid_arg ~msg:"zero p"
          "spectral_bandwidth: cannot raise deviations to the power 0 (p must \
           be finite and positive)" (fun () ->
            ignore (spectral_bandwidth ~p:0. ~sample_rate:sr (spectrum 3)) ) )
  ; test "bandwidth validates a reused centroid" (fun () ->
        raises_invalid_arg ~msg:"rank-one centroid"
          "spectral_bandwidth: cannot reuse a rank-1 centroid (centroid must \
           be [...; 1; frames])" (fun () ->
            ignore
              (spectral_bandwidth
                 ~centroid:(Nx.zeros Nx.float64 [|3|])
                 ~sample_rate:sr (spectrum 3) ) ) ;
        raises_invalid_arg ~msg:"misshapen centroid"
          "spectral_bandwidth: cannot reuse a centroid with 2 rows over 7 \
           frames for a 3-frame spectrogram (centroid must be [...; 1; \
           frames], one frequency per frame)" (fun () ->
            ignore
              (spectral_bandwidth
                 ~centroid:(Nx.zeros Nx.float64 [|2; 7|])
                 ~sample_rate:sr (spectrum 3) ) ) )
  ; test "rolloff validates roll_percent" (fun () ->
        raises_invalid_arg ~msg:"one keeps everything"
          "spectral_rolloff: cannot keep 1 of the spectral energy \
           (roll_percent must lie strictly between 0 and 1)" (fun () ->
            ignore
              (spectral_rolloff ~roll_percent:1. ~sample_rate:sr (spectrum 3)) ) ;
        raises_invalid_arg ~msg:"zero keeps nothing"
          "spectral_rolloff: cannot keep 0 of the spectral energy \
           (roll_percent must lie strictly between 0 and 1)" (fun () ->
            ignore
              (spectral_rolloff ~roll_percent:0. ~sample_rate:sr (spectrum 3)) ) )
  ; test "flatness validates amin and power" (fun () ->
        raises_invalid_arg ~msg:"zero amin"
          "spectral_flatness: cannot floor the spectrum at 0 (amin must be \
           finite and positive)" (fun () ->
            ignore (spectral_flatness ~amin:0. (spectrum 3)) ) ;
        raises_invalid_arg ~msg:"infinite power"
          "spectral_flatness: cannot raise magnitudes to the power inf (power \
           must be finite and positive)" (fun () ->
            ignore (spectral_flatness ~power:Float.infinity (spectrum 3)) ) )
  ; test "stage construction validates chunk-independent parameters" (fun () ->
        raises_invalid_arg ~msg:"zero rate at build time"
          "spectral_centroid_stage: cannot use a sample rate of 0 Hz \
           (sample_rate must be at least 1)" (fun () ->
            ignore
              ( spectral_centroid_stage ~sample_rate:0 ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.offline )
                  Pipeline.t ) ) ;
        raises_invalid_arg ~msg:"rank-two freqs at build time"
          "spectral_centroid_stage: cannot use a rank-2 freqs tensor (freqs is \
           rank-one, one frequency per bin)" (fun () ->
            ignore
              ( spectral_centroid_stage
                  ~freqs:(Nx.zeros Nx.float64 [|bins; 1|])
                  ~sample_rate:sr ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.offline )
                  Pipeline.t ) ) ;
        raises_invalid_arg ~msg:"nan p at build time"
          "spectral_bandwidth_stage: cannot raise deviations to the power nan \
           (p must be finite and positive)" (fun () ->
            ignore
              ( spectral_bandwidth_stage ~p:Float.nan ~sample_rate:sr ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.offline )
                  Pipeline.t ) ) ;
        raises_invalid_arg ~msg:"roll_percent at build time"
          "spectral_rolloff_stage: cannot keep 2 of the spectral energy \
           (roll_percent must lie strictly between 0 and 1)" (fun () ->
            ignore
              ( spectral_rolloff_stage ~roll_percent:2. ~sample_rate:sr ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.offline )
                  Pipeline.t ) ) ;
        raises_invalid_arg ~msg:"zero amin at build time"
          "spectral_flatness_stage: cannot floor the spectrum at 0 (amin must \
           be finite and positive)" (fun () ->
            ignore
              ( spectral_flatness_stage ~amin:0. ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.offline )
                  Pipeline.t ) ) ) ]

(* {2 Shapes} *)

let shape_tests =
  [ test "each feature keeps the bin axis as one row" (fun () ->
        let s = spectrum 4 in
        List.iter
          (fun (name, feature) ->
            equal ~msg:name (array int) [|1; 4|] (Nx.shape (feature s)) )
          [ ("centroid", fun s -> spectral_centroid ~sample_rate:sr s)
          ; ("bandwidth", fun s -> spectral_bandwidth ~sample_rate:sr s)
          ; ("rolloff", fun s -> spectral_rolloff ~sample_rate:sr s)
          ; ("flatness", fun s -> spectral_flatness s) ] )
  ; test "leading axes broadcast" (fun () ->
        let batch =
          Nx.init Nx.float64 [|2; 3; bins; 4|] (fun i ->
              Float.exp
                (Float.sin
                   ( 0.31
                   *. Float.of_int
                        ((i.(0) * 60) + (i.(1) * 20) + (i.(2) * 4) + i.(3)) ) ) )
        in
        let out = spectral_centroid ~sample_rate:sr batch in
        equal ~msg:"batched shape" (array int) [|2; 3; 1; 4|] (Nx.shape out) ;
        for i = 0 to 1 do
          for j = 0 to 2 do
            equal
              ~msg:(Printf.sprintf "slice %d,%d" i j)
              farray
              (Nx.to_array
                 (spectral_centroid ~sample_rate:sr
                    (Nx.slice [I i; I j; A; A] batch) ) )
              (Nx.to_array (Nx.slice [I i; I j; A; A] out))
          done
        done )
  ; test "empty inputs map to zeros of the result shape" (fun () ->
        equal ~msg:"zero frames" (array int) [|1; 0|]
          (Nx.shape (spectral_flatness (Nx.zeros Nx.float64 [|bins; 0|]))) ;
        equal ~msg:"zero-size leading axis" (array int) [|0; 1; 4|]
          (Nx.shape
             (spectral_rolloff ~sample_rate:sr
                (Nx.zeros Nx.float32 [|0; bins; 4|]) ) ) ) ]

(* {2 Semantics on closed forms} *)

let semantic_tests =
  [ test "a single-peak frame reads its own bin frequency" (fun () ->
        let s = peak ~bin:3 ~frames:2 in
        (* bin 3 of the exact grid sits at 3 * 1024 / 8 = 384 Hz *)
        equal ~msg:"centroid at the peak" farray [|384.; 384.|]
          (Nx.to_array (spectral_centroid ~sample_rate:sr s)) ;
        equal ~msg:"bandwidth zero at the peak" farray [|0.; 0.|]
          (Nx.to_array (spectral_bandwidth ~sample_rate:sr s)) ;
        equal ~msg:"rolloff at the peak for low roll_percent" farray
          [|384.; 384.|]
          (Nx.to_array (spectral_rolloff ~roll_percent:0.1 ~sample_rate:sr s)) ;
        equal ~msg:"rolloff at the peak for high roll_percent" farray
          [|384.; 384.|]
          (Nx.to_array (spectral_rolloff ~roll_percent:0.85 ~sample_rate:sr s)) )
  ; test "an all-zero frame has centroid zero and rolls off at bin zero"
      (fun () ->
        let s = Nx.zeros Nx.float64 [|bins; 3|] in
        equal ~msg:"unnormalised silent centroid" farray [|0.; 0.; 0.|]
          (Nx.to_array (spectral_centroid ~sample_rate:sr s)) ;
        equal ~msg:"silent rolloff" farray [|0.; 0.; 0.|]
          (Nx.to_array (spectral_rolloff ~sample_rate:sr s)) )
  ; test "a flat frame has flatness one" (fun () ->
        let s = Nx.full Nx.float64 [|bins; 3|] 0.5 in
        Tutils.check_close ~msg:"flatness of a constant spectrum"
          ~expected:[|1.; 1.; 1.|] (spectral_flatness s) )
  ; test "a reused double-precision centroid changes nothing" (fun () ->
        let s = spectrum 6 in
        let c = spectral_centroid ~sample_rate:sr s in
        equal ~msg:"bandwidth with and without reuse" farray
          (Nx.to_array (spectral_bandwidth ~sample_rate:sr s))
          (Nx.to_array (spectral_bandwidth ~centroid:c ~sample_rate:sr s)) )
  ; test "defaults are pinned" (fun () ->
        let s = spectrum 5 in
        equal ~msg:"bandwidth p defaults to 2" farray
          (Nx.to_array (spectral_bandwidth ~p:2. ~sample_rate:sr s))
          (Nx.to_array (spectral_bandwidth ~sample_rate:sr s)) ;
        equal ~msg:"rolloff roll_percent defaults to 0.85" farray
          (Nx.to_array (spectral_rolloff ~roll_percent:0.85 ~sample_rate:sr s))
          (Nx.to_array (spectral_rolloff ~sample_rate:sr s)) ;
        equal ~msg:"flatness amin and power default to 1e-10 and 2" farray
          (Nx.to_array (spectral_flatness ~amin:1e-10 ~power:2. s))
          (Nx.to_array (spectral_flatness s)) )
  ; test "explicit freqs equal to the derived grid change nothing" (fun () ->
        let s = spectrum 4 in
        let step =
          1.0 /. (Float.of_int (2 * (bins - 1)) *. (1.0 /. Float.of_int sr))
        in
        let grid =
          Nx.create Nx.float64 [|bins|]
            (Array.init bins (fun k -> Float.of_int k *. step))
        in
        equal ~msg:"same grid, same centroid" farray
          (Nx.to_array (spectral_centroid ~sample_rate:sr s))
          (Nx.to_array (spectral_centroid ~freqs:grid ~sample_rate:sr s)) )
  ; test "a float32 result is the float64 result rounded once" (fun () ->
        let values =
          Array.init (bins * 6) (fun i ->
              Float.abs (Float.sin (0.9 *. Float.of_int i)) )
        in
        let s32 = Nx.create Nx.float32 [|bins; 6|] values in
        (* the float64 twin holds the float32-quantized values exactly *)
        let s64 = Nx.cast Nx.float64 s32 in
        List.iter
          (fun (name, f32, f64) ->
            equal ~msg:name farray
              (Nx.to_array (Nx.cast Nx.float32 (f64 s64)))
              (Nx.to_array (f32 s32)) )
          [ ( "centroid"
            , (fun s -> spectral_centroid ~sample_rate:sr s)
            , fun s -> spectral_centroid ~sample_rate:sr s )
          ; ( "bandwidth"
            , (fun s -> spectral_bandwidth ~sample_rate:sr s)
            , fun s -> spectral_bandwidth ~sample_rate:sr s )
          ; ( "rolloff"
            , (fun s -> spectral_rolloff ~sample_rate:sr s)
            , fun s -> spectral_rolloff ~sample_rate:sr s )
          ; ( "flatness"
            , (fun s -> spectral_flatness s)
            , fun s -> spectral_flatness s ) ] ) ]

let suite =
  [ group "preconditions" precondition_tests
  ; group "shapes" shape_tests
  ; group "semantics" semantic_tests ]
