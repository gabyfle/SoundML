(* Contracts of the phase vocoder that the golden vectors do not carry: the
   output lengths and their tie rounding, the identity settings and what they
   actually return, the degenerate inputs, batch broadcasting, the ratios
   {!Effects.semitones} names, the validation messages, and the size of the
   resampler substitution the pitch vectors are compared at. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

let sample_rate = Pvoc_signals.sample_rate

let config ?(fft_size = 256) ?(hop = 64) () =
  Stft.Config.create ~fft_size ~hop ~pad:(`Constant 0.) ()

let peak t = Nx.item [] (Nx.max (Nx.abs t))

let deviation a b = peak (Nx.sub a b)

(* {2 Lengths} *)

(* [round (n / rate)] with ties to even, which the four rates below all hit: 5 /
   2 and 1 / 2 round down, 3 / 2 and 7 / 2 round up. *)
let length_tests =
  [ test "the output length is the input length over the rate" (fun () ->
        let c = config () in
        List.iter
          (fun (n, rate, expected) ->
            let y = Effects.time_stretch c ~rate (Nx.zeros Nx.float64 [|n|]) in
            equal
              ~msg:(Printf.sprintf "n = %d at rate %g" n rate)
              int expected (Nx.dim 0 y) )
          [ (1000, 1.37, 730)
          ; (1000, 0.5, 2000)
          ; (1000, 2.0, 500)
          ; (1000, 1.0, 1000)
          ; (997, 1.37, 728) ] )
  ; test "length ties round to even" (fun () ->
        let c = config () in
        List.iter
          (fun (n, expected) ->
            let y =
              Effects.time_stretch c ~rate:2.0 (Nx.zeros Nx.float64 [|n|])
            in
            equal
              ~msg:
                (Printf.sprintf "n = %d at rate 2 (n / 2 = %g)" n
                   (Float.of_int n /. 2.) )
              int expected (Nx.dim 0 y) )
          [(1, 0); (3, 2); (5, 2); (7, 4); (9, 4); (11, 6)] )
  ; test "an empty signal stretches to an empty signal" (fun () ->
        let c = config () in
        List.iter
          (fun rate ->
            equal ~msg:"no samples in, no samples out" int 0
              (Nx.dim 0
                 (Effects.time_stretch c ~rate (Nx.zeros Nx.float64 [|0|])) ) )
          [0.5; 1.0; 1.37; 2.0] )
  ; test "a spectrum with no frames vocodes to a spectrum with no frames"
      (fun () ->
        let c = config () in
        let y =
          Effects.phase_vocoder c ~rate:1.37
            (Nx.zeros Nx.complex128 [|Stft.Config.bins c; 0|])
        in
        equal ~msg:"shape" (array int) [|Stft.Config.bins c; 0|] (Nx.shape y) )
  ; test "the vocoded frame count is the analysis count over the rate"
      (fun () ->
        let c = config () in
        let x = Pvoc_signals.sweep (sample_rate / 4) in
        let spectrum = Stft.transform Nx.complex128 c x in
        let frames = Nx.dim 1 spectrum in
        List.iter
          (fun rate ->
            let y = Effects.phase_vocoder c ~rate spectrum in
            equal
              ~msg:(Printf.sprintf "rate %g over %d frames" rate frames)
              int
              (Float.to_int (Float.ceil (Float.of_int frames /. rate)))
              (Nx.dim 1 y) )
          [0.5; 0.75; 1.0; 1.25; 1.37; 2.0] )
  ; test "pitch shifting keeps the input length" (fun () ->
        let c = config () in
        let x = Pvoc_signals.sweep 4000 in
        List.iter
          (fun (num, den) ->
            let y = Effects.pitch_shift c ~ratio:{Pipeline.Rate.num; den} x in
            equal
              ~msg:(Printf.sprintf "ratio %d/%d" num den)
              int 4000 (Nx.dim 0 y) )
          [(2, 1); (1, 2); (3524, 2797); (1, 1)] ) ]

(* {2 The identity settings} *)

(* Neither [rate = 1.] nor the unit ratio is a shortcut: the analysis, the
   recurrence and the synthesis all run, so the residual is the round trip's,
   amplified by however many frames the recurrence carries. It grows as the
   transform shrinks, because a shorter transform means more frames over the
   same signal. *)
let identity_tests =
  [ test "rate 1 is the identity to within the round trip" (fun () ->
        let x = Pvoc_signals.sweep sample_rate in
        List.iter
          (fun (fft_size, hop, bound) ->
            let c = config ~fft_size ~hop () in
            let y = Effects.time_stretch c ~rate:1.0 x in
            equal ~msg:"length" int (Nx.dim 0 x) (Nx.dim 0 y) ;
            let residual = deviation y x /. peak x in
            is_true
              ~msg:
                (Printf.sprintf "fft %d hop %d: %.3e of peak" fft_size hop
                   residual )
              (residual <= bound) )
          [ (2048, 512, 1e-11)
          ; (1024, 256, 1e-11)
          ; (256, 64, 1e-11)
          ; (64, 16, 1e-10) ] )
  ; test "rate 1 at float32 is the identity to the storage rounding" (fun () ->
        let x = Nx.cast Nx.float32 (Pvoc_signals.sweep sample_rate) in
        let c = config ~fft_size:2048 ~hop:512 () in
        let y = Effects.time_stretch c ~rate:1.0 x in
        is_true
          ~msg:(Printf.sprintf "%.3e of peak" (deviation y x /. peak x))
          (deviation y x /. peak x <= 1e-6) )
  ; test "the unit ratio passes through the resampler untouched" (fun () ->
        let x = Pvoc_signals.sweep sample_rate in
        let c = config ~fft_size:2048 ~hop:512 () in
        let shifted = Effects.pitch_shift c ~ratio:Pipeline.Rate.identity x in
        let stretched = Effects.time_stretch c ~rate:1.0 x in
        equal ~msg:"the identity conversion changes nothing" (float 0.) 0.
          (deviation shifted stretched) ) ]

(* {2 Batching} *)

let batch_tests =
  [ test "a batch is the rows computed one by one" (fun () ->
        let c = config () in
        let rows = [Pvoc_signals.sweep 4000; Pvoc_signals.buzz 4000] in
        let batch = Nx.stack ~axis:0 rows in
        let stretched = Effects.time_stretch c ~rate:1.37 batch in
        List.iteri
          (fun i row ->
            let alone = Effects.time_stretch c ~rate:1.37 row in
            equal
              ~msg:(Printf.sprintf "row %d" i)
              (float 0.) 0.
              (deviation
                 ( Nx.shrink [|(i, i + 1); (0, Nx.dim 1 stretched)|] stretched
                 |> Nx.reshape [|Nx.dim 1 stretched|] )
                 alone ) )
          rows ) ]

(* {2 Equal temperament} *)

let semitone_tests =
  [ test "whole octaves are exact" (fun () ->
        List.iter
          (fun (steps, num, den) ->
            let r = Effects.semitones (Float.of_int steps) in
            equal
              ~msg:(Printf.sprintf "%d steps" steps)
              (pair int int) (num, den) (r.num, r.den) )
          [(12, 2, 1); (-12, 1, 2); (24, 4, 1); (-24, 1, 4); (0, 1, 1)] )
  ; test "twelve-tone equal temperament lands within 0.027 cents" (fun () ->
        for steps = -12 to 12 do
          let r = Effects.semitones (Float.of_int steps) in
          let target = Float.pow 2. (Float.of_int steps /. 12.) in
          let cents =
            Float.abs
              ( 1200.
              *. Float.log2 (Float.of_int r.num /. Float.of_int r.den /. target)
              )
          in
          is_true
            ~msg:
              (Printf.sprintf "%d steps: %d/%d, %.3e cents" steps r.num r.den
                 cents )
            (cents <= 0.027)
        done )
  ; test "fractional steps and other divisions of the octave" (fun () ->
        List.iter
          (fun (steps, bins_per_octave) ->
            let r = Effects.semitones ~bins_per_octave steps in
            let target = Float.pow 2. (steps /. Float.of_int bins_per_octave) in
            let cents =
              Float.abs
                ( 1200.
                *. Float.log2
                     (Float.of_int r.num /. Float.of_int r.den /. target) )
            in
            is_true
              ~msg:
                (Printf.sprintf "%g steps of %d: %d/%d, %.3e cents" steps
                   bins_per_octave r.num r.den cents )
              (cents <= 0.1 && r.num <= 512 && r.den <= 512) )
          [(0.5, 12); (3., 24); (-7.5, 24); (1., 53); (19., 31)] )
  ; test "the ratio drives the shift" (fun () ->
        (* A tone shifted up a whole octave lands an octave up: the strongest
           component of the result sits at twice the input frequency. *)
        let c = config ~fft_size:2048 ~hop:512 () in
        let x = Pvoc_signals.tone ~f0:440. (2 * sample_rate) in
        let y = Effects.pitch_shift c ~ratio:(Effects.semitones 12.) x in
        let n = Nx.dim 0 y in
        let trimmed = Nx.shrink [|(4096, n - 4096)|] y in
        let magnitudes =
          Nx.magnitude Nx.float64 (Nx.rfft Nx.complex128 trimmed)
        in
        let bins = Nx.dim 0 magnitudes in
        let best = ref 1 in
        for i = 1 to bins - 2 do
          if Nx.item [i] magnitudes > Nx.item [!best] magnitudes then best := i
        done ;
        let frequency =
          Float.of_int !best *. Float.of_int sample_rate
          /. Float.of_int (Nx.dim 0 trimmed)
        in
        is_true
          ~msg:(Printf.sprintf "strongest component at %.2f Hz" frequency)
          (Float.abs (frequency -. 880.) <= 10.) ) ]

(* {2 The resampler substitution} *)

(* The pitch vectors are librosa's whole composition, soxr included, so the
   difference between them and this library's is the resampler substitution and
   nothing else — the stretch stage of the same cells is pinned at 1e-11 by its
   own file. This test names the bound as one number over every case: the same
   4e-2 of peak {!Pvoc_goldens} compares at, which the cases clear at 2.7e-2. *)
let substitution_bound = 4e-2

let substitution_tests =
  [ test "the resampler substitution stays inside its stated bound" (fun () ->
        let worst = ref 0. in
        List.iter
          (fun (file : Tutils.Golden.file) ->
            List.iter
              (fun (case : Tutils.Golden.case) ->
                if Tutils.Golden.string_param case "dtype" = "float64" then begin
                  let n = Tutils.Golden.int_param case "n" in
                  let c =
                    config
                      ~fft_size:(Tutils.Golden.int_param case "fft_size")
                      ~hop:(Tutils.Golden.int_param case "hop")
                      ()
                  in
                  let ratio =
                    { Pipeline.Rate.num= Tutils.Golden.int_param case "num"
                    ; den= Tutils.Golden.int_param case "den" }
                  in
                  let x =
                    Nx.create Nx.float64 [|n|]
                      (let state = ref 20250803 in
                       Array.init n (fun _ ->
                           state :=
                             ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
                           (Float.of_int !state /. Float.of_int (1 lsl 30))
                           -. 1. ) )
                  in
                  let expected = Nx.create Nx.float64 case.shape case.values in
                  let got = Effects.pitch_shift c ~ratio x in
                  let fraction = deviation got expected /. peak expected in
                  if fraction > !worst then worst := fraction
                end )
              file.cases )
          (Tutils.Golden.load_dir
             ~filter:(fun name -> name = "pitch.json")
             vectors_dir ) ;
        is_true
          ~msg:(Printf.sprintf "worst substitution %.3e of peak" !worst)
          (!worst <= substitution_bound) ) ]

(* {2 Validation} *)

let validation_tests =
  [ test "the rate must be finite and positive" (fun () ->
        let c = config () in
        let x = Nx.zeros Nx.float64 [|64|] in
        raises_invalid_arg ~msg:"zero rate"
          "time_stretch: cannot stretch by a rate of 0 (the rate must be \
           finite and positive)" (fun () ->
            ignore (Effects.time_stretch c ~rate:0. x) ) ;
        raises_invalid_arg ~msg:"negative rate"
          "time_stretch: cannot stretch by a rate of -1 (the rate must be \
           finite and positive)" (fun () ->
            ignore (Effects.time_stretch c ~rate:(-1.) x) ) ;
        raises_invalid_arg ~msg:"infinite rate"
          "phase_vocoder: cannot stretch by a rate of inf (the rate must be \
           finite and positive)" (fun () ->
            ignore
              (Effects.phase_vocoder c ~rate:Float.infinity
                 (Nx.zeros Nx.complex128 [|Stft.Config.bins c; 3|]) ) ) ;
        raises_invalid_arg ~msg:"nan rate"
          "phase_vocoder: cannot stretch by a rate of nan (the rate must be \
           finite and positive)" (fun () ->
            ignore
              (Effects.phase_vocoder c ~rate:Float.nan
                 (Nx.zeros Nx.complex128 [|Stft.Config.bins c; 3|]) ) ) )
  ; test "the spectrum must have a bin axis of the configured height" (fun () ->
        let c = config () in
        raises_invalid_arg ~msg:"rank one"
          "phase_vocoder: cannot vocode a rank-1 tensor (the bin and frame \
           axes must exist)" (fun () ->
            ignore
              (Effects.phase_vocoder c ~rate:1.37
                 (Nx.zeros Nx.complex128 [|4|]) ) ) ;
        raises_invalid_arg ~msg:"wrong bin count"
          "phase_vocoder: cannot vocode 64 frequency bins of a 256-point \
           transform (the bin axis must hold fft_size / 2 + 1 = 129 values)"
          (fun () ->
            ignore
              (Effects.phase_vocoder c ~rate:1.37
                 (Nx.zeros Nx.complex128 [|64; 3|]) ) ) )
  ; test "signals must have a time axis" (fun () ->
        let c = config () in
        raises_invalid_arg ~msg:"rank zero"
          "time_stretch: cannot process a rank-zero tensor (the time axis must \
           exist)" (fun () ->
            ignore
              (Effects.time_stretch c ~rate:1.37 (Nx.zeros Nx.float64 [||])) ) ;
        raises_invalid_arg ~msg:"rank zero"
          "pitch_shift: cannot process a rank-zero tensor (the time axis must \
           exist)" (fun () ->
            ignore
              (Effects.pitch_shift c ~ratio:(Effects.semitones 4.)
                 (Nx.zeros Nx.float64 [||]) ) ) )
  ; test "both terms of the ratio must be positive" (fun () ->
        let c = config () in
        let x = Nx.zeros Nx.float64 [|64|] in
        raises_invalid_arg ~msg:"zero numerator"
          "pitch_shift: cannot shift by a frequency ratio of 0/1 (both terms \
           must be at least 1)" (fun () ->
            ignore
              (Effects.pitch_shift c ~ratio:{Pipeline.Rate.num= 0; den= 1} x) ) ;
        raises_invalid_arg ~msg:"negative denominator"
          "pitch_shift: cannot shift by a frequency ratio of 3/-2 (both terms \
           must be at least 1)" (fun () ->
            ignore
              (Effects.pitch_shift c ~ratio:{Pipeline.Rate.num= 3; den= -2} x) ) )
  ; test "semitones validates its arguments" (fun () ->
        raises_invalid_arg ~msg:"zero resolution"
          "semitones: cannot divide the octave into 0 steps (bins_per_octave \
           must be at least 1)" (fun () ->
            ignore (Effects.semitones ~bins_per_octave:0 4.) ) ;
        raises_invalid_arg ~msg:"nan steps"
          "semitones: cannot shift by nan steps (the step count must be finite)"
          (fun () -> ignore (Effects.semitones Float.nan) ) ;
        raises_invalid_arg ~msg:"past the cap"
          "semitones: cannot represent a frequency ratio of 8192 within 512 \
           (the step count is too far from unity)" (fun () ->
            ignore (Effects.semitones 156.) ) )
  ; test "a configuration that cannot be inverted is rejected" (fun () ->
        (* A periodic Hann advanced by its own length overlap-adds to zero at
           the frame boundary, which the synthesis cannot divide by. *)
        let c =
          Stft.Config.create ~fft_size:64 ~hop:64 ~pad:(`Constant 0.) ()
        in
        raises_invalid_arg ~msg:"no overlap"
          "invert: cannot invert a 64-point window advanced by 64 samples \
           inside a 64-point frame (the overlap-added squared window must stay \
           above 1e-10 of its largest value at every position)" (fun () ->
            ignore
              (Effects.time_stretch c ~rate:1.37 (Nx.zeros Nx.float64 [|512|])) ) )
  ]

let suite =
  [ group "lengths" length_tests
  ; group "identity" identity_tests
  ; group "batching" batch_tests
  ; group "semitones" semitone_tests
  ; group "substitution" substitution_tests
  ; group "validation" validation_tests ]
