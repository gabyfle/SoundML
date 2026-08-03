(* Contracts of the flat spectral_contrast and onset_strength features and their
   stages: parameter validation (every precondition raises its documented
   message, in ascending band order for the contrast plan), defaults, output
   shapes and leading-axis broadcast, the exact agreement of the
   spectral_contrast_of_spectrogram face with the audio face, the zero prefixes
   of the onset envelope (the lag and the centered compensation shift), and the
   per-chunk shape preconditions of the stages. *)

open Windtrap
open Soundml

let farray = array (float 0.)

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:1000 ~channels:1

let signal n = Array.init n (fun i -> Float.sin (0.4 *. Float.of_int i))

let stft8 () = Stft.Config.create ~fft_size:8 ~hop:2 ()

let mel8 () = Mel.Config.create ~n_mels:3 ~sample_rate:1000 ~fft_size:8 ()

(* a plan that fits the fft-8 grid at 1000 Hz: bins at 0, 125, .., 500 Hz, bands
   [0, 150], [150, 300] and [300, 600] Hz *)
let contrast8 ?(n_bands = 2) ?(f_min = 150.) ?quantile ?linear x =
  spectral_contrast (stft8 ()) ~n_bands ~f_min ?quantile ?linear
    ~sample_rate:1000 x

(* {2 spectral_contrast: validation} *)

let contrast_validation_tests =
  [ test "n_bands must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero bands"
          "spectral_contrast: cannot divide the spectrum into 0 octave bands \
           (n_bands must be at least 1)" (fun () ->
            ignore
              (spectral_contrast (stft8 ()) ~n_bands:0 ~sample_rate:1000
                 (Nx.zeros Nx.float64 [|32|]) ) ) )
  ; test "f_min must be finite and positive" (fun () ->
        raises_invalid_arg ~msg:"zero f_min"
          "spectral_contrast: cannot start the first octave band at 0 Hz \
           (f_min must be finite and positive)" (fun () ->
            ignore
              (spectral_contrast (stft8 ()) ~f_min:0. ~sample_rate:1000
                 (Nx.zeros Nx.float64 [|32|]) ) ) ;
        raises_invalid_arg ~msg:"nan f_min"
          "spectral_contrast: cannot start the first octave band at nan Hz \
           (f_min must be finite and positive)" (fun () ->
            ignore
              (spectral_contrast (stft8 ()) ~f_min:Float.nan ~sample_rate:1000
                 (Nx.zeros Nx.float64 [|32|]) ) ) )
  ; test "quantile must lie strictly between 0 and 1" (fun () ->
        List.iter
          (fun (name, quantile) ->
            raises_invalid_arg ~msg:name
              (Printf.sprintf
                 "spectral_contrast: cannot average the %g quantile of each \
                  band (quantile must lie strictly between 0 and 1)"
                 quantile ) (fun () ->
                ignore
                  (spectral_contrast (stft8 ()) ~quantile ~sample_rate:1000
                     (Nx.zeros Nx.float64 [|32|]) ) ) )
          [("zero", 0.); ("one", 1.); ("nan", Float.nan)] )
  ; test "sample_rate must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"zero rate"
          "spectral_contrast: cannot use a sample rate of 0 Hz (sample_rate \
           must be at least 1)" (fun () ->
            ignore
              (spectral_contrast (stft8 ()) ~sample_rate:0
                 (Nx.zeros Nx.float64 [|32|]) ) ) )
  ; test "every band must start below Nyquist" (fun () ->
        (* the librosa defaults at 8000 Hz: the top band starts at 200 * 2^5 =
           6400 Hz, beyond the 4000 Hz Nyquist *)
        raises_invalid_arg ~msg:"top band beyond Nyquist"
          "spectral_contrast: cannot start the top octave band at 6400 Hz at a \
           sample rate of 8000 Hz (every band must start below the Nyquist \
           frequency 4000; lower f_min or n_bands)" (fun () ->
            ignore
              (spectral_contrast
                 (Stft.Config.create ~fft_size:64 ())
                 ~sample_rate:8000
                 (Nx.zeros Nx.float64 [|128|]) ) ) )
  ; test "a band spanning no bin raises instead of degrading" (fun () ->
        (* fft 32 at 22050 Hz: bins are 689 Hz apart, so [200, 400] Hz holds
           none — librosa would produce NaN means there *)
        raises_invalid_arg ~msg:"empty band"
          "spectral_contrast: cannot resolve the octave band [200, 400] Hz \
           with an FFT of size 32 at a sample rate of 22050 Hz (the band spans \
           no FFT bin; raise fft_size or lower n_bands)" (fun () ->
            ignore
              (spectral_contrast
                 (Stft.Config.create ~fft_size:32 ())
                 ~sample_rate:22050
                 (Nx.zeros Nx.float64 [|64|]) ) ) )
  ; test "a band emptied by yielding its top bin raises too" (fun () ->
        (* fft 64 at 8000 Hz: [0, 100] Hz holds bin 0 alone, and band 0 yields
           its only bin to band 1 *)
        raises_invalid_arg ~msg:"single-bin band 0"
          "spectral_contrast: cannot resolve the octave band [0, 100] Hz with \
           an FFT of size 64 at a sample rate of 8000 Hz (the band spans no \
           FFT bin below its top edge; raise fft_size or lower n_bands)"
          (fun () ->
            ignore
              (spectral_contrast
                 (Stft.Config.create ~fft_size:64 ())
                 ~n_bands:4 ~f_min:100. ~sample_rate:8000
                 (Nx.zeros Nx.float64 [|128|]) ) ) )
  ; test "the audio must have a time axis" (fun () ->
        raises_invalid_arg ~msg:"rank zero"
          "spectral_contrast: cannot analyse a rank-zero tensor (the time axis \
           must exist)" (fun () ->
            ignore (contrast8 (Nx.scalar Nx.float64 1.)) ) ) ]

(* {2 spectral_contrast: shapes and defaults} *)

let contrast_shape_tests =
  [ test "defaults are n_bands 6, f_min 200, quantile 0.02, logarithmic"
      (fun () ->
        let c = Stft.Config.create ~fft_size:512 ~hop:128 () in
        let x = Nx.create Nx.float64 [|700|] (signal 700) in
        equal ~msg:"explicit defaults" farray
          (Nx.to_array
             (spectral_contrast c ~n_bands:6 ~f_min:200. ~quantile:0.02
                ~linear:false ~sample_rate:22050 x ) )
          (Nx.to_array (spectral_contrast c ~sample_rate:22050 x)) )
  ; test "output shape is [n_bands + 1; frames]" (fun () ->
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        let frames = Stft.frames (stft8 ()) ~n:41 in
        equal ~msg:"mono" (array int) [|3; frames|] (Nx.shape (contrast8 x)) )
  ; test "leading axes broadcast" (fun () ->
        let batch =
          Nx.init Nx.float64 [|2; 3; 41|] (fun i ->
              Float.sin
                (0.4 *. Float.of_int ((i.(0) * 200) + (i.(1) * 50) + i.(2))) )
        in
        let out = contrast8 batch in
        let frames = Stft.frames (stft8 ()) ~n:41 in
        equal ~msg:"batched shape" (array int) [|2; 3; 3; frames|]
          (Nx.shape out) ;
        for i = 0 to 1 do
          for j = 0 to 2 do
            equal
              ~msg:(Printf.sprintf "slice %d,%d" i j)
              farray
              (Nx.to_array (contrast8 (Nx.slice [I i; I j; A] batch)))
              (Nx.to_array (Nx.slice [I i; I j; A; A] out))
          done
        done )
  ; test "the empty signal maps to the empty contrast" (fun () ->
        equal ~msg:"zero samples" (array int) [|3; 0|]
          (Nx.shape (contrast8 (Nx.zeros Nx.float64 [|0|]))) )
  ; test "a flat spectrum has zero contrast in both forms" (fun () ->
        (* peak and valley coincide on a constant-magnitude spectrum *)
        let chunk = Nx.full Nx.float64 [|5; 4|] 0.25 in
        List.iter
          (fun (name, linear) ->
            let p =
              spectral_contrast_stage (stft8 ()) ~n_bands:2 ~f_min:150. ~linear
                ~sample_rate:1000 ()
            in
            equal ~msg:name farray (Array.make 12 0.)
              (Nx.to_array (Pipeline.run ~source:(source ()) p chunk)) )
          [("linear", true); ("logarithmic", false)] ) ]

(* {2 spectral_contrast_of_spectrogram: the S= face} *)

let contrast_spectrogram_tests =
  [ test "the spectrogram face agrees exactly with the audio face" (fun () ->
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        let s = Stft.power_spectrum ~power:1. (stft8 ()) x in
        equal ~msg:"same contrast" farray
          (Nx.to_array (contrast8 x))
          (Nx.to_array
             (spectral_contrast_of_spectrogram ~n_bands:2 ~f_min:150.
                ~sample_rate:1000 s ) ) )
  ; test "the spectrogram must be spectral" (fun () ->
        raises_invalid_arg ~msg:"rank one"
          "spectral_contrast_of_spectrogram: cannot analyse a rank-1 tensor \
           (spectral contrast needs [...; bins; frames])" (fun () ->
            ignore
              (spectral_contrast_of_spectrogram ~sample_rate:1000
                 (Nx.zeros Nx.float64 [|5|]) ) ) ;
        raises_invalid_arg ~msg:"single bin"
          "spectral_contrast_of_spectrogram: cannot derive bin frequencies for \
           a 1-bin spectrogram (the implied FFT size is 0; a magnitude \
           spectrogram holds fft_size / 2 + 1 bins)" (fun () ->
            ignore
              (spectral_contrast_of_spectrogram ~sample_rate:1000
                 (Nx.zeros Nx.float64 [|1; 4|]) ) ) )
  ; test "magnitudes must be non-negative" (fun () ->
        let poisoned =
          Nx.init Nx.float64 [|5; 2|] (fun i ->
              if i.(0) = 2 && i.(1) = 1 then -3. else 1. )
        in
        raises_invalid_arg ~msg:"negative entry"
          "spectral_contrast_of_spectrogram: cannot analyse a spectrogram with \
           negative or NaN values (a magnitude spectrogram is non-negative)"
          (fun () ->
            ignore
              (spectral_contrast_of_spectrogram ~n_bands:2 ~f_min:150.
                 ~sample_rate:1000 poisoned ) ) ) ]

(* {2 spectral_contrast_stage: per-chunk preconditions} *)

let contrast_stage_tests =
  [ test "chunks must be spectral" (fun () ->
        let p =
          spectral_contrast_stage (stft8 ()) ~n_bands:2 ~f_min:150.
            ~sample_rate:1000 ()
        in
        raises_invalid_arg ~msg:"rank one"
          "spectral_contrast_stage: cannot analyse a rank-1 tensor (spectral \
           contrast needs [...; bins; frames])" (fun () ->
            ignore
              (Pipeline.run ~source:(source ()) p (Nx.zeros Nx.float64 [|5|])) ) ;
        raises_invalid_arg ~msg:"bins mismatch"
          "spectral_contrast_stage: cannot band 4 frequency bins with a plan \
           built for an FFT of size 8 (5 bins)" (fun () ->
            ignore
              (Pipeline.run ~source:(source ())
                 (spectral_contrast_stage (stft8 ()) ~n_bands:2 ~f_min:150.
                    ~sample_rate:1000 () )
                 (Nx.zeros Nx.float64 [|4; 3|]) ) ) )
  ; test "stage construction validates the plan" (fun () ->
        raises_invalid_arg ~msg:"invalid quantile at build time"
          "spectral_contrast_stage: cannot average the 1 quantile of each band \
           (quantile must lie strictly between 0 and 1)" (fun () ->
            ignore
              ( spectral_contrast_stage (stft8 ()) ~quantile:1.
                  ~sample_rate:1000 ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.offline )
                  Pipeline.t ) ) ) ]

(* {2 onset_strength: validation} *)

let onset_validation_tests =
  [ test "lag must be at least 1, on both faces" (fun () ->
        raises_invalid_arg ~msg:"flat zero lag"
          "onset_strength: cannot difference frames at a lag of 0 (lag must be \
           at least 1)" (fun () ->
            ignore
              (onset_strength (stft8 ()) (mel8 ()) ~lag:0
                 (Nx.zeros Nx.float64 [|32|]) ) ) ;
        raises_invalid_arg ~msg:"stage negative lag"
          "onset_strength_stage: cannot difference frames at a lag of -1 (lag \
           must be at least 1)" (fun () ->
            ignore
              ( onset_strength_stage ~lag:(-1) ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.offline )
                  Pipeline.t ) ) )
  ; test "the configurations must agree on fft_size" (fun () ->
        raises_invalid_arg ~msg:"disagreeing configs"
          "onset_strength: cannot project a 16-point STFT through a filterbank \
           built for an FFT of size 8 (the two configurations must agree on \
           fft_size)" (fun () ->
            ignore
              (onset_strength
                 (Stft.Config.create ~fft_size:16 ())
                 (mel8 ())
                 (Nx.zeros Nx.float64 [|32|]) ) ) )
  ; test "the audio must have a time axis" (fun () ->
        raises_invalid_arg ~msg:"rank zero"
          "onset_strength: cannot analyse a rank-zero tensor (the time axis \
           must exist)" (fun () ->
            ignore
              (onset_strength (stft8 ()) (mel8 ()) (Nx.scalar Nx.float64 1.)) ) )
  ]

(* {2 onset_strength: shapes and the zero prefixes} *)

let onset_shape_tests =
  [ test "a lag of 1 is the default" (fun () ->
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        equal ~msg:"lag 1 = absent" farray
          (Nx.to_array (onset_strength (stft8 ()) (mel8 ()) ~lag:1 x))
          (Nx.to_array (onset_strength (stft8 ()) (mel8 ()) x)) )
  ; test "output shape drops the mel axis" (fun () ->
        let frames = Stft.frames (stft8 ()) ~n:41 in
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        equal ~msg:"mono" (array int) [|frames|]
          (Nx.shape (onset_strength (stft8 ()) (mel8 ()) x)) ;
        let batch =
          Nx.create Nx.float32 [|2; 41|]
            (Array.init 82 (fun i -> Float.of_int i /. 82.))
        in
        equal ~msg:"batched" (array int) [|2; frames|]
          (Nx.shape (onset_strength (stft8 ()) (mel8 ()) batch)) ;
        equal ~msg:"empty signal" (array int) [|0|]
          (Nx.shape
             (onset_strength (stft8 ()) (mel8 ()) (Nx.zeros Nx.float64 [|0|])) ) )
  ; test "centered analysis zeroes lag plus shift frames" (fun () ->
        (* fft 8, hop 2: the compensation shift is 8 / (2 * 2) = 2 frames *)
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        let envelope =
          Nx.to_array (onset_strength (stft8 ()) (mel8 ()) ~lag:1 x)
        in
        equal ~msg:"first three frames" farray [|0.; 0.; 0.|]
          (Array.sub envelope 0 3) ;
        is_true ~msg:"the envelope is not identically zero"
          (Array.exists (fun v -> v > 0.) envelope) )
  ; test "left alignment zeroes only the lag" (fun () ->
        let left = Stft.Config.create ~alignment:`Left ~fft_size:8 ~hop:2 () in
        let x = Nx.create Nx.float64 [|41|] (signal 41) in
        let envelope = Nx.to_array (onset_strength left (mel8 ()) ~lag:1 x) in
        equal ~msg:"first frame" (float 0.) 0. envelope.(0) ;
        is_true ~msg:"the second frame moved"
          (Array.exists (fun v -> v > 0.) (Array.sub envelope 1 2)) )
  ; test "a lag beyond the frame count degenerates to zeros" (fun () ->
        let x = Nx.create Nx.float64 [|21|] (signal 21) in
        let frames = Stft.frames (stft8 ()) ~n:21 in
        equal ~msg:"all zero" farray (Array.make frames 0.)
          (Nx.to_array
             (onset_strength (stft8 ()) (mel8 ()) ~lag:(frames + 1) x) ) ) ]

(* {2 onset_strength_stage: per-chunk preconditions} *)

let onset_stage_tests =
  [ test "chunks must be spectral" (fun () ->
        raises_invalid_arg ~msg:"rank one"
          "onset_strength_stage: cannot aggregate a rank-1 tensor (the onset \
           flux needs [...; bins; frames])" (fun () ->
            ignore
              (Pipeline.run ~source:(source ()) (onset_strength_stage ())
                 (Nx.zeros Nx.float64 [|5|]) ) ) ;
        raises_invalid_arg ~msg:"zero-size bins axis"
          "onset_strength_stage: cannot aggregate a chunk with a zero-size \
           axis (bins and channels must be at least 1)" (fun () ->
            ignore
              (Pipeline.run ~source:(source ()) (onset_strength_stage ())
                 (Nx.zeros Nx.float64 [|0; 3|]) ) ) ) ]

let suite =
  [ group "contrast-validation" contrast_validation_tests
  ; group "contrast-shapes" contrast_shape_tests
  ; group "contrast-spectrogram" contrast_spectrogram_tests
  ; group "contrast-stage" contrast_stage_tests
  ; group "onset-validation" onset_validation_tests
  ; group "onset-shapes" onset_shape_tests
  ; group "onset-stage" onset_stage_tests ]
