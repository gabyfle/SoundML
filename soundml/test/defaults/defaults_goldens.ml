(* Golden parity against librosa 1.0 at its defaults: every reference call
   leaves pad_mode, center and window alone — zero padding, centered frames, the
   Hann window — and every configuration here is built from the arguments the
   reference was given and nothing else, so each case states that a default of
   this library is librosa's. The signals are the 31-bit LCG stream under this
   suite's seed, reproduced with the generator's integer arithmetic; float32
   cases quantize the signal against a float64 reference. The generator asserts
   that each case accepting pad_mode moves by more than a thousandth of its peak
   under pad_mode="reflect" — measured, from 0.6% of it for the centroid to more
   than the whole peak for Griffin-Lim — so none of them would pass over another
   padding. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

let seed = 20261026

let sample_rate = 22050

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let int_param = Tutils.Golden.int_param

let signal dtype case =
  let n = int_param case "length" in
  Nx.create dtype [|n|] (lcg_values n)

(* [stft_config case] is the analysis geometry given its transform size alone.
   Where the reference was handed a hop — librosa's feature functions fix 512
   whatever the transform size — it is the configuration's default hop too,
   checked here so that a change on either side reads as a hop mismatch rather
   than as a frame count. *)
let stft_config (case : Tutils.Golden.case) =
  let c = Stft.Config.create ~fft_size:(int_param case "fft_size") () in
  if List.mem_assoc "hop" case.params then
    equal
      ~msg:(case.name ^ ": default hop")
      int (int_param case "hop") (Stft.Config.hop c) ;
  c

let mel_config case =
  Mel.Config.create ~n_mels:(int_param case "n_mels") ~sample_rate
    ~fft_size:(int_param case "fft_size")
    ()

(* [complex_part kind z] is the real or imaginary component of [z] as float64,
   as in the stft suite: the cast reads the real component and multiplying by
   [-i] rotates the imaginary one into it exactly. *)
let complex_part kind z =
  match kind with
  | "real" ->
      Nx.cast Nx.float64 z
  | _ ->
      Nx.cast Nx.float64 (Nx.mul_s z Complex.{re= 0.; im= -1.})

(* [compute dtype case] is the library's side of [case] at [dtype], widened to
   float64 (exactly) for the comparison. *)
let compute (type a) (dtype : (float, a) Nx.dtype) (case : Tutils.Golden.case) =
  let f64 t = Nx.cast Nx.float64 t in
  match Tutils.Golden.string_param case "kind" with
  | "magnitude" ->
      f64 (Stft.power_spectrum ~power:1. (stft_config case) (signal dtype case))
  | ("real" | "imag") as kind ->
      complex_part kind
        (Stft.transform Nx.complex128 (stft_config case) (signal dtype case))
  | "melspectrogram" ->
      f64
        (mel_spectrogram (stft_config case) (mel_config case)
           (signal dtype case) )
  | "mfcc" ->
      f64 (mfcc (stft_config case) (mel_config case) (signal dtype case))
  | "chroma_stft" ->
      f64
        (chroma_stft (stft_config case)
           (Chroma.Config.create ~sample_rate
              ~fft_size:(int_param case "fft_size")
              () )
           (signal dtype case) )
  | "spectral_contrast" ->
      f64
        (spectral_contrast (stft_config case) ~sample_rate (signal dtype case))
  | "spectral_centroid" ->
      f64
        (spectral_centroid ~sample_rate
           (Stft.power_spectrum ~power:1. (stft_config case) (signal dtype case)) )
  | "onset_strength" ->
      f64
        (onset_strength (stft_config case) (mel_config case) (signal dtype case))
  | "rms" ->
      f64 (rms (signal dtype case))
  | "zero_crossing_rate" ->
      f64 (zero_crossing_rate (signal dtype case))
  | "griffinlim" ->
      (* librosa reads the transform size off the bin count and defaults the hop
         to a quarter of it *)
      let bins = int_param case "bins" and frames = int_param case "frames" in
      let magnitudes =
        Nx.create dtype [|bins; frames|]
          (Array.map (fun v -> Float.abs v +. 0.5) (lcg_values (bins * frames)))
      in
      f64 (Stft.griffin_lim (stft_config case) magnitudes)
  | "time_stretch" ->
      f64
        (Effects.time_stretch (stft_config case)
           ~rate:(Tutils.Golden.float_param case "rate")
           (signal dtype case) )
  | "harmonic" ->
      f64 (harmonic (stft_config case) (signal dtype case))
  | "percussive" ->
      f64 (percussive (stft_config case) (signal dtype case))
  | other ->
      failf "golden case %s: unknown kind %s" case.name other

(* Tolerances. The float64 analysis cases — spectra, the features over them and
   the separation — agree within 2.9e-14 absolute (1.6e-15 of peak), and the
   centroid, in hertz, within 1.2e-15 relative; the bounds keep a factor of ten
   over both. *)
let analysis64_rtol = 1e-14

let analysis64_atol = 3e-13

(* Griffin-Lim and the time stretch iterate — 32 analysis-synthesis pairs, a
   phase recurrence — and agree within 1.5e-13 on signals peaking near one; the
   bound keeps a factor of ten. *)
let synthesis64_atol = 1.5e-12

(* float32 cases agree within 1.1e-7 relative and take the suite-wide float32
   tolerances, a factor of nine above that. *)
let tolerances (case : Tutils.Golden.case) =
  match
    ( Tutils.Golden.string_param case "dtype"
    , Tutils.Golden.string_param case "kind" )
  with
  | "float32", _ ->
      (Tutils.float32_rtol, Tutils.float32_atol)
  | _, ("griffinlim" | "time_stretch") ->
      (0., synthesis64_atol)
  | _ ->
      (analysis64_rtol, analysis64_atol)

let golden_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let rtol, atol = tolerances case in
      let check got =
        Tutils.check_close ~rtol ~atol ~shape:case.shape ~msg:case.name
          ~expected:case.values got
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          check (compute Nx.float64 case)
      | "float32" ->
          check (compute Nx.float32 case)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

(* The configuration a transform size alone produces, field by field: librosa's
   [window="hann"], [center=True], [pad_mode="constant"], [win_length=n_fft] and
   [hop_length=win_length // 4], with no window normalisation. *)
let configuration_tests =
  [ test "a transform size alone is librosa's stft geometry" (fun () ->
        let c = Stft.Config.create ~fft_size:2048 () in
        is_true ~msg:"Hann window"
          (match Stft.Config.window c with Window.Hann -> true | _ -> false) ;
        is_true ~msg:"centered frames"
          (match Stft.Config.alignment c with `Centered -> true | _ -> false) ;
        is_true ~msg:"zero padding"
          ( match Stft.Config.pad c with
          | `Constant v ->
              Float.equal v 0.
          | `Reflect | `Edge ->
              false ) ;
        equal ~msg:"win_length" int 2048 (Stft.Config.win_length c) ;
        equal ~msg:"hop" int 512 (Stft.Config.hop c) ;
        is_true ~msg:"no window normalisation"
          (match Stft.Config.scale c with `None -> true | _ -> false) )
  ; (* The one input on which the defaults part ways, a deviation stft.mli
       names: librosa's zero padding turns an empty signal into one all-zero
       frame for an even n_fft and rejects it for an odd one — the spectrum of
       [librosa.stft(np.zeros(0))] is 1025 bins by one frame — while this
       library produces no frames whatever the transform size. *)
    test "an empty signal produces no frames, a named deviation" (fun () ->
        List.iter
          (fun fft_size ->
            let c = Stft.Config.create ~fft_size () in
            let x = Nx.create Nx.float64 [|0|] [||] in
            let bins = (fft_size / 2) + 1 in
            equal
              ~msg:(Printf.sprintf "frames, fft %d" fft_size)
              int 0 (Stft.frames c ~n:0) ;
            equal
              ~msg:(Printf.sprintf "power spectrum shape, fft %d" fft_size)
              (array int) [|bins; 0|]
              (Nx.shape (Stft.power_spectrum c x)) ;
            equal
              ~msg:(Printf.sprintf "transform shape, fft %d" fft_size)
              (array int) [|bins; 0|]
              (Nx.shape (Stft.transform Nx.complex128 c x)) )
          [2048; 2047; 64] ) ]

let suite =
  group "configuration" configuration_tests
  :: List.map
       (fun (file : Tutils.Golden.file) ->
         group (Filename.basename file.path) (List.map golden_case file.cases) )
       (Tutils.Golden.load_dir vectors_dir)
