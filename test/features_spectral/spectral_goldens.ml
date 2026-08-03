(* Golden parity against librosa 0.11: spectral_centroid, spectral_bandwidth,
   spectral_rolloff and spectral_flatness, replayed from the committed vectors
   under test/vectors/features_spectral/. Direct-spectrogram cases rebuild the
   magnitude tensor bit-exactly — the stft suite's 31-bit LCG under this suite's
   seed, absolute value, C-order reshape — and cover batched inputs, a custom
   non-uniform frequency grid (exact triangular-number arithmetic) and both
   element dtypes; float32 cases quantize the magnitudes against a float64
   reference, exactly like the stft and mel suites. End-to-end cases recompute
   the magnitude spectrogram with Stft.power_spectrum ~power:1. from the LCG
   signal and follow the librosa reference through the same feature. *)

open Windtrap
open Soundml

let vectors_dir =
  Filename.concat ".." (Filename.concat "vectors" "features_spectral")

(* float64 tolerances: the features are linear-domain reductions over the
   spectrogram (flatness's log/exp cancels within the same magnitude), so the
   float64 pipeline matches librosa up to FFT and summation-order noise, far
   below 1e-9 relative. *)
let float64_rtol = 1e-9

let float64_atol = 1e-12

let seed = 20261024

let lcg_signal n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let magnitudes ~squared n =
  Array.map
    (fun v ->
      let a = Float.abs v in
      if squared then a *. a else a )
    (lcg_signal n)

let int_list_param (case : Tutils.Golden.case) key =
  match Tutils.Golden.param case key with
  | `List l ->
      Array.of_list (List.map Yojson.Safe.Util.to_int l)
  | _ ->
      failf "golden case %s: parameter %s is not a list" case.name key

(* The custom grid of the generator: 10 * (k + 1) * (k + 2) / 2 for bin [k] —
   exact integer arithmetic, so both sides hold identical doubles. *)
let triangular_freqs dtype bins =
  Nx.create dtype [|bins|]
    (Array.init bins (fun k -> 10. *. Float.of_int ((k + 1) * (k + 2)) /. 2.))

let freqs_of_params (case : Tutils.Golden.case) dtype bins =
  match Tutils.Golden.string_param case "freqs" with
  | "fft" ->
      None
  | "triangular" ->
      Some (triangular_freqs dtype bins)
  | other ->
      failf "golden case %s: unknown freqs grid %s" case.name other

let run_feature stem (case : Tutils.Golden.case) freqs s =
  match stem with
  | "spectral_centroid" ->
      spectral_centroid ?freqs
        ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
        s
  | "spectral_bandwidth" ->
      spectral_bandwidth
        ~p:(Tutils.Golden.float_param case "p")
        ?freqs
        ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
        s
  | "spectral_rolloff" ->
      spectral_rolloff
        ~roll_percent:(Tutils.Golden.float_param case "roll_percent")
        ?freqs
        ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
        s
  | "spectral_flatness" ->
      spectral_flatness
        ~amin:(Tutils.Golden.float_param case "amin")
        ~power:(Tutils.Golden.float_param case "power")
        s
  | other ->
      failf "golden case %s: unknown feature %s" case.name other

(* Direct-spectrogram cases: the magnitude tensor is rebuilt bit-exactly, so the
   only imprecision under test is the feature computation itself. *)
let spectrogram_case stem (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let shape = int_list_param case "shape_s" in
      let bins = shape.(Array.length shape - 2) in
      let squared = Tutils.Golden.bool_param case "squared" in
      let values = magnitudes ~squared (Array.fold_left ( * ) 1 shape) in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let s = Nx.create Nx.float64 shape values in
          let freqs = freqs_of_params case Nx.float64 bins in
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (run_feature stem case freqs s)
      | "float32" ->
          let s = Nx.create Nx.float32 shape values in
          let freqs = freqs_of_params case Nx.float32 bins in
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (run_feature stem case freqs s)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

(* End-to-end cases: the magnitude spectrogram comes from Stft.power_spectrum
   ~power:1. on the LCG signal, against librosa.feature.* called on the same
   signal with the analysis geometry passed explicitly. *)
let signal_case stem (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let length = Tutils.Golden.int_param case "length" in
      let config =
        Stft.Config.create
          ~fft_size:(Tutils.Golden.int_param case "fft_size")
          ~hop:(Tutils.Golden.int_param case "hop")
          ()
      in
      let signal = lcg_signal length in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|length|] signal in
          let s = Stft.power_spectrum ~power:1. config x in
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (run_feature stem case None s)
      | "float32" ->
          let x = Nx.create Nx.float32 [|length|] signal in
          let s = Stft.power_spectrum ~power:1. config x in
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (run_feature stem case None s)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let feature_case stem (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "source" with
  | "spectrogram" ->
      spectrogram_case stem case
  | "signal" ->
      signal_case stem case
  | other ->
      test case.name (fun () ->
          failf "golden case %s: unknown source %s" case.name other )

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      group ("golden-" ^ stem) (List.map (feature_case stem) file.cases) )
    files
