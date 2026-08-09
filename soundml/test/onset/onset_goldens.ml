(* Golden parity against librosa 0.11: spectral_contrast over magnitude
   spectrograms (librosa.feature.spectral_contrast) and onset_strength
   end-to-end on its default log-power-mel chain (librosa.onset.onset_strength),
   replayed from the committed vectors under the suite's vectors/ directory. The
   test signal is the stft suite's 31-bit LCG under this suite's seed,
   reproduced with the same integer arithmetic; the envelope case shapes it by
   exp(-12 i / n) to drive the log-mel through librosa's top_db clamp, and the
   silence-tail case zeroes its second half so the clamp inside the logarithmic
   contrast difference binds. float32 cases quantize the signal to float32
   against a float64 reference, exactly like the stft and mel suites. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 tolerances by domain: linear-domain results (the linear contrast
   difference) match librosa's float64 pipeline up to FFT and summation-order
   noise, far below 1e-9 relative; the logarithmic contrast and the onset
   envelope live in the log domain, where cancellations make the relative term
   meaningless near zero, so an absolute term at the same magnitude joins it
   (the mfcc precedent). float32 log-domain cases follow the db/mfcc precedent
   at 1e-4. *)
let float64_rtol = 1e-9

let float64_atol = 1e-12

let log64_atol = 1e-9

let log32_rtol = 1e-4

let log32_atol = 1e-4

let seed = 20260802

let lcg_signal ?(envelope = false) ?(silence_tail = false) n =
  let state = ref seed in
  Array.init n (fun i ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      let v = (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. in
      let v =
        if envelope then
          v *. Float.exp (-12. *. Float.of_int i /. Float.of_int n)
        else v
      in
      if silence_tail && i >= n / 2 then 0. else v )

let alignment_of_param (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "alignment" with
  | "centered" ->
      `Centered
  | "left" ->
      `Left
  | other ->
      failf "golden case %s: unknown alignment %s" case.Tutils.Golden.name other

let stft_config_of_params (case : Tutils.Golden.case) =
  Stft.Config.create
    ~hop:(Tutils.Golden.int_param case "hop")
    ~alignment:(alignment_of_param case)
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let signal_of_params (case : Tutils.Golden.case) =
  lcg_signal
    ~envelope:(Tutils.Golden.bool_param case "envelope")
    ~silence_tail:(Tutils.Golden.bool_param case "silence_tail")
    (Tutils.Golden.int_param case "length")

let contrast_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let stft_config = stft_config_of_params case in
      let sample_rate = Tutils.Golden.int_param case "sample_rate" in
      let n_bands = Tutils.Golden.int_param case "n_bands" in
      let f_min = Tutils.Golden.float_param case "f_min" in
      let quantile = Tutils.Golden.float_param case "quantile" in
      let linear = Tutils.Golden.bool_param case "linear" in
      let signal = signal_of_params case in
      let contrast dtype =
        spectral_contrast stft_config ~n_bands ~f_min ~quantile ~linear
          ~sample_rate
          (Nx.create dtype [|Array.length signal|] signal)
      in
      match (Tutils.Golden.string_param case "dtype", linear) with
      | "float64", true ->
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (contrast Nx.float64)
      | "float64", false ->
          Tutils.check_close ~rtol:float64_rtol ~atol:log64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (contrast Nx.float64)
      | "float32", true ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (contrast Nx.float32)
      | "float32", false ->
          Tutils.check_close ~rtol:log32_rtol ~atol:log32_atol ~shape:case.shape
            ~msg:case.name ~expected:case.values (contrast Nx.float32)
      | other, _ ->
          failf "golden case %s: unknown dtype %s" case.name other )

let onset_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let stft_config = stft_config_of_params case in
      let mel_config =
        Mel.Config.create
          ~n_mels:(Tutils.Golden.int_param case "n_mels")
          ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
          ~fft_size:(Tutils.Golden.int_param case "fft_size")
          ()
      in
      let lag = Tutils.Golden.int_param case "lag" in
      let signal = signal_of_params case in
      let envelope dtype =
        onset_strength stft_config mel_config ~lag
          (Nx.create dtype [|Array.length signal|] signal)
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~rtol:float64_rtol ~atol:log64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (envelope Nx.float64)
      | "float32" ->
          Tutils.check_close ~rtol:log32_rtol ~atol:log32_atol ~shape:case.shape
            ~msg:case.name ~expected:case.values (envelope Nx.float32)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      let case =
        match stem with
        | "spectral_contrast" ->
            contrast_case
        | "onset_strength" ->
            onset_case
        | other ->
            fun case ->
              test case.Tutils.Golden.name (fun () ->
                  failf "unknown golden file %s" other )
      in
      group ("golden-" ^ stem) (List.map case file.cases) )
    files
