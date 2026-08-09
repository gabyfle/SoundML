(* Golden parity against librosa 0.11: filterbank weights (librosa.filters.mel),
   mel spectrograms (librosa.feature.melspectrogram) and MFCCs
   (librosa.feature.mfcc), replayed from the committed vectors under the suite's
   vectors/ directory. The test signal is the stft suite's 31-bit LCG under this
   suite's seed, reproduced with the same integer arithmetic; the clamped cases
   shape it by an exp(-12 i / n) envelope whose ~104 dB power decay exercises
   librosa's top_db clamp and amin floor inside the log-mel. float32 cases
   quantize the signal to float32 against a float64 reference, exactly like the
   stft suite. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 tolerances by domain. Linear-domain results (weights, mel powers)
   match librosa's float64 pipeline up to FFT and summation-order noise, far
   below 1e-9 relative. MFCCs live in the log domain and end in a DCT whose
   coefficients cancel, so an absolute term at the same magnitude replaces the
   meaningless relative one near zero. *)
let float64_rtol = 1e-9

let float64_atol = 1e-12

let mfcc64_atol = 1e-9

(* float32 MFCC cases follow the db suite's log-domain precedent: the reference
   is float64 on the quantized signal, and float32 log arithmetic is only
   comparable at 1e-4. *)
let mfcc32_rtol = 1e-4

let mfcc32_atol = 1e-4

let seed = 20260803

let lcg_signal ?(envelope = false) n =
  let state = ref seed in
  Array.init n (fun i ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      let v = (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. in
      if envelope then v *. Float.exp (-12. *. Float.of_int i /. Float.of_int n)
      else v )

let scale_of_param case =
  match Tutils.Golden.string_param case "scale" with
  | "slaney" ->
      `Slaney
  | "htk" ->
      `Htk
  | other ->
      failf "golden case %s: unknown scale %s" case.Tutils.Golden.name other

let norm_of_param case =
  match Tutils.Golden.string_param case "norm" with
  | "slaney" ->
      `Slaney
  | "none" ->
      `None
  | other ->
      failf "golden case %s: unknown norm %s" case.Tutils.Golden.name other

let f_max_of_param case =
  match Tutils.Golden.param case "f_max" with
  | `Null ->
      None
  | json ->
      Some (Yojson.Safe.Util.to_number json)

let mel_config_of_params (case : Tutils.Golden.case) =
  Mel.Config.create
    ~f_min:(Tutils.Golden.float_param case "f_min")
    ?f_max:(f_max_of_param case) ~scale:(scale_of_param case)
    ~norm:(norm_of_param case)
    ~n_mels:(Tutils.Golden.int_param case "n_mels")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let stft_config_of_params (case : Tutils.Golden.case) =
  let alignment =
    match Tutils.Golden.string_param case "alignment" with
    | "centered" ->
        `Centered
    | "left" ->
        `Left
    | other ->
        failf "golden case %s: unknown alignment %s" case.Tutils.Golden.name
          other
  in
  Stft.Config.create
    ~hop:(Tutils.Golden.int_param case "hop")
    ~alignment
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let filterbank_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let config = mel_config_of_params case in
      Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol ~shape:case.shape
        ~msg:case.name ~expected:case.values
        (Mel.filterbank Nx.float64 config) ;
      Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
        ~shape:case.shape ~msg:(case.name ^ "/float32") ~expected:case.values
        (Mel.filterbank Nx.float32 config) )

let signal_of_params (case : Tutils.Golden.case) =
  lcg_signal
    ~envelope:(Tutils.Golden.bool_param case "envelope")
    (Tutils.Golden.int_param case "length")

let spectrogram_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let stft_config = stft_config_of_params case in
      let mel_config = mel_config_of_params case in
      let power = Tutils.Golden.float_param case "power" in
      let signal = signal_of_params case in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|Array.length signal|] signal in
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (mel_spectrogram stft_config mel_config ~power x)
      | "float32" ->
          let x = Nx.create Nx.float32 [|Array.length signal|] signal in
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (mel_spectrogram stft_config mel_config ~power x)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let mfcc_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let stft_config = stft_config_of_params case in
      let mel_config = mel_config_of_params case in
      let n_mfcc = Tutils.Golden.int_param case "n_mfcc" in
      (* a zero lifter is omitted, so the librosa lifter=0 goldens pin the
         default at the same time *)
      let lifter =
        match Tutils.Golden.float_param case "lifter" with
        | 0. ->
            None
        | l ->
            Some l
      in
      let signal = signal_of_params case in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|Array.length signal|] signal in
          Tutils.check_close ~rtol:float64_rtol ~atol:mfcc64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (mfcc stft_config mel_config ~n_mfcc ?lifter x)
      | "float32" ->
          let x = Nx.create Nx.float32 [|Array.length signal|] signal in
          Tutils.check_close ~rtol:mfcc32_rtol ~atol:mfcc32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (mfcc stft_config mel_config ~n_mfcc ?lifter x)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      let case =
        match stem with
        | "filterbank" ->
            filterbank_case
        | "mel_spectrogram" ->
            spectrogram_case
        | "mfcc" ->
            mfcc_case
        | other ->
            fun case ->
              test case.Tutils.Golden.name (fun () ->
                  failf "unknown golden file %s" other )
      in
      group ("golden-" ^ stem) (List.map case file.cases) )
    files
