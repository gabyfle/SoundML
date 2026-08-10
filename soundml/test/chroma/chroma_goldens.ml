(* Golden parity against librosa 0.11: the chroma filterbank
   (librosa.filters.chroma), the constant-Q assignment matrix
   (librosa.filters.cq_to_chroma), and both chromagrams
   (librosa.feature.chroma_stft and librosa.feature.chroma_cqt), replayed from
   the committed vectors under the suite's vectors/ directory.

   The test signal is the cqt suite's harmonic recipe. float32 cases quantize it
   to float32 against a float64 reference, exactly like the stft suite.

   Every generated case passes dtype=np.float64 to the filterbank: librosa
   builds it in single precision by default, which would put a 1e-7 floor under
   a comparison that is otherwise closed-form. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* Tolerances are quoted against each case's own peak — [max |expected|] — the
   unit the deviations below were measured in.

   The matrices and the linear-frequency chromagram are closed-form doubles up
   to one matrix product and one per-frame division: both sides evaluate the
   same expressions, and the measured worst is 3.2e-16 of peak on the
   filterbanks and 1.3e-15 on chroma_stft. The 0/1 assignment matrices are
   compared exactly: a deviation there is a wrong entry, not a rounding. *)
let closed_form_rtol = 1e-12

let closed_form_atol = 1e-13

(* The constant-Q chromagram inherits the transform underneath it, whose octave
   recursion runs this library's resampler rather than librosa's. Measured worst
   1.42e-5 of the frame peak at hop 512 (the 24-band unnormalised case), and
   5.7e-8 at the odd hop, where neither side resamples. *)
let cqt_atol = 1e-4

let cqt_odd_hop_atol = 1e-6

let cqt_rtol = 1e-5

let peak_of values =
  Array.fold_left (fun acc v -> Float.max acc (Float.abs v)) 0. values

let seed = 20260803

let lcg n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

(* The generator's harmonic recipe, shared with the cqt suite: three decaying
   harmonic notes whose onsets are placed as fractions of the signal, a
   transient a quarter of the way in, and a low noise floor. *)
let harmonic n sample_rate =
  let base = lcg n in
  let y = Array.make n 0. in
  let sr = Float.of_int sample_rate in
  List.iter
    (fun (f0, fraction) ->
      let onset = fraction *. Float.of_int n /. sr in
      for i = 0 to n - 1 do
        let t = Float.of_int i /. sr in
        let env =
          if t >= onset then Float.exp (-3.0 *. Float.max (t -. onset) 0.)
          else 0.
        in
        for h = 1 to 24 do
          let h = Float.of_int h in
          if f0 *. h < 0.45 *. sr then
            y.(i) <-
              y.(i)
              +. Float.pow 0.7 h *. env
                 *. Float.sin (2. *. Float.pi *. f0 *. h *. (t -. onset))
        done
      done )
    [(65.406, 0.0); (130.813, 0.23); (246.942, 0.55)] ;
  y.(n / 4) <- y.(n / 4) +. 3.0 ;
  Array.mapi (fun i v -> (0.2 *. v) +. (0.002 *. base.(i))) y

let quantize v = Int32.float_of_bits (Int32.bits_of_float v)

let norm_of_param (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "norm" with
  | "inf" ->
      `Inf
  | "l1" ->
      `P 1.
  | "l2" ->
      `P 2.
  | "none" ->
      `None
  | other ->
      failf "golden case %s: unknown norm %s" case.name other

let chroma_config (case : Tutils.Golden.case) =
  Chroma.Config.create
    ~n_chroma:(Tutils.Golden.int_param case "n_chroma")
    ~tuning:(Tutils.Golden.float_param case "tuning")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let filterbank_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let octwidth =
        match Tutils.Golden.param case "octwidth" with
        | `Null ->
            None
        | json ->
            Some (Yojson.Safe.Util.to_number json)
      in
      let config =
        Chroma.Config.create
          ~n_chroma:(Tutils.Golden.int_param case "n_chroma")
          ~tuning:(Tutils.Golden.float_param case "tuning")
          ~ctroct:(Tutils.Golden.float_param case "ctroct")
          ~octwidth
          ~base_c:(Tutils.Golden.bool_param case "base_c")
          ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
          ~fft_size:(Tutils.Golden.int_param case "fft_size")
          ()
      in
      Tutils.check_close ~rtol:closed_form_rtol
        ~atol:(closed_form_atol *. peak_of case.values)
        ~shape:case.shape ~msg:case.name ~expected:case.values
        (Chroma.filterbank Nx.float64 config) ;
      Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
        ~shape:case.shape ~msg:(case.name ^ "/float32") ~expected:case.values
        (Chroma.filterbank Nx.float32 config) )

let cqt_config (case : Tutils.Golden.case) =
  Cqt.Config.create
    ~fmin:(Tutils.Golden.float_param case "fmin")
    ~bins_per_octave:(Tutils.Golden.int_param case "bins_per_octave")
    ~tuning:(Tutils.Golden.float_param case "tuning")
    ~hop:(Tutils.Golden.int_param case "hop")
    ~n_bins:(Tutils.Golden.int_param case "n_bins")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    ()

let projection_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let config = cqt_config case in
      let n_chroma = Tutils.Golden.int_param case "n_chroma" in
      (* The assignment is exactly zero or one, so any deviation is a wrong
         entry, not a rounding. *)
      Tutils.check_close ~rtol:0. ~atol:0. ~shape:case.shape ~msg:case.name
        ~expected:case.values
        (Chroma.cqt_projection Nx.float64 ~n_chroma config) )

let signal_of_params (case : Tutils.Golden.case) =
  let n = Tutils.Golden.int_param case "length" in
  match Tutils.Golden.string_param case "signal" with
  | "harmonic" ->
      harmonic n (Tutils.Golden.int_param case "sample_rate")
  | other ->
      failf "golden case %s: unknown signal %s" case.name other

let spectrum_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let stft =
        Stft.Config.create
          ~hop:(Tutils.Golden.int_param case "hop")
          ~pad:(`Constant 0.)
          ~fft_size:(Tutils.Golden.int_param case "fft_size")
          ()
      in
      let config = chroma_config case in
      let norm = norm_of_param case in
      let power = Tutils.Golden.float_param case "power" in
      let signal = signal_of_params case in
      let n = Array.length signal in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|n|] signal in
          Tutils.check_close ~rtol:closed_form_rtol
            ~atol:(closed_form_atol *. peak_of case.values)
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (chroma_stft stft config ~power ~norm x)
      | "float32" ->
          let x = Nx.create Nx.float32 [|n|] (Array.map quantize signal) in
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:(case.name ^ "/float32")
            ~expected:case.values
            (chroma_stft stft config ~power ~norm x)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let constant_q_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let config = cqt_config case in
      let n_chroma = Tutils.Golden.int_param case "n_chroma" in
      let norm = norm_of_param case in
      let signal = signal_of_params case in
      let n = Array.length signal in
      let atol =
        peak_of case.values
        *.
        if Tutils.Golden.int_param case "hop" mod 2 = 1 then cqt_odd_hop_atol
        else cqt_atol
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|n|] signal in
          Tutils.check_close ~rtol:cqt_rtol ~atol ~shape:case.shape
            ~msg:case.name ~expected:case.values
            (chroma_cqt config ~n_chroma ~norm x)
      | "float32" ->
          let x = Nx.create Nx.float32 [|n|] (Array.map quantize signal) in
          Tutils.check_close ~rtol:cqt_rtol ~atol ~shape:case.shape
            ~msg:(case.name ^ "/float32") ~expected:case.values
            (chroma_cqt config ~n_chroma ~norm x)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let case_of_kind (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "kind" with
  | "filterbank" ->
      filterbank_case case
  | "cqt_projection" ->
      projection_case case
  | "chroma_stft" ->
      spectrum_case case
  | "chroma_cqt" ->
      constant_q_case case
  | other ->
      test case.name (fun () -> failf "unknown golden kind %s" other)

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      group ("golden-" ^ stem) (List.map case_of_kind file.cases) )
    files
