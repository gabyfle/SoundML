(* Golden parity against librosa 0.11: constant-Q and variable-Q magnitudes
   (librosa.vqt, of which librosa.cqt is the gamma = 0 instance), the frequency
   ladder (librosa.cqt_frequencies), the fractional filter lengths and main-lobe
   cutoffs (librosa.filters.wavelet_lengths), librosa's own early-downsampling
   schedule and its Nyquist admissibility boundary, replayed from the committed
   vectors under the suite's vectors/ directory.

   Two test signals, both reproduced in OCaml: the stft suite's 31-bit LCG under
   this suite's seed, bit for bit; and a harmonic recipe of three decaying notes
   over a transient and a noise floor, whose sines and exponentials carry a libm
   ulp of variance the wide tier's tolerance absorbs. float32 cases quantize the
   signal to float32 against a float64 reference, exactly like the stft suite.

   Tolerances are quoted against each case's own peak — [max |expected|] — the
   unit the deviations below were measured in. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* Cases whose octave recursion never resamples: an odd hop stops the halving
   after the first octave, and a single-octave configuration never halves at
   all. Nothing but double-precision arithmetic separates the two
   implementations there, and the measured worst over the tight file is 1.13e-7
   of the case peak (t3, the erb variable-Q ladder). *)
let tight_atol = 1e-6

(* float32 cases carry the storage granularity of the result on top: at a peak
   near two, one float32 ulp is already 6e-8 of peak. Measured worst 1.05e-7 —
   the same double-precision floor, since the interior runs in double either
   way. *)
let tight32_atol = 5e-6

(* Cases that do resample. SoundML decimates each octave with its own resampler,
   librosa with SoX Resampler at soxr_hq; two filters meeting one specification
   are not the same filter. Measured worst over the wide file: 1.08e-4 of peak
   on the harmonic signal (w4, variable-Q at gamma = 20) and 1.40e-4 on the
   short broadband cases (s1, 1024 samples). For scale, running the same
   comparison with soxr's own next quality tier in place of soxr_hq moves those
   two cases by 2.28e-4 and 4.14e-4 — SoundML sits closer to librosa's default
   than librosa's own higher tier does. *)
let wide_atol = 6e-4

(* Relative slack on top of the peak-anchored absolute term, for the entries
   near the peak where a relative comparison is the meaningful one. *)
let rtol = 1e-5

let seed = 20260803

let lcg n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

(* The generator's harmonic recipe: three decaying harmonic notes whose onsets
   are placed as fractions of the signal, a transient a quarter of the way in,
   and a low noise floor. *)
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

let peak_of values =
  Array.fold_left (fun acc v -> Float.max acc (Float.abs v)) 0. values

let optional_param (case : Tutils.Golden.case) key =
  List.assoc_opt key case.params

let gamma_of_param (case : Tutils.Golden.case) =
  match optional_param case "gamma" with
  | None ->
      `Constant_q
  | Some json -> (
      let text = Yojson.Safe.Util.to_string json in
      match text with
      | "erb" ->
          `Erb
      | "0.0" ->
          `Constant_q
      | other -> (
        match float_of_string_opt other with
        | Some g ->
            `Fixed g
        | None ->
            failf "golden case %s: unknown gamma %s" case.name other ) )

let window_of_param (case : Tutils.Golden.case) =
  match optional_param case "window" with
  | None ->
      Window.Hann
  | Some json -> (
    match Yojson.Safe.Util.to_string json with
    | "hann" ->
        Window.Hann
    | "blackmanharris" ->
        Window.Blackman_harris
    | "kaiser" ->
        Window.Kaiser (Tutils.Golden.float_param case "beta")
    | other ->
        failf "golden case %s: unknown window %s" case.name other )

let filter_scale_of_param (case : Tutils.Golden.case) =
  match optional_param case "filter_scale" with
  | None ->
      1.
  | Some json ->
      Yojson.Safe.Util.to_number json

let hop_of_param (case : Tutils.Golden.case) =
  match optional_param case "hop" with
  | None ->
      512
  | Some json ->
      Yojson.Safe.Util.to_int json

let scale_of_param (case : Tutils.Golden.case) =
  match optional_param case "scale" with
  | None ->
      true
  | Some json ->
      Yojson.Safe.Util.to_bool json

let config_of_params (case : Tutils.Golden.case) =
  Cqt.Config.create
    ~fmin:(Tutils.Golden.float_param case "fmin")
    ~bins_per_octave:(Tutils.Golden.int_param case "bins_per_octave")
    ~gamma:(gamma_of_param case)
    ~tuning:(Tutils.Golden.float_param case "tuning")
    ~window:(window_of_param case)
    ~filter_scale:(filter_scale_of_param case)
    ~hop:(hop_of_param case) ~scale:(scale_of_param case)
    ~n_bins:(Tutils.Golden.int_param case "n_bins")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    ()

(* {2 Transform cases} *)

let transform_case ~atol64 ~atol32 (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let sample_rate = Tutils.Golden.int_param case "sample_rate" in
      let n = Tutils.Golden.int_param case "length" in
      let signal =
        match Tutils.Golden.string_param case "signal" with
        | "lcg" ->
            lcg n
        | "harmonic" ->
            harmonic n sample_rate
        | other ->
            failf "golden case %s: unknown signal %s" case.name other
      in
      let config = config_of_params case in
      let peak = peak_of case.values in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|n|] signal in
          Tutils.check_close ~rtol ~atol:(atol64 *. peak) ~shape:case.shape
            ~msg:case.name ~expected:case.values
            (Cqt.power_spectrum ~power:1. config x)
      | "float32" ->
          let x = Nx.create Nx.float32 [|n|] (Array.map quantize signal) in
          Tutils.check_close ~rtol ~atol:(atol32 *. peak) ~shape:case.shape
            ~msg:case.name ~expected:case.values
            (Cqt.power_spectrum ~power:1. config x)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

(* {2 Support cases}

   The ladder, the filter lengths and the cutoff are closed-form doubles: both
   sides evaluate the same expression, so the default float64 grade applies. The
   ladder is walked octave by octave here and in one step by librosa's
   [cqt_frequencies] helper, which moves at most three of the last bits — four
   orders of magnitude inside that grade. The early-downsampling count and the
   Nyquist verdict are integers, compared exactly. *)

let ladder_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let config = config_of_params case in
      Tutils.check_close ~shape:case.shape ~msg:case.name ~expected:case.values
        (Cqt.frequencies Nx.float64 config) ;
      Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
        ~shape:case.shape ~msg:(case.name ^ "/float32") ~expected:case.values
        (Cqt.frequencies Nx.float32 config) )

let lengths_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      Tutils.check_close ~shape:case.shape ~msg:case.name ~expected:case.values
        (Cqt.filter_lengths Nx.float64 (config_of_params case)) )

let cutoff_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let config = config_of_params case in
      Tutils.check_close ~shape:[|1|] ~msg:case.name ~expected:case.values
        (Nx.create Nx.float64 [|1|] [|Cqt.Config.cutoff config|]) )

(* The early-decimation count is not an accessor; it is read back from the frame
   grid, which is where it is user-visible. Octave zero analyses [ceil (n / 2 ^
   e)] samples at hop [hop / 2 ^ e] and is the octave with the fewest frames, so
   the shortest signal that yields a second frame has [n = hop - 2 ^ e + 1]. *)
let early_of config =
  let hop = Cqt.Config.hop config in
  let rec search n =
    if n > hop then failf "early: no second frame at or below one hop"
    else if Cqt.frames config ~n >= 2 then hop + 1 - n
    else search (n + 1)
  in
  Float.log2 (Float.of_int (search 1))

let early_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      equal ~msg:case.name (float 0.) case.values.(0)
        (early_of (config_of_params case)) )

let nyquist_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let accepted =
        match config_of_params case with
        | _ ->
            true
        | exception Invalid_argument _ ->
            false
      in
      equal ~msg:case.name (float 0.) case.values.(0)
        (if accepted then 1. else 0.) )

let support_case (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "kind" with
  | "frequencies" ->
      ladder_case case
  | "filter_lengths" ->
      lengths_case case
  | "cutoff" ->
      cutoff_case case
  | "early" ->
      early_case case
  | "nyquist" ->
      nyquist_case case
  | other ->
      test case.name (fun () -> failf "unknown support kind %s" other)

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      let case =
        match stem with
        | "cqt_tight" ->
            transform_case ~atol64:tight_atol ~atol32:tight32_atol
        | "cqt_wide" ->
            transform_case ~atol64:wide_atol ~atol32:wide_atol
        | "cqt_support" ->
            support_case
        | other ->
            fun case ->
              test case.Tutils.Golden.name (fun () ->
                  failf "unknown golden file %s" other )
      in
      group ("golden-" ^ stem) (List.map case file.cases) )
    files
