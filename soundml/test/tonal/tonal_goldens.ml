(* Golden parity against librosa 1.0: librosa.feature.chroma_cens and
   librosa.feature.tonnetz, of given chromagrams and of constant-Q chromagrams
   computed from audio, replayed from the committed vectors. The inputs are
   rebuilt by Tonal_inputs. float32 cases quantize the input to float32 against
   a float64 reference.

   CENS of a given chromagram reaches librosa through its constant-Q path with
   an identity fold (bins_per_octave = n_chroma), so both sides quantise the
   same shares, computed by the same arithmetic. From audio, the chromagram
   carries the constant-Q departures documented under Cqt; every reference share
   was asserted by the generator to lie at least a margin from every threshold,
   many times the measured share deviation, so no quantisation decision differs
   and the CENS tolerances stay at double-precision rounding. The stages are
   checked against the flat functions at every given chromagram, bit for bit. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* CENS, float64, of given chromagrams and of audio alike. The quantised
   sequence is the same on both sides; what is left is the smoothing window —
   Window's exactly symmetric evaluation against scipy's linspace form, a
   rounding apart at some taps — its sum, and the final norm. Measured worst
   4.4e-16 absolute (the values lie in [0, 1], the unnormalised ones too, and a
   third of the cases agree bit for bit); the bound keeps a factor of ten. The
   float32 cases round the same values once, to within float32_atol (measured
   3.0e-8). *)
let cens64_atol = 5e-15

(* The tonal centroid of a given chromagram, float64: both sides build the
   projection by the same expressions, bit for bit; librosa adds the terms
   through Accelerate's matrix product, here in ascending order. Measured worst
   1.1e-16 absolute (the coordinates lie in [-1, 1]; every case but the batch
   agrees bit for bit); the bound keeps a factor of ten. *)
let tonnetz64_atol = 1e-15

(* The tonal centroid from audio follows the chromagram proportionally, so it
   inherits the constant-Q deviation: measured worst 2.0e-8 absolute at an odd
   hop, where neither side resamples (4.0e-8 from float32 audio), and 9.3e-5 at
   hop 512, where the octave recursion runs this library's resampler. Bounds at
   ten times the worst. *)
let tonnetz_odd_hop_atol = 4e-7

let tonnetz_even_hop_atol = 1e-3

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

let smoothing_of_param (case : Tutils.Golden.case) =
  match Tutils.Golden.param case "smoothing" with
  | `Null ->
      None
  | `Int l ->
      Some l
  | _ ->
      failf "golden case %s: smoothing is not an integer or null" case.name

let window_of_param (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "window" with
  | "hann" ->
      Window.Hann
  | "hamming" ->
      Window.Hamming
  | "boxcar" ->
      Window.Rectangular
  | "blackman" ->
      Window.Blackman
  | "kaiser" ->
      Window.Kaiser (Tutils.Golden.float_param case "beta")
  | other ->
      failf "golden case %s: unknown window %s" case.name other

let input_shape (case : Tutils.Golden.case) =
  match Tutils.Golden.param case "input_shape" with
  | `List l ->
      Array.of_list
        (List.map (function `Int i -> i | _ -> failf "%s: shape" case.name) l)
  | _ ->
      failf "%s: input_shape is not a list" case.name

let source dtype = Pipeline.Format.audio dtype ~sample_rate:22050 ~channels:1

let same_bits ~msg a b =
  equal ~msg (array float_exact) (Nx.to_array a) (Nx.to_array b)

let cqt_config (case : Tutils.Golden.case) =
  Cqt.Config.create
    ~fmin:(Tutils.Golden.float_param case "fmin")
    ~bins_per_octave:(Tutils.Golden.int_param case "bins_per_octave")
    ~tuning:(Tutils.Golden.float_param case "tuning")
    ~hop:(Tutils.Golden.int_param case "hop")
    ~pad:(`Constant 0.)
    ~n_bins:(Tutils.Golden.int_param case "n_bins")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    ()

(* A check run at the case's dtype: the dtype, the input values and the
   tolerances. *)
type check =
  {check: 'b. (float, 'b) Nx.dtype -> float array -> atol:float -> unit}

(* [at_dtype case ~atol values c] runs [c] at the case's dtype, on [values]
   quantized to float32 for the float32 cases, which also widen [atol] to
   [float32_atol] and take [float32_rtol]. *)
let at_dtype (case : Tutils.Golden.case) ~atol values c =
  match Tutils.Golden.string_param case "dtype" with
  | "float64" ->
      c.check Nx.float64 values ~atol
  | "float32" ->
      c.check Nx.float32
        (Array.map Tonal_inputs.quantize values)
        ~atol:(Float.max atol Tutils.float32_atol)
  | other ->
      failf "golden case %s: unknown dtype %s" case.name other

let rtol_of : type b. (float, b) Nx.dtype -> float = function
  | Nx.Float32 ->
      Tutils.float32_rtol
  | _ ->
      0.

let check_case (case : Tutils.Golden.case) ~atol got =
  Tutils.check_close
    ~rtol:(rtol_of (Nx.dtype got))
    ~atol ~shape:case.shape ~msg:case.name ~expected:case.values got

let cens_chroma_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let shape = input_shape case in
      let values =
        Tonal_inputs.chroma (Tutils.Golden.string_param case "chroma") shape
      in
      let norm = norm_of_param case
      and smoothing = smoothing_of_param case
      and smoothing_window = window_of_param case in
      at_dtype case ~atol:cens64_atol values
        { check=
            (fun dtype values ~atol ->
              let c = Nx.create dtype shape values in
              let got =
                chroma_cens_of_chroma ~norm ~smoothing ~smoothing_window c
              in
              check_case case ~atol got ;
              same_bits ~msg:(case.name ^ "/stage") got
                (Pipeline.run ~source:(source dtype)
                   (chroma_cens_stage ~norm ~smoothing ~smoothing_window ())
                   c ) ) } )

let cens_audio_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let n = Tutils.Golden.int_param case "length" in
      let y =
        Tonal_inputs.signal (Tutils.Golden.string_param case "signal") n
      in
      let cqt = cqt_config case in
      let n_chroma = Tutils.Golden.int_param case "n_chroma" in
      let norm = norm_of_param case
      and smoothing = smoothing_of_param case
      and smoothing_window = window_of_param case in
      at_dtype case ~atol:cens64_atol y
        { check=
            (fun dtype y ~atol ->
              check_case case ~atol
                (chroma_cens cqt ~n_chroma ~norm ~smoothing ~smoothing_window
                   (Nx.create dtype [|n|] y) ) ) } )

let tonnetz_chroma_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let shape = input_shape case in
      let values =
        Tonal_inputs.chroma (Tutils.Golden.string_param case "chroma") shape
      in
      at_dtype case ~atol:tonnetz64_atol values
        { check=
            (fun dtype values ~atol ->
              let c = Nx.create dtype shape values in
              let got = tonnetz_of_chroma c in
              check_case case ~atol got ;
              same_bits ~msg:(case.name ^ "/stage") got
                (Pipeline.run ~source:(source dtype) (tonnetz_stage ()) c) ) } )

let tonnetz_audio_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let n = Tutils.Golden.int_param case "length" in
      let y =
        Tonal_inputs.signal (Tutils.Golden.string_param case "signal") n
      in
      let cqt = cqt_config case in
      let n_chroma = Tutils.Golden.int_param case "n_chroma" in
      let atol =
        if Tutils.Golden.int_param case "hop" mod 2 = 1 then
          tonnetz_odd_hop_atol
        else tonnetz_even_hop_atol
      in
      at_dtype case ~atol y
        { check=
            (fun dtype y ~atol ->
              check_case case ~atol
                (tonnetz cqt ~n_chroma (Nx.create dtype [|n|] y)) ) } )

let case_of (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "kind" with
  | "cens_chroma" ->
      cens_chroma_case case
  | "cens_audio" ->
      cens_audio_case case
  | "tonnetz_chroma" ->
      tonnetz_chroma_case case
  | "tonnetz_audio" ->
      tonnetz_audio_case case
  | other ->
      failf "golden case %s: unknown kind %s" case.name other

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      group (Filename.basename file.path) (List.map case_of file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
