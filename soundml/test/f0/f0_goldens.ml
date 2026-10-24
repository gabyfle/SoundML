(* Golden parity against librosa 1.0: librosa.yin and librosa.pyin over the
   synthetic signals of F0VectorGenerator (see f0_signals.ml), replayed from the
   committed vectors. float32 cases quantize the input to float32 against a
   float64 reference, like the energy suite.

   What the tolerances cover. Every discrete decision — the troughs, the
   absolute threshold, the global minimum, the pyin thresholds, the parabola's
   branch, the pitch-bin rounding and the Viterbi maxima — must come out the
   same, and no tolerance could absorb one that did not: the generator asserts
   that each lies at least 1e-9 from its tie on librosa's own difference
   function (5.6e-7 at the closest) and that a 2 ** -40 relative perturbation of
   the input moves none of them, far beyond the few ulps by which the two
   difference functions differ. What is left is continuous: a refined period
   carries the rounding of the FFT autocorrelation (Nx's own transform, where
   librosa runs scipy's pocketfft) into the parabola's vertex, and a voiced
   probability the order of its sums. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* yin estimates. The two agree within 3.9e-15 relative on every case here (the
   uncentered 44.1 kHz chirp); the bound keeps a factor of ten over it. *)
let yin_rtol = 4e-14

(* pyin frequencies are values of the pitch grid [fmin * 2 ** (k / bins per
   octave)]: once the decoded states agree they agree bit for bit here (or are
   both NaN), as the voiced flags do, since numpy's power is this machine's libm
   pow. The bound allows another platform's pow an ulp or so. *)
let pyin_f0_rtol = 1e-15

(* Voiced probabilities. They agree within 8.9e-16 absolute (librosa's prior is
   a BLAS product, and it sums a one-frame signal pairwise, here in ascending
   order); the bound keeps a factor of ten over it. *)
let pyin_probability_atol = 1e-14

let has (case : Tutils.Golden.case) key = List.mem_assoc key case.params

let int_or case key default =
  if has case key then Tutils.Golden.int_param case key else default

let float_or case key default =
  if has case key then Tutils.Golden.float_param case key else default

let alignment center = if center then `Centered else `Left

let pad (case : Tutils.Golden.case) =
  match
    if has case "pad" then Tutils.Golden.string_param case "pad" else "constant"
  with
  | "constant" ->
      `Constant (float_or case "pad_value" 0.)
  | "edge" ->
      `Edge
  | "reflect" ->
      `Reflect
  | other ->
      failf "%s: unknown pad %s" case.name other

let input dtype (case : Tutils.Golden.case) =
  F0_signals.of_params dtype
    ~sr:(Tutils.Golden.int_param case "sample_rate")
    ~lead:(Tutils.Golden.param case "lead")
    ~channels:(Tutils.Golden.param case "channels")

let geometry (case : Tutils.Golden.case) =
  let frame_length = int_or case "frame_length" 2048 in
  ( frame_length
  , int_or case "hop" (frame_length / 4)
  , (if has case "center" then Tutils.Golden.bool_param case "center" else true)
  , pad case )

let run_yin case x =
  let frame_length, hop, center, pad = geometry case in
  yin ~frame_length ~hop ~alignment:(alignment center) ~pad
    ~trough_threshold:(float_or case "trough_threshold" 0.1)
    ~fmin:(Tutils.Golden.float_param case "fmin")
    ~fmax:(Tutils.Golden.float_param case "fmax")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    x

let run_pyin (case : Tutils.Golden.case) x =
  let frame_length, hop, center, pad = geometry case in
  let beta_parameters =
    match
      if has case "beta_parameters" then
        Some (Tutils.Golden.param case "beta_parameters")
      else None
    with
    | None ->
        (2., 18.)
    | Some (`List [a; b]) ->
        (F0_signals.number a, F0_signals.number b)
    | Some _ ->
        failf "%s: malformed beta_parameters" case.name
  in
  let fill =
    match List.assoc_opt "fill" case.params with
    | None ->
        `Value Float.nan
    | Some (`String "decoded") ->
        `Decoded
    | Some v ->
        `Value (F0_signals.number v)
  in
  pyin ~frame_length ~hop ~alignment:(alignment center) ~pad
    ~n_thresholds:(int_or case "n_thresholds" 100)
    ~beta_parameters
    ~boltzmann_parameter:(float_or case "boltzmann_parameter" 2.)
    ~resolution:(float_or case "resolution" 0.1)
    ~max_transition_rate:(float_or case "max_transition_rate" 35.92)
    ~switch_prob:(float_or case "switch_prob" 0.01)
    ~no_trough_prob:(float_or case "no_trough_prob" 0.01)
    ~transition_min_prob:(float_or case "transition_min_prob" 1e-4)
    ~fill
    ~fmin:(Tutils.Golden.float_param case "fmin")
    ~fmax:(Tutils.Golden.float_param case "fmax")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    x

let dtype_tolerances (case : Tutils.Golden.case) ~rtol ~atol =
  match Tutils.Golden.string_param case "dtype" with
  | "float64" ->
      (rtol, atol)
  | "float32" ->
      (Tutils.float32_rtol, Tutils.float32_atol)
  | other ->
      failf "golden case %s: unknown dtype %s" case.name other

let yin_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let rtol, atol = dtype_tolerances case ~rtol:yin_rtol ~atol:0. in
      let check x =
        Tutils.check_close ~rtol ~atol ~shape:case.shape ~msg:case.name
          ~expected:case.values (run_yin case x)
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float32" ->
          check (input Nx.float32 case)
      | _ ->
          check (input Nx.float64 case) )

(* A pyin golden stacks f0, the voiced flag as 0/1 and the voiced probability on
   a new leading axis. *)
let pyin_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let block = Array.length case.values / 3 in
      let part k = Array.sub case.values (k * block) block in
      let shape = Array.sub case.shape 1 (Array.length case.shape - 1) in
      let f0_rtol, f0_atol =
        dtype_tolerances case ~rtol:pyin_f0_rtol ~atol:0.
      in
      let p_rtol, p_atol =
        dtype_tolerances case ~rtol:0. ~atol:pyin_probability_atol
      in
      let check (type a) (x : (float, a) Nx.t) =
        let r = run_pyin case x in
        Tutils.check_close ~rtol:f0_rtol ~atol:f0_atol ~shape
          ~msg:(case.name ^ " f0") ~expected:(part 0) r.f0 ;
        Tutils.check_close ~rtol:0. ~atol:0. ~shape ~msg:(case.name ^ " voiced")
          ~expected:(part 1)
          (Nx.cast Nx.float64 r.voiced) ;
        Tutils.check_close ~rtol:p_rtol ~atol:p_atol ~shape
          ~msg:(case.name ^ " voiced_probability")
          ~expected:(part 2) r.voiced_probability
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float32" ->
          check (input Nx.float32 case)
      | _ ->
          check (input Nx.float64 case) )

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      let replay =
        if Filename.basename file.path = "pyin.json" then pyin_case
        else yin_case
      in
      group (Filename.basename file.path) (List.map replay file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
