(* Golden parity against librosa 1.0: harmonic/percussive separation, replayed
   from the committed vectors under the suite's vectors/ directory.

   The spectrogram-domain files (hpss, hpss_masks, hpss_boundary) rebuild their
   input bit-exactly — the stft suite's 31-bit LCG under this suite's seed,
   absolute value, the ridge and column constants added in the generator's
   order, and the silent top band — so the only imprecision under test is the
   separation itself. Their float32 cases carry a genuine float32 reference:
   separation selects values and multiplies them, with no accumulation whose
   order could differ, so a float32-in/float32-out reference is exactly
   reproducible, and the generator's own float32 run is the honest reference.
   That departs from the float64-reference convention of the stft and mel
   suites, which exists for chains that accumulate.

   The effects file goes through the STFT round trip, which does accumulate, and
   follows the usual convention instead: the signal is quantized to float32 and
   the reference computed in float64.

   Hard masks (power = infinity) are exact 0/1 decisions where one flipped
   comparison changes a whole bin, so they are checked by exact agreement rather
   than by tolerance: the fraction of bins where the mask matches the reference
   bit for bit must be 1.

   Every case is also replayed through the pipeline stages — the offline driver
   and a stream of one-frame pushes for the spectrogram files, the analysis,
   separation and synthesis chain for the effects file — under the same
   tolerances, so the stages' parity with the reference is checked directly and
   not only through their agreement with the offline faces. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 tolerances: separation is value selection followed by elementwise
   arithmetic on the selected values — no accumulation, no cancellation — so the
   float64 pipeline matches librosa far below 1e-9 relative. The pair is the one
   the spectral, onset and istft suites use. *)
let float64_rtol = 1e-9

let float64_atol = 1e-12

(* The 31-bit LCG of the stft suite, under the seed the generator records. *)
let lcg_signal n seed =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

(* The generator's deterministic spectrogram: one folded LCG draw per cell, plus
   a constant ridge on every seventh bin and a constant column on every fifth
   frame, in that order, with the top [silent_bins] bins zeroed. *)
let spectrogram ~bins ~frames ~seed ~silent_bins =
  let noise = lcg_signal (bins * frames) seed in
  Array.init (bins * frames) (fun i ->
      let b = i / frames and t = i mod frames in
      if b >= bins - silent_bins then 0.
      else
        let v = Float.abs noise.(i) in
        let v = v +. if b mod 7 = 3 then 3. else 0. in
        v +. if t mod 5 = 2 then 2. else 0. )

(* The batched cell stacks the spectrogram with its half. *)
let planed ~planes values =
  if planes = 1 then values
  else Array.concat [values; Array.map (fun v -> v *. 0.5) values]

let source_of (case : Tutils.Golden.case) =
  let bins = Tutils.Golden.int_param case "bins" in
  let frames = Tutils.Golden.int_param case "frames" in
  let planes = Tutils.Golden.int_param case "planes" in
  let values =
    planed ~planes
      (spectrogram ~bins ~frames
         ~seed:(Tutils.Golden.int_param case "seed")
         ~silent_bins:(Tutils.Golden.int_param case "silent_bins") )
  in
  let shape =
    if planes = 1 then [|bins; frames|] else [|planes; bins; frames|]
  in
  (shape, values)

let kernel_of (case : Tutils.Golden.case) =
  ( Tutils.Golden.int_param case "kernel_h"
  , Tutils.Golden.int_param case "kernel_p" )

let power_of (case : Tutils.Golden.case) =
  float_of_string (Tutils.Golden.string_param case "power")

let margin_of (case : Tutils.Golden.case) =
  ( Tutils.Golden.float_param case "margin_h"
  , Tutils.Golden.float_param case "margin_p" )

let component_of (case : Tutils.Golden.case) pair =
  match Tutils.Golden.string_param case "component" with
  | "harmonic" ->
      fst pair
  | "percussive" ->
      snd pair
  | other ->
      failf "golden case %s: unknown component %s" case.name other

(* Hard masks decide bin by bin, so agreement is counted, not measured: the
   fraction of bins equal to the reference bit for bit must be 1, and any
   shortfall is reported with the count of flipped bins. *)
let check_exact_agreement ~msg ~expected actual =
  let got = Nx.to_array (Nx.cast Nx.float64 actual) in
  if Array.length got <> Array.length expected then
    failf "%s: %d values, expected %d" msg (Array.length got)
      (Array.length expected) ;
  let flipped = ref 0 in
  Array.iteri (fun i v -> if v <> expected.(i) then incr flipped) got ;
  let total = Array.length expected in
  if !flipped <> 0 then
    failf "%s: mask agreement %.6f (%d of %d bins flipped)" msg
      (Float.of_int (total - !flipped) /. Float.of_int total)
      !flipped total

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:22050 ~channels:1

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let frames_of t start stop =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
    t

(* [streamed p x ~chunk] is [p] streamed over [x] in pushes of [chunk] items,
   its chunks joined by [join]. *)
let streamed p x ~chunk ~join =
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:chunk in
  let n = last_dim x in
  let pushed =
    List.filter_map (Pipeline.Stream.push s)
      (List.init
         ((n + chunk - 1) / chunk)
         (fun i -> frames_of x (i * chunk) (Int.min n ((i + 1) * chunk))) )
  in
  join (pushed @ Pipeline.Stream.flush s)

let join_pairs parts =
  ( Nx.concatenate ~axis:(-1) (List.map fst parts)
  , Nx.concatenate ~axis:(-1) (List.map snd parts) )

(* [separations ~masks ... s] is the separation of [s] by the offline face, by
   its stage under the offline driver, and by its stage streamed one frame at a
   time, each labelled. *)
let separations ~masks ~kernel_size ~power ~margin s =
  let stage () =
    if masks then hpss_masks_stage ~kernel_size ~power ~margin ()
    else hpss_of_spectrogram_stage ~kernel_size ~power ~margin ()
  in
  [ ( ""
    , if masks then hpss_masks ~kernel_size ~power ~margin s
      else hpss_of_spectrogram ~kernel_size ~power ~margin s )
  ; (" (stage, run)", Pipeline.run ~source:(source ()) (stage ()) s)
  ; ( " (stage, one-frame pushes)"
    , streamed (stage ()) s ~chunk:1 ~join:join_pairs ) ]

let run_spectrogram_case ~masks (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let shape, values = source_of case in
      let kernel_size = kernel_of case in
      let power = power_of case in
      let margin = margin_of case in
      let faces s = separations ~masks ~kernel_size ~power ~margin s in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          List.iter
            (fun (how, pair) ->
              let msg = case.name ^ how and actual = component_of case pair in
              if masks && not (Float.is_finite power) then
                check_exact_agreement ~msg ~expected:case.values actual
              else
                Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
                  ~shape:case.shape ~msg ~expected:case.values actual )
            (faces (Nx.create Nx.float64 shape values))
      | "float32" ->
          List.iter
            (fun (how, pair) ->
              let msg = case.name ^ how and actual = component_of case pair in
              if masks && not (Float.is_finite power) then
                check_exact_agreement ~msg ~expected:case.values actual
              else
                Tutils.check_close ~rtol:Tutils.float32_rtol
                  ~atol:Tutils.float32_atol ~shape:case.shape ~msg
                  ~expected:case.values actual )
            (faces (Nx.create Nx.float32 shape values))
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let run_effects_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let length = Tutils.Golden.int_param case "length" in
      let config =
        Stft.Config.create
          ~fft_size:(Tutils.Golden.int_param case "fft_size")
          ~hop:(Tutils.Golden.int_param case "hop")
          ~pad:(`Constant 0.) ()
      in
      let signal = lcg_signal length (Tutils.Golden.int_param case "seed") in
      let kernel_size = kernel_of case in
      let power = power_of case in
      let margin = margin_of case in
      let face x =
        match Tutils.Golden.string_param case "face" with
        | "hpss" ->
            component_of case (hpss config ~kernel_size ~power ~margin x)
        | "harmonic" ->
            harmonic config ~kernel_size ~power ~margin x
        | "percussive" ->
            percussive config ~kernel_size ~power ~margin x
        | other ->
            failf "golden case %s: unknown face %s" case.name other
      in
      (* The streamed chain ends at the synthesis's default length, which is the
         input's own here: centred frames on a hop that divides it. *)
      let pick pair =
        match Tutils.Golden.string_param case "face" with
        | "harmonic" ->
            fst pair
        | "percussive" ->
            snd pair
        | _ ->
            component_of case pair
      in
      let chain cdtype dtype =
        Pipeline.(
          Stft.stage cdtype config
          >> map pick (hpss_of_stft_stage ~kernel_size ~power ~margin ())
          >> Stft.synthesis_stage dtype config )
      in
      let join parts = Nx.concatenate ~axis:(-1) parts in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|length|] signal in
          List.iter
            (fun (how, actual) ->
              Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
                ~shape:case.shape ~msg:(case.name ^ how) ~expected:case.values
                actual )
            [ ("", face x)
            ; ( " (stage chain, run)"
              , Pipeline.run ~source:(source ())
                  (chain Nx.complex128 Nx.float64)
                  x )
            ; ( " (stage chain, 100-sample pushes)"
              , streamed (chain Nx.complex128 Nx.float64) x ~chunk:100 ~join )
            ]
      | "float32" ->
          let x = Nx.create Nx.float32 [|length|] signal in
          List.iter
            (fun (how, actual) ->
              Tutils.check_close ~rtol:Tutils.float32_rtol
                ~atol:Tutils.float32_atol ~shape:case.shape
                ~msg:(case.name ^ how) ~expected:case.values actual )
            [ ("", face x)
            ; ( " (stage chain, run)"
              , Pipeline.run ~source:(source ())
                  (chain Nx.complex64 Nx.float32)
                  x )
            ; ( " (stage chain, 100-sample pushes)"
              , streamed (chain Nx.complex64 Nx.float32) x ~chunk:100 ~join ) ]
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let case_of stem =
  match stem with
  | "hpss" | "hpss_boundary" ->
      run_spectrogram_case ~masks:false
  | "hpss_masks" ->
      run_spectrogram_case ~masks:true
  | "hpss_effects" ->
      run_effects_case
  | other ->
      fun (case : Tutils.Golden.case) ->
        test case.name (fun () -> failf "unknown golden file %s" other)

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      group ("golden-" ^ stem) (List.map (case_of stem) file.cases) )
    files
