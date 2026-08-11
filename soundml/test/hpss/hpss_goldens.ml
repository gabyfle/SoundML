(* Golden parity against librosa 0.11: harmonic/percussive separation, replayed
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
   bit for bit must be 1. *)

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

let run_spectrogram_case ~masks (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let shape, values = source_of case in
      let kernel_size = kernel_of case in
      let power = power_of case in
      let margin = margin_of case in
      let face s =
        component_of case
          ( if masks then hpss_masks ~kernel_size ~power ~margin s
            else hpss_of_spectrogram ~kernel_size ~power ~margin s )
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let actual = face (Nx.create Nx.float64 shape values) in
          if masks && not (Float.is_finite power) then
            check_exact_agreement ~msg:case.name ~expected:case.values actual
          else
            Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
              ~shape:case.shape ~msg:case.name ~expected:case.values actual
      | "float32" ->
          let actual = face (Nx.create Nx.float32 shape values) in
          if masks && not (Float.is_finite power) then
            check_exact_agreement ~msg:case.name ~expected:case.values actual
          else
            Tutils.check_close ~rtol:Tutils.float32_rtol
              ~atol:Tutils.float32_atol ~shape:case.shape ~msg:case.name
              ~expected:case.values actual
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
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (face (Nx.create Nx.float64 [|length|] signal))
      | "float32" ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (face (Nx.create Nx.float32 [|length|] signal))
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
