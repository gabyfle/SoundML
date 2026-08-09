(* Golden parity against librosa 0.11: librosa.feature.rms over audio and over
   magnitude spectrograms (the S= path), and librosa.feature.zero_crossing_rate,
   replayed from the committed vectors under the suite's vectors/ directory. The
   test input is the stft suite's 31-bit LCG under this suite's seed, reproduced
   with the same integer arithmetic; spectrogram cases take its absolute values
   reshaped in C order. float32 cases quantize the input to float32 against a
   float64 reference, exactly like the stft suite. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 tolerances by domain. The rms cases are linear-domain means of
   squares, matching librosa's float64 pipeline up to summation-order noise, far
   below 1e-9 relative. Zero-crossing counts are integers, exact in any
   summation order, and the closing division is the same operation on both
   sides, so those cases hold at the harness's near-exact defaults. *)
let rms64_rtol = 1e-9

let rms64_atol = 1e-12

let seed = 20260812

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

(* [signal_of_params case] is the audio input of [case]: [channels * length] LCG
   values, reshaped [[channels; length]] in C order when multichannel. *)
let signal_of_params dtype (case : Tutils.Golden.case) =
  let length = Tutils.Golden.int_param case "length" in
  let channels = Tutils.Golden.int_param case "channels" in
  let values = lcg_values (channels * length) in
  let shape = if channels > 1 then [|channels; length|] else [|length|] in
  Nx.create dtype shape values

(* [spectrogram_of_params case] is the magnitude-spectrogram input of [case]:
   absolute LCG values reshaped [[...; bins; frames]] in C order. *)
let spectrogram_of_params dtype (case : Tutils.Golden.case) =
  let bins = Tutils.Golden.int_param case "bins" in
  let frames = Tutils.Golden.int_param case "frames" in
  let channels = Tutils.Golden.int_param case "channels" in
  let values = Array.map Float.abs (lcg_values (channels * bins * frames)) in
  let shape =
    if channels > 1 then [|channels; bins; frames|] else [|bins; frames|]
  in
  Nx.create dtype shape values

let rms_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let frame_length = Tutils.Golden.int_param case "frame_length" in
      let hop = Tutils.Golden.int_param case "hop" in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~rtol:rms64_rtol ~atol:rms64_atol ~shape:case.shape
            ~msg:case.name ~expected:case.values
            (rms ~frame_length ~hop (signal_of_params Nx.float64 case))
      | "float32" ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (rms ~frame_length ~hop (signal_of_params Nx.float32 case))
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let rms_spectrogram_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let frame_length = Tutils.Golden.int_param case "frame_length" in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~rtol:rms64_rtol ~atol:rms64_atol ~shape:case.shape
            ~msg:case.name ~expected:case.values
            (rms_of_spectrogram ~frame_length
               (spectrogram_of_params Nx.float64 case) )
      | "float32" ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (rms_of_spectrogram ~frame_length
               (spectrogram_of_params Nx.float32 case) )
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let zcr_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let frame_length = Tutils.Golden.int_param case "frame_length" in
      let hop = Tutils.Golden.int_param case "hop" in
      (* a null threshold is omitted, so the librosa-default goldens pin the
         1e-10 default at the same time *)
      let threshold =
        match Tutils.Golden.param case "threshold" with
        | `Null ->
            None
        | json ->
            Some (Yojson.Safe.Util.to_number json)
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~shape:case.shape ~msg:case.name
            ~expected:case.values
            (zero_crossing_rate ~frame_length ~hop ?threshold
               (signal_of_params Nx.float64 case) )
      | "float32" ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (zero_crossing_rate ~frame_length ~hop ?threshold
               (signal_of_params Nx.float32 case) )
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      let case =
        match stem with
        | "rms" ->
            rms_case
        | "rms_spectrogram" ->
            rms_spectrogram_case
        | "zero_crossing_rate" ->
            zcr_case
        | other ->
            fun case ->
              test case.Tutils.Golden.name (fun () ->
                  failf "unknown golden file %s" other )
      in
      group ("golden-" ^ stem) (List.map case file.cases) )
    files
