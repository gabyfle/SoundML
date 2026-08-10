(* Golden parity against librosa 0.11 for the least-squares synthesis, replayed
   from the committed vectors under the suite's vectors/ directory. The spectra
   are two 31-bit LCG streams — one per complex component — reproduced here with
   the same integer arithmetic as the generator, so the float64 inputs are
   bit-identical; the imaginary parts of the DC and Nyquist bins are zero
   because the inverse real transform ignores them. float32 cases quantize both
   components exactly like the generator does.

   The spectra are inconsistent on purpose: no signal has exactly this
   transform, so every case measures the least-squares solution rather than a
   round trip, which the property suite covers separately.

   Synthesis reads no padding mode — the boundary extension is trimmed off, not
   recomputed — so the configurations below fix one and the padding grid lives
   in the property and Griffin-Lim suites. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 synthesis matches librosa's float64 pipeline up to FFT implementation
   differences, far below this tolerance — the stft suite's pair, for the same
   reason. *)
let float64_rtol = 1e-9

let float64_atol = 1e-12

let lcg_stream seed n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

(* [spectrum_values ~fft_size ~frames] is the generator's synthetic spectrum,
   flattened in C order over [[bins; frames]]. *)
let spectrum_values ~fft_size ~frames =
  let bins = (fft_size / 2) + 1 in
  let re = lcg_stream 20250803 (bins * frames) in
  let im = lcg_stream 20250804 (bins * frames) in
  Array.init (bins * frames) (fun k ->
      let bin = k / frames in
      let real_only = bin = 0 || (fft_size mod 2 = 0 && bin = bins - 1) in
      Complex.{re= re.(k); im= (if real_only then 0. else im.(k))} )

let alignment_of_param (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "alignment" with
  | "centered" ->
      `Centered
  | "left" ->
      `Left
  | "right" ->
      `Right
  | other ->
      failf "golden case %s: unknown alignment %s" case.name other

let config_of_params (case : Tutils.Golden.case) =
  Stft.Config.create
    ~win_length:(Tutils.Golden.int_param case "win_length")
    ~hop:(Tutils.Golden.int_param case "hop")
    ~alignment:(alignment_of_param case) ~pad:(`Constant 0.)
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let inverse_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let config = config_of_params case in
      let fft_size = Tutils.Golden.int_param case "fft_size" in
      let frames = Tutils.Golden.int_param case "frames" in
      let length =
        Option.map Yojson.Safe.Util.to_int (List.assoc_opt "length" case.params)
      in
      let shape = [|(fft_size / 2) + 1; frames|] in
      let values = spectrum_values ~fft_size ~frames in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let z = Nx.create Nx.complex128 shape values in
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (Stft.invert Nx.float64 config ?length z)
      | "float32" ->
          (* the component width matching the output dtype, which is what the
             generator quantizes the spectrum to *)
          let z = Nx.create Nx.complex64 shape values in
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (Stft.invert Nx.float32 config ?length z)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let suite =
  let files =
    Tutils.Golden.load_dir
      ~filter:(fun name -> not (String.starts_with ~prefix:"griffinlim" name))
      vectors_dir
  in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      group ("golden-" ^ stem) (List.map inverse_case file.cases) )
    files
