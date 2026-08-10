(* Golden parity against librosa 0.11 for Griffin-Lim phase reconstruction, at
   the settings where the reference is deterministic: the all-ones initial
   phase, which is our [`Zero_phase], and zero padding, which is what the
   reference pads with. The padding mode is geometry here rather than a free
   knob — the iteration re-analyses what it synthesises, so it reads [pad] — and
   is therefore pinned to the reference's rather than varied; the property suite
   crosses the three modes. The reference's default initial phase is random, so
   no committed vector could pin it; the property suite carries the rest.

   The magnitudes are one 31-bit LCG stream shifted into [0, 2), reproduced here
   with the same integer arithmetic as the generator. They are inconsistent — no
   signal has exactly these magnitudes — so every iteration moves, and the cases
   cross the classic algorithm (momentum 0) with the accelerated one at several
   iteration counts. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* The iteration is a non-expansive map, so rounding accumulates linearly in
   [n_iter] rather than compounding: at 32 iterations the worst case still sits
   more than an order of magnitude inside the stft suite's float64 pair, which
   this reuses. *)
let float64_rtol = 1e-9

let float64_atol = 1e-12

let lcg_stream seed n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let magnitude_values ~fft_size ~frames =
  let bins = (fft_size / 2) + 1 in
  Array.map (fun v -> v +. 1.) (lcg_stream 20250803 (bins * frames))

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

let griffin_lim_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let fft_size = Tutils.Golden.int_param case "fft_size" in
      let frames = Tutils.Golden.int_param case "frames" in
      let config =
        Stft.Config.create
          ~win_length:(Tutils.Golden.int_param case "win_length")
          ~hop:(Tutils.Golden.int_param case "hop")
          ~alignment:(alignment_of_param case) ~pad:(`Constant 0.) ~fft_size ()
      in
      let n_iter = Tutils.Golden.int_param case "n_iter" in
      let momentum = Tutils.Golden.float_param case "momentum" in
      let shape = [|(fft_size / 2) + 1; frames|] in
      let values = magnitude_values ~fft_size ~frames in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (Stft.griffin_lim ~n_iter ~momentum ~init:`Zero_phase config
               (Nx.create Nx.float64 shape values) )
      | "float32" ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (Stft.griffin_lim ~n_iter ~momentum ~init:`Zero_phase config
               (Nx.create Nx.float32 shape values) )
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let suite =
  let files =
    Tutils.Golden.load_dir
      ~filter:(String.starts_with ~prefix:"griffinlim")
      vectors_dir
  in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      group ("golden-" ^ stem) (List.map griffin_lim_case file.cases) )
    files
