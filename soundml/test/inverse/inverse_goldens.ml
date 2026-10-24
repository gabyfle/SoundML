(* Golden parity against librosa 1.0's feature inversions
   (librosa.feature.inverse), replayed from the committed vectors. Every input
   is stored with its case, flattened in C order, so no forward feature is
   recomputed here; float32 cases quantize the input to float32 against a
   float64 reference.

   mel_to_stft is compared in two ways, as its parity contract states. The
   "start" cases are frames where librosa returns its clipped pseudo-inverse
   start, so the spectra themselves are compared. The "objective" cases compare
   fits instead: each frame's 1/2 |A S^2 - m|^2 must not exceed librosa's beyond
   rounding, among them tonal mfcc_to_mel envelopes whose start goes negative.
   The audio cases run Griffin-Lim on the start cases' spectra. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 tolerances, each about ten times the worst error measured here.
   mfcc_to_mel is one matrix and one power per value, against scipy's inverse
   DCT and numpy's power: 2.6e-15 relative. The start spectra differ only by the
   two pseudo-inverses (Nx's SVD against LAPACK's) and the square root: 1.4e-14
   relative on every bin a filter reaches, and the bins no filter reaches are
   exactly 0 on both sides (the generator pins librosa's ~1e-15 residue
   there). *)
let mfcc_to_mel_rtol = 3e-14

let start_rtol = 1e-13

(* An objective case passes when each frame's fit is at most librosa's, with
   room for rounding where both reach the optimum. Measured: ours 4e-34 to 4e-2
   of |m|^2, librosa's 1e-16 to 5e-2; on the rank-deficient filterbank, where
   L-BFGS-B reaches the optimum and both fits are the least-squares residual,
   ours exceeds librosa's by at most a relative 4.3e-13, and the slack is about
   ten times that. *)
let fit_slack = 5e-12

(* Griffin-Lim amplifies the 1e-14 agreement of the spectra through the bins of
   near-zero magnitude, whose phase is ill-defined, and does so erratically:
   measured relative to the signal's peak, 1.0e-12 after one iteration, 5.3e-12
   after eight and 5.5e-11 after thirty-two, the last 2.2e-10 with a
   pseudo-inverse rounded differently. The bounds are about ten times the worst
   of those, as an absolute tolerance scaled by the peak (a sample near zero has
   no meaningful relative error). *)
let audio_peak_atol n_iter =
  if n_iter <= 1 then 1e-11 else if n_iter <= 8 then 6e-11 else 3e-9

let ints case key =
  match Tutils.Golden.param case key with
  | `List l ->
      Array.of_list
        (List.map
           (function
             | `Int i ->
                 i
             | _ ->
                 failf "%s: %s is not a list of ints" case.Tutils.Golden.name
                   key )
           l )
  | _ ->
      failf "%s: %s is not a list" case.Tutils.Golden.name key

let floats case key =
  match Tutils.Golden.param case key with
  | `List l ->
      Array.of_list
        (List.map
           (function
             | `Float f ->
                 f
             | `Int i ->
                 Float.of_int i
             | _ ->
                 failf "%s: %s is not a list of floats" case.Tutils.Golden.name
                   key )
           l )
  | _ ->
      failf "%s: %s is not a list" case.Tutils.Golden.name key

let input dtype case =
  Nx.create dtype (ints case "input_shape") (floats case "input")

let has (case : Tutils.Golden.case) key = List.mem_assoc key case.params

let mel_config case =
  let bound key =
    if has case key then Some (Tutils.Golden.float_param case key) else None
  in
  Mel.Config.create ?f_min:(bound "f_min") ?f_max:(bound "f_max")
    ~n_mels:(Tutils.Golden.int_param case "n_mels")
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let stft_config case =
  Stft.Config.create ~pad:(`Constant 0.)
    ~hop:(Tutils.Golden.int_param case "hop")
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

(* A function at both dtypes, for the cases that run either. *)
type f = {f: 'a. (float, 'a) Nx.t -> (float, 'a) Nx.t}

let check ~rtol ~atol (case : Tutils.Golden.case) {f} =
  match Tutils.Golden.string_param case "dtype" with
  | "float64" ->
      Tutils.check_close ~rtol ~atol ~shape:case.shape ~msg:case.name
        ~expected:case.values
        (f (input Nx.float64 case))
  | "float32" ->
      Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
        ~shape:case.shape ~msg:case.name ~expected:case.values
        (f (input Nx.float32 case))
  | other ->
      failf "golden case %s: unknown dtype %s" case.name other

let mfcc_to_mel_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let mel = mel_config case in
      let lifter = Tutils.Golden.float_param case "lifter" in
      let reference = Tutils.Golden.float_param case "reference" in
      check ~rtol:mfcc_to_mel_rtol ~atol:0. case
        {f= (fun c -> mfcc_to_mel mel ~lifter ~reference c)} )

(* [fits mel m s] is each frame's [1/2 |A s^2 - m|^2] and [|m|^2], for [m]
   [[n_mels; frames]] and [s] [[bins; frames]], flattened in C order. *)
let fits mel m s =
  let a = Nx.to_array (Mel.filterbank Nx.float64 mel) in
  let bands = Mel.Config.n_mels mel and bins = Mel.Config.bins mel in
  let frames = Array.length m / bands in
  Array.init frames (fun t ->
      let f = ref 0. and n = ref 0. in
      for i = 0 to bands - 1 do
        let acc = ref 0. in
        for j = 0 to bins - 1 do
          let x = s.((j * frames) + t) in
          acc := !acc +. (a.((i * bins) + j) *. x *. x)
        done ;
        let b = m.((i * frames) + t) in
        let d = !acc -. b in
        f := !f +. (0.5 *. d *. d) ;
        n := !n +. (b *. b)
      done ;
      (!f, !n) )

let mel_to_stft_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let mel = mel_config case in
      let power = Tutils.Golden.float_param case "power" in
      match Tutils.Golden.string_param case "kind" with
      | "start" ->
          check ~rtol:start_rtol ~atol:0. case
            {f= (fun m -> mel_to_stft mel ~power m)}
      | "objective" ->
          let m = floats case "input" in
          let ours =
            fits mel m
              (Nx.to_array (mel_to_stft mel ~power (input Nx.float64 case)))
          in
          let theirs = fits mel m case.values in
          Array.iteri
            (fun t (f, n) ->
              let f_lib, _ = theirs.(t) in
              if not (f <= (f_lib *. (1. +. fit_slack)) +. (1e-20 *. n)) then
                failf "%s, frame %d: fit %g exceeds librosa's %g" case.name t f
                  f_lib )
            ours
      | other ->
          failf "golden case %s: unknown kind %s" case.name other )

let audio_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let stft = stft_config case and mel = mel_config case in
      let n_iter = Tutils.Golden.int_param case "n_iter" in
      let peak =
        Array.fold_left (fun a v -> Float.max a (Float.abs v)) 0. case.values
      in
      let atol = audio_peak_atol n_iter *. peak in
      match Tutils.Golden.string_param case "kind" with
      | "mel_to_audio" ->
          check ~rtol:0. ~atol case
            {f= (fun m -> mel_to_audio stft mel ~n_iter m)}
      | "mfcc_to_audio" ->
          check ~rtol:0. ~atol case
            {f= (fun c -> mfcc_to_audio stft mel ~n_iter c)}
      | other ->
          failf "golden case %s: unknown kind %s" case.name other )

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      let name = Filename.basename file.path in
      let run =
        match name with
        | "mfcc_to_mel.json" ->
            mfcc_to_mel_case
        | "mel_to_stft.json" ->
            mel_to_stft_case
        | "audio.json" ->
            audio_case
        | other ->
            fun _ -> failf "unexpected golden file %s" other
      in
      group name (List.map run file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
