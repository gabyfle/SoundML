(* Golden parity against librosa 0.11: magnitude and power spectra, frame times
   and bin frequencies, replayed from the committed vectors under
   test/vectors/stft/. The test signal is a 31-bit LCG reproduced here with the
   same integer arithmetic as the generator, so the float64 inputs are
   bit-identical; float32 cases quantize the signal exactly like the generator
   does. *)

open Windtrap
open Soundml

let vectors_dir = Filename.concat ".." (Filename.concat "vectors" "stft")

(* float64 spectra: the interior matches librosa's float64 pipeline up to FFT
   implementation differences, far below this tolerance. *)
let float64_rtol = 1e-9

let float64_atol = 1e-12

let lcg_signal n =
  let state = ref 20250803 in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let alignment_of_param case =
  match Tutils.Golden.string_param case "alignment" with
  | "centered" ->
      `Centered
  | "left" ->
      `Left
  | other ->
      failf "golden case %s: unknown alignment %s" case.name other

let config_of_params (case : Tutils.Golden.case) =
  Stft.Config.create
    ~win_length:(Tutils.Golden.int_param case "win_length")
    ~hop:(Tutils.Golden.int_param case "hop")
    ~alignment:(alignment_of_param case)
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let power_of_kind case =
  match Tutils.Golden.string_param case "kind" with
  | "magnitude" ->
      1.
  | "power" ->
      2.
  | other ->
      failf "golden case %s: unknown kind %s" case.name other

let spectrum_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let config = config_of_params case in
      let signal = lcg_signal (Tutils.Golden.int_param case "length") in
      let power = power_of_kind case in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          let x = Nx.create Nx.float64 [|Array.length signal|] signal in
          Tutils.check_close ~rtol:float64_rtol ~atol:float64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (Stft.power_spectrum ~power config x)
      | "float32" ->
          let x = Nx.create Nx.float32 [|Array.length signal|] signal in
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (Stft.power_spectrum ~power config x) ;
          (* the complex witness path: a complex64 transform's magnitudes carry
             the same values through the boundary cast *)
          if Float.equal power 1. then
            Tutils.check_close ~rtol:Tutils.float32_rtol
              ~atol:Tutils.float32_atol ~shape:case.shape
              ~msg:(case.name ^ "/complex64-witness")
              ~expected:case.values
              (Nx.cast Nx.float32
                 (Nx.abs (Stft.transform Nx.complex64 config x)) )
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let coordinate_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let sample_rate = Tutils.Golden.int_param case "sample_rate" in
      match Tutils.Golden.string_param case "kind" with
      | "frequencies" ->
          let config =
            Stft.Config.create
              ~hop:(Tutils.Golden.int_param case "hop")
              ~fft_size:(Tutils.Golden.int_param case "fft_size")
              ()
          in
          Tutils.check_close ~shape:case.shape ~msg:case.name
            ~expected:case.values
            (Stft.frequencies Nx.float64 config ~sample_rate) ;
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:(case.name ^ "/float32")
            ~expected:case.values
            (Stft.frequencies Nx.float32 config ~sample_rate)
      | "times" ->
          let config =
            Stft.Config.create
              ~hop:(Tutils.Golden.int_param case "hop")
              ~alignment:(alignment_of_param case)
              ~fft_size:(Tutils.Golden.int_param case "fft_size")
              ()
          in
          let n = Tutils.Golden.int_param case "length" in
          Tutils.check_close ~shape:case.shape ~msg:case.name
            ~expected:case.values
            (Stft.times Nx.float64 config ~sample_rate ~n) ;
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:(case.name ^ "/float32")
            ~expected:case.values
            (Stft.times Nx.float32 config ~sample_rate ~n)
      | other ->
          failf "golden case %s: unknown kind %s" case.name other )

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      let case =
        if String.equal stem "coordinates" then coordinate_case
        else spectrum_case
      in
      group ("golden-" ^ stem) (List.map case file.cases) )
    files
