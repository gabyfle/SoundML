(* Golden parity against librosa 1.0: librosa.piptrack over |LCG| spectrograms
   under this suite's seed (reshaped in C order, reproduced bit-exactly here)
   and over STFT magnitude and power spectrograms of tones; librosa.pitch_tuning
   over the frequency sets stored in the params; and librosa.estimate_tuning
   over tones, mixtures and noise, by audio and by power spectrogram — replayed
   from the committed vectors. float32 cases quantize the input (the
   spectrogram, or the audio) to float32 against a float64 reference.

   The generator keeps every discrete decision of the STFT cases away from its
   tie (see its docstring), so the peak supports and the tuning estimates
   compare exactly; only the interpolated values carry a tolerance. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* Tolerances. On |LCG| spectrograms both sides evaluate the same IEEE
   operations on the same doubles — the reference value, the rfftfreq band, the
   parabola, [x0 + (0.5 b) shift] and [((k + shift) sr) / n] — and agree bit for
   bit (measured 0 on every case), so those cases compare exactly.

   The STFT cases inherit the two transforms' rounding (and the two libms' sines
   beneath the test signals), about 1e-16 of the frame peak per bin. Measured
   worst: pitches 2.9e-16 relative, magnitudes 6.4e-16 of the case's largest
   magnitude (the power spectrogram); the bounds keep a factor of ten over that.
   float32 cases use the library-wide float32 tolerances, the magnitudes'
   relative to the case's largest: the outputs are the float64 values rounded
   once, measured within 5.7e-8 relative. *)
let stft_pitch_rtol = 3e-15

let stft_mag_rtol = 7e-15

let seed = 20261107

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let ints_of json =
  match json with
  | `List l ->
      Array.of_list (List.map Yojson.Safe.Util.to_int l)
  | _ ->
      failf "not a list of integers"

let floats_of json =
  match json with
  | `List l ->
      Array.of_list (List.map Yojson.Safe.Util.to_number l)
  | _ ->
      failf "not a list of numbers"

(* [audio spec] rebuilds the generator's signal: per channel, the partials [a *
   sin (2 pi f t)] summed in order at [t = i / sample_rate], then the noise, its
   level times the LCG stream — numpy's expressions and order, each product
   rounded on its own so no platform fuses it into the sum. *)
let audio spec =
  let open Yojson.Safe.Util in
  let n = to_int (member "n" spec) in
  let sr = Float.of_int (to_int (member "sample_rate" spec)) in
  let channel c =
    let y = Array.make n 0. in
    List.iter
      (fun partial ->
        match partial with
        | `List [f; a] ->
            let f = to_number f and a = to_number a in
            for i = 0 to n - 1 do
              let t = Float.of_int i /. sr in
              let v = a *. Float.sin (2. *. Float.pi *. f *. t) in
              y.(i) <- y.(i) +. v
            done
        | _ ->
            failf "malformed partial" )
      (to_list (member "partials" c)) ;
    let noise = to_number (member "noise" c) in
    if noise <> 0. then begin
      let stream = lcg_values n in
      Array.iteri
        (fun i s ->
          let v = noise *. s in
          y.(i) <- y.(i) +. v )
        stream
    end ;
    y
  in
  match to_list (member "channels" spec) with
  | [c] ->
      ([|n|], channel c)
  | cs ->
      (Array.of_list [List.length cs; n], Array.concat (List.map channel cs))

(* [signal dtype case] is the case's audio in [dtype]: float32 values are the
   float64 ones rounded once, as the generator quantizes them. *)
let signal dtype (case : Tutils.Golden.case) =
  let shape, values = audio (Tutils.Golden.param case "signal") in
  Nx.create dtype shape values

let stft_config (case : Tutils.Golden.case) =
  Stft.Config.create
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ~hop:(Tutils.Golden.int_param case "hop")
    ~pad:(`Constant 0.) ()

let has (case : Tutils.Golden.case) key = List.mem_assoc key case.params

let opt_float case key =
  if has case key then Some (Tutils.Golden.float_param case key) else None

let reference case = Option.map (fun r -> `Value r) (opt_float case "reference")

(* {1 piptrack} *)

(* [spectrogram case] is the case's float64 spectrogram, before any
   quantization: the |LCG| values (floored to quarters for the plateau case) or
   the STFT magnitudes raised to the case's power. *)
let spectrogram (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "source" with
  | ("lcg" | "plateau") as source ->
      let shape = ints_of (Tutils.Golden.param case "input_shape") in
      let n = Array.fold_left ( * ) 1 shape in
      let values =
        Array.map
          (fun v ->
            let m = Float.abs v in
            if source = "plateau" then Float.floor (m *. 4.) /. 4. else m )
          (lcg_values n)
      in
      Nx.create Nx.float64 shape values
  | "stft" ->
      Stft.power_spectrum
        ~power:(Tutils.Golden.float_param case "power")
        (stft_config case) (signal Nx.float64 case)
  | other ->
      failf "%s: unknown source %s" case.name other

let run_piptrack (case : Tutils.Golden.case) s =
  piptrack ?fmin:(opt_float case "fmin") ?fmax:(opt_float case "fmax")
    ?threshold:(opt_float case "threshold")
    ?reference:(reference case)
    ?fft_size:
      ( if has case "fft_size" then
          Some (Tutils.Golden.int_param case "fft_size")
        else None )
    ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
    s

(* [check_peaks ~exact ~pitch_rtol ~mag_atol case (pitches, mags)] compares the
   result with the sparse rows [index; pitch; magnitude]: the support of both
   tensors must be the stored one exactly, and the values agree exactly or
   within the tolerances. *)
let check_peaks ~exact ~pitch_rtol ~mag_atol (case : Tutils.Golden.case)
    (pitches, mags) =
  let p = Nx.to_array (Nx.cast Nx.float64 pitches)
  and m = Nx.to_array (Nx.cast Nx.float64 mags) in
  let rows = case.shape.(0) in
  let expected = Array.init rows (fun r -> Float.to_int case.values.(3 * r)) in
  let support a =
    Array.of_list
      (List.rev
         (snd
            (Array.fold_left
               (fun (i, acc) v -> (i + 1, if v <> 0. then i :: acc else acc))
               (0, []) a ) ) )
  in
  equal ~msg:(case.name ^ ": pitch support") (array int) expected (support p) ;
  equal
    ~msg:(case.name ^ ": magnitude support")
    (array int) expected (support m) ;
  for r = 0 to rows - 1 do
    let i = expected.(r) in
    let ep = case.values.((3 * r) + 1) and em = case.values.((3 * r) + 2) in
    let msg what = Printf.sprintf "%s: %s at %d" case.name what i in
    if exact then begin
      equal ~msg:(msg "pitch") float_exact ep p.(i) ;
      equal ~msg:(msg "magnitude") float_exact em m.(i)
    end
    else begin
      equal ~msg:(msg "pitch") (float_rel ~rel:pitch_rtol ~abs:0.) ep p.(i) ;
      equal ~msg:(msg "magnitude") (float mag_atol) em m.(i)
    end
  done

let piptrack_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let s = spectrogram case in
      let peak =
        Array.fold_left
          (fun acc r -> Float.max acc case.values.((3 * r) + 2))
          0.
          (Array.init case.shape.(0) Fun.id)
      in
      let stft = Tutils.Golden.string_param case "source" = "stft" in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          check_peaks ~exact:(not stft) ~pitch_rtol:stft_pitch_rtol
            ~mag_atol:(stft_mag_rtol *. peak) case (run_piptrack case s)
      | "float32" ->
          (* the |LCG| values quantized exactly as the generator did; the STFT
             magnitudes rounded once from this library's own transform *)
          let s32 = Nx.cast Nx.float32 s in
          check_peaks ~exact:false ~pitch_rtol:Tutils.float32_rtol
            ~mag_atol:(Tutils.float32_atol +. (Tutils.float32_rtol *. peak))
            case (run_piptrack case s32)
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

(* {1 pitch_tuning} *)

let pitch_tuning_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let f = floats_of (Tutils.Golden.param case "frequencies") in
      let resolution = Tutils.Golden.float_param case "resolution"
      and bins_per_octave = Tutils.Golden.int_param case "bins_per_octave" in
      let got =
        match Tutils.Golden.string_param case "dtype" with
        | "float64" ->
            pitch_tuning ~resolution ~bins_per_octave
              (Nx.create Nx.float64 [|Array.length f|] f)
        | "float32" ->
            pitch_tuning ~resolution ~bins_per_octave
              (Nx.create Nx.float32 [|Array.length f|] f)
        | other ->
            failf "golden case %s: unknown dtype %s" case.name other
      in
      equal ~msg:case.name float_exact case.values.(0) got )

(* {1 estimate_tuning} *)

let estimate_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let resolution = Tutils.Golden.float_param case "resolution"
      and bins_per_octave = Tutils.Golden.int_param case "bins_per_octave"
      and sample_rate = Tutils.Golden.int_param case "sample_rate" in
      let fmin = opt_float case "fmin"
      and fmax = opt_float case "fmax"
      and threshold = opt_float case "threshold"
      and reference = reference case in
      let c = stft_config case in
      let got =
        match
          ( Tutils.Golden.string_param case "face"
          , Tutils.Golden.string_param case "dtype" )
        with
        | "audio", "float64" ->
            estimate_tuning c ~resolution ~bins_per_octave ?fmin ?fmax
              ?threshold ?reference ~sample_rate (signal Nx.float64 case)
        | "audio", "float32" ->
            estimate_tuning c ~resolution ~bins_per_octave ?fmin ?fmax
              ?threshold ?reference ~sample_rate (signal Nx.float32 case)
        | "spectrogram", dtype ->
            (* the face's input is the spectrogram: float32 cases quantize it,
               not the audio *)
            let s =
              Stft.power_spectrum
                ~power:(Tutils.Golden.float_param case "power")
                c (signal Nx.float64 case)
            in
            let estimate s =
              estimate_tuning_of_spectrogram ~resolution ~bins_per_octave ?fmin
                ?fmax ?threshold ?reference ~sample_rate s
            in
            if dtype = "float32" then estimate (Nx.cast Nx.float32 s)
            else estimate s
        | face, dtype ->
            failf "golden case %s: unknown face %s or dtype %s" case.name face
              dtype
      in
      equal ~msg:case.name float_exact case.values.(0) got )

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      let run =
        match Filename.basename file.path with
        | "piptrack.json" ->
            piptrack_case
        | "pitch_tuning.json" ->
            pitch_tuning_case
        | "estimate_tuning.json" ->
            estimate_case
        | other ->
            failf "unknown golden file %s" other
      in
      group (Filename.basename file.path) (List.map run file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
