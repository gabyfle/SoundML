(* Golden parity against librosa 1.0: librosa.feature.tempogram,
   fourier_tempogram and tempo, librosa.beat.beat_track and plp, and
   librosa.tempo_frequencies / fourier_tempo_frequencies, replayed from the
   committed vectors. Inputs are rebuilt from the case parameters: formula
   envelopes over the stft suite's 31-bit LCG (a floor of |values| with spikes
   added, or their fourth power), librosa's onset envelopes of the click tracks,
   read back from envelopes.json, and the click tracks themselves, whose
   envelope Soundml.onset_strength computes here on the geometry librosa was
   given. float32 cases quantize the envelope (or the audio) to float32 against
   a float64 reference. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* {1 Tolerances}

   Each bound is about ten times the worst error measured over its cases, the
   envelope and the audio ones alike: Soundml.onset_strength agrees with
   librosa's envelopes of the click tracks within 7e-14 at values of up to 67,
   which no later stage magnifies.

   The autocorrelation tempogram sums each lag directly where librosa
   transforms: the two agree within 1.4e-15 of the frame's zero lag, which the
   inf norm makes 1. *)
let tempogram_atol = 1.5e-14

(* The Fourier tempogram is Stft.transform, within 3.6e-15 of librosa's FFT at
   coefficients of up to 15. *)
let fourier_atol = 4e-14

(* The pulse is a least-squares inverse whose frames are weighted by up to
   1e154, normalised by its maximum: within 2.9e-15 of it. *)
let plp_atol = 3e-14

(* The click envelopes themselves, rebuilt here: within 7e-14, as above. *)
let envelope_atol = 7e-13

(* {1 Inputs} *)

let lcg n seed =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let field key (json : Yojson.Safe.t) =
  match json with
  | `Assoc l -> (
    match List.assoc_opt key l with
    | Some v ->
        v
    | None ->
        failf "missing field %s" key )
  | _ ->
      failf "field %s of a non-object" key

let to_int = function `Int i -> i | _ -> failf "not an int"

let to_float = function
  | `Float f ->
      f
  | `Int i ->
      Float.of_int i
  | _ ->
      failf "not a float"

let to_list = function `List l -> l | _ -> failf "not a list"

let has key (json : Yojson.Safe.t) =
  match json with `Assoc l -> List.mem_assoc key l | _ -> false

(* [envelope spec] is the formula envelope [spec] describes: zeros, a floor of
   |LCG| or the fourth power of |LCG|, spikes added one at a time in order,
   optionally negated. *)
let envelope spec =
  let n = to_int (field "n" spec) in
  let o =
    match field "kind" spec with
    | `String "zeros" ->
        Array.make n 0.
    | `String "abs" ->
        let floor = to_float (field "floor" spec) in
        Array.map
          (fun v -> floor *. Float.abs v)
          (lcg n (to_int (field "seed" spec)))
    | `String "pow4" ->
        Array.map
          (fun v ->
            let a = Float.abs v in
            let s = a *. a in
            s *. s )
          (lcg n (to_int (field "seed" spec)))
    | _ ->
        failf "unknown envelope kind"
  in
  if has "spikes" spec then
    List.iter
      (fun pair ->
        match to_list pair with
        | [f; h] ->
            let f = to_int f in
            o.(f) <- o.(f) +. to_float h
        | _ ->
            failf "spike" )
      (to_list (field "spikes" spec)) ;
  if has "negate" spec && field "negate" spec = `Bool true then
    Array.map Float.neg o
  else o

(* [audio spec] is the click track [spec] describes: a 220-sample burst of the
   LCG under the ramp [1 - i / 220], scaled, at each position, plus LCG noise.
   The products are let-bound, as numpy forms them. *)
let audio spec =
  let n = to_int (field "length" spec) in
  let seed = to_int (field "seed" spec) in
  let scale = to_float (field "scale" spec) in
  let noise = to_float (field "noise" spec) in
  let len = 220 in
  let burst = lcg len seed in
  let y = Array.make n 0. in
  List.iter
    (fun p ->
      let p = to_int p in
      for i = 0 to Int.min len (n - p) - 1 do
        let ramp = 1. -. (Float.of_int i /. Float.of_int len) in
        let c = scale *. ramp *. burst.(i) in
        y.(p + i) <- y.(p + i) +. c
      done )
    (to_list (field "positions" spec)) ;
  if noise > 0. then begin
    let l = lcg n (seed + 1) in
    for i = 0 to n - 1 do
      let q = noise *. l.(i) in
      y.(i) <- y.(i) +. q
    done
  end ;
  y

(* [onset dtype spec] is Soundml.onset_strength of the click track [spec] in
   [dtype], on librosa's geometry: a 2048-point centred STFT over zeros and 128
   Slaney mel bands up to Nyquist. *)
let onset dtype spec =
  let sample_rate = to_int (field "sample_rate" spec) in
  let hop = to_int (field "hop" spec) in
  let y = audio spec in
  let stft = Stft.Config.create ~fft_size:2048 ~hop ~pad:(`Constant 0.) () in
  let mel = Mel.Config.create ~n_mels:128 ~sample_rate ~fft_size:2048 () in
  onset_strength stft mel (Nx.create dtype [|Array.length y|] y)

let click_envelopes =
  lazy
    (let file =
       Tutils.Golden.load (Filename.concat vectors_dir "envelopes.json")
     in
     List.map
       (fun (case : Tutils.Golden.case) ->
         (case.name, (case.params, case.values)) )
       file.cases )

let track key = List.assoc key (Lazy.force click_envelopes)

(* [input dtype case] is the onset envelope of [case] in [dtype]. *)
let input (type a) (dtype : (float, a) Nx.dtype) (case : Tutils.Golden.case) :
    (float, a) Nx.t =
  let has key = List.mem_assoc key case.params in
  if has "envelope" then
    let o = envelope (Tutils.Golden.param case "envelope") in
    Nx.create dtype [|Array.length o|] o
  else if has "envelopes" then
    let rows =
      List.map envelope (to_list (Tutils.Golden.param case "envelopes"))
    in
    let n = Array.length (List.hd rows) in
    Nx.create dtype [|List.length rows; n|] (Array.concat rows)
  else
    let key = Tutils.Golden.string_param case "clicks" in
    let spec, values = track key in
    match Tutils.Golden.string_param case "origin" with
    | "envelope" ->
        Nx.create dtype [|Array.length values|] values
    | "audio" ->
        onset dtype (`Assoc spec)
    | other ->
        failf "%s: unknown origin %s" case.name other

let int_or case key default =
  if List.mem_assoc key case.Tutils.Golden.params then
    Tutils.Golden.int_param case key
  else default

let float_or case key default =
  if List.mem_assoc key case.Tutils.Golden.params then
    match Tutils.Golden.param case key with
    | `String "none" ->
        None
    | _ ->
        Some (Tutils.Golden.float_param case key)
  else Some default

(* scipy.stats.uniform(30, 300).logpdf and lognorm(s=1, loc=log 120,
   scale=120).logpdf, in closed form. *)
let uniform b =
  let y = (b -. 30.) /. 300. in
  if y >= 0. && y <= 1. then Float.log 1. -. Float.log 300.
  else Float.neg_infinity

let lognorm b =
  let y = (b -. Float.log 120.) /. 120. in
  if y > 0. then
    let l = Float.log y in
    (-.(l *. l) /. 2.)
    -. Float.log (y *. Float.sqrt (2. *. Float.pi))
    -. Float.log 120.
  else Float.neg_infinity

let prior (case : Tutils.Golden.case) =
  if List.mem_assoc "prior" case.params then
    match Tutils.Golden.string_param case "prior" with
    | "uniform" ->
        Some uniform
    | "lognorm" ->
        Some lognorm
    | other ->
        failf "%s: unknown prior %s" case.name other
  else None

let dtype_param case = Tutils.Golden.string_param case "dtype"

let compare ~atol ~rtol (case : Tutils.Golden.case) actual =
  Tutils.check_close ~rtol ~atol ~shape:case.shape ~msg:case.name
    ~expected:case.values actual

(* [columns case t] is the frames of [t] ([[...; rows; frames]]) the case
   stores. *)
let columns (case : Tutils.Golden.case) t =
  let cols = List.map to_int (to_list (Tutils.Golden.param case "columns")) in
  let nd = Nx.ndim t in
  Nx.concatenate ~axis:(-1)
    (List.map
       (fun c ->
         Nx.slice
           (List.init nd (fun i -> if i = nd - 1 then Nx.R (c, c + 1) else Nx.A))
           t )
       cols )

let alignment case =
  match Tutils.Golden.string_param case "alignment" with
  | "centered" ->
      `Centered
  | "left" ->
      `Left
  | other ->
      failf "unknown alignment %s" other

(* {1 Cases} *)

let envelope_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      (* the track rebuilt here, through Soundml.onset_strength *)
      let got = onset Nx.float64 (`Assoc case.params) in
      compare ~atol:envelope_atol ~rtol:0. case got ;
      (* and silence is exactly silent on both sides *)
      let got = Nx.to_array got in
      Array.iteri
        (fun i e ->
          if Float.equal e 0. <> Float.equal got.(i) 0. then
            failf "%s: frame %d is %g here and %g in librosa" case.name i
              got.(i) e )
        case.values )

let frequency_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let n = Tutils.Golden.int_param case "n" in
      let sample_rate = Tutils.Golden.int_param case "sample_rate" in
      let hop = Tutils.Golden.int_param case "hop" in
      match Tutils.Golden.string_param case "kind" with
      | "tempo" ->
          let f = tempo_frequencies Nx.float64 ~hop ~sample_rate n in
          compare ~atol:0. ~rtol:Tutils.float64_rtol case
            (Nx.slice [Nx.R (1, n)] f)
      | _ ->
          compare ~atol:0. ~rtol:Tutils.float64_rtol case
            (fourier_tempo_frequencies Nx.float64 ~hop ~sample_rate n) )

let norm_param case =
  match Tutils.Golden.string_param case "norm" with
  | "inf" ->
      `Inf
  | "none" ->
      `None
  | p ->
      `P (float_of_string p)

let tempogram_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let run x =
        columns case
          (tempogram
             ~win_length:(Tutils.Golden.int_param case "win_length")
             ~alignment:(alignment case) ~norm:(norm_param case) x )
      in
      match dtype_param case with
      | "float64" ->
          compare ~atol:tempogram_atol ~rtol:0. case
            (run (input Nx.float64 case))
      | _ ->
          compare ~atol:Tutils.float32_atol ~rtol:Tutils.float32_rtol case
            (run (input Nx.float32 case)) )

(* [parts z] is the complex [z] as [[...; 2]] real and imaginary parts. *)
let parts (type c) (z : (Complex.t, c) Nx.t) =
  let z = Nx.cast Nx.complex128 z in
  Nx.stack ~axis:(-1) [Nx.real Nx.float64 z; Nx.imag Nx.float64 z]

let fourier_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let win_length = Tutils.Golden.int_param case "win_length" in
      let alignment = alignment case in
      match dtype_param case with
      | "float64" ->
          compare ~atol:fourier_atol ~rtol:0. case
            (parts
               (columns case
                  (fourier_tempogram Nx.complex128 ~win_length ~alignment
                     (input Nx.float64 case) ) ) )
      | _ ->
          compare ~atol:Tutils.float32_atol ~rtol:Tutils.float32_rtol case
            (parts
               (columns case
                  (fourier_tempogram Nx.complex64 ~win_length ~alignment
                     (input Nx.float32 case) ) ) ) )

let aggregate case =
  if List.mem_assoc "aggregate" case.Tutils.Golden.params then
    match Tutils.Golden.string_param case "aggregate" with
    | "none" ->
        `None
    | "median" ->
        `Median
    | other ->
        failf "unknown aggregate %s" other
  else `Mean

let tempo_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let run x =
        tempo
          ~hop:(Tutils.Golden.int_param case "hop")
          ?start_bpm:(float_or case "start_bpm" 120.)
          ?std_bpm:(float_or case "std_bpm" 1.)
          ?ac_size:(float_or case "ac_size" 8.)
          ~tempo_max:
            (Option.value ~default:Float.infinity
               (float_or case "max_tempo" 320.) )
          ~aggregate:(aggregate case) ?prior:(prior case)
          ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
          x
      in
      (* a tempo is a bin's, rounded to the dtype *)
      match dtype_param case with
      | "float64" ->
          compare ~atol:0. ~rtol:0. case (run (input Nx.float64 case))
      | _ ->
          compare ~atol:0. ~rtol:Tutils.float32_rtol case
            (run (input Nx.float32 case)) )

let beat_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let run x =
        beat_track
          ~hop:(Tutils.Golden.int_param case "hop")
          ?start_bpm:(float_or case "start_bpm" 120.)
          ?tightness:(float_or case "tightness" 100.)
          ?trim:
            ( if List.mem_assoc "trim" case.params then
                Some (Tutils.Golden.bool_param case "trim")
              else None )
          ?bpm:
            ( if List.mem_assoc "bpm" case.params then
                Some (Tutils.Golden.float_param case "bpm")
              else None )
          ?prior:(prior case)
          ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
          x
      in
      let bpm, beats =
        match dtype_param case with
        | "float64" ->
            run (input Nx.float64 case)
        | _ ->
            run (input Nx.float32 case)
      in
      let is_tempo =
        String.length case.name > 6 && String.sub case.name 0 6 = "tempo_"
      in
      if is_tempo then
        (* the tempo in double, whatever the dtype *)
        equal ~msg:case.name float_exact case.values.(0) bpm
      else
        equal ~msg:case.name (array int)
          (Array.map Float.to_int case.values)
          beats )

let plp_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let run x =
        plp
          ~hop:(Tutils.Golden.int_param case "hop")
          ~win_length:(int_or case "win_length" 384)
          ~tempo_min:(Option.value ~default:0. (float_or case "tempo_min" 30.))
          ~tempo_max:
            (Option.value ~default:Float.infinity
               (float_or case "tempo_max" 300.) )
          ?prior:(prior case)
          ~kernels:
            ( match Tutils.Golden.string_param case "kernels" with
            | "unit" ->
                `Unit
            | _ ->
                `Reference )
          ~sample_rate:(Tutils.Golden.int_param case "sample_rate")
          x
      in
      match dtype_param case with
      | "float64" ->
          compare ~atol:plp_atol ~rtol:0. case (run (input Nx.float64 case))
      | _ ->
          compare ~atol:Tutils.float32_atol ~rtol:Tutils.float32_rtol case
            (run (input Nx.float32 case)) )

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      let kind = Filename.remove_extension (Filename.basename file.path) in
      let case =
        match kind with
        | "envelopes" ->
            envelope_case
        | "frequencies" ->
            frequency_case
        | "tempogram" ->
            tempogram_case
        | "fourier_tempogram" ->
            fourier_case
        | "tempo" ->
            tempo_case
        | "beat_track" ->
            beat_case
        | "plp" ->
            plp_case
        | other ->
            failwith ("unknown golden file " ^ other)
      in
      group (Filename.basename file.path) (List.map case file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
