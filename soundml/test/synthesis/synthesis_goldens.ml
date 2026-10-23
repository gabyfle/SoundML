(* Golden parity against librosa 1.0: librosa.tone, librosa.chirp (over
   scipy.signal.chirp), librosa.clicks and librosa.mu_compress / mu_expand,
   replayed from the committed vectors; dev/generate_vectors.py's
   SynthesisVectorGenerator says what each case pins. float32 cases compare a
   float32 result against a float64 reference, their input (where they have one)
   quantized to float32 first, except the float32 click tracks, whose reference
   is librosa's own float32 track. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* Tolerances. On macOS arm64 every case matches bit for bit: numpy calls the
   same libm functions as OCaml, and the library spells the reference's
   arithmetic operation for operation. Under another C library (glibc on the
   Linux CI) a libm call may differ by an ulp, which the tolerances must allow.
   It was measured with a proxy that keeps every IEEE operation of the reference
   and replaces each libm call by its correctly rounded value (mpmath at 200
   bits):

   - the tones, the linear chirps and the mu-law values move by at most 1.1e-16
   (one ulp of a cosine), the codes not at all: the float64 defaults (rtol
   1e-12, atol 1e-15) keep ten times that;

   - an exponential sweep multiplies the error of its power by its phase, so a
   one-ulp difference in pow or log moves a sample by up to about 2^-52 times
   the total phase swept (in radians): ten times that is each exponential case's
   bound, [exponential_atol] below. The proxy moves the widest sweep here (20 Hz
   to 20 kHz in a second at 48 kHz, 18000 radians) by 1.4e-12 against a bound of
   4e-11, the 357-sample one (38 radians) by 2.3e-15 against 8e-14, the others
   by 1.1e-16;

   - the float64 click tracks move by 4.4e-16 (three overlapping clicks of an
   ulp each), hence 5e-15; the float32 tracks not at all, and an ulp of the
   float64 click could move a float32 sum by one float32 ulp, which the float32
   tolerances allow. *)
let clicks64_atol = 5e-15

(* [exponential_atol ~fmin ~fmax ~t1] is ten times 2^-52 times the phase an
   exponential sweep of [t1] seconds covers: [2 pi fmin t1 (r - 1) / ln r] with
   [r = fmax / fmin], [2 pi fmin t1] when [r = 1]. *)
let exponential_atol ~fmin ~fmax ~t1 =
  let r = fmax /. fmin in
  let phase =
    if Float.equal r 1. then 2. *. Float.pi *. fmin *. t1
    else 2. *. Float.pi *. fmin *. t1 *. (r -. 1.) /. Float.log r
  in
  Float.max Tutils.float64_atol (10. *. epsilon_float *. Float.abs phase)

let has (case : Tutils.Golden.case) key = List.mem_assoc key case.params

let opt_int case key =
  if has case key then Some (Tutils.Golden.int_param case key) else None

let opt_float case key =
  if has case key then Some (Tutils.Golden.float_param case key) else None

let int_list (case : Tutils.Golden.case) key =
  match Tutils.Golden.param case key with
  | `List l ->
      Array.of_list
        (List.map
           (function `Int i -> i | _ -> failf "%s: %s" case.name key)
           l )
  | _ ->
      failf "%s: %s is not a list" case.name key

let float_list (case : Tutils.Golden.case) key =
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
                 failf "%s: %s" case.name key )
           l )
  | _ ->
      failf "%s: %s is not a list" case.name key

(* [tail case t] is the last [tail] samples of [t] when the case keeps only
   those, [t] otherwise. *)
let tail case t =
  match opt_int case "tail" with
  | None ->
      t
  | Some n ->
      let len = Nx.dim 0 t in
      Nx.slice [Nx.R (len - n, len)] t

let check (case : Tutils.Golden.case) ?(atol64 = Tutils.float64_atol) run64
    run32 =
  let expected = case.values in
  match Tutils.Golden.string_param case "dtype" with
  | "float64" ->
      Tutils.check_close ~atol:atol64 ~shape:case.shape ~msg:case.name ~expected
        (run64 ())
  | "float32" ->
      Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
        ~shape:case.shape ~msg:case.name ~expected (run32 ())
  | other ->
      failf "golden case %s: unknown dtype %s" case.name other

(* [exact case run64 run32] checks the case's result bit for bit. *)
let exact (case : Tutils.Golden.case) run64 run32 =
  let got =
    match Tutils.Golden.string_param case "dtype" with
    | "float64" ->
        Nx.to_array (run64 ())
    | "float32" ->
        Nx.to_array (run32 ())
    | other ->
        failf "golden case %s: unknown dtype %s" case.name other
  in
  equal ~msg:case.name (array float_exact) case.values got

(* {1 Tones} *)

let tone_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let length = opt_int case "length"
      and duration = opt_float case "duration"
      and phase = opt_float case "phase"
      and sample_rate = Tutils.Golden.int_param case "sample_rate"
      and f = Tutils.Golden.float_param case "frequency" in
      let run dtype () =
        tail case (tone dtype ?length ?duration ?phase ~sample_rate f)
      in
      check case (run Nx.float64) (run Nx.float32) )

(* {1 Chirps} *)

let chirp_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let length = opt_int case "length"
      and duration = opt_float case "duration"
      and phase = opt_float case "phase"
      and linear = Tutils.Golden.bool_param case "linear"
      and sample_rate = Tutils.Golden.int_param case "sample_rate"
      and fmin = Tutils.Golden.float_param case "fmin"
      and fmax = Tutils.Golden.float_param case "fmax" in
      let run dtype () =
        tail case
          (chirp dtype ?length ?duration ~linear ?phase ~sample_rate ~fmin ~fmax
             () )
      in
      (* librosa's sweep time; the reference is scipy's sweep over all the
         samples asked for, librosa's own output on the first [reference_length]
         (the generator asserts it) *)
      let t1 =
        match (length, duration) with
        | Some l, _ ->
            1. /. Float.of_int sample_rate *. Float.of_int l
        | None, Some d ->
            d
        | None, None ->
            failf "%s: no length" case.name
      in
      let atol64 =
        if linear then Tutils.float64_atol else exponential_atol ~fmin ~fmax ~t1
      in
      check case ~atol64 (run Nx.float64) (run Nx.float32) )

(* {1 Click tracks} *)

let lcg_click shape =
  (* the generator's custom click: the LCG stream at its seed + 1 *)
  let n = Array.fold_left ( * ) 1 shape in
  let state = ref (20261103 + 1) in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let clicks_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let sample_rate = Tutils.Golden.int_param case "sample_rate"
      and hop = opt_int case "hop"
      and length = opt_int case "length"
      and click_frequency = opt_float case "click_frequency"
      and click_duration = opt_float case "click_duration" in
      let positions =
        if has case "times" then `Times (float_list case "times")
        else if has case "frames" then `Frames (int_list case "frames")
        else `Samples (int_list case "samples")
      in
      let run (type a) (dtype : (float, a) Nx.dtype) () : (float, a) Nx.t =
        let click =
          if has case "click_shape" then
            let shape = int_list case "click_shape" in
            Some (Nx.create dtype shape (lcg_click shape))
          else None
        in
        clicks dtype ?hop ?click ?click_frequency ?click_duration ?length
          ~sample_rate positions
      in
      check case ~atol64:clicks64_atol (run Nx.float64) (run Nx.float32) )

(* {1 Mu-law companding} *)

let mu_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let mu = Tutils.Golden.float_param case "mu"
      and quantize = Tutils.Golden.bool_param case "quantize"
      and shape = int_list case "input_shape"
      and input = float_list case "input" in
      let op = Tutils.Golden.string_param case "op" in
      let run dtype () =
        let x = Nx.create dtype shape input in
        match op with
        | "compress" ->
            mu_compress ~mu ~quantize x
        | "expand" ->
            mu_expand ~mu ~quantize x
        | "roundtrip" ->
            mu_expand ~mu ~quantize (mu_compress ~mu ~quantize x)
        | other ->
            failf "%s: unknown op %s" case.name other
      in
      (* codes are whole numbers on both sides, kept clear of every level edge
         by the generator: exact *)
      if op = "compress" && quantize then
        exact case (run Nx.float64) (run Nx.float32)
      else check case (run Nx.float64) (run Nx.float32) )

let suite =
  let file name case =
    let f = Tutils.Golden.load (Filename.concat vectors_dir (name ^ ".json")) in
    group (Filename.basename f.path) (List.map case f.cases)
  in
  [ file "tone" tone_case
  ; file "chirp" chirp_case
  ; file "clicks" clicks_case
  ; file "mu" mu_case ]
