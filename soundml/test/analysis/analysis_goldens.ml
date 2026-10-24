(* Golden parity against librosa 1.0: librosa.autocorrelate, librosa.lpc and
   librosa.feature.poly_features, replayed from the committed vectors. Inputs
   are the LCG stream under this suite's seed, reshaped in C order, or the
   stored signals of inputs.json (sinusoids and lfilter-made autoregressive
   processes, which OCaml cannot rebuild bit for bit). float32 cases quantize
   the input to float32 against a float64 reference. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

let seed = 20261103

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let load name = Tutils.Golden.load (Filename.concat vectors_dir name)

let has (case : Tutils.Golden.case) key = List.mem_assoc key case.params

let number = function
  | `Int i ->
      Float.of_int i
  | `Float f ->
      f
  | _ ->
      failwith "not a number"

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
      Array.of_list (List.map number l)
  | _ ->
      failf "%s: %s is not a list" case.name key

let signals = lazy (load "inputs.json")

let signal name =
  match
    List.find_opt
      (fun (c : Tutils.Golden.case) -> c.name = name)
      (Lazy.force signals).cases
  with
  | Some c ->
      Nx.create Nx.float64 c.shape c.values
  | None ->
      failf "no stored signal %s" name

(* [input case] is the case's float64 input: a stored signal, a constant, or the
   LCG stream in the case's shape. *)
let input (case : Tutils.Golden.case) =
  if has case "signal" then signal (Tutils.Golden.string_param case "signal")
  else
    let shape = int_list case "input_shape" in
    let n = Array.fold_left ( * ) 1 shape in
    if has case "constant" then
      match Tutils.Golden.string_param case "constant" with
      | "zeros" ->
          Nx.zeros Nx.float64 shape
      | "ones" ->
          Nx.ones Nx.float64 shape
      | other ->
          failf "%s: unknown constant %s" case.name other
    else Nx.create Nx.float64 shape (lcg_values n)

let dtype_of (case : Tutils.Golden.case) =
  Tutils.Golden.string_param case "dtype"

let check_shape (case : Tutils.Golden.case) t =
  equal ~msg:(case.name ^ " shape") (array int) case.shape (Nx.shape t)

(* {1 autocorrelate}

   An FFT rounds relative to the whole row: every lag carries a few ulps of the
   row's energy, lag 0, which bounds every lag (Cauchy-Schwarz) — so the error
   is measured against each row's lag 0. Measured worst agreement, float64:
   5.9e-16 of lag 0 (scipy's transform against Nx's, on the tone under noise);
   float32, where the double lags round once to float32: 3.4e-8 of lag 0. Each
   bound is ten times its worst. *)
let autocorrelate64_tol = 6e-15

let autocorrelate32_tol = 3.5e-7

let autocorrelate_error ~expected got =
  let m = Nx.dim (Nx.ndim got - 1) got in
  let got = Nx.to_array (Nx.cast Nx.float64 got) in
  let worst = ref 0. in
  Array.iteri
    (fun i e ->
      let r0 = Float.abs expected.(i / m * m) in
      worst := Float.max !worst (Float.abs (got.(i) -. e) /. r0) )
    expected ;
  !worst

let autocorrelate_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let max_size =
        match Tutils.Golden.param case "max_size" with
        | `Null ->
            None
        | `Int m ->
            Some m
        | _ ->
            failf "%s: max_size" case.name
      in
      let x = input case in
      let err, tol =
        match dtype_of case with
        | "float64" ->
            let y = autocorrelate ?max_size x in
            check_shape case y ;
            (autocorrelate_error ~expected:case.values y, autocorrelate64_tol)
        | "float32" ->
            let y = autocorrelate ?max_size (Nx.cast Nx.float32 x) in
            check_shape case y ;
            (autocorrelate_error ~expected:case.values y, autocorrelate32_tol)
        | other ->
            failf "%s: unknown dtype %s" case.name other
      in
      at_most ~msg:case.name float_exact ~than:tol err )

(* {1 lpc}

   float64: librosa's numba kernel sums from zero in index order and never fuses
   a multiply-add, and the implementation spells the same operations, so the
   coefficients are the reference's bits on every case — the ill-conditioned
   tone under weak noise included. The check is exact. float32: both sides read
   the same quantized samples; the double coefficients are then the float64
   reference's bits, rounded once to float32, so the check is exact against that
   rounding. *)
let lpc_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let order = Tutils.Golden.int_param case "order" in
      let x = input case in
      match dtype_of case with
      | "float64" ->
          let a = lpc ~order x in
          check_shape case a ;
          equal ~msg:case.name (array float_exact) case.values (Nx.to_array a)
      | "float32" ->
          let a = lpc ~order (Nx.cast Nx.float32 x) in
          check_shape case a ;
          let expected =
            Nx.to_array
              (Nx.cast Nx.float64
                 (Nx.cast Nx.float32
                    (Nx.create Nx.float64 case.shape case.values) ) )
          in
          equal ~msg:case.name (array float_exact) expected
            (Nx.to_array (Nx.cast Nx.float64 a))
      | other ->
          failf "%s: unknown dtype %s" case.name other )

(* {1 poly_features}

   A fit's coefficients span many orders of magnitude (the cubic term in Hz
   units is near 1e-12) and some sit near zero, where an elementwise relative
   error means nothing. Each coefficient is instead measured against the size of
   the terms it sums, the case's "err_scale" — [Σⱼ |P i j| * |S j t|], P numpy's
   least-squares map — which bounds the rounding of any accurate solver:

   |got - expected| <= tol * err_scale, elementwise.

   numpy solves the column-scaled problem by LAPACK's gelsd and the library by a
   one-sided Jacobi SVD folded into one matrix, every sum over the bins
   compensated. Measured worst agreement, float64: 4.1e-15 at orders 0 to 3, on
   FFT grids of 3 to 8939 bins, flat frames and the rank-deficient fits included
   — the order-0 fit over 1089 bins, where numpy's own fit misses the exact
   frame mean by 4.0e-15 and the library's by under 3e-16; every other case is
   within 3.1e-15 — and 9.5e-15 at order 5, where the scaled problem's condition
   number has grown to 2.5e3. (Plain running sums put the order-0 fits 2.5e-13
   off over 8939 bins, and 4.4e-14 over 2049.) float32, where the double
   coefficients round once to float32 (and the audio face's magnitudes before
   them): 3.9e-8. Each bound is ten times its worst. *)
let poly64_tol order = if order <= 3 then 4e-14 else 1e-13

let poly32_tol = 4e-7

(* [spectrogram case] is the case's float64 spectrogram, built from the LCG
   stream exactly as the generator builds it. *)
let spectrogram (case : Tutils.Golden.case) =
  let shape = int_list case "input_shape" in
  let n = Array.fold_left ( * ) 1 shape in
  let base = lcg_values n in
  let nd = Array.length shape in
  let bins = shape.(nd - 2) and frames = shape.(nd - 1) in
  let values =
    match Tutils.Golden.string_param case "kind" with
    | "abs" ->
        Array.map Float.abs base
    | "signed" ->
        Array.map (fun v -> 40.0 *. v) base
    | "tonal" ->
        let s = Array.map (fun v -> 1e-3 *. Float.abs v) base in
        ( match Tutils.Golden.param case "peaks" with
        | `List peaks ->
            List.iter
              (function
                | `List [`Int k; v] ->
                    let v = number v in
                    for i = 0 to (n / (bins * frames)) - 1 do
                      for t = 0 to frames - 1 do
                        let o = (((i * bins) + k) * frames) + t in
                        s.(o) <- s.(o) +. v
                      done
                    done
                | _ ->
                    failf "%s: peaks" case.name )
              peaks
        | _ ->
            failf "%s: peaks" case.name ) ;
        s
    | "wide" ->
        let scales = float_list case "frame_scales" in
        Array.mapi (fun i v -> Float.abs v *. scales.(i mod frames)) base
    | "flat" ->
        Array.make n (number (Tutils.Golden.param case "level"))
    | other ->
        failf "%s: unknown kind %s" case.name other
  in
  Nx.create Nx.float64 shape values

let poly_error ~err_scale ~expected got =
  let got = Nx.to_array (Nx.cast Nx.float64 got) in
  let worst = ref 0. in
  Array.iteri
    (fun i e ->
      let d = Float.abs (got.(i) -. e) in
      let r = if d = 0. then 0. else d /. err_scale.(i) in
      worst := Float.max !worst r )
    expected ;
  !worst

let poly_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let order = Tutils.Golden.int_param case "order" in
      let sample_rate = Tutils.Golden.int_param case "sample_rate" in
      let err_scale = float_list case "err_scale" in
      let run (type a) (dtype : (float, a) Nx.dtype) : (float, a) Nx.t =
        match Tutils.Golden.string_param case "source" with
        | "spectrogram" ->
            let s = Nx.cast dtype (spectrogram case) in
            let freqs =
              if has case "freqs" then
                let f = float_list case "freqs" in
                Some (Nx.cast dtype (Nx.create Nx.float64 [|Array.length f|] f))
              else None
            in
            poly_features_of_spectrogram ~order ?freqs ~sample_rate s
        | "audio" ->
            let fft_size = Tutils.Golden.int_param case "fft_size" in
            let hop = Tutils.Golden.int_param case "hop" in
            let c =
              Stft.Config.create ~fft_size ~hop ~alignment:`Centered
                ~pad:(`Constant 0.) ()
            in
            poly_features c ~order ~sample_rate (Nx.cast dtype (input case))
        | other ->
            failf "%s: unknown source %s" case.name other
      in
      let err, tol =
        match dtype_of case with
        | "float64" ->
            let got = run Nx.float64 in
            check_shape case got ;
            (poly_error ~err_scale ~expected:case.values got, poly64_tol order)
        | "float32" ->
            let got = run Nx.float32 in
            check_shape case got ;
            (poly_error ~err_scale ~expected:case.values got, poly32_tol)
        | other ->
            failf "%s: unknown dtype %s" case.name other
      in
      at_most ~msg:case.name float_exact ~than:tol err )

let suite =
  [ group "autocorrelate.json"
      (List.map autocorrelate_case (load "autocorrelate.json").cases)
  ; group "lpc.json" (List.map lpc_case (load "lpc.json").cases)
  ; group "poly_features.json"
      (List.map poly_case (load "poly_features.json").cases) ]
