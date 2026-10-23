(* Golden parity against scipy 1.18 and librosa 1.0: butter, lfilter_zi,
   sosfilt_zi, lfilter, sosfilt, filtfilt, sosfiltfilt and
   librosa.effects.preemphasis/deemphasis over the LCG stream under this suite's
   seed, reshaped in C order, replayed from the committed vectors. Every
   filtering case carries the coefficients and the state it was run with.
   float32 cases quantize the input to float32 against a float64 reference.

   The bounds are on the largest error over a case relative to the case's
   largest magnitude: elementwise relative error is meaningless for the
   near-zero coefficients of a band design or the zero crossings of a filtered
   signal. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* Tolerances. The vectors come from scipy's arm64 build, which contracts
   multiply-adds exactly where this implementation does on arm64, and an x86_64
   build of this implementation contracts none. Each bound is about ten times
   the worst error of its family over two measurements: this implementation on
   arm64, and an operation-for-operation emulation of it with every product
   rounded on its own (the x86_64 build), which reproduces the arm64 measurement
   exactly when its multiply-adds are contracted. *)

(* Design: bit for bit on arm64 in every case; 2.9e-15 unfused, at an
   eighth-order bandpass as a transfer function. Away from the vectors a band
   design can also take a complex square root that rounds otherwise than the
   platform's (see the interface). *)
let butter_tol = 3e-14

(* Steady states: no product is contracted on either side and no libm function
   is called, so both measurements are bit for bit; the bound is a floor of a
   few units in the last place of the largest value. *)
let zi_tol = 1e-15

(* Transfer functions: bit for bit on arm64 but for the 16-tap FIR filter, which
   scipy convolves in another order (3.4e-16); unfused, 1.2e-13 on the outputs
   (a third-order bandstop) and 1.1e-12 on the final states (a third-order
   highpass). *)
let lfilter_tol = 1.2e-12

let lfilter_zf_tol = 1.2e-11

(* Sections: bit for bit on arm64; unfused, 3.3e-12 on the outputs and 1.6e-11
   on the final states, both at the fifth-order 100-6500 Hz bandstop. *)
let sosfilt_tol = 3.3e-11

let sosfilt_zf_tol = 1.6e-10

(* Zero-phase filtering: bit for bit on arm64; 3.6e-14 unfused. *)
let filtfilt_tol = 3.6e-13

(* Emphasis: bit for bit in both measurements, two-tap arithmetic contracting to
   the same bits as its unfused form. What a platform can still move is the libm
   power in librosa's de-emphasis correction, by an ulp of a term no larger than
   the signal; the bound is a floor of a few ulps. *)
let emphasis_tol = 1e-15

let seed = 20261109

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let has (case : Tutils.Golden.case) key = List.mem_assoc key case.params

let floats (case : Tutils.Golden.case) key =
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
                 failf "%s: %s is not a list of numbers" case.name key )
           l )
  | _ ->
      failf "%s: %s is not a list" case.name key

let ints (case : Tutils.Golden.case) key =
  match Tutils.Golden.param case key with
  | `List l ->
      Array.of_list
        (List.map
           (function `Int i -> i | _ -> failf "%s: %s" case.name key)
           l )
  | _ ->
      failf "%s: %s is not a list" case.name key

let vector v = Nx.create Nx.float64 [|Array.length v|] v

let input dtype (case : Tutils.Golden.case) =
  let shape = ints case "input_shape" in
  Nx.create dtype shape (lcg_values (Array.fold_left ( * ) 1 shape))

let zi (case : Tutils.Golden.case) =
  if has case "zi" then
    Some (Nx.create Nx.float64 (ints case "zi_shape") (floats case "zi"))
  else None

let sos (case : Tutils.Golden.case) =
  Nx.create Nx.float64
    [|Tutils.Golden.int_param case "n_sections"; 6|]
    (floats case "sos")

let output (case : Tutils.Golden.case) =
  if has case "output" then Tutils.Golden.string_param case "output" else "y"

(* [scaled_error ~expected actual] is the largest elementwise error of [actual]
   relative to the largest magnitude of [expected]. *)
let scaled_error ~expected (actual : (float, 'a) Nx.t) =
  let got = Nx.to_array (Nx.cast Nx.float64 actual) in
  let scale =
    Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. expected
  in
  let worst = ref 0. in
  Array.iteri
    (fun i e -> worst := Float.max !worst (Float.abs (got.(i) -. e)))
    expected ;
  if scale = 0. then !worst else !worst /. scale

let check ~tol (case : Tutils.Golden.case) actual =
  equal ~msg:(case.name ^ ": shape") (array int) case.shape (Nx.shape actual) ;
  let err = scaled_error ~expected:case.values actual in
  if not (err <= tol) then
    failf "%s: error %.3e of the largest magnitude, above the bound %.1e"
      case.name err tol

(* float32 output against the float64 reference: the rounding of the output *)
let check32 (case : Tutils.Golden.case) actual =
  Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
    ~shape:case.shape ~msg:case.name ~expected:case.values actual

(* [dtype_case ~tol ~zf_tol case ~f64 ~f32] checks the output or the final state
   of the run at the case's dtype. *)
let dtype_case ~tol ~zf_tol (case : Tutils.Golden.case) ~f64 ~f32 =
  match (Tutils.Golden.string_param case "dtype", output case) with
  | "float64", "y" ->
      check ~tol case (fst (f64 ()))
  | "float64", "zf" ->
      check ~tol:zf_tol case (snd (f64 ()))
  | "float32", "y" ->
      check32 case (fst (f32 ()))
  | "float32", "zf" ->
      (* the state is float64 at both dtypes; from a float32 input it is the
         float64 filter's state on the quantized input *)
      check ~tol:zf_tol case (snd (f32 ()))
  | dtype, out ->
      failf "golden case %s: unknown dtype %s or output %s" case.name dtype out

(* {1 Design} *)

let band (case : Tutils.Golden.case) =
  let c = floats case "cutoffs" in
  match Tutils.Golden.string_param case "kind" with
  | "lowpass" ->
      `Lowpass c.(0)
  | "highpass" ->
      `Highpass c.(0)
  | "bandpass" ->
      `Bandpass (c.(0), c.(1))
  | "bandstop" ->
      `Bandstop (c.(0), c.(1))
  | other ->
      failf "%s: unknown band type %s" case.name other

let butter_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let order = Tutils.Golden.int_param case "order"
      and sample_rate = Tutils.Golden.int_param case "sample_rate" in
      match Tutils.Golden.string_param case "output" with
      | "ba" ->
          let b, a = Filter.butter ~order ~sample_rate (band case) in
          check ~tol:butter_tol case (Nx.stack [b; a])
      | "sos" ->
          check ~tol:butter_tol case
            (Filter.butter_sos ~order ~sample_rate (band case))
      | other ->
          failf "%s: unknown output %s" case.name other )

(* {1 Steady states} *)

let zi_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      match Tutils.Golden.string_param case "function" with
      | "lfilter_zi" ->
          check ~tol:zi_tol case
            (Filter.lfilter_zi
               (vector (floats case "b"))
               (vector (floats case "a")) )
      | "sosfilt_zi" ->
          check ~tol:zi_tol case (Filter.sosfilt_zi (sos case))
      | other ->
          failf "%s: unknown function %s" case.name other )

(* {1 Filtering} *)

let lfilter_case (case : Tutils.Golden.case) =
  test
    (case.name ^ " " ^ output case)
    (fun () ->
      let b = vector (floats case "b") and a = vector (floats case "a") in
      let zi = zi case in
      dtype_case ~tol:lfilter_tol ~zf_tol:lfilter_zf_tol case
        ~f64:(fun () -> Filter.lfilter_zf ?zi b a (input Nx.float64 case))
        ~f32:(fun () -> Filter.lfilter_zf ?zi b a (input Nx.float32 case)) )

let sosfilt_case (case : Tutils.Golden.case) =
  test
    (case.name ^ " " ^ output case)
    (fun () ->
      let sos = sos case and zi = zi case in
      dtype_case ~tol:sosfilt_tol ~zf_tol:sosfilt_zf_tol case
        ~f64:(fun () -> Filter.sosfilt_zf ?zi sos (input Nx.float64 case))
        ~f32:(fun () -> Filter.sosfilt_zf ?zi sos (input Nx.float32 case)) )

let pad (case : Tutils.Golden.case) =
  match Tutils.Golden.string_param case "pad" with
  | "odd" ->
      `Odd
  | "even" ->
      `Reflect
  | "constant" ->
      `Edge
  | other ->
      failf "%s: unknown padding %s" case.name other

let filtfilt_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let pad = pad case
      and padlen =
        if has case "padlen" then Some (Tutils.Golden.int_param case "padlen")
        else None
      in
      let run (type a) (dtype : (float, a) Nx.dtype) : (float, a) Nx.t =
        let x = input dtype case in
        match Tutils.Golden.string_param case "function" with
        | "filtfilt" ->
            Filter.filtfilt ~pad ?padlen
              (vector (floats case "b"))
              (vector (floats case "a"))
              x
        | "sosfiltfilt" ->
            Filter.sosfiltfilt ~pad ?padlen (sos case) x
        | other ->
            failf "%s: unknown function %s" case.name other
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          check ~tol:filtfilt_tol case (run Nx.float64)
      | "float32" ->
          check32 case (run Nx.float32)
      | other ->
          failf "%s: unknown dtype %s" case.name other )

let emphasis_case (case : Tutils.Golden.case) =
  test
    (case.name ^ " " ^ output case)
    (fun () ->
      let coef =
        if has case "coef" then Some (Tutils.Golden.float_param case "coef")
        else None
      and zi = zi case in
      let run (type a) (dtype : (float, a) Nx.dtype) =
        let x = input dtype case in
        match Tutils.Golden.string_param case "function" with
        | "preemphasis" ->
            Filter.preemphasis_zf ?coef ?zi x
        | "deemphasis" ->
            Filter.deemphasis_zf ?coef ?zi x
        | "round_trip" ->
            (Filter.deemphasis (Filter.preemphasis x), Nx.zeros Nx.float64 [|0|])
        | other ->
            failf "%s: unknown function %s" case.name other
      in
      dtype_case ~tol:emphasis_tol ~zf_tol:emphasis_tol case
        ~f64:(fun () -> run Nx.float64)
        ~f32:(fun () -> run Nx.float32) )

let suite =
  let file name f =
    match
      Tutils.Golden.load_dir ~filter:(String.equal (name ^ ".json")) vectors_dir
    with
    | [file] ->
        group (name ^ ".json") (List.map f file.cases)
    | _ ->
        failwith ("missing golden file " ^ name ^ ".json")
  in
  [ file "butter" butter_case
  ; file "zi" zi_case
  ; file "lfilter" lfilter_case
  ; file "sosfilt" sosfilt_case
  ; file "filtfilt" filtfilt_case
  ; file "emphasis" emphasis_case ]
