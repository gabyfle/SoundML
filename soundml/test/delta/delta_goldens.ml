(* Golden parity against librosa 1.0: librosa.feature.delta over the LCG stream
   under this suite's seed, reshaped in C order, replayed from the committed
   vectors. float32 cases quantize the input to float32 against a float64
   reference, like the energy suite. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 tolerances. Each output is a weighted sum of [width] frames whose
   weights the library solves for in double precision; scipy solves the same
   least-squares system by SVD and fits the "interp" borders with polyfit. The
   two agree within 1e-13 relative and 1e-14 absolute on every case here (inputs
   |x| < 1); the bounds below keep a factor of ten over that. *)
let delta64_rtol = 1e-12

let delta64_atol = 1e-13

let seed = 20261023

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let input dtype (case : Tutils.Golden.case) =
  let shape =
    match Tutils.Golden.param case "input_shape" with
    | `List l ->
        Array.of_list
          (List.map
             (function `Int i -> i | _ -> failf "%s: shape" case.name)
             l )
    | _ ->
        failf "%s: input_shape is not a list" case.name
  in
  Nx.create dtype shape (lcg_values (Array.fold_left ( * ) 1 shape))

let mode_of case =
  match Tutils.Golden.string_param case "mode" with
  | "interp" ->
      `Interp
  | "nearest" ->
      `Nearest
  | other ->
      failf "golden case %s: unknown mode %s" case.name other

let delta_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let width = Tutils.Golden.int_param case "width" in
      let order = Tutils.Golden.int_param case "order" in
      let mode = mode_of case in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~rtol:delta64_rtol ~atol:delta64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (delta ~width ~order ~mode (input Nx.float64 case))
      | "float32" ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (delta ~width ~order ~mode (input Nx.float32 case))
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      group (Filename.basename file.path) (List.map delta_case file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
