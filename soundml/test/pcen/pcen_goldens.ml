(* Golden parity against librosa 1.0: librosa.pcen over |LCG values| under this
   suite's seed, scaled by 2 ** 15 and reshaped in C order, replayed from the
   committed vectors. float32 cases quantize the input to float32 against a
   float64 reference, like the energy suite. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* float64 tolerances. The smoothing runs scipy's lfilter arithmetic exactly;
   what is left is exp, log1p and expm1, which the C library and numpy each
   round within an ulp or so. The two agree within 1e-14 relative on every case
   here, some bit for bit; the bounds below keep a factor of ten over that. *)
let pcen64_rtol = 1e-13

let pcen64_atol = 1e-14

let seed = 20261024

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
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create dtype shape
    (Array.map (fun v -> Float.abs v *. 32768.) (lcg_values n))

let pcen_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let has key = List.mem_assoc key case.params in
      let float key default =
        if has key then Tutils.Golden.float_param case key else default
      in
      let int key default =
        if has key then Tutils.Golden.int_param case key else default
      in
      let b =
        if has "b" then Some (Tutils.Golden.float_param case "b") else None
      in
      let run s =
        pcen ~hop:(int "hop" 512) ~gain:(float "gain" 0.98)
          ~bias:(float "bias" 2.) ~power:(float "power" 0.5)
          ~time_constant:(float "time_constant" 0.4)
          ~eps:(float "eps" 1e-6) ?b ~max_size:(int "max_size" 1)
          ~sample_rate:(int "sample_rate" 22050) s
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          Tutils.check_close ~rtol:pcen64_rtol ~atol:pcen64_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (run (input Nx.float64 case))
      | "float32" ->
          Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
            ~shape:case.shape ~msg:case.name ~expected:case.values
            (run (input Nx.float32 case))
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      group (Filename.basename file.path) (List.map pcen_case file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
