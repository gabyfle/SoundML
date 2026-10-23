(* Golden parity against librosa 1.0: librosa.effects.trim and split over
   piecewise signals — runs of [length, scale] segments cut from one LCG stream
   under this suite's seed, mono or [channels; samples] in C order — replayed
   from the committed vectors. The generator keeps every frame at least 0.1 dB
   from the threshold, so the intervals compare exactly. Also here: the trimmed
   tensor is the slice, and the errors. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

let seed = 20261025

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let segments (case : Tutils.Golden.case) =
  match Tutils.Golden.param case "segments" with
  | `List l ->
      List.map
        (function
          | `List [length; scale] ->
              (Yojson.Safe.Util.to_int length, Yojson.Safe.Util.to_number scale)
          | _ ->
              failf "%s: malformed segment" case.name )
        l
  | _ ->
      failf "%s: segments is not a list" case.name

let segment_list name = function
  | `List l ->
      List.map
        (function
          | `List [length; scale] ->
              (Yojson.Safe.Util.to_int length, Yojson.Safe.Util.to_number scale)
          | _ ->
              failf "%s: malformed segment" name )
        l
  | _ ->
      failf "%s: a row's segments are not a list" name

(* [batch dtype case] rebuilds a batch case's signal: the stream cut into rows,
   each row times its own segments' scales, shaped [lead; samples]. *)
let batch dtype (case : Tutils.Golden.case) =
  let rows =
    match Tutils.Golden.param case "row_segments" with
    | `List rows ->
        List.map (segment_list case.name) rows
    | _ ->
        failf "%s: row_segments is not a list" case.name
  in
  let lead =
    match Tutils.Golden.param case "lead" with
    | `List l ->
        List.map Yojson.Safe.Util.to_int l
    | _ ->
        failf "%s: lead is not a list" case.name
  in
  let n = List.fold_left (fun a (l, _) -> a + l) 0 (List.hd rows) in
  let scales =
    Array.concat
      (List.concat_map (List.map (fun (l, s) -> Array.make l s)) rows)
  in
  let stream = lcg_values (List.length rows * n) in
  Nx.create dtype
    (Array.of_list (lead @ [n]))
    (Array.mapi (fun i v -> v *. scales.(i)) stream)

(* [signal dtype case] rebuilds the case's signal: the stream times each
   segment's scale, tiled over the channels. *)
let signal dtype (case : Tutils.Golden.case) =
  if List.mem_assoc "row_segments" case.params then batch dtype case
  else
    let segs = segments case in
    let channels = Tutils.Golden.int_param case "channels" in
    let n = List.fold_left (fun a (l, _) -> a + l) 0 segs in
    let scales = Array.concat (List.map (fun (l, s) -> Array.make l s) segs) in
    let stream = lcg_values (channels * n) in
    let values = Array.mapi (fun i v -> v *. scales.(i mod n)) stream in
    let shape = if channels > 1 then [|channels; n|] else [|n|] in
    Nx.create dtype shape values

let options (case : Tutils.Golden.case) =
  let has key = List.mem_assoc key case.params in
  let top_db =
    if has "top_db" then Some (Tutils.Golden.float_param case "top_db")
    else None
  in
  let reference =
    if has "reference" then
      Some (`Value (Tutils.Golden.float_param case "reference"))
    else None
  in
  let frame_length =
    if has "frame_length" then
      Some (Tutils.Golden.int_param case "frame_length")
    else None
  in
  let hop =
    if has "hop" then Some (Tutils.Golden.int_param case "hop") else None
  in
  (top_db, reference, frame_length, hop)

let ints values = Array.to_list (Array.map Float.to_int values)

let golden_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let top_db, reference, frame_length, hop = options case in
      let run (type a) (dtype : (float, a) Nx.dtype) =
        let x = signal dtype case in
        if String.starts_with ~prefix:"trim_" case.name then begin
          let y, (start, stop) =
            Effects.trim ?top_db ?reference ?frame_length ?hop x
          in
          equal ~msg:"interval" (list int) (ints case.values) [start; stop] ;
          let nd = Nx.ndim x in
          let expected =
            Nx.slice
              (List.init nd (fun i ->
                   if i = nd - 1 then Nx.R (start, stop) else Nx.A ) )
              x
          in
          equal ~msg:"trimmed shape" (array int) (Nx.shape expected)
            (Nx.shape y) ;
          equal ~msg:"trimmed samples" (array float_exact)
            (Nx.to_array (Nx.cast Nx.float64 expected))
            (Nx.to_array (Nx.cast Nx.float64 y))
        end
        else
          let got =
            Effects.split ?top_db ?reference ?frame_length ?hop x
            |> Array.to_list
            |> List.concat_map (fun (a, b) -> [a; b])
          in
          equal ~msg:"intervals" (list int) (ints case.values) got
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          run Nx.float64
      | "float32" ->
          run Nx.float32
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let error_tests =
  let x = Nx.create Nx.float64 [|100|] (Array.init 100 Float.of_int) in
  [ test "top_db must be finite" (fun () ->
        raises ~msg:"top_db"
          (Invalid_argument
             "trim: cannot threshold at inf dB below the reference (top_db \
              must be finite)" ) (fun () ->
            ignore (Effects.trim ~top_db:Float.infinity x) ) )
  ; test "a fixed reference must be finite and positive" (fun () ->
        raises ~msg:"reference"
          (Invalid_argument
             "split: cannot measure decibels against 0 (the reference must be \
              finite and positive)" ) (fun () ->
            ignore (Effects.split ~reference:(`Value 0.) x) ) )
  ; test "the framing is validated under the caller's name" (fun () ->
        raises ~msg:"hop"
          (Invalid_argument
             "split: cannot advance frames by 0 samples (hop must be at least \
              1)" ) (fun () -> ignore (Effects.split ~hop:0 x) ) ;
        raises ~msg:"frame_length"
          (Invalid_argument
             "trim: cannot analyse frames of 0 samples (frame_length must be \
              at least 1)" ) (fun () ->
            ignore (Effects.trim ~frame_length:0 x) ) )
  ; test "rank zero is refused" (fun () ->
        raises ~msg:"rank"
          (Invalid_argument
             "trim: cannot process a rank-zero tensor (the time axis must \
              exist)" ) (fun () ->
            ignore (Effects.trim (Nx.scalar Nx.float64 1.)) ) ) ]

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      group (Filename.basename file.path) (List.map golden_case file.cases) )
    (Tutils.Golden.load_dir vectors_dir)
  @ [group "errors" error_tests]
