(* The pipeline law for delta_stage:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   exactly, over seeded partitionings — one-frame chunks, the whole sequence,
   random sizes and occasional empty chunks — for both borders, several window
   widths and orders, and sequence lengths around the window (the fewest frames
   `Interp accepts, and shorter ones under `Nearest). Also here: the stage
   against the flat function, the border property that makes `Interp streamable,
   the latency queries, and the errors. *)

open Windtrap
open Soundml

let seed = 0xde17a

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (min n 7) in
    s :: random_sizes rng (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

let partitions rng n =
  let named =
    [("ones", List.init n (fun _ -> 1)); ("whole", if n = 0 then [] else [n])]
  in
  let random =
    List.init 4 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))
  in
  let random_empty =
    List.init 2 (fun i ->
        ( Printf.sprintf "random-with-empty-chunks-%d" i
        , sprinkle_empty rng (random_sizes rng n) ) )
  in
  named @ random @ random_empty

let chunks_of x sizes =
  let nd = Nx.ndim x in
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        Nx.slice
          (List.init nd (fun i ->
               if i = nd - 1 then Nx.R (off, off + s) else Nx.A ) )
          x
        :: go (off + s) rest
  in
  go 0 sizes

let stream_outputs p ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

let features dtype lead frames =
  let shape = Array.append lead [|frames|] in
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create dtype shape
    (Array.init n (fun i -> Float.sin (0.37 *. Float.of_int (i * i mod 101))))

let law_case ~name ~width ~order ~mode ~lead frame_counts =
  test name (fun () ->
      let rng = Random.State.make [|seed; width; order|] in
      let p = delta_stage ~width ~order ~mode () in
      List.iter
        (fun frames ->
          let x = features Nx.float64 lead frames in
          let expected = Nx.to_array (delta ~width ~order ~mode x) in
          let offline = Nx.to_array (Pipeline.run ~source:(source ()) p x) in
          equal
            ~msg:(Printf.sprintf "run n=%d" frames)
            (array float_exact) expected offline ;
          List.iter
            (fun (pname, sizes) ->
              let max_chunk = List.fold_left max 1 sizes in
              let outs = stream_outputs p ~max_chunk (chunks_of x sizes) in
              let got =
                match outs with
                | [] ->
                    [||]
                | l ->
                    Nx.to_array (Nx.concatenate ~axis:(-1) l)
              in
              equal
                ~msg:(Printf.sprintf "%s n=%d" pname frames)
                (array float_exact) expected got )
            (partitions rng frames) )
        frame_counts )

let law_tests =
  [ law_case ~name:"interp, width 9, order 1, [3; frames]" ~width:9 ~order:1
      ~mode:`Interp ~lead:[|3|] [9; 10; 17; 40]
  ; law_case ~name:"interp, width 5, order 2, [2; 3; frames]" ~width:5 ~order:2
      ~mode:`Interp ~lead:[|2; 3|] [5; 6; 23]
  ; law_case ~name:"interp, width 3, order 1, [frames]" ~width:3 ~order:1
      ~mode:`Interp ~lead:[||] [3; 4; 11]
  ; law_case ~name:"nearest, width 9, order 1, [3; frames]" ~width:9 ~order:1
      ~mode:`Nearest ~lead:[|3|] [0; 1; 2; 8; 9; 40]
  ; law_case ~name:"nearest, width 11, order 3, [2; frames]" ~width:11 ~order:3
      ~mode:`Nearest ~lead:[|2|] [1; 5; 12; 30] ]

let property_tests =
  [ test "interp borders repeat the first and last full windows" (fun () ->
        let x = features Nx.float64 [|4|] 30 in
        let d = delta ~width:9 x in
        let col t = Nx.to_array (Nx.slice [Nx.A; Nx.R (t, t + 1)] d) in
        for t = 0 to 3 do
          equal
            ~msg:(Printf.sprintf "frame %d" t)
            (array float_exact) (col 4) (col t)
        done ;
        for t = 26 to 29 do
          equal
            ~msg:(Printf.sprintf "frame %d" t)
            (array float_exact) (col 25) (col t)
        done )
  ; test "the first-order weights are k over the sum of squares" (fun () ->
        (* a ramp's first derivative is its slope, at every frame and on both
           borders *)
        let x =
          Nx.create Nx.float64 [|12|]
            (Array.init 12 (fun i -> 0.5 *. Float.of_int i))
        in
        List.iter
          (fun mode ->
            let d = Nx.to_array (delta ~width:5 ~mode x) in
            Array.iteri
              (fun t v ->
                let expected =
                  match mode with
                  | `Interp ->
                      0.5
                  | `Nearest ->
                      (* the copied edge frames flatten the ramp *)
                      if t = 0 || t = 11 then 0.25
                      else if t = 1 || t = 10 then 0.4
                      else 0.5
                in
                equal
                  ~msg:(Printf.sprintf "frame %d" t)
                  (float 1e-15) expected v )
              d )
          [`Interp; `Nearest] )
  ; test "float32 input stays float32" (fun () ->
        let d = delta (features Nx.float32 [|2|] 12) in
        equal ~msg:"dtype" bool true (Nx.dtype d = Nx.float32) ) ]

let query_tests =
  [ test "latency is width - 1 frames under interp, width / 2 under nearest"
      (fun () ->
        equal ~msg:"interp" rate_t (r 8 1)
          (Pipeline.latency (delta_stage ~width:9 ())) ;
        equal ~msg:"nearest" rate_t (r 4 1)
          (Pipeline.latency (delta_stage ~width:9 ~mode:`Nearest ())) ;
        equal ~msg:"rate" rate_t (r 1 1) (Pipeline.rate (delta_stage ())) ) ]

let error_tests =
  [ test "width must be odd and at least 3" (fun () ->
        raises ~msg:"even"
          (Invalid_argument
             "delta: cannot differentiate over a window of 4 frames (width \
              must be an odd integer of at least 3)" ) (fun () ->
            ignore (delta ~width:4 (features Nx.float64 [||] 10)) ) ;
        raises ~msg:"stage"
          (Invalid_argument
             "delta_stage: cannot differentiate over a window of 1 frames \
              (width must be an odd integer of at least 3)" ) (fun () ->
            ignore (delta_stage ~width:1 ()) ) )
  ; test "order must lie in [1, width - 1]" (fun () ->
        raises ~msg:"order"
          (Invalid_argument
             "delta: cannot take a derivative of order 5 over 5 frames (order \
              must lie in [1, width - 1])" ) (fun () ->
            ignore (delta ~width:5 ~order:5 (features Nx.float64 [||] 10)) ) )
  ; test "interp needs at least width frames, offline and at the flush"
      (fun () ->
        let short = features Nx.float64 [|2|] 8 in
        let msg fn =
          Invalid_argument
            (Printf.sprintf
               "%s: cannot fit a window of 9 frames to 8 frames (mode `Interp \
                needs at least width frames)"
               fn )
        in
        raises ~msg:"offline" (msg "delta") (fun () -> ignore (delta short)) ;
        raises ~msg:"stream" (msg "delta_stage") (fun () ->
            ignore (stream_outputs (delta_stage ()) ~max_chunk:8 [short]) ) )
  ; test "a chunk must keep the stream's leading shape" (fun () ->
        raises ~msg:"leading shape"
          (Invalid_argument
             "delta_stage: cannot continue a stream of leading shape [3] with \
              a chunk of leading shape [4]" ) (fun () ->
            ignore
              (stream_outputs (delta_stage ()) ~max_chunk:10
                 [features Nx.float64 [|3|] 10; features Nx.float64 [|4|] 10] ) ) )
  ; test "rank zero is refused" (fun () ->
        raises ~msg:"rank"
          (Invalid_argument
             "delta: cannot differentiate a rank-zero tensor (the frame axis \
              must exist)" ) (fun () ->
            ignore (delta (Nx.scalar Nx.float64 1.)) ) ) ]

let suite =
  [ group "law" law_tests
  ; group "properties" property_tests
  ; group "queries" query_tests
  ; group "errors" error_tests ]
