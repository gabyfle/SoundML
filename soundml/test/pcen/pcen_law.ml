(* The pipeline law for pcen_stage:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   exactly, over seeded partitionings — one-frame chunks, the whole spectrogram,
   random sizes and occasional empty chunks — with and without the band maximum,
   at both dtypes and over batched input. Also here: the stage against the flat
   function, reset, the latency query, and the errors. *)

open Windtrap
open Soundml

let seed = 0x9ce17

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

let spectrogram dtype shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create dtype shape
    (Array.init n (fun i ->
         1e4 *. Float.abs (Float.sin (0.37 *. Float.of_int (i * i mod 211))) )
    )

let law_case (type a) ~name (dtype : (float, a) Nx.dtype) ~max_size shape =
  test name (fun () ->
      let rng = Random.State.make [|seed; max_size|] in
      let p = pcen_stage ~max_size ~sample_rate:16000 ~hop:160 () in
      let x = spectrogram dtype shape in
      let frames = shape.(Array.length shape - 1) in
      let expected =
        Nx.to_array (pcen ~max_size ~sample_rate:16000 ~hop:160 x)
      in
      equal ~msg:"run" (array float_exact) expected
        (Nx.to_array (Pipeline.run ~source:(source ()) p x)) ;
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
          equal ~msg:pname (array float_exact) expected got )
        (partitions rng frames) )

let law_tests =
  [ law_case ~name:"float32, [8; 37]" Nx.float32 ~max_size:1 [|8; 37|]
  ; law_case ~name:"float64, [2; 8; 23], max_size 4" Nx.float64 ~max_size:4
      [|2; 8; 23|]
  ; law_case ~name:"float64, one band [29]" Nx.float64 ~max_size:1 [|29|]
  ; law_case ~name:"float32, [5; 31], max_size 3" Nx.float32 ~max_size:3
      [|5; 31|] ]

let stage_tests =
  [ test "reset restarts the smoothing" (fun () ->
        let p = pcen_stage ~sample_rate:22050 () in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:16 in
        let x = spectrogram Nx.float64 [|4; 16|] in
        let first = Option.get (Pipeline.Stream.push s x) in
        ignore (Pipeline.Stream.push s x) ;
        Pipeline.Stream.reset s ;
        let again = Option.get (Pipeline.Stream.push s x) in
        equal ~msg:"after reset" (array float_exact) (Nx.to_array first)
          (Nx.to_array again) )
  ; test "no latency, frame rate kept" (fun () ->
        equal ~msg:"latency" rate_t (r 0 1)
          (Pipeline.latency (pcen_stage ~sample_rate:22050 ())) ;
        equal ~msg:"rate" rate_t (r 1 1)
          (Pipeline.rate (pcen_stage ~sample_rate:22050 ())) )
  ; test "b = 1 is the plain gain law" (fun () ->
        (* with no smoothing, M = E: pcen = (E / (eps + E) ** gain + bias) **
           power - bias ** power *)
        let x = spectrogram Nx.float64 [|3; 9|] in
        let got = Nx.to_array (pcen ~b:1. ~sample_rate:22050 x) in
        Array.iteri
          (fun i e ->
            let expected =
              Float.pow ((e /. Float.pow (1e-6 +. e) 0.98) +. 2.) 0.5
              -. Float.sqrt 2.
            in
            equal ~msg:(string_of_int i) (float 1e-12) expected got.(i) )
          (Nx.to_array x) ) ]

let error_tests =
  [ test "parameters are validated" (fun () ->
        raises ~msg:"gain"
          (Invalid_argument
             "pcen: cannot use gain = -1 (gain must be finite and non-negative)"
          ) (fun () ->
            ignore
              (pcen ~gain:(-1.) ~sample_rate:22050
                 (spectrogram Nx.float64 [|2; 3|]) ) ) ;
        raises ~msg:"eps"
          (Invalid_argument
             "pcen_stage: cannot use eps = 0 (eps must be finite and positive)"
          ) (fun () -> ignore (pcen_stage ~eps:0. ~sample_rate:22050 ()) ) ;
        raises ~msg:"b"
          (Invalid_argument
             "pcen: cannot use b = 0 (b must be finite and in (0, 1])" )
          (fun () ->
            ignore
              (pcen ~b:0. ~sample_rate:22050 (spectrogram Nx.float64 [|2; 3|])) ) ;
        raises ~msg:"sample_rate"
          (Invalid_argument
             "pcen: cannot smooth at 0 Hz (sample_rate must be at least 1)" )
          (fun () ->
            ignore (pcen ~sample_rate:0 (spectrogram Nx.float64 [|2; 3|])) ) )
  ; test "the band maximum needs a band axis" (fun () ->
        raises ~msg:"rank 1"
          (Invalid_argument
             "pcen: cannot take the maximum over bands of a rank-1 tensor \
              (max_size 3 needs [...; bands; frames])" ) (fun () ->
            ignore
              (pcen ~max_size:3 ~sample_rate:22050
                 (spectrogram Nx.float64 [|5|]) ) ) )
  ; test "a chunk must keep the stream's leading shape" (fun () ->
        raises ~msg:"leading shape"
          (Invalid_argument
             "pcen_stage: cannot continue a stream of leading shape [3] with a \
              chunk of leading shape [4]" ) (fun () ->
            ignore
              (stream_outputs
                 (pcen_stage ~sample_rate:22050 ())
                 ~max_chunk:5
                 [ spectrogram Nx.float64 [|3; 5|]
                 ; spectrogram Nx.float64 [|4; 5|] ] ) ) ) ]

let suite =
  [group "law" law_tests; group "stage" stage_tests; group "errors" error_tests]
