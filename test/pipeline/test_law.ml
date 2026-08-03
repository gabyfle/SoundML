(* The pipeline law, tested over randomized partitionings:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   for every partitioning of [x] — one-item chunks, chunks larger than any
   internal window, occasional empty chunks, the empty partition, and inputs
   shorter than the pipeline's total latency. The PRNG is seeded so CI is
   deterministic. *)

open Soundml

let seed = 0x5eed

let source ?(sample_rate = 1000) ?(channels = 1) () =
  Pipeline.Format.audio Nx.float32 ~sample_rate ~channels

let rate_t = Alcotest.testable Pipeline.Rate.pp Pipeline.Rate.equal

let farray = Alcotest.(array (float 0.))

(* {1 Partitionings} *)

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (min n 7) in
    s :: random_sizes rng (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

(* [partitions rng n] is a named list of size lists, each summing to [n]. *)
let partitions rng n =
  let named =
    [ ("ones", List.init n (fun _ -> 1))
    ; ("whole", if n = 0 then [] else [n])
    ; ("empty", []) ]
  in
  let named = List.filter (fun (_, s) -> List.fold_left ( + ) 0 s = n) named in
  let random =
    List.init 5 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))
  in
  let random_empty =
    List.init 2 (fun i ->
        ( Printf.sprintf "random-with-empty-chunks-%d" i
        , sprinkle_empty rng (random_sizes rng n) ) )
  in
  named @ random @ random_empty

let split x sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        Array.sub x off s :: go (off + s) rest
  in
  go 0 sizes

let stream_outputs p ~source ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

(* {1 The law over float arrays} *)

let check_law ~name p x =
  let rng = Random.State.make [|seed|] in
  let src = source () in
  let expected = Pipeline.run ~source:src p x in
  List.iter
    (fun (pname, sizes) ->
      let chunks = split x sizes in
      let max_chunk = List.fold_left max 1 sizes in
      let got = Array.concat (stream_outputs p ~source:src ~max_chunk chunks) in
      Alcotest.check farray
        (Printf.sprintf "%s/%s/n=%d" name pname (Array.length x))
        expected got )
    (partitions rng (Array.length x))

let inputs =
  [ Array.init 61 (fun i -> float_of_int i -. 30.5)
  ; Array.init 17 (fun i -> float_of_int (i * 7 mod 5) -. 2.)
  ; [|1.; -2.|] (* shorter than most latencies and windows *)
  ; [||] (* empty input, empty partition *) ]

let law_case name mk =
  Alcotest.test_case name `Quick (fun () ->
      List.iter (fun x -> check_law ~name (mk ()) x) inputs )

let law_tests =
  [ law_case "gain" (fun () -> Toys.gain 2.0)
  ; law_case "block_sum" (fun () -> Toys.block_sum 4)
  ; law_case "lookahead" (fun () -> Toys.lookahead 3)
  ; law_case "decim4" (fun () -> Toys.decim4 ())
  ; law_case "lookahead-longer-than-input" (fun () -> Toys.lookahead 100)
  ; law_case "gain>>block_sum" (fun () ->
        Pipeline.( >> ) (Toys.gain 2.0) (Toys.block_sum 4) )
  ; law_case "decim4>>lookahead" (fun () ->
        Pipeline.( >> ) (Toys.decim4 ()) (Toys.lookahead 3) )
  ; law_case "lookahead>>block_sum" (fun () ->
        Pipeline.( >> ) (Toys.lookahead 3) (Toys.block_sum 4) )
  ; law_case "block_sum>>decim4" (fun () ->
        Pipeline.( >> ) (Toys.block_sum 2) (Toys.decim4 ()) )
  ; law_case "gain>>lookahead>>decim4>>block_sum" (fun () ->
        Pipeline.(
          Toys.gain 0.5 >> Toys.lookahead 5 >> Toys.decim4 ()
          >> Toys.block_sum 2 ) )
  ; law_case "map-over-composite" (fun () ->
        Pipeline.map (Array.map Float.abs)
          (Pipeline.( >> ) (Toys.gain (-1.5)) (Toys.block_sum 3)) ) ]

(* {1 The law for fanout: compare against the pair of independent runs} *)

let check_fanout ~name f g x =
  let rng = Random.State.make [|seed + 1|] in
  let src = source () in
  let p = Pipeline.fanout f g in
  let expected_f = Pipeline.run ~source:src f x in
  let expected_g = Pipeline.run ~source:src g x in
  ( match Pipeline.run ~source:src p x with
  | Some bf, Some bg ->
      Alcotest.check farray (name ^ "/run/left") expected_f bf ;
      Alcotest.check farray (name ^ "/run/right") expected_g bg
  | _ ->
      Alcotest.fail (name ^ "/run: fanout run must pair both branches") ) ;
  List.iter
    (fun (pname, sizes) ->
      let chunks = split x sizes in
      let max_chunk = List.fold_left max 1 sizes in
      let outs = stream_outputs p ~source:src ~max_chunk chunks in
      let left = Array.concat (List.filter_map fst outs) in
      let right = Array.concat (List.filter_map snd outs) in
      Alcotest.check farray
        (Printf.sprintf "%s/%s/left" name pname)
        expected_f left ;
      Alcotest.check farray
        (Printf.sprintf "%s/%s/right" name pname)
        expected_g right )
    (partitions rng (Array.length x))

let fanout_tests =
  [ Alcotest.test_case "fanout gain/block_sum" `Quick (fun () ->
        List.iter
          (fun x ->
            check_fanout ~name:"fanout" (Toys.gain 2.0) (Toys.block_sum 4) x )
          inputs )
  ; Alcotest.test_case "fanout differing rate and latency" `Quick (fun () ->
        List.iter
          (fun x ->
            check_fanout ~name:"fanout-rl"
              (Pipeline.( >> ) (Toys.lookahead 3) (Toys.gain 0.5))
              (Toys.decim4 ()) x )
          inputs ) ]

(* {1 The law over Nx tensors} *)

let nx_of_array a = Nx.create Nx.float32 [|Array.length a|] a

(* [Nx.shrink] rejects zero-width slices; empty chunks are fresh tensors *)
let nx_slice t off s =
  if s = 0 then Nx.zeros Nx.float32 [|0|] else Nx.shrink [|(off, off + s)|] t

let check_nx_law ~name p x =
  let rng = Random.State.make [|seed + 2|] in
  let src = source () in
  let t = nx_of_array x in
  let expected = Nx.to_array (Pipeline.run ~source:src p t) in
  List.iter
    (fun (pname, sizes) ->
      let chunks =
        List.map
          (fun (off, s) -> nx_slice t off s)
          (List.rev
             (fst
                (List.fold_left
                   (fun (acc, off) s -> ((off, s) :: acc, off + s))
                   ([], 0) sizes ) ) )
      in
      let max_chunk = List.fold_left max 1 sizes in
      let outs = stream_outputs p ~source:src ~max_chunk chunks in
      let got = Nx.to_array (Toys.nx_concat outs) in
      Alcotest.check farray
        (Printf.sprintf "%s/%s/n=%d" name pname (Array.length x))
        expected got )
    (partitions rng (Array.length x))

let nx_tests =
  [ Alcotest.test_case "nx gain>>lookahead" `Quick (fun () ->
        List.iter
          (fun x ->
            check_nx_law ~name:"nx"
              (Pipeline.( >> ) (Toys.nx_gain 2.0) (Toys.nx_lookahead 4))
              x )
          inputs )
  ; Alcotest.test_case "push borrows: no aliasing of the caller's buffer" `Quick
      (fun () ->
        (* the caller reuses one block buffer across pushes — the audio callback
           pattern; previously returned chunks must not change *)
        let s =
          Pipeline.Stream.prepare (Toys.nx_lookahead 4) ~source:(source ())
            ~max_chunk:6
        in
        let buf = Nx.create Nx.float32 [|6|] [|1.; 2.; 3.; 4.; 5.; 6.|] in
        let out1 =
          match Pipeline.Stream.push s buf with
          | Some out ->
              out
          | None ->
              Alcotest.fail "first push must emit"
        in
        let snap = Nx.to_array out1 in
        Nx.blit (Nx.create Nx.float32 [|6|] [|7.; 8.; 9.; 10.; 11.; 12.|]) buf ;
        let rest = Option.to_list (Pipeline.Stream.push s buf) in
        let tail = Pipeline.Stream.flush s in
        Alcotest.check farray "returned chunk survives buffer reuse" snap
          (Nx.to_array out1) ;
        Alcotest.check farray "stream equals the logical signal"
          (Array.init 12 (fun i -> float_of_int (i + 1)))
          (Nx.to_array (Toys.nx_concat ((out1 :: rest) @ tail))) ) ]

(* {1 Reset: a reset plan replays the law from scratch} *)

let reset_tests =
  [ Alcotest.test_case "reset restores the freshly prepared state" `Quick
      (fun () ->
        let p =
          Pipeline.(Toys.lookahead 3 >> Toys.decim4 () >> Toys.block_sum 2)
        in
        let src = source () in
        let x = List.hd inputs in
        let expected = Pipeline.run ~source:src p x in
        let s = Pipeline.Stream.prepare p ~source:src ~max_chunk:16 in
        (* pollute the state, then reset *)
        ignore (Pipeline.Stream.push s (Array.sub x 0 13)) ;
        ignore (Pipeline.Stream.push s (Array.sub x 0 7)) ;
        Pipeline.Stream.reset s ;
        let pushed =
          List.filter_map (Pipeline.Stream.push s) (split x [16; 16; 16; 13])
        in
        let outs = pushed @ Pipeline.Stream.flush s in
        Alcotest.check farray "post-reset law" expected (Array.concat outs) )
  ; Alcotest.test_case "flush drains: a second flush emits nothing" `Quick
      (fun () ->
        let q = Pipeline.( >> ) (Toys.lookahead 3) (Toys.block_sum 2) in
        let s = Pipeline.Stream.prepare q ~source:(source ()) ~max_chunk:8 in
        ignore (Pipeline.Stream.push s [|1.; 2.; 3.; 4.; 5.|]) ;
        ignore (Pipeline.Stream.flush s) ;
        Alcotest.(check (list farray))
          "second flush" [] (Pipeline.Stream.flush s) ;
        let nx =
          Pipeline.Stream.prepare (Toys.nx_lookahead 3) ~source:(source ())
            ~max_chunk:8
        in
        ignore (Pipeline.Stream.push nx (Nx.create Nx.float32 [|2|] [|1.; 2.|])) ;
        ignore (Pipeline.Stream.flush nx) ;
        Alcotest.(check int)
          "second nx flush" 0
          (List.length (Pipeline.Stream.flush nx)) ) ]

(* {1 Static queries: latency and rate fold exactly, as rationals} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ Alcotest.test_case "latency folds through rate changes" `Quick (fun () ->
        Alcotest.check rate_t "lookahead 3" (r 3 1)
          (Pipeline.latency (Toys.lookahead 3)) ;
        Alcotest.check rate_t "gain" (r 0 1) (Pipeline.latency (Toys.gain 2.)) ;
        Alcotest.check rate_t "lookahead 3 >> decim4" (r 3 1)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.lookahead 3) (Toys.decim4 ())) ) ;
        Alcotest.check rate_t "decim4 >> lookahead 3" (r 12 1)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.decim4 ()) (Toys.lookahead 3)) ) ;
        Alcotest.check rate_t "block_sum 3 >> lookahead 2" (r 6 1)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.block_sum 3) (Toys.lookahead 2)) ) ;
        Alcotest.check rate_t "fanout latency is the max" (r 12 1)
          (Pipeline.latency
             (Pipeline.fanout
                (Pipeline.( >> ) (Toys.decim4 ()) (Toys.lookahead 3))
                (Toys.lookahead 2) ) ) )
  ; Alcotest.test_case "rate folds and normalises" `Quick (fun () ->
        Alcotest.check rate_t "decim4 >> block_sum 2" (r 1 8)
          (Pipeline.rate (Pipeline.( >> ) (Toys.decim4 ()) (Toys.block_sum 2))) ;
        Alcotest.check rate_t "stateless is 1:1" (r 1 1)
          (Pipeline.rate (Toys.gain 2.)) ;
        Alcotest.check rate_t "Rate.( * ) normalises" (r 1 2)
          Pipeline.Rate.(r 2 4 * identity) )
  ; Alcotest.test_case "stream latency equals pipeline latency" `Quick
      (fun () ->
        let p = Pipeline.( >> ) (Toys.decim4 ()) (Toys.lookahead 3) in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:32 in
        Alcotest.check rate_t "latency" (Pipeline.latency p)
          (Pipeline.Stream.latency s) ) ]

(* {1 Format threading: each stage sees its own input format} *)

let probe seen =
  Pipeline.kernel ~concat:Array.concat
    ~prepare:(fun fmt -> seen := Some fmt)
    ~step:(fun () (c : float array) -> Some c)
    ()

let format_tests =
  [ Alcotest.test_case "formats thread left to right at prepare" `Quick
      (fun () ->
        let seen = ref None in
        let p = Pipeline.(Toys.decim4 () >> Toys.lookahead 3 >> probe seen) in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:100 in
        ignore s ;
        match !seen with
        | None ->
            Alcotest.fail "probe prepare did not run"
        | Some fmt ->
            Alcotest.check rate_t "items/s after 1:4 decimation" (r 250 1)
              (Pipeline.Format.items_per_second fmt) ;
            Alcotest.(check (option int))
              "bound shrunk by the decimator" (Some 25)
              (Pipeline.Format.max_items fmt) ;
            Alcotest.(check int) "channels" 1 (Pipeline.Format.channels fmt) ;
            Alcotest.check rate_t
              "upstream latency in source samples (3 items at 250/s = 12)"
              (r 12 1)
              (Pipeline.Format.upstream_latency fmt) )
  ; Alcotest.test_case "run threads formats too" `Quick (fun () ->
        let seen = ref None in
        let p = Pipeline.( >> ) (Toys.block_sum 5) (probe seen) in
        ignore (Pipeline.run ~source:(source ()) p (Array.make 10 1.)) ;
        match !seen with
        | None ->
            Alcotest.fail "probe prepare did not run"
        | Some fmt ->
            Alcotest.check rate_t "items/s after 1:5 summing" (r 200 1)
              (Pipeline.Format.items_per_second fmt) ;
            Alcotest.(check (option int))
              "offline bound stays unbounded" None
              (Pipeline.Format.max_items fmt) )
  ; Alcotest.test_case "latency widens the threaded bound for drain" `Quick
      (fun () ->
        (* a latency-50 stage may flush its whole 50-item tail as one chunk: the
           stage downstream must be sized for it at prepare *)
        let seen = ref None in
        let p = Pipeline.( >> ) (Toys.lookahead 50) (probe seen) in
        ignore (Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:4) ;
        match !seen with
        | None ->
            Alcotest.fail "probe prepare did not run"
        | Some fmt ->
            Alcotest.(check (option int))
              "bound covers the drained tail" (Some 50)
              (Pipeline.Format.max_items fmt) )
  ; Alcotest.test_case "Format.equal and equal_dtype observe every field" `Quick
      (fun () ->
        let f = source () in
        let open Pipeline.Format in
        Alcotest.(check bool) "reflexive" true (equal f f) ;
        Alcotest.(check bool)
          "dtype differs" false
          (equal f (with_dtype Nx.float64 f)) ;
        Alcotest.(check bool)
          "channels differ" false
          (equal f (with_channels 2 f)) ;
        Alcotest.(check bool)
          "bound differs" false
          (equal f (with_max_items (Some 8) f)) ;
        Alcotest.(check bool)
          "equal_dtype float32" true
          (equal_dtype (dtype f) (Dtype Nx.float32)) ;
        Alcotest.(check bool)
          "equal_dtype float64" false
          (equal_dtype (dtype f) (Dtype Nx.float64)) ) ]

(* {1 Prepare-time validation: Invalid_argument, never mid-stream} *)

let expect_invalid_arg name f =
  match f () with
  | exception Invalid_argument _ ->
      ()
  | _ ->
      Alcotest.fail (name ^ ": expected Invalid_argument")

(* a kernel whose [out_format] forgets to scale items/s: must be rejected
   wherever it sits — standalone, composed, or inside a fanout branch. The
   [stepped] flag proves no data flowed before the rejection. *)
let bad_rate_kernel stepped =
  Pipeline.kernel
    ~rate:{Pipeline.Rate.num= 1; den= 2}
    ~out_format:(fun fmt -> fmt)
    ~concat:Array.concat
    ~prepare:(fun _ -> ())
    ~step:(fun () (c : float array) ->
      stepped := true ;
      Some c )
    ()

let validation_tests =
  [ Alcotest.test_case "stage validates its format at prepare" `Quick (fun () ->
        let stereo = source ~channels:2 () in
        expect_invalid_arg "stereo block_sum" (fun () ->
            Pipeline.Stream.prepare (Toys.block_sum 4) ~source:stereo
              ~max_chunk:8 ) ;
        expect_invalid_arg "stereo block_sum offline" (fun () ->
            Pipeline.run ~source:stereo (Toys.block_sum 4) [|1.; 2.|] ) )
  ; Alcotest.test_case "out_format must agree with the declared rate" `Quick
      (fun () ->
        let bad = bad_rate_kernel (ref false) in
        expect_invalid_arg "inconsistent out_format" (fun () ->
            Pipeline.Stream.prepare bad ~source:(source ()) ~max_chunk:8 ) )
  ; Alcotest.test_case "fanout validates both branches at prepare" `Quick
      (fun () ->
        let stepped = ref false in
        expect_invalid_arg "bad left branch" (fun () ->
            Pipeline.Stream.prepare
              (Pipeline.fanout (bad_rate_kernel stepped) (Toys.gain 1.))
              ~source:(source ()) ~max_chunk:8 ) ;
        (* a bad terminal stage of a multi-stage branch, through run: the
           rejection must come before any data flows *)
        expect_invalid_arg "bad nested right branch" (fun () ->
            Pipeline.run ~source:(source ())
              (Pipeline.fanout (Toys.gain 1.)
                 (Pipeline.( >> ) (Toys.gain 1.) (bad_rate_kernel stepped)) )
              [|1.; 2.; 3.|] ) ;
        Alcotest.(check bool)
          "no data flowed through the bad stage" false !stepped )
  ; Alcotest.test_case "out_format built from scratch is rejected" `Quick
      (fun () ->
        let fabricator =
          Pipeline.kernel
            ~out_format:(fun _ ->
              Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1 )
            ~concat:Array.concat
            ~prepare:(fun _ -> ())
            ~step:(fun () (c : float array) -> Some c)
            ()
        in
        expect_invalid_arg "from-scratch out_format at prepare" (fun () ->
            Pipeline.Stream.prepare fabricator ~source:(source ()) ~max_chunk:8 ) ;
        expect_invalid_arg "from-scratch out_format offline" (fun () ->
            Pipeline.run ~source:(source ()) fabricator [|1.|] ) )
  ; Alcotest.test_case "constructor preconditions" `Quick (fun () ->
        expect_invalid_arg "negative latency" (fun () ->
            Pipeline.kernel ~latency:(-1) ~concat:Array.concat
              ~prepare:(fun _ -> ())
              ~step:(fun () (c : float array) -> Some c)
              () ) ;
        expect_invalid_arg "zero rate" (fun () ->
            Pipeline.kernel
              ~rate:{Pipeline.Rate.num= 0; den= 1}
              ~concat:Array.concat
              ~prepare:(fun _ -> ())
              ~step:(fun () (c : float array) -> Some c)
              () ) ;
        expect_invalid_arg "audio sample_rate" (fun () ->
            Pipeline.Format.audio Nx.float32 ~sample_rate:0 ~channels:1 ) ;
        expect_invalid_arg "audio channels" (fun () ->
            Pipeline.Format.audio Nx.float32 ~sample_rate:44100 ~channels:0 ) ;
        expect_invalid_arg "max_chunk" (fun () ->
            Pipeline.Stream.prepare (Toys.gain 1.) ~source:(source ())
              ~max_chunk:0 ) ) ]

(* {1 Offline pipelines run; concat [] covers the empty tail} *)

let offline_tests =
  [ Alcotest.test_case "offline_only composes and runs" `Quick (fun () ->
        let p = Pipeline.( >> ) (Toys.gain 2.0) (Toys.normalize ()) in
        let y = Pipeline.run ~source:(source ()) p [|1.; -4.; 2.|] in
        Alcotest.check farray "peak-normalised" [|0.25; -1.; 0.5|] y ;
        let q = Pipeline.( >> ) (Toys.normalize ()) (Toys.block_sum 2) in
        let z = Pipeline.run ~source:(source ()) q [|2.; 2.; -4.|] in
        Alcotest.check farray "offline mid-chain" [|1.; -1.|] z )
  ; Alcotest.test_case "input shorter than latency yields the concat [] chunk"
      `Quick (fun () ->
        let p = Pipeline.( >> ) (Toys.lookahead 8) (Toys.block_sum 100) in
        let y = Pipeline.run ~source:(source ()) p [||] in
        Alcotest.check farray "empty in, empty out" [||] y ) ]

let tests =
  [ ("law", law_tests)
  ; ("law-fanout", fanout_tests)
  ; ("law-nx", nx_tests)
  ; ("reset", reset_tests)
  ; ("static", static_tests)
  ; ("format-threading", format_tests)
  ; ("validation", validation_tests)
  ; ("offline", offline_tests) ]
