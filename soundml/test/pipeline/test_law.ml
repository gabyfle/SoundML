(* The pipeline law, tested over randomized partitionings:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   for every partitioning of [x] — one-item chunks, chunks larger than any
   internal window, occasional empty chunks, the empty partition, and inputs
   shorter than the pipeline's total latency. The PRNG is seeded so CI is
   deterministic. *)

open Windtrap
open Soundml

let seed = 0x5eed

let source ?(sample_rate = 1000) ?(channels = 1) () =
  Pipeline.Format.audio Nx.float32 ~sample_rate ~channels

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

let farray = array (float 0.)

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
      equal
        ~msg:(Printf.sprintf "%s/%s/n=%d" name pname (Array.length x))
        farray expected got )
    (partitions rng (Array.length x))

let inputs =
  [ Array.init 61 (fun i -> float_of_int i -. 30.5)
  ; Array.init 17 (fun i -> float_of_int (i * 7 mod 5) -. 2.)
  ; [|1.; -2.|] (* shorter than most latencies and windows *)
  ; [||] (* empty input, empty partition *) ]

let law_case name mk =
  test name (fun () -> List.iter (fun x -> check_law ~name (mk ()) x) inputs)

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
      equal ~msg:(name ^ "/run/left") farray expected_f bf ;
      equal ~msg:(name ^ "/run/right") farray expected_g bg
  | _ ->
      fail (name ^ "/run: fanout run must pair both branches") ) ;
  List.iter
    (fun (pname, sizes) ->
      let chunks = split x sizes in
      let max_chunk = List.fold_left max 1 sizes in
      let outs = stream_outputs p ~source:src ~max_chunk chunks in
      let left = Array.concat (List.filter_map fst outs) in
      let right = Array.concat (List.filter_map snd outs) in
      equal ~msg:(Printf.sprintf "%s/%s/left" name pname) farray expected_f left ;
      equal
        ~msg:(Printf.sprintf "%s/%s/right" name pname)
        farray expected_g right )
    (partitions rng (Array.length x))

let fanout_tests =
  [ test "fanout gain/block_sum" (fun () ->
        List.iter
          (fun x ->
            check_fanout ~name:"fanout" (Toys.gain 2.0) (Toys.block_sum 4) x )
          inputs )
  ; test "fanout differing rate and latency" (fun () ->
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
      equal
        ~msg:(Printf.sprintf "%s/%s/n=%d" name pname (Array.length x))
        farray expected got )
    (partitions rng (Array.length x))

let nx_tests =
  [ test "nx gain>>lookahead" (fun () ->
        List.iter
          (fun x ->
            check_nx_law ~name:"nx"
              (Pipeline.( >> ) (Toys.nx_gain 2.0) (Toys.nx_lookahead 4))
              x )
          inputs )
  ; test "push borrows: no aliasing of the caller's buffer" (fun () ->
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
              fail "first push must emit"
        in
        let snap = Nx.to_array out1 in
        Nx.blit (Nx.create Nx.float32 [|6|] [|7.; 8.; 9.; 10.; 11.; 12.|]) buf ;
        let rest = Option.to_list (Pipeline.Stream.push s buf) in
        let tail = Pipeline.Stream.flush s in
        equal ~msg:"returned chunk survives buffer reuse" farray snap
          (Nx.to_array out1) ;
        equal ~msg:"stream equals the logical signal" farray
          (Array.init 12 (fun i -> float_of_int (i + 1)))
          (Nx.to_array (Toys.nx_concat ((out1 :: rest) @ tail))) ) ]

(* {1 Reset: a reset plan replays the law from scratch} *)

let reset_tests =
  [ test "reset restores the freshly prepared state" (fun () ->
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
        equal ~msg:"post-reset law" farray expected (Array.concat outs) )
  ; test "flush drains: a second flush emits nothing" (fun () ->
        let q = Pipeline.( >> ) (Toys.lookahead 3) (Toys.block_sum 2) in
        let s = Pipeline.Stream.prepare q ~source:(source ()) ~max_chunk:8 in
        ignore (Pipeline.Stream.push s [|1.; 2.; 3.; 4.; 5.|]) ;
        ignore (Pipeline.Stream.flush s) ;
        equal ~msg:"second flush" (list farray) [] (Pipeline.Stream.flush s) ;
        let nx =
          Pipeline.Stream.prepare (Toys.nx_lookahead 3) ~source:(source ())
            ~max_chunk:8
        in
        ignore (Pipeline.Stream.push nx (Nx.create Nx.float32 [|2|] [|1.; 2.|])) ;
        ignore (Pipeline.Stream.flush nx) ;
        equal ~msg:"second nx flush" int 0
          (List.length (Pipeline.Stream.flush nx)) ) ]

(* {1 Static queries: latency and rate fold exactly, as rationals} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ test "latency folds through rate changes" (fun () ->
        equal ~msg:"lookahead 3" rate_t (r 3 1)
          (Pipeline.latency (Toys.lookahead 3)) ;
        equal ~msg:"gain" rate_t (r 0 1) (Pipeline.latency (Toys.gain 2.)) ;
        equal ~msg:"lookahead 3 >> decim4" rate_t (r 3 1)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.lookahead 3) (Toys.decim4 ())) ) ;
        equal ~msg:"decim4 >> lookahead 3" rate_t (r 12 1)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.decim4 ()) (Toys.lookahead 3)) ) ;
        equal ~msg:"block_sum 3 >> lookahead 2" rate_t (r 6 1)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.block_sum 3) (Toys.lookahead 2)) ) ;
        equal ~msg:"fanout latency is the max" rate_t (r 12 1)
          (Pipeline.latency
             (Pipeline.fanout
                (Pipeline.( >> ) (Toys.decim4 ()) (Toys.lookahead 3))
                (Toys.lookahead 2) ) ) )
  ; test "rate folds and normalises" (fun () ->
        equal ~msg:"decim4 >> block_sum 2" rate_t (r 1 8)
          (Pipeline.rate (Pipeline.( >> ) (Toys.decim4 ()) (Toys.block_sum 2))) ;
        equal ~msg:"stateless is 1:1" rate_t (r 1 1)
          (Pipeline.rate (Toys.gain 2.)) ;
        equal ~msg:"Rate.( * ) normalises" rate_t (r 1 2)
          Pipeline.Rate.(r 2 4 * identity) )
  ; test "stream latency equals pipeline latency" (fun () ->
        let p = Pipeline.( >> ) (Toys.decim4 ()) (Toys.lookahead 3) in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:32 in
        equal ~msg:"latency" rate_t (Pipeline.latency p)
          (Pipeline.Stream.latency s) ) ]

(* {1 Format threading: each stage sees its own input format} *)

let probe seen =
  Pipeline.kernel ~concat:Array.concat
    ~prepare:(fun fmt -> seen := Some fmt)
    ~step:(fun () (c : float array) -> Some c)
    ()

let format_tests =
  [ test "formats thread left to right at prepare" (fun () ->
        let seen = ref None in
        let p = Pipeline.(Toys.decim4 () >> Toys.lookahead 3 >> probe seen) in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:100 in
        ignore s ;
        match !seen with
        | None ->
            fail "probe prepare did not run"
        | Some fmt ->
            equal ~msg:"items/s after 1:4 decimation" rate_t (r 250 1)
              (Pipeline.Format.items_per_second fmt) ;
            equal ~msg:"bound shrunk by the decimator" (option int) (Some 25)
              (Pipeline.Format.max_items fmt) ;
            equal ~msg:"channels" int 1 (Pipeline.Format.channels fmt) ;
            equal
              ~msg:"upstream latency in source samples (3 items at 250/s = 12)"
              rate_t (r 12 1)
              (Pipeline.Format.upstream_latency fmt) )
  ; test "run threads formats too" (fun () ->
        let seen = ref None in
        let p = Pipeline.( >> ) (Toys.block_sum 5) (probe seen) in
        ignore (Pipeline.run ~source:(source ()) p (Array.make 10 1.)) ;
        match !seen with
        | None ->
            fail "probe prepare did not run"
        | Some fmt ->
            equal ~msg:"items/s after 1:5 summing" rate_t (r 200 1)
              (Pipeline.Format.items_per_second fmt) ;
            equal ~msg:"offline bound stays unbounded" (option int) None
              (Pipeline.Format.max_items fmt) )
  ; test "latency widens the threaded bound for drain" (fun () ->
        (* a latency-50 stage may flush its whole 50-item tail as one chunk: the
           stage downstream must be sized for it at prepare *)
        let seen = ref None in
        let p = Pipeline.( >> ) (Toys.lookahead 50) (probe seen) in
        ignore (Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:4) ;
        match !seen with
        | None ->
            fail "probe prepare did not run"
        | Some fmt ->
            equal ~msg:"bound covers the drained tail" (option int) (Some 50)
              (Pipeline.Format.max_items fmt) )
  ; test "Format.equal and equal_dtype observe every field" (fun () ->
        let f = source () in
        let open Pipeline.Format in
        is_true ~msg:"reflexive" (equal f f) ;
        is_false ~msg:"dtype differs" (equal f (with_dtype Nx.float64 f)) ;
        is_false ~msg:"channels differ" (equal f (with_channels 2 f)) ;
        is_false ~msg:"bound differs" (equal f (with_max_items (Some 8) f)) ;
        is_true ~msg:"equal_dtype float32"
          (equal_dtype (dtype f) (Dtype Nx.float32)) ;
        is_false ~msg:"equal_dtype float64"
          (equal_dtype (dtype f) (Dtype Nx.float64)) ) ]

(* {1 Prepare-time validation: Invalid_argument, never mid-stream} *)

let expect_invalid_arg name f =
  raises_match
    ~msg:(name ^ ": expected Invalid_argument")
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> ignore (f ()))

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
  [ test "stage validates its format at prepare" (fun () ->
        let stereo = source ~channels:2 () in
        expect_invalid_arg "stereo block_sum" (fun () ->
            Pipeline.Stream.prepare (Toys.block_sum 4) ~source:stereo
              ~max_chunk:8 ) ;
        expect_invalid_arg "stereo block_sum offline" (fun () ->
            Pipeline.run ~source:stereo (Toys.block_sum 4) [|1.; 2.|] ) )
  ; test "out_format must agree with the declared rate" (fun () ->
        let bad = bad_rate_kernel (ref false) in
        expect_invalid_arg "inconsistent out_format" (fun () ->
            Pipeline.Stream.prepare bad ~source:(source ()) ~max_chunk:8 ) )
  ; test "fanout validates both branches at prepare" (fun () ->
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
        is_false ~msg:"no data flowed through the bad stage" !stepped )
  ; test "out_format built from scratch is rejected" (fun () ->
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
  ; test "constructor preconditions" (fun () ->
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
  [ test "offline_only composes and runs" (fun () ->
        let p = Pipeline.( >> ) (Toys.gain 2.0) (Toys.normalize ()) in
        let y = Pipeline.run ~source:(source ()) p [|1.; -4.; 2.|] in
        equal ~msg:"peak-normalised" farray [|0.25; -1.; 0.5|] y ;
        let q = Pipeline.( >> ) (Toys.normalize ()) (Toys.block_sum 2) in
        let z = Pipeline.run ~source:(source ()) q [|2.; 2.; -4.|] in
        equal ~msg:"offline mid-chain" farray [|1.; -1.|] z )
  ; test "input shorter than latency yields the concat [] chunk" (fun () ->
        let p = Pipeline.( >> ) (Toys.lookahead 8) (Toys.block_sum 100) in
        let y = Pipeline.run ~source:(source ()) p [||] in
        equal ~msg:"empty in, empty out" farray [||] y ) ]

let suite =
  [ group "law" law_tests
  ; group "law-fanout" fanout_tests
  ; group "law-nx" nx_tests
  ; group "reset" reset_tests
  ; group "static" static_tests
  ; group "format-threading" format_tests
  ; group "validation" validation_tests
  ; group "offline" offline_tests ]
