(* The lookahead algebra of Soundml.Pipeline, on both sides of a stage:

   lookahead = latency + output_latency / rate

   lookahead (f >> g) = lookahead f + lookahead g / rate f

   with the guarantee that after [n] input items a stage has emitted at least
   [ceil ((n - lookahead) * rate)] output items. The toys here are the two
   shapes the algebra has to carry: a rate-changing stage that trails its own
   output by a whole number of output items — a fraction of an input item, which
   no integral [latency] can name — and an identity that withholds on the output
   side what an equal [latency] would withhold on the input side, so that the
   two declarations must report the same number.

   Every declaration is checked twice: against the equation, and against a
   brute-force measurement of what the prepared stream actually emits. The
   pipeline law itself is checked for the new toys, since a stage that reports
   its lookahead wrongly is a bug, but a stage that emits the wrong samples is a
   worse one. *)

open Windtrap
open Soundml

let source ?(sample_rate = 1000) ?(channels = 1) () =
  Pipeline.Format.audio Nx.float32 ~sample_rate ~channels

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

let farray = array (float 0.)

let r num den = {Pipeline.Rate.num; den}

(* {1 Toys declaring an output-side lookahead} *)

let rate_3 = {Pipeline.Rate.num= 3; den= 1}

(* Each input item becomes three output items ([v], [2v], [3v]); the stage holds
   the last [hold] of them back and [flush] releases them. Its lookahead is
   [hold / 3] input items, which is not an integer. *)
let expand3 ~hold =
  let expand chunk =
    Array.init
      (3 * Array.length chunk)
      (fun i -> Float.of_int ((i mod 3) + 1) *. chunk.(i / 3))
  in
  Pipeline.kernel ~output_latency:hold ~rate:rate_3
    ~out_format:(fun fmt ->
      let open Pipeline in
      let ips = Rate.(Format.items_per_second fmt * rate_3) in
      let bound = Option.map (fun n -> 3 * n) (Format.max_items fmt) in
      Format.with_max_items bound (Format.with_items_per_second ips fmt) )
    ~flush:(fun st ->
      if Array.length !st = 0 then []
      else begin
        let tail = !st in
        st := [||] ;
        [tail]
      end )
    ~reset:(fun st -> st := [||])
    ~concat:Array.concat
    ~prepare:(fun _ -> ref [||])
    ~step:(fun st chunk ->
      let data = Array.append !st (expand chunk) in
      let n = Array.length data in
      if n <= hold then (
        st := data ;
        None )
      else begin
        st := Array.sub data (n - hold) hold ;
        Some (Array.sub data 0 (n - hold))
      end )
    ()

(* The identity withholding [d] items, declared on the output side. At rate
   [1:1] it is {!Toys.lookahead} [d] in every observable way, including what it
   reports — which is the point. *)
let hold_out d =
  Pipeline.kernel ~output_latency:d
    ~flush:(fun st ->
      if Array.length !st = 0 then []
      else begin
        let tail = !st in
        st := [||] ;
        [tail]
      end )
    ~reset:(fun st -> st := [||])
    ~concat:Array.concat
    ~prepare:(fun _ -> ref [||])
    ~step:(fun st chunk ->
      let data = Array.append !st chunk in
      let n = Array.length data in
      if n <= d then (
        st := data ;
        None )
      else begin
        st := Array.sub data (n - d) d ;
        Some (Array.sub data 0 (n - d))
      end )
    ()

(* {1 The equation} *)

let declaration_tests =
  [ test "output latency reports as a rational" (fun () ->
        equal ~msg:"2 items of output at rate 3" rate_t (r 2 3)
          (Pipeline.latency (expand3 ~hold:2)) ;
        equal ~msg:"nothing withheld" rate_t (r 0 1)
          (Pipeline.latency (expand3 ~hold:0)) ;
        equal ~msg:"5 items of output at rate 3" rate_t (r 5 3)
          (Pipeline.latency (expand3 ~hold:5)) ;
        equal ~msg:"rate 3 is unaffected" rate_t (r 3 1)
          (Pipeline.rate (expand3 ~hold:2)) )
  ; test "at rate 1:1 the two sides declare the same number" (fun () ->
        equal ~msg:"hold_out 3" rate_t
          (Pipeline.latency (Toys.lookahead 3))
          (Pipeline.latency (hold_out 3)) ;
        equal ~msg:"hold_out 3 is 3" rate_t (r 3 1)
          (Pipeline.latency (hold_out 3)) )
  ; test "composition folds the rational lookahead" (fun () ->
        equal ~msg:"decim4 >> expand3(2): 0 + (2/3) / (1/4)" rate_t (r 8 3)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.decim4 ()) (expand3 ~hold:2)) ) ;
        equal ~msg:"expand3(2) >> lookahead 3: 2/3 + 3/3" rate_t (r 5 3)
          (Pipeline.latency
             (Pipeline.( >> ) (expand3 ~hold:2) (Toys.lookahead 3)) ) ;
        equal ~msg:"expand3(2) >> expand3(2): 2/3 + (2/3)/3" rate_t (r 8 9)
          (Pipeline.latency
             (Pipeline.( >> ) (expand3 ~hold:2) (expand3 ~hold:2)) ) ;
        equal ~msg:"lookahead 3 >> expand3(2): 3 + 2/3" rate_t (r 11 3)
          (Pipeline.latency
             (Pipeline.( >> ) (Toys.lookahead 3) (expand3 ~hold:2)) ) ;
        equal ~msg:"hold_out 3 >> decim4: 3 + 0" rate_t (r 3 1)
          (Pipeline.latency (Pipeline.( >> ) (hold_out 3) (Toys.decim4 ()))) ;
        equal ~msg:"expand3(2) >> block_sum 2 >> lookahead 1" rate_t (r 4 3)
          (Pipeline.latency
             Pipeline.(expand3 ~hold:2 >> Toys.block_sum 2 >> Toys.lookahead 1) ) )
  ; test "omitting output_latency leaves every declaration unmoved" (fun () ->
        (* the values the suite has always pinned, restated here so that the new
           parameter is seen not to touch them *)
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
                (Toys.lookahead 2) ) ) ) ]

(* {1 The guarantee, brute-forced}

   [emitted_after n] is what a prepared stream has emitted after [n] one-item
   pushes; the declaration promises at least [ceil ((n - lookahead) * rate)]. *)

let ceil_div a b = if a <= 0 then 0 else (a + b - 1) / b

let promised ~lookahead ~rate n =
  let l = lookahead.Pipeline.Rate.num and ld = lookahead.Pipeline.Rate.den in
  let rn = rate.Pipeline.Rate.num and rd = rate.Pipeline.Rate.den in
  ceil_div (((n * ld) - l) * rn) (ld * rd)

let emission_profile p ~n =
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:1 in
  let emitted = ref 0 in
  let profile =
    List.init n (fun i ->
        ( match Pipeline.Stream.push s [|Float.of_int i|] with
        | None ->
            ()
        | Some out ->
            emitted := !emitted + Array.length out ) ;
        !emitted )
  in
  let tail =
    List.fold_left ( + ) 0 (List.map Array.length (Pipeline.Stream.flush s))
  in
  (profile, !emitted + tail)

let brute_force_case name p ~n =
  test name (fun () ->
      let lookahead = Pipeline.latency p and rate = Pipeline.rate p in
      let profile, total = emission_profile p ~n in
      List.iteri
        (fun i emitted ->
          let want = promised ~lookahead ~rate (i + 1) in
          is_true
            ~msg:
              (Printf.sprintf
                 "after %d items: emitted %d, declaration promises %d" (i + 1)
                 emitted want )
            (emitted >= want) )
        profile ;
      equal ~msg:"the drain releases everything" int
        (n * rate.Pipeline.Rate.num / rate.Pipeline.Rate.den)
        total )

(* The measured deficit against the naive rate map, in output items: what the
   declaration must be reporting, converted. *)
let measured_deficit p ~n =
  let rate = Pipeline.rate p in
  let profile, _ = emission_profile p ~n in
  List.fold_left max 0
    (List.mapi
       (fun i emitted ->
         ((i + 1) * rate.Pipeline.Rate.num / rate.Pipeline.Rate.den) - emitted )
       profile )

let brute_force_tests =
  [ brute_force_case "expand3(2) meets its declaration" (expand3 ~hold:2) ~n:40
  ; brute_force_case "expand3(5) meets its declaration" (expand3 ~hold:5) ~n:40
  ; brute_force_case "hold_out 7 meets its declaration" (hold_out 7) ~n:40
  ; brute_force_case "decim4 >> expand3(2) meets its declaration"
      (Pipeline.( >> ) (Toys.decim4 ()) (expand3 ~hold:2))
      ~n:40
  ; test "the declaration is the deficit, not a bound over it" (fun () ->
        equal ~msg:"expand3(2) withholds 2 output items" int 2
          (measured_deficit (expand3 ~hold:2) ~n:40) ;
        equal ~msg:"expand3(5) withholds 5 output items" int 5
          (measured_deficit (expand3 ~hold:5) ~n:40) ;
        equal ~msg:"hold_out 7 withholds 7 output items" int 7
          (measured_deficit (hold_out 7) ~n:40) ) ]

(* {1 Threading: the bound covers the withheld tail, and the format carries
   it} *)

let probe seen =
  Pipeline.kernel ~concat:Array.concat
    ~prepare:(fun fmt -> seen := Some fmt)
    ~step:(fun () (c : float array) -> Some c)
    ()

let threading_tests =
  [ test "output latency widens the threaded bound for drain" (fun () ->
        let seen = ref None in
        let p = Pipeline.( >> ) (hold_out 50) (probe seen) in
        ignore (Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:4) ;
        match !seen with
        | None ->
            fail "probe prepare did not run"
        | Some fmt ->
            equal ~msg:"bound covers the withheld tail" (option int) (Some 50)
              (Pipeline.Format.max_items fmt) )
  ; test "every flush chunk fits the threaded bound" (fun () ->
        let seen = ref None in
        let p = Pipeline.( >> ) (expand3 ~hold:5) (probe seen) in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:1 in
        let bound =
          match !seen with
          | Some fmt ->
              Option.get (Pipeline.Format.max_items fmt)
          | None ->
              fail "probe prepare did not run"
        in
        ignore (Pipeline.Stream.push s [|1.|]) ;
        List.iter
          (fun chunk ->
            is_true
              ~msg:
                (Printf.sprintf "flush chunk of %d items fits the bound %d"
                   (Array.length chunk) bound )
              (Array.length chunk <= bound) )
          (Pipeline.Stream.flush s) )
  ; test "upstream latency reaches downstream in source samples" (fun () ->
        let seen = ref None in
        let p = Pipeline.( >> ) (expand3 ~hold:2) (probe seen) in
        ignore (Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:8) ;
        match !seen with
        | None ->
            fail "probe prepare did not run"
        | Some fmt ->
            equal ~msg:"items/s tripled" rate_t (r 3000 1)
              (Pipeline.Format.items_per_second fmt) ;
            equal ~msg:"2 output items are 2/3 of a source sample" rate_t
              (r 2 3)
              (Pipeline.Format.upstream_latency fmt) )
  ; test "stream latency equals pipeline latency" (fun () ->
        let p = Pipeline.( >> ) (Toys.decim4 ()) (expand3 ~hold:2) in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:32 in
        equal ~msg:"latency" rate_t (Pipeline.latency p)
          (Pipeline.Stream.latency s) ) ]

(* {1 The pipeline law for the new toys} *)

let split x sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        Array.sub x off s :: go (off + s) rest
  in
  go 0 sizes

let partitions n =
  let rng = Random.State.make [|0x1a7|] in
  let rec random_sizes n =
    if n = 0 then []
    else
      let s = 1 + Random.State.int rng (min n 7) in
      s :: random_sizes (n - s)
  in
  [("ones", List.init n (fun _ -> 1)); ("whole", if n = 0 then [] else [n])]
  @ List.init 5 (fun i -> (Printf.sprintf "random-%d" i, random_sizes n))

let law_case name mk =
  test name (fun () ->
      List.iter
        (fun x ->
          let p = mk () in
          let src = source () in
          let expected = Pipeline.run ~source:src p x in
          List.iter
            (fun (pname, sizes) ->
              let s =
                Pipeline.Stream.prepare p ~source:src
                  ~max_chunk:(List.fold_left max 1 sizes)
              in
              let pushed =
                List.filter_map (Pipeline.Stream.push s) (split x sizes)
              in
              equal
                ~msg:(Printf.sprintf "%s/%s/n=%d" name pname (Array.length x))
                farray expected
                (Array.concat (pushed @ Pipeline.Stream.flush s)) )
            (partitions (Array.length x)) )
        [Array.init 37 (fun i -> Float.of_int i -. 18.5); [|1.; -2.|]; [||]] )

let law_tests =
  [ law_case "expand3(2)" (fun () -> expand3 ~hold:2)
  ; law_case "expand3(5)" (fun () -> expand3 ~hold:5)
  ; law_case "hold_out 7" (fun () -> hold_out 7)
  ; law_case "decim4 >> expand3(2)" (fun () ->
        Pipeline.( >> ) (Toys.decim4 ()) (expand3 ~hold:2) )
  ; law_case "expand3(2) >> block_sum 2" (fun () ->
        Pipeline.( >> ) (expand3 ~hold:2) (Toys.block_sum 2) ) ]

(* {1 Validation} *)

let validation_tests =
  [ test "a negative output latency is rejected" (fun () ->
        raises_invalid_arg ~msg:"negative output_latency"
          "Soundml.Pipeline.kernel: output_latency must be non-negative"
          (fun () ->
            ignore
              (Pipeline.kernel ~output_latency:(-1) ~concat:Array.concat
                 ~prepare:(fun _ -> ())
                 ~step:(fun () (c : float array) -> Some c)
                 () ) ) ) ]

let suite =
  [ group "declaration" declaration_tests
  ; group "brute-force" brute_force_tests
  ; group "threading" threading_tests
  ; group "law" law_tests
  ; group "validation" validation_tests ]
