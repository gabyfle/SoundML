(* The pipeline law for Effects.split_stage:

   run ~source p x = Effects.split ~reference:(`Value r) x = Array.concat
   (List.filter_map (Stream.push s) (chunks x) @ Stream.flush s)

   exactly — the intervals are integers — over seeded partitionings: the whole
   signal, one-sample pushes, odd sizes, sizes below a frame and below a third
   of one, sizes up to three frames, and empty chunks among them; over frame
   geometries with even and odd frame lengths, frame_length 1, hops above half
   the frame and above the frame, and a reference small enough that digital
   silence is non-silent; at both dtypes, over the leading shapes [], [2] and
   [2; 3], on piecewise signals whose rows are loud at different times, of
   lengths around the frame and the hop. Every interval is also checked to
   arrive on the push that completes its first silent frame, and on no other.
   The offline function is itself checked against librosa's edge rule, every
   edge clamped. Also here: the stage over the golden vectors' fixed-reference
   cases, the span identity with trim, the queries, the state, and the
   errors. *)

open Windtrap
open Soundml

let seed = 0x5917

let source dtype = Pipeline.Format.audio dtype ~sample_rate:16000 ~channels:1

let intervals = array (pair int int)

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let last_dim t = Nx.dim (Nx.ndim t - 1) t

(* {1 Signals} *)

let quiet = [|0.; 1e-4|]

let loud = [|1e-2; 0.3; 0.8|]

(* [signal rng dtype lead n ~frame_length] is a signal shaped [lead @ [n]] whose
   rows each run through their own segments, a segment a scale times uniform
   draws in [-1, 1): loud scales over at most [frame_length] samples, quiet ones
   over at most six frames. Loud segments are rarer the more rows there are, so
   the rows, loud at different times, still leave silent frames between them. *)
let signal rng dtype lead n ~frame_length =
  let rows = Array.fold_left ( * ) 1 lead in
  let values = Array.make (rows * n) 0. in
  for r = 0 to rows - 1 do
    let i = ref 0 in
    while !i < n do
      let scales, span =
        if Random.State.int rng (rows + 1) = 0 then (loud, frame_length)
        else (quiet, 6 * frame_length)
      in
      let len = Int.min (n - !i) (1 + Random.State.int rng span) in
      let s = scales.(Random.State.int rng (Array.length scales)) in
      for k = !i to !i + len - 1 do
        values.((r * n) + k) <- s *. (Random.State.float rng 2. -. 1.)
      done ;
      i := !i + len
    done
  done ;
  Nx.create dtype (Array.append lead [|n|]) values

(* {1 Partitionings} *)

let rec sizes_below rng bound n =
  if n = 0 then []
  else
    let s = Int.min n (1 + Random.State.int rng bound) in
    s :: sizes_below rng bound (n - s)

let rec odd_sizes rng n =
  if n = 0 then []
  else
    let s = Int.min n (1 + (2 * Random.State.int rng 4)) in
    s :: odd_sizes rng (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

(* [partitions rng ~frame_length ~ones n] is the named partitionings of [n]
   samples, one-sample pushes among them when [ones]. *)
let partitions rng ~frame_length ~ones n =
  [ ("whole", if n = 0 then [] else [n])
  ; ("odd", odd_sizes rng n)
  ; ("below-frame", sizes_below rng (Int.max 1 (frame_length - 1)) n)
  ; ("below-third", sizes_below rng (Int.max 1 (frame_length / 3)) n)
  ; ("up-to-three-frames", sizes_below rng (3 * frame_length) n)
  ; ( "with-empty"
    , 0 :: sprinkle_empty rng (sizes_below rng (Int.max 1 (frame_length / 2)) n)
    ) ]
  @ if ones then [("ones", List.init n (fun _ -> 1))] else []

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

(* [stream p dtype chunks] pushes [chunks] and flushes, and is every interval in
   order with where it arrived: [`Push (before, after)], the samples seen before
   and after the push that emitted it, or [`Flush]. *)
let stream p dtype chunks =
  let max_chunk = List.fold_left (fun m c -> Int.max m (last_dim c)) 1 chunks in
  let s = Pipeline.Stream.prepare p ~source:(source dtype) ~max_chunk in
  let seen = ref 0 and out = ref [] in
  List.iter
    (fun c ->
      let before = !seen in
      seen := before + last_dim c ;
      match Pipeline.Stream.push s c with
      | None ->
          ()
      | Some a ->
          Array.iter (fun i -> out := (i, `Push (before, !seen)) :: !out) a )
    chunks ;
  List.iter
    (Array.iter (fun i -> out := (i, `Flush) :: !out))
    (Pipeline.Stream.flush s) ;
  List.rev !out

(* [check_arrivals ~frame_length ~n arrived] checks that each interval arrived
   on the push that completed its first silent frame — the one bringing the
   samples seen to [stop + (frame_length + 1) / 2] — or, when that frame reaches
   past the [n] samples, at the flush. *)
let check_arrivals ~msg ~frame_length ~n arrived =
  let reach = (frame_length + 1) / 2 in
  List.iter
    (fun ((a, e), where) ->
      match where with
      | `Push (before, after) ->
          if not (before < e + reach && e + reach <= after) then
            failf "%s: (%d, %d) arrived on the push from %d to %d samples" msg a
              e before after
      | `Flush ->
          if e + reach <= n then
            failf "%s: (%d, %d) waited for the flush of %d samples" msg a e n )
    arrived

(* [librosa_split ~top_db ~frame_length ~hop ~r x] is split as librosa spells
   it, over the frames of [Soundml.rms]: the edges where the verdict flips, the
   frame count appended when the last frame is non-silent, and every edge turned
   into samples and clamped to [n] — the oracle for the stage's and the offline
   function's unclamped interior edges. *)
let librosa_split ~top_db ~frame_length ~hop ~r x =
  let e = rms ~frame_length ~hop x in
  let frames = last_dim e in
  let mse = Nx.to_array (Nx.cast Nx.float64 e) in
  let rows = if frames = 0 then 0 else Array.length mse / frames in
  let amin = 1e-5 *. 1e-5 in
  let offset = 10. *. Float.log10 (Float.max amin (r *. r)) in
  let loud =
    Array.init frames (fun t ->
        let loudest = ref Float.neg_infinity in
        for row = 0 to rows - 1 do
          let v = mse.((row * frames) + t) in
          let db = (10. *. Float.log10 (Float.max amin (v *. v))) -. offset in
          if db > !loudest then loudest := db
        done ;
        !loudest > -.top_db )
  in
  let n = last_dim x in
  let edges = ref (if frames > 0 && loud.(0) then [0] else []) in
  for t = 1 to frames - 1 do
    if loud.(t) <> loud.(t - 1) then edges := t :: !edges
  done ;
  if frames > 0 && loud.(frames - 1) then edges := frames :: !edges ;
  let samples =
    Array.of_list (List.rev_map (fun f -> Int.min n (f * hop)) !edges)
  in
  Array.init
    (Array.length samples / 2)
    (fun i -> (samples.(2 * i), samples.((2 * i) + 1)))

let streamed p dtype x sizes =
  Array.of_list (List.map fst (stream p dtype (chunks_of x sizes)))

let span split =
  let k = Array.length split in
  if k = 0 then (0, 0) else (fst split.(0), snd split.(k - 1))

(* {1 The law} *)

type geometry =
  {frame_length: int; hop: int; top_db: float; r: float; lengths: int list}

let small ~frame_length ~hop ~top_db ~r =
  { frame_length
  ; hop
  ; top_db
  ; r
  ; lengths=
      List.sort_uniq Int.compare
        [ 0
        ; 1
        ; 2
        ; 7
        ; 8
        ; 64
        ; Int.max 0 (frame_length - 1)
        ; frame_length
        ; frame_length + 1
        ; 12 * hop
        ; 40 * hop
        ; 400 * hop ] }

let geometries =
  [ { frame_length= 2048
    ; hop= 512
    ; top_db= 60.
    ; r= 1.0
    ; lengths= [0; 300; 9000; 17500; 60000] }
  ; { frame_length= 2048
    ; hop= 512
    ; top_db= 30.
    ; r= 0.1
    ; lengths= [0; 300; 9000; 40000] }
  ; { frame_length= 2048
    ; hop= 512
    ; top_db= 60.
    ; r= 1e-3
    ; lengths= [0; 300; 9000; 40000] }
  ; { frame_length= 512
    ; hop= 128
    ; top_db= 60.
    ; r= 1.0
    ; lengths= [0; 300; 2048; 5000; 15000] }
  ; { frame_length= 511
    ; hop= 128
    ; top_db= 60.
    ; r= 1.0
    ; lengths= [0; 300; 2047; 5000; 15000] }
  ; small ~frame_length:5 ~hop:2 ~top_db:20. ~r:0.5
  ; small ~frame_length:4 ~hop:2 ~top_db:20. ~r:0.5
  ; small ~frame_length:2 ~hop:4 ~top_db:20. ~r:0.5
  ; small ~frame_length:1 ~hop:3 ~top_db:20. ~r:0.5
  ; small ~frame_length:3 ~hop:4 ~top_db:20. ~r:0.5
  ; small ~frame_length:7 ~hop:3 ~top_db:40. ~r:1e-3
  ; small ~frame_length:6 ~hop:6 ~top_db:30. ~r:0.2 ]

let law_case (type a) (dtype : (float, a) Nx.dtype) dname lead g =
  let name =
    Printf.sprintf "fl %d, hop %d, top_db %g, ref %g, %s, [%s]" g.frame_length
      g.hop g.top_db g.r dname
      (String.concat "; "
         (List.map string_of_int (Array.to_list lead) @ ["n"]) )
  in
  test name (fun () ->
      let rng =
        Random.State.make [|seed; g.frame_length; g.hop; Array.length lead|]
      in
      let {frame_length; hop; top_db; r; _} = g in
      let p =
        Effects.split_stage ~top_db ~frame_length ~hop ~reference:(`Value r) ()
      in
      List.iter
        (fun n ->
          let x = signal rng dtype lead n ~frame_length in
          let expected =
            Effects.split ~top_db ~frame_length ~hop ~reference:(`Value r) x
          in
          let msg what = Printf.sprintf "%s n=%d" what n in
          equal ~msg:(msg "librosa's edges") intervals
            (librosa_split ~top_db ~frame_length ~hop ~r x)
            expected ;
          equal ~msg:(msg "run") intervals expected
            (Pipeline.run ~source:(source dtype) p x) ;
          let _, kept =
            Effects.trim ~top_db ~frame_length ~hop ~reference:(`Value r) x
          in
          equal ~msg:(msg "trim is the span") (pair int int) (span expected)
            kept ;
          List.iter
            (fun (pname, sizes) ->
              let arrived = stream p dtype (chunks_of x sizes) in
              equal ~msg:(msg pname) intervals expected
                (Array.of_list (List.map fst arrived)) ;
              check_arrivals ~msg:(msg pname) ~frame_length ~n arrived )
            (partitions rng ~frame_length ~ones:true n) )
        g.lengths )

let law_tests =
  List.concat_map
    (fun g ->
      List.concat_map
        (fun lead ->
          [ law_case Nx.float64 "float64" lead g
          ; law_case Nx.float32 "float32" lead g ] )
        [[||]; [|2|]; [|2; 3|]] )
    geometries

(* {1 The stage over the golden vectors}

   Every split case with a fixed reference, replayed through the stage on the
   whole signal, one-sample pushes, odd sizes and sizes below a frame: the stage
   against librosa directly. *)

let golden_tests =
  List.concat_map
    (fun (file : Tutils.Golden.file) ->
      List.filter_map
        (fun (case : Tutils.Golden.case) ->
          if
            String.starts_with ~prefix:"split_" case.name
            && List.mem_assoc "reference" case.params
          then
            Some
              (test case.name (fun () ->
                   let top_db, reference, frame_length, hop =
                     Trim_goldens.options case
                   in
                   let reference =
                     match reference with
                     | Some (`Value r) ->
                         `Value r
                     | None ->
                         failf "%s: no reference" case.name
                   in
                   let expected =
                     let v = Array.map Float.to_int case.values in
                     Array.init
                       (Array.length v / 2)
                       (fun i -> (v.(2 * i), v.((2 * i) + 1)))
                   in
                   let run (type a) (dtype : (float, a) Nx.dtype) =
                     let x = Trim_goldens.signal dtype case in
                     let p =
                       Effects.split_stage ?top_db ?frame_length ?hop ~reference
                         ()
                     in
                     equal ~msg:"run" intervals expected
                       (Pipeline.run ~source:(source dtype) p x) ;
                     let rng = Random.State.make [|seed; last_dim x|] in
                     let fl = Option.value frame_length ~default:2048 in
                     List.iter
                       (fun (pname, sizes) ->
                         equal ~msg:pname intervals expected
                           (streamed p dtype x sizes) )
                       (List.filter
                          (fun (pname, _) ->
                            List.mem pname
                              ["whole"; "ones"; "odd"; "below-frame"] )
                          (partitions rng ~frame_length:fl ~ones:true
                             (last_dim x) ) )
                   in
                   match Tutils.Golden.string_param case "dtype" with
                   | "float64" ->
                       run Nx.float64
                   | "float32" ->
                       run Nx.float32
                   | other ->
                       failf "golden case %s: unknown dtype %s" case.name other )
              )
          else None )
        file.cases )
    (Tutils.Golden.load_dir Trim_goldens.vectors_dir)

(* {1 Emission, queries and state} *)

let burst () =
  Nx.create Nx.float64 [|8000|]
    (Array.init 8000 (fun i -> if i < 3000 then 0.5 else 0.))

let emission_tests =
  [ test "an interval arrives with its first silent frame" (fun () ->
        (* the run of frames 0 to 7 closes at frame 8, centred at 4096, whose
           window ends with sample 5119 *)
        let p = Effects.split_stage ~reference:(`Value 1.) () in
        let x = burst () in
        equal ~msg:"offline" intervals
          [|(0, 4096)|]
          (Effects.split ~reference:(`Value 1.) x) ;
        let arrived =
          stream p Nx.float64 (chunks_of x (List.init 8000 (fun _ -> 1)))
        in
        equal ~msg:"arrival"
          (list (pair (pair int int) int))
          [((0, 4096), 5120)]
          (List.map
             (function
               | i, `Push (_, after) -> (i, after) | i, `Flush -> (i, -1) )
             arrived ) )
  ; test "the last frame alone gives the empty interval (n, n)" (fun () ->
        let x = Nx.create Nx.float64 [|8|] [|0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.|] in
        let p =
          Effects.split_stage ~frame_length:2 ~hop:4 ~reference:(`Value 1.) ()
        in
        equal ~msg:"offline" intervals
          [|(8, 8)|]
          (Effects.split ~frame_length:2 ~hop:4 ~reference:(`Value 1.) x) ;
        let rng = Random.State.make [|seed; 8|] in
        List.iter
          (fun (pname, sizes) ->
            equal ~msg:pname intervals
              [|(8, 8)|]
              (streamed p Nx.float64 x sizes) )
          ( partitions rng ~frame_length:2 ~ones:true 8
          @ [("3+5", [3; 5]); ("7+1", [7; 1])] ) )
  ; test "a run open at the end closes at the flush, at n" (fun () ->
        let x =
          Nx.create Nx.float32 [|2; 1500|]
            (Array.init 3000 (fun i -> if i < 1500 then 0. else 0.3))
        in
        let p =
          Effects.split_stage ~frame_length:512 ~hop:128 ~reference:(`Value 1.)
            ()
        in
        match stream p Nx.float32 [x] with
        | [((_, stop), `Flush)] ->
            equal ~msg:"stop" int 1500 stop
        | _ ->
            fail "expected one interval, at the flush" ) ]

let query_tests =
  [ test "latency is frame_length / 2 and rate 1/hop, as rms_stage" (fun () ->
        let p = Effects.split_stage ~reference:(`Value 1.) () in
        equal ~msg:"latency" rate_t
          Pipeline.Rate.{num= 1024; den= 1}
          (Pipeline.latency p) ;
        equal ~msg:"rate" rate_t
          Pipeline.Rate.{num= 1; den= 512}
          (Pipeline.rate p) ;
        equal ~msg:"odd latency" rate_t
          Pipeline.Rate.{num= 255; den= 1}
          (Pipeline.latency
             (Effects.split_stage ~frame_length:511 ~hop:128
                ~reference:(`Value 1.) () ) ) ;
        equal ~msg:"as rms_stage" rate_t
          (Pipeline.latency (rms_stage ~frame_length:511 ~hop:128 ()))
          (Pipeline.latency
             (Effects.split_stage ~frame_length:511 ~hop:128
                ~reference:(`Value 1.) () ) ) ;
        equal ~msg:"no frame border" rate_t
          Pipeline.Rate.{num= 0; den= 1}
          (Pipeline.latency
             (Effects.split_stage ~frame_length:1 ~hop:3 ~reference:(`Value 1.)
                () ) ) ) ]

let state_tests =
  let p =
    Effects.split_stage ~frame_length:512 ~hop:128 ~reference:(`Value 1.) ()
  in
  (* three bursts, the last on the second row alone *)
  let x =
    Nx.create Nx.float64 [|2; 6000|]
      (Array.init 12000 (fun i ->
           let row = i / 6000 and t = i mod 6000 in
           let loud =
             if row = 0 then (t >= 500 && t < 1500) || (t >= 3000 && t < 3500)
             else t >= 4500 && t < 5200
           in
           if loud then 0.5 *. Float.sin (0.1 *. Float.of_int t) else 0. ) )
  in
  let prepare () =
    Pipeline.Stream.prepare p ~source:(source Nx.float64) ~max_chunk:1000
  in
  let feed s =
    List.filter_map (Pipeline.Stream.push s)
      (chunks_of x (List.init 6 (fun _ -> 1000)))
  in
  [ test "flush drains: a second flush emits nothing" (fun () ->
        let s = prepare () in
        let pushed = feed s in
        let flushed = Pipeline.Stream.flush s in
        equal ~msg:"total" intervals
          (Effects.split ~frame_length:512 ~hop:128 ~reference:(`Value 1.) x)
          (Array.concat (pushed @ flushed)) ;
        equal ~msg:"second flush" (list intervals) [] (Pipeline.Stream.flush s) )
  ; test "a push after the flush is refused" (fun () ->
        let s = prepare () in
        ignore (feed s) ;
        ignore (Pipeline.Stream.flush s) ;
        raises ~msg:"drained"
          (Invalid_argument
             "step: cannot feed a drained kernel (flush consumed the tail; \
              reset before reusing)" ) (fun () ->
            ignore (Pipeline.Stream.push s (List.hd (chunks_of x [1000]))) ) )
  ; test "reset restarts the stream's sample count" (fun () ->
        let s = prepare () in
        let first = feed s in
        let first = Array.concat (first @ Pipeline.Stream.flush s) in
        Pipeline.Stream.reset s ;
        let second = feed s in
        let second = Array.concat (second @ Pipeline.Stream.flush s) in
        equal ~msg:"rerun" intervals first second ;
        equal ~msg:"three intervals" int 3 (Array.length first) )
  ; test "an empty stream has no interval" (fun () ->
        let s = prepare () in
        equal ~msg:"flush alone" (list intervals) [] (Pipeline.Stream.flush s) ;
        equal ~msg:"run" intervals [||]
          (Pipeline.run ~source:(source Nx.float64) p
             (Nx.zeros Nx.float64 [|2; 0|]) ) ;
        equal ~msg:"offline" intervals [||]
          (Effects.split ~reference:(`Value 1e-3) (Nx.zeros Nx.float64 [|0|])) )
  ; test "a stream of no signals has no interval" (fun () ->
        let none = Nx.zeros Nx.float64 [|0; 3000|] in
        equal ~msg:"offline" intervals [||]
          (Effects.split ~frame_length:512 ~hop:128 ~reference:(`Value 1e-3)
             none ) ;
        let q =
          Effects.split_stage ~frame_length:512 ~hop:128
            ~reference:(`Value 1e-3) ()
        in
        equal ~msg:"stream"
          (list (pair int int))
          []
          (List.map fst (stream q Nx.float64 (chunks_of none [1000; 0; 2000]))) )
  ]

let error_tests =
  [ test "the reference must be finite and positive" (fun () ->
        List.iter
          (fun (r, shown) ->
            raises ~msg:shown
              (Invalid_argument
                 (Printf.sprintf
                    "split_stage: cannot measure decibels against %s (the \
                     reference must be finite and positive)"
                    shown ) )
              (fun () -> ignore (Effects.split_stage ~reference:(`Value r) ())) )
          [(0., "0"); (Float.nan, "nan"); (Float.infinity, "inf"); (-1., "-1")] )
  ; test "the framing and the threshold are validated at construction"
      (fun () ->
        raises ~msg:"hop"
          (Invalid_argument
             "split_stage: cannot advance frames by 0 samples (hop must be at \
              least 1)" ) (fun () ->
            ignore (Effects.split_stage ~hop:0 ~reference:(`Value 1.) ()) ) ;
        raises ~msg:"frame_length"
          (Invalid_argument
             "split_stage: cannot analyse frames of 0 samples (frame_length \
              must be at least 1)" ) (fun () ->
            ignore
              (Effects.split_stage ~frame_length:0 ~reference:(`Value 1.) ()) ) ;
        raises ~msg:"top_db"
          (Invalid_argument
             "split_stage: cannot threshold at inf dB below the reference \
              (top_db must be finite)" ) (fun () ->
            ignore
              (Effects.split_stage ~top_db:Float.infinity ~reference:(`Value 1.)
                 () ) ) )
  ; test "a chunk of rank zero is refused" (fun () ->
        let p = Effects.split_stage ~reference:(`Value 1.) () in
        raises ~msg:"rank"
          (Invalid_argument
             "split_stage: cannot process a rank-zero tensor (the time axis \
              must exist)" ) (fun () ->
            let s =
              Pipeline.Stream.prepare p ~source:(source Nx.float64) ~max_chunk:1
            in
            ignore (Pipeline.Stream.push s (Nx.scalar Nx.float64 1.)) ) )
  ; test "a chunk must keep the stream's leading shape" (fun () ->
        let p = Effects.split_stage ~reference:(`Value 1.) () in
        raises ~msg:"leading shape"
          (Invalid_argument
             "split_stage: cannot continue a stream of leading shape [2] with \
              a chunk of leading shape [3]" ) (fun () ->
            ignore
              (stream p Nx.float64
                 [Nx.zeros Nx.float64 [|2; 10|]; Nx.zeros Nx.float64 [|3; 10|]] ) ) )
  ]

let suite =
  [ group "law" law_tests
  ; group "goldens through the stage" golden_tests
  ; group "emission" emission_tests
  ; group "queries" query_tests
  ; group "state" state_tests
  ; group "errors" error_tests ]
