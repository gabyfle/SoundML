(* Par, compiled from the library's own source.

   Every chunk runs exactly once whatever the pool is doing — spinning, parked,
   retiring or coming back — for runs from several domains at once, nested runs
   and raising chunks. Also here: a parked pool outlives its linger and wakes
   for the next run, it retires once major cycles pass without a run, and the
   domain watching it retires it when it exits.

   The first test must stay first: it needs a pool that no domain watches
   yet. *)

open Windtrap

(* The domains running before any pool. *)
let base = Domain.count ()

(* With a single recommended domain every run computes its chunks in place, and
   no pool ever forms. *)
let parallel = Domain.recommended_domain_count () > 1

let with_pool () = if not parallel then skip ~reason:"no worker domains" ()

(* Bounds every wait on another domain, so that a lost wakeup fails the test
   instead of hanging it. *)
let patience = 1_000_000_000

let await cond =
  let rec go n =
    cond ()
    || n > 0
       &&
       ( Domain.cpu_relax () ;
         go (n - 1) )
  in
  go patience

(* Waits through forced major cycles, which run the pool's alarm on the main
   domain. *)
let await_cycles cond =
  let rec go n =
    cond ()
    || n > 0
       && begin
         Gc.full_major () ;
         for _ = 1 to 100_000 do
           Domain.cpu_relax ()
         done ;
         go (n - 1)
       end
  in
  go 1_000

(* Far longer than a worker's linger, and without allocating, so that no major
   cycle ends meanwhile. *)
let pause () =
  for _ = 1 to 1_000_000 do
    Domain.cpu_relax ()
  done

let counted ?(body = ignore) chunks =
  let hits = Array.init chunks (fun _ -> Atomic.make 0) in
  Par.run ~chunks (fun i ->
      Atomic.incr hits.(i) ;
      body i ) ;
  Array.map Atomic.get hits

let once name hits =
  Array.iteri
    (fun i n -> if n <> 1 then failf "%s: chunk %d ran %d times" name i n)
    hits

(* Two chunks, the first of which waits for the second to start: the run returns
   only once another domain has taken a chunk. *)
let rendezvous () =
  let started = Atomic.make false in
  once "rendezvous"
    (counted 2 ~body:(fun i ->
         if i = 1 then Atomic.set started true
         else if not (await (fun () -> Atomic.get started)) then
           failwith "rendezvous: no worker took the second chunk" ) )

let raising chunks bad =
  let hits = Array.init chunks (fun _ -> Atomic.make 0) in
  match
    Par.run ~chunks (fun i ->
        Atomic.incr hits.(i) ;
        if i = bad then failwith "chunk" )
  with
  | () ->
      failf "raising: run %d of %d returned" bad chunks
  | exception Failure msg when msg = "chunk" ->
      Array.iteri
        (fun i h ->
          let n = Atomic.get h in
          if n > 1 || (i = bad && n <> 1) then
            failf "raising: chunk %d of %d ran %d times" i chunks n )
        hits

let covered n grain =
  let hits = Array.init n (fun _ -> Atomic.make 0) in
  Par.ranges ~n ~grain (fun lo hi ->
      if hi - lo < Int.min n (Int.max 1 grain) then
        failf "ranges: [%d, %d) is shorter than the grain %d" lo hi grain ;
      for i = lo to hi - 1 do
        Atomic.incr hits.(i)
      done ) ;
  once (Printf.sprintf "ranges %d %d" n grain) (Array.map Atomic.get hits)

(* One domain's share of the stress: runs of random sizes, some nested, some
   raising, with pauses that let the pool park and major cycles that let it
   retire. *)
let stress seed rounds =
  let rng = Random.State.make [|0x9a7; seed|] in
  for _ = 1 to rounds do
    let chunks =
      if Random.State.int rng 16 = 0 then 1000 else Random.State.int rng 40
    in
    match Random.State.int rng 8 with
    | 0 ->
        let chunks = Int.max 1 chunks in
        raising chunks (Random.State.int rng chunks)
    | 1 ->
        once "nested"
          (counted (Int.min 40 chunks) ~body:(fun _ ->
               once "inner" (counted 3) ) )
    | 2 ->
        pause ()
    | 3 ->
        Gc.full_major ()
    | _ ->
        once "plain" (counted chunks)
  done

let pool_tests =
  [ test "the domain watching the pool retires it as it exits" (fun () ->
        with_pool () ;
        Domain.join
          (Domain.spawn (fun () ->
               rendezvous () ; pause () ; rendezvous () ; pause () ) ) ;
        is_true ~msg:"every worker exits"
          (await (fun () -> Domain.count () = base)) )
  ; test "a parked pool outlives its linger and wakes for the next run"
      (fun () ->
        with_pool () ;
        for _ = 1 to 5 do
          rendezvous () ;
          pause () ;
          is_true ~msg:"workers stay parked" (Domain.count () > base)
        done )
  ; test "the pool retires after major cycles without a run, then comes back"
      (fun () ->
        with_pool () ;
        for _ = 1 to 5 do
          rendezvous () ;
          is_true ~msg:"the pool retires"
            (await_cycles (fun () -> Domain.count () = base))
        done ;
        rendezvous () ) ]

let chunk_tests =
  [ test "every chunk runs once, whatever the pool is doing" (fun () ->
        List.iter
          (fun chunks ->
            for round = 0 to 11 do
              once (Printf.sprintf "%d chunks" chunks) (counted chunks) ;
              if round mod 3 = 0 then pause () ;
              if round mod 4 = 0 then Gc.full_major ()
            done )
          [0; 1; 2; 3; 7; 17; 18; 19; 64; 1000] )
  ; test "ranges cover [0, n) once, in ranges of at least the grain" (fun () ->
        List.iter
          (fun (n, grain) -> covered n grain)
          [ (0, 1)
          ; (1, 1)
          ; (5, 3)
          ; (7, 0)
          ; (100, 1)
          ; (100, 7)
          ; (12345, 64)
          ; (12345, 100_000) ] )
  ; test "a raising chunk is re-raised and the pool keeps working" (fun () ->
        List.iter
          (fun chunks ->
            for bad = 0 to chunks - 1 do
              raising chunks bad
            done ;
            once "after" (counted chunks) )
          [1; 2; 5; 40] )
  ; test
      "runs from several domains, nested and raising, as the pool parks and \
       retires" (fun () ->
        let others =
          List.init 4 (fun d -> Domain.spawn (fun () -> stress (d + 1) 300))
        in
        stress 0 300 ;
        List.iter Domain.join others ;
        if parallel then rendezvous () ) ]

let suite = [group "pool" pool_tests; group "chunks" chunk_tests]

let () = exit (run "Soundml.par" suite)
