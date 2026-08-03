(* Allocation-budget gate.

   The steady-state contract is BOUNDED allocation per push — O(stages) fresh
   chunks of bounded size, nothing proportional to history — not zero
   allocation: Nx operations return freshly allocated tensors, and the toy
   kernels allocate their output chunks. The gate therefore measures, after a
   warmup that fills every internal buffer:

   - [Gc.minor_words] delta per push, against an explicit per-pipeline budget
   derived from the chunk size (each stage may allocate a handful of chunks of
   at most [max_chunk] items plus closure/option noise; budgets below are the
   back-of-envelope word count times a ~4x safety factor), and - major-heap
   words growth ([Gc.quick_stat]) across the window: promotions cannot exceed
   what was allocated, and steady state means no unbounded growth, so the
   per-push major delta is asserted under the same budget. Bigarray payloads are
   malloc'd outside the OCaml heap; what shows up here is their custom-block
   accounting.

   Two disjoint measurement windows are asserted so a per-push cost growing with
   history fails the gate. *)

open Windtrap
open Soundml

let chunk_len = 256

let alloc_case name ~minor_budget p mk_chunk summarize =
  test name (fun () ->
      let source =
        Pipeline.Format.audio Nx.float32 ~sample_rate:44100 ~channels:1
      in
      let plan = Pipeline.Stream.prepare p ~source ~max_chunk:chunk_len in
      let sink = ref 0.0 in
      let consume = function
        | None ->
            ()
        | Some b ->
            sink := !sink +. summarize b
      in
      (* warmup: fill lookahead buffers, settle phases and leftovers *)
      for i = 0 to 63 do
        consume (Pipeline.Stream.push plan (mk_chunk i))
      done ;
      let measure_window () =
        Gc.full_major () ;
        let minor0 = Gc.minor_words () in
        let major0 = (Gc.quick_stat ()).Gc.major_words in
        let pushes = 512 in
        for i = 0 to pushes - 1 do
          consume (Pipeline.Stream.push plan (mk_chunk i))
        done ;
        let minor1 = Gc.minor_words () in
        let major1 = (Gc.quick_stat ()).Gc.major_words in
        ( (minor1 -. minor0) /. float_of_int pushes
        , (major1 -. major0) /. float_of_int pushes )
      in
      let check window (minor_per_push, major_per_push) =
        if minor_per_push > minor_budget then
          failf "%s/%s: %.1f minor words/push exceeds the budget of %.1f words"
            name window minor_per_push minor_budget ;
        if major_per_push > minor_budget then
          failf
            "%s/%s: %.1f promoted words/push exceeds the budget of %.1f words"
            name window major_per_push minor_budget
      in
      check "window-1" (measure_window ()) ;
      check "window-2" (measure_window ()) ;
      ignore !sink )

let float_chunks =
  Array.init 8 (fun i ->
      Array.init chunk_len (fun j ->
          float_of_int (((i * chunk_len) + j) mod 32) ) )

let nx_chunks = Array.map (Nx.create Nx.float32 [|chunk_len|]) float_chunks

let sum_array = Array.fold_left ( +. ) 0.

(* Budgets, in words per push, with chunk_len = 256. Each budget is roughly 2x
   the value measured on OCaml 5.5 (in parentheses); per-push allocation is
   deterministic — the same code path allocates the same words every push — so
   2x absorbs compiler/runtime variation across CI hosts while still catching
   any O(history) regression, which grows without bound.

   - gain>>decim4 (measured 2125): gain allocates one 256-float array (257
   words); decim4 builds a 64-element list of boxed floats (~64 x 6 words) and
   the 64-float output array; plus options, closures and the [Array.sub]
   leftovers. - lookahead>>block_sum (measured 1944): lookahead appends
   buffer+chunk (~262 words) and takes two subs; block_sum appends its leftover,
   takes one 4-float sub per window (64 x ~7 words) and allocates the 64-float
   output. - nx gain>>lookahead (measured 1821): three Nx ops per push (mul_s,
   concatenate, two shrink views), each allocating tensor records, shape
   metadata and a custom block on the OCaml heap — the bigarray payloads are
   malloc'd outside the heap. *)

let suite =
  [ group "alloc"
      [ alloc_case "gain>>decim4" ~minor_budget:4500.
          (Pipeline.( >> ) (Toys.gain 2.0) (Toys.decim4 ()))
          (fun i -> float_chunks.(i land 7))
          sum_array
      ; alloc_case "lookahead>>block_sum" ~minor_budget:4000.
          (Pipeline.( >> ) (Toys.lookahead 5) (Toys.block_sum 4))
          (fun i -> float_chunks.(i land 7))
          sum_array
      ; alloc_case "nx gain>>lookahead" ~minor_budget:4000.
          (Pipeline.( >> ) (Toys.nx_gain 2.0) (Toys.nx_lookahead 4))
          (fun i -> nx_chunks.(i land 7))
          (fun t -> Nx.item [0] t) ] ]
