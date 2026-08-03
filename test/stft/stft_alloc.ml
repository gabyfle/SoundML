(* Allocation-budget gate for the streaming STFT.

   The steady-state contract is BOUNDED allocation per push — a handful of fresh
   tensors of chunk-proportional size, nothing proportional to history — not
   zero allocation: this nx revision returns freshly allocated tensors from
   every operation. After a warmup that settles the boundary carry, two disjoint
   windows measure [Gc.minor_words] and major-heap growth per push, so a
   per-push cost growing with history fails the gate.

   Budget rationale, chunk_len = 512, fft_size = 64, hop = 16 (32 frames per
   push): one push runs one boundary-carry copy, one concatenation, one
   [extract_patches] gather, one cast to double, one window multiply, one
   batched rfft, one complex [abs], one cast, one square and one cast back —
   each allocating a tensor record, shape metadata and a custom block on the
   OCaml heap (the bigarray payloads are malloc'd outside it) — plus closures,
   options and small slice copies. Measured on OCaml 5.5: ~3.1k minor words/push
   and ~6 promoted words/push, identical across windows; the budget is ~2x the
   minor figure, absorbing runtime variation across hosts while still failing
   any O(history) regression. *)

open Windtrap
open Soundml

let chunk_len = 512

let minor_budget = 6000.

let suite =
  [ group "alloc"
      [ test "power_stage push stays within budget" (fun () ->
            let config = Stft.Config.create ~fft_size:64 ~hop:16 () in
            let p = Stft.power_stage config in
            let source =
              Pipeline.Format.audio Nx.float32 ~sample_rate:44100 ~channels:1
            in
            let plan = Pipeline.Stream.prepare p ~source ~max_chunk:chunk_len in
            let chunks =
              Array.init 8 (fun i ->
                  Nx.create Nx.float32 [|chunk_len|]
                    (Array.init chunk_len (fun j ->
                         Float.of_int (((i * chunk_len) + j) mod 32) /. 32. ) ) )
            in
            let sink = ref 0.0 in
            let consume = function
              | Some b when Nx.numel b > 0 ->
                  sink := !sink +. Nx.item [0; 0] b
              | _ ->
                  ()
            in
            (* warmup: install the left border, settle the carry *)
            for i = 0 to 63 do
              consume (Pipeline.Stream.push plan chunks.(i land 7))
            done ;
            let measure_window () =
              Gc.full_major () ;
              let minor0 = Gc.minor_words () in
              let major0 = (Gc.quick_stat ()).Gc.major_words in
              let pushes = 256 in
              for i = 0 to pushes - 1 do
                consume (Pipeline.Stream.push plan chunks.(i land 7))
              done ;
              let minor1 = Gc.minor_words () in
              let major1 = (Gc.quick_stat ()).Gc.major_words in
              ( (minor1 -. minor0) /. float_of_int pushes
              , (major1 -. major0) /. float_of_int pushes )
            in
            let check window (minor_per_push, major_per_push) =
              if minor_per_push > minor_budget then
                failf
                  "%s: %.1f minor words/push exceeds the budget of %.1f words"
                  window minor_per_push minor_budget ;
              if major_per_push > minor_budget then
                failf
                  "%s: %.1f promoted words/push exceeds the budget of %.1f \
                   words"
                  window major_per_push minor_budget
            in
            check "window-1" (measure_window ()) ;
            check "window-2" (measure_window ()) ;
            ignore !sink ) ] ]
