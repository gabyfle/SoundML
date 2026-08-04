(* Allocation-budget gate for the streaming resampler.

   The steady-state contract is BOUNDED allocation per step — the freshly
   computed output tensor and a fixed handful of records around it, nothing
   proportional to history, channels-times-history or samples-consumed-so-far —
   not zero allocation: this nx revision returns freshly allocated tensors from
   every operation. After a warmup that settles the phase state, two disjoint
   windows measure [Gc.minor_words] and major-heap growth per call, so a
   per-call cost growing with accumulated state fails the gate.

   Budget rationale, chunk_len = 512, `High 11025 -> 8000 (L/M = 320/441, a
   single direct stage — the near-unity class whose block rule sits past the FFT
   executor's emission ceiling, so it pins the direct path now that the 44.1 <->
   48 kHz plans execute their sharp stage by overlap-save): one step wraps three
   bigarrays around the kernel state and the chunk, allocates the output tensor
   (record, shape metadata and a custom block on the OCaml heap — the payload is
   malloc'd outside it) and a few options and closures. Measured on OCaml 5.5:
   ~493 minor words/step mono float32, ~508 stereo float64, ~534 per stage push,
   identical across windows; the minor budget is ~2x the measured figure,
   absorbing runtime variation across hosts while still failing any O(history)
   regression. The promoted budget is its own, an order of magnitude over the
   measured ~0 words/step: comparing promotions against the minor budget would
   let a steady promoted-heap leak pass unnoticed.

   [apply] carries its own per-call budget: it prepares a kernel (the bank cast
   plus zeroed history and scratch) and concatenates the stepped and drained
   tensors — ~1.9k minor and ~18 promoted words per call, measured, fixed in the
   input length. *)

open Windtrap
open Soundml

let chunk_len = 512

let minor_budget = 1200.

let promoted_budget = 60.

let apply_minor_budget = 4000.

let apply_promoted_budget = 120.

let measure_windows ~warmup ~calls f =
  for _ = 1 to warmup do
    f ()
  done ;
  let window () =
    Gc.full_major () ;
    let minor0 = Gc.minor_words () in
    let major0 = (Gc.quick_stat ()).Gc.major_words in
    for _ = 1 to calls do
      f ()
    done ;
    let minor1 = Gc.minor_words () in
    let major1 = (Gc.quick_stat ()).Gc.major_words in
    ( (minor1 -. minor0) /. float_of_int calls
    , (major1 -. major0) /. float_of_int calls )
  in
  [("window-1", window ()); ("window-2", window ())]

let check ~minor_budget ~promoted_budget windows =
  List.iter
    (fun (window, (minor_per_call, major_per_call)) ->
      if minor_per_call > minor_budget then
        failf "%s: %.1f minor words/call exceeds the budget of %.1f words"
          window minor_per_call minor_budget ;
      if major_per_call > promoted_budget then
        failf "%s: %.1f promoted words/call exceeds the budget of %.1f words"
          window major_per_call promoted_budget )
    windows

let cfg () = Resample.Config.create ~sample_rate:11025 ~target:8000 ()

(* FFT-executed plans (the `High 8 -> 48 k and 44.1 -> 16 k plans of record run
   their sharp stage by overlap-save) allocate transform transients per executed
   block — the assembly, the spectra, the inverse — on top of the output tensor:
   the documented deviation until nx grows destination passing, the same class
   the Stft kernel lives with. The budgets are still BOUNDED per call and
   independent of accumulated state: at chunk 512 the 8 -> 48 k stage-1 grid (B
   = 824) completes a block on ~2 of 3 steps, and a block is a fixed handful of
   tensors whatever the history. Measured on OCaml 5.5: ~2.7k minor words/step
   and ~3 promoted, mono float32, identical across windows; budgets are ~2x the
   measured figures, same discipline as the direct rows above. [apply] on a
   one-second 44.1 -> 16 k clip batches every block of the whole signal through
   one stacked transform: ~11.9k minor and ~410 promoted words per call, fixed
   in the input length. *)
let ols_minor_budget = 5500.

let ols_promoted_budget = 60.

let ols_apply_minor_budget = 24000.

let ols_apply_promoted_budget = 800.

let ols_cfg () = Resample.Config.create ~sample_rate:8000 ~target:48000 ()

let suite =
  [ group "alloc"
      [ test "kernel step stays within budget (mono float32)" (fun () ->
            let kernel =
              Resample.Kernel.prepare (cfg ()) Nx.float32 ~channels:1
                ~max_block:chunk_len
            in
            let chunk = Nx.rand Nx.float32 [|chunk_len|] in
            check ~minor_budget ~promoted_budget
              (measure_windows ~warmup:64 ~calls:256 (fun () ->
                   ignore
                     (Sys.opaque_identity (Resample.Kernel.step kernel chunk)) )
              ) )
      ; test "kernel step stays within budget (stereo float64)" (fun () ->
            let kernel =
              Resample.Kernel.prepare (cfg ()) Nx.float64 ~channels:2
                ~max_block:chunk_len
            in
            let chunk = Nx.rand Nx.float64 [|2; chunk_len|] in
            check ~minor_budget ~promoted_budget
              (measure_windows ~warmup:64 ~calls:256 (fun () ->
                   ignore
                     (Sys.opaque_identity (Resample.Kernel.step kernel chunk)) )
              ) )
      ; test "stage push stays within budget" (fun () ->
            let source =
              Pipeline.Format.audio Nx.float32 ~sample_rate:11025 ~channels:1
            in
            let plan =
              Pipeline.Stream.prepare
                (Resample.stage (cfg ()))
                ~source ~max_chunk:chunk_len
            in
            let chunk = Nx.rand Nx.float32 [|chunk_len|] in
            check ~minor_budget ~promoted_budget
              (measure_windows ~warmup:64 ~calls:256 (fun () ->
                   ignore
                     (Sys.opaque_identity (Pipeline.Stream.push plan chunk)) )
              ) )
      ; test "apply stays within its per-call budget" (fun () ->
            let c = cfg () in
            let clip = Nx.rand Nx.float32 [|4410|] in
            check ~minor_budget:apply_minor_budget
              ~promoted_budget:apply_promoted_budget
              (measure_windows ~warmup:16 ~calls:128 (fun () ->
                   ignore (Sys.opaque_identity (Resample.apply c clip)) ) ) )
      ; test "ols kernel step stays within budget (mono float32, 8k -> 48k)"
          (fun () ->
            let kernel =
              Resample.Kernel.prepare (ols_cfg ()) Nx.float32 ~channels:1
                ~max_block:chunk_len
            in
            let chunk = Nx.rand Nx.float32 [|chunk_len|] in
            check ~minor_budget:ols_minor_budget
              ~promoted_budget:ols_promoted_budget
              (measure_windows ~warmup:64 ~calls:256 (fun () ->
                   ignore
                     (Sys.opaque_identity (Resample.Kernel.step kernel chunk)) )
              ) )
      ; test "ols apply stays within its per-call budget (44.1k -> 16k)"
          (fun () ->
            let c =
              Resample.Config.create ~sample_rate:44100 ~target:16000 ()
            in
            let clip = Nx.rand Nx.float32 [|44100|] in
            check ~minor_budget:ols_apply_minor_budget
              ~promoted_budget:ols_apply_promoted_budget
              (measure_windows ~warmup:8 ~calls:64 (fun () ->
                   ignore (Sys.opaque_identity (Resample.apply c clip)) ) ) ) ]
  ]
