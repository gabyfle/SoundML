(* Profiling probe for the Resample cost-structure decomposition.

   Emits TSV lines: GEOM (filter geometry per pair/tier), APPLY (offline
   throughput), STREAM (chunked kernel throughput), STEPFIT (per-step fixed
   cost vs chunk size), OVERHEAD (apply decomposition on a 1 s clip). Times
   are wall-clock, min/median/max over n reps after warmup. Run on a quiet
   host, single invocation. *)

module R = Soundml.Resample

let now () = Unix.gettimeofday ()

let time ?(warmup = 2) ?(n = 10) f =
  for _ = 1 to warmup do
    ignore (Sys.opaque_identity (f ()))
  done ;
  let ts =
    Array.init n (fun _ ->
        let t0 = now () in
        ignore (Sys.opaque_identity (f ())) ;
        now () -. t0 )
  in
  Array.sort compare ts ;
  (ts.(0), ts.(n / 2), ts.(n - 1))

let pairs =
  [ ("44k1->48k", 44100, 48000)
  ; ("48k->44k1", 48000, 44100)
  ; ("44k1->16k", 44100, 16000)
  ; ("16k->44k1", 16000, 44100)
  ; ("8k->48k", 8000, 48000)
  ; ("48k->8k", 48000, 8000) ]

let tiers = [("fast", `Fast); ("high", `High); ("best", `Best)]

let geom () =
  Printf.printf "# GEOM\tpair\ttier\tL\tM\tK\ttaps\tbank_f32_kb\tmac_per_out\n" ;
  List.iter
    (fun (name, sr, tgt) ->
      List.iter
        (fun (tname, tier) ->
          let c = R.Config.create ~quality:tier ~sample_rate:sr ~target:tgt () in
          let rate = R.Config.rate c in
          let l = rate.Soundml.Pipeline.Rate.num
          and m = rate.Soundml.Pipeline.Rate.den in
          let k = R.Config.latency c in
          let taps = (2 * k) + 1 in
          Printf.printf "GEOM\t%s\t%s\t%d\t%d\t%d\t%d\t%.1f\t%d\n" name tname l
            m k taps
            (float_of_int (l * taps * 4) /. 1024.)
            taps )
        tiers )
    pairs

let apply_bench () =
  Printf.printf
    "# APPLY\tpair\ttier\tdtype\tdur_s\tn_in\tn_out\tt_min\tt_med\tt_max\n" ;
  List.iter
    (fun dur ->
      List.iter
        (fun (name, sr, tgt) ->
          let c = R.Config.create ~quality:`High ~sample_rate:sr ~target:tgt () in
          let n_in = sr * dur in
          let n_out = R.Config.output_frames c ~n:n_in in
          let x32 = Nx.rand Nx.float32 [|n_in|] in
          let x64 = Nx.rand Nx.float64 [|n_in|] in
          let tmin, tmed, tmax = time (fun () -> R.apply c x32) in
          Printf.printf "APPLY\t%s\thigh\tf32\t%d\t%d\t%d\t%.6e\t%.6e\t%.6e\n"
            name dur n_in n_out tmin tmed tmax ;
          let tmin, tmed, tmax = time (fun () -> R.apply c x64) in
          Printf.printf "APPLY\t%s\thigh\tf64\t%d\t%d\t%d\t%.6e\t%.6e\t%.6e\n"
            name dur n_in n_out tmin tmed tmax )
        pairs )
    [1; 30]

let stream_bench () =
  Printf.printf "# STREAM\tpair\tchunk\tdur_s\tn_in\tt_min\tt_med\tt_max\n" ;
  let dur = 30 in
  List.iter
    (fun (name, sr, tgt) ->
      let c = R.Config.create ~quality:`High ~sample_rate:sr ~target:tgt () in
      let n_in = sr * dur in
      let x = Nx.rand Nx.float32 [|n_in|] in
      List.iter
        (fun chunk ->
          let k = R.Kernel.prepare c Nx.float32 ~channels:1 ~max_block:chunk in
          let run () =
            R.Kernel.reset k ;
            let i = ref 0 in
            let acc = ref 0 in
            while !i < n_in do
              let len = min chunk (n_in - !i) in
              let sl = Nx.shrink [|(!i, !i + len)|] x in
              ( match R.Kernel.step k sl with
              | Some o ->
                  acc := !acc + Nx.dim 0 o
              | None ->
                  () ) ;
              i := !i + len
            done ;
            ( match R.Kernel.flush k with
            | Some o ->
                acc := !acc + Nx.dim 0 o
            | None ->
                () ) ;
            !acc
          in
          let tmin, tmed, tmax = time ~warmup:1 ~n:5 run in
          Printf.printf "STREAM\t%s\t%d\t%d\t%d\t%.6e\t%.6e\t%.6e\n" name chunk
            dur n_in tmin tmed tmax )
        [1024; 4096; 16384] )
    [("44k1->48k", 44100, 48000); ("44k1->16k", 44100, 16000)]

let stepfit () =
  Printf.printf "# STEPFIT\tchunk\tsteps\tt_per_step\n" ;
  let c = R.Config.create ~quality:`High ~sample_rate:44100 ~target:48000 () in
  List.iter
    (fun chunk ->
      let k = R.Kernel.prepare c Nx.float32 ~channels:1 ~max_block:chunk in
      let x = Nx.rand Nx.float32 [|chunk|] in
      let steps = max 200 (2_000_000 / chunk) in
      let run () =
        R.Kernel.reset k ;
        for _ = 1 to steps do
          ignore (Sys.opaque_identity (R.Kernel.step k x))
        done
      in
      let tmin, _, _ = time ~warmup:1 ~n:5 run in
      Printf.printf "STEPFIT\t%d\t%d\t%.6e\n" chunk steps
        (tmin /. float_of_int steps) )
    [16; 64; 256; 1024; 4096; 16384]

let overhead () =
  Printf.printf "# OVERHEAD\tcase\tt_min\tt_med\tt_max\n" ;
  let c = R.Config.create ~quality:`High ~sample_rate:44100 ~target:48000 () in
  let n = 44100 in
  let x = Nx.rand Nx.float32 [|n|] in
  let n_out = R.Config.output_frames c ~n in
  let p label f =
    let tmin, tmed, tmax = time ~n:20 f in
    Printf.printf "OVERHEAD\t%s\t%.6e\t%.6e\t%.6e\n" label tmin tmed tmax
  in
  p "apply_1s" (fun () -> R.apply c x) ;
  let k = R.Kernel.prepare c Nx.float32 ~channels:1 ~max_block:n in
  p "reset_step_flush_1s" (fun () ->
      R.Kernel.reset k ;
      let a = R.Kernel.step k x in
      let b = R.Kernel.flush k in
      (a, b) ) ;
  p "reset_step_only_1s" (fun () ->
      R.Kernel.reset k ;
      R.Kernel.step k x ) ;
  p "kernel_prepare" (fun () ->
      R.Kernel.prepare c Nx.float32 ~channels:1 ~max_block:n ) ;
  p "nx_empty_out" (fun () -> Nx.empty Nx.float32 [|n_out|]) ;
  p "nx_rand_touch" (fun () -> Nx.copy x)

let () =
  let want s = Array.length Sys.argv < 2 || Array.exists (( = ) s) Sys.argv in
  Printf.printf "# machine: single run, %s\n" (Unix.gethostname ()) ;
  if want "geom" then geom () ;
  if want "overhead" then overhead () ;
  if want "stepfit" then stepfit () ;
  if want "apply" then apply_bench () ;
  if want "stream" then stream_bench ()
