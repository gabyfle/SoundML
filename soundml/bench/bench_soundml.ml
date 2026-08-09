(* SoundML benchmarks.

   The window group times [Window.make] for every spec at n = 2048, one row per
   spec, mirroring the librosa rows of [bench_soundml.py] one to one.

   The pipeline group prices the abstraction: [Pipeline.run] of a three-stage
   stateless toy chain over a one-million-sample float32 tensor, next to the
   hand-written sequence of the same three Nx calls. The two rows must stay
   within a few percent of each other; the committed baseline plus the
   suite-level budgets below turn any drift into a failed build.

   The stft and mel groups time the offline analysis paths end to end —
   [Stft.power_spectrum] and [mel_spectrogram] over 30 s of mono audio at 22.05
   kHz, fft 2048 and hop 512, in both float32 and float64 — one row per dtype,
   mirroring the librosa rows of [bench_soundml.py] one to one.

   The resample group covers every face of [Resample] on one-second mono clips:
   [apply] across the three presets, the six headline rate pairs — near-unity
   both ways and every pair the planner splits into a cascade, so a stage-level
   regression on any plan shape trips the ratchet — and both dtypes; the GEMM
   surface next to the executor at [`High]; the streaming kernel at [`High]
   float32 across chunk sizes (the 1024 row keeps the per-chunk dispatch
   overhead honest); the stage inside a resample-then-STFT front next to the
   same computation hand-written; the identity-rate passthrough; and
   [Config.create] itself, priced separately: the OLS plans design far smaller
   banks, so creation now costs a fraction of the one-second conversion it
   configures (0.1-0.6x, from ~2x before) — reuse is still the documented
   contract, and this row keeps the design cost visible.

   The committed baseline gates drift (the suite budgets below), not absolute
   throughput: the resample rows record what the executor delivers on the
   reference machine — the dot-product kernel measured standalone runs within a
   few percent of these numbers, so they are the compute ceiling of this
   formulation there, not a tuning shortfall. The Python twin [bench_soundml.py]
   carries the cross-library rows; the honest comparative position is recorded
   in README.md. *)

let n_window = 2048

(* One (name, spec) pair per Window.t constructor; parametrised specs use the
   same representative parameters as bench_soundml.py. *)
let window_specs =
  [ ("hann", Soundml.Window.Hann)
  ; ("hamming", Soundml.Window.Hamming)
  ; ("blackman", Soundml.Window.Blackman)
  ; ("blackman_harris", Soundml.Window.Blackman_harris)
  ; ("nuttall", Soundml.Window.Nuttall)
  ; ("bartlett", Soundml.Window.Bartlett)
  ; ("kaiser 8.6", Soundml.Window.Kaiser 8.6)
  ; ("gaussian 256", Soundml.Window.Gaussian 256.)
  ; ("tukey 0.5", Soundml.Window.Tukey 0.5)
  ; ("flat_top", Soundml.Window.Flat_top)
  ; ("rectangular", Soundml.Window.Rectangular) ]

let window_benchmarks () =
  List.map
    (fun (name, spec) ->
      Thumper.bench (Printf.sprintf "%s %d" name n_window) (fun () ->
          Soundml.Window.make Nx.float64 spec n_window ) )
    window_specs

(* 30 s of mono audio at librosa's default rate, analysed at librosa's default
   fft size and hop; the mel projection uses librosa's default 128 bands. *)
let sample_rate = 22050

let n_audio = 30 * sample_rate

let stft_config = Soundml.Stft.Config.create ~fft_size:2048 ~hop:512 ()

let mel_config =
  Soundml.Mel.Config.create ~n_mels:128 ~sample_rate ~fft_size:2048 ()

let stft_benchmarks () =
  let x32 = Nx.rand Nx.float32 [|n_audio|] in
  let x64 = Nx.rand Nx.float64 [|n_audio|] in
  [ Thumper.bench "power_spectrum 30s f32 fft2048 hop512" (fun () ->
        Soundml.Stft.power_spectrum stft_config x32 )
  ; Thumper.bench "power_spectrum 30s f64 fft2048 hop512" (fun () ->
        Soundml.Stft.power_spectrum stft_config x64 ) ]

let mel_benchmarks () =
  let x32 = Nx.rand Nx.float32 [|n_audio|] in
  let x64 = Nx.rand Nx.float64 [|n_audio|] in
  [ Thumper.bench "mel_spectrogram 30s f32 fft2048 hop512" (fun () ->
        Soundml.mel_spectrogram stft_config mel_config x32 )
  ; Thumper.bench "mel_spectrogram 30s f64 fft2048 hop512" (fun () ->
        Soundml.mel_spectrogram stft_config mel_config x64 ) ]

let n = 1_000_000

let source =
  Soundml.Pipeline.Format.audio Nx.float32 ~sample_rate:44100 ~channels:1

(* The toy chain: gain, bias, rectify — three stateless stages whose
   hand-written equivalent is exactly three Nx calls. *)
let chain =
  let open Soundml.Pipeline in
  stateless (fun t -> Nx.mul_s t 0.5)
  >> stateless (fun t -> Nx.add_s t 0.1)
  >> stateless Nx.abs

let direct t = Nx.abs (Nx.add_s (Nx.mul_s t 0.5) 0.1)

let pipeline_benchmarks () =
  let x = Nx.rand Nx.float32 [|n|] in
  [ Thumper.bench "run gain>>bias>>abs 1M" (fun () ->
        Soundml.Pipeline.run ~source chain x )
  ; Thumper.bench "direct gain+bias+abs 1M" (fun () -> direct x) ]

(* One-second mono clips; every row's throughput in input Msamples/s is the
   clip's sample rate over the row's wall time. *)

let resample_pairs =
  [ ("44k1-48k", 44100, 48000)
  ; ("48k-44k1", 48000, 44100)
  ; ("44k1-16k", 44100, 16000)
  ; ("16k-44k1", 16000, 44100)
  ; ("8k-48k", 8000, 48000)
  ; ("48k-8k", 48000, 8000) ]

let resample_tiers = [("fast", `Fast); ("high", `High); ("best", `Best)]

let resample_apply_benchmarks () =
  List.concat_map
    (fun (pair, sample_rate, target) ->
      let clip32 = Nx.rand Nx.float32 [|sample_rate|] in
      let clip64 = Nx.rand Nx.float64 [|sample_rate|] in
      List.concat_map
        (fun (tier, quality) ->
          let cfg =
            Soundml.Resample.Config.create ~quality ~sample_rate ~target ()
          in
          [ Thumper.bench (Printf.sprintf "apply %s %s f32" tier pair) (fun () ->
                Soundml.Resample.apply cfg clip32 )
          ; Thumper.bench (Printf.sprintf "apply %s %s f64" tier pair)
              (fun () -> Soundml.Resample.apply cfg clip64 ) ] )
        resample_tiers )
    resample_pairs

let resample_gemm_benchmarks () =
  List.concat_map
    (fun (pair, sample_rate, target) ->
      let clip32 = Nx.rand Nx.float32 [|sample_rate|] in
      let clip64 = Nx.rand Nx.float64 [|sample_rate|] in
      let cfg = Soundml.Resample.Config.create ~sample_rate ~target () in
      [ Thumper.bench (Printf.sprintf "gemm high %s f32" pair) (fun () ->
            Soundml.Resample.apply_gemm cfg clip32 )
      ; Thumper.bench (Printf.sprintf "gemm high %s f64" pair) (fun () ->
            Soundml.Resample.apply_gemm cfg clip64 ) ] )
    [("44k1-48k", 44100, 48000); ("44k1-16k", 44100, 16000)]

let resample_stream_benchmarks () =
  (* every streaming cadence: 44.1 -> 48 k feeds the near-unity GEMM-executed
     plan (call-sized bursts), 48 -> 8 k the /F-last FFT-executed class — whose
     float32 streaming pays the FFT executor's few-line stacking price, so it
     stays ratcheted — and 11.025 -> 8 k the direct dot-product kernel; the 1024
     rows keep each executor's per-chunk overhead story honest *)
  let one (name, sample_rate, target, chunk_sizes) =
    let clip = Nx.rand Nx.float32 [|sample_rate|] in
    let cfg = Soundml.Resample.Config.create ~sample_rate ~target () in
    List.map
      (fun max_chunk ->
        let kernel =
          Soundml.Resample.Kernel.prepare cfg Nx.float32 ~channels:1
            ~max_block:max_chunk
        in
        let chunks =
          let rec cut off =
            if off >= sample_rate then []
            else
              let stop = min sample_rate (off + max_chunk) in
              Nx.shrink [|(off, stop)|] clip :: cut stop
          in
          cut 0
        in
        Thumper.bench
          (Printf.sprintf "stream high %s f32 chunk %d" name max_chunk)
          (fun () ->
            Soundml.Resample.Kernel.reset kernel ;
            List.iter
              (fun c -> ignore (Soundml.Resample.Kernel.step kernel c))
              chunks ;
            ignore (Soundml.Resample.Kernel.flush kernel) ) )
      chunk_sizes
  in
  List.concat_map one
    [ ("44k1-48k", 44100, 48000, [1024; 4096; 16384])
    ; ("48k-8k", 48000, 8000, [1024; 4096])
    ; ("11k025-8k", 11025, 8000, [1024; 4096]) ]

let resample_config_benchmarks () =
  [ Thumper.bench "config create high 44k1-48k" (fun () ->
        Soundml.Resample.Config.create ~sample_rate:44100 ~target:48000 () )
  ; Thumper.bench "config create high 44k1-16k" (fun () ->
        Soundml.Resample.Config.create ~sample_rate:44100 ~target:16000 () ) ]

let resample_stage_benchmarks () =
  let sample_rate = 44100 in
  let clip = Nx.rand Nx.float32 [|sample_rate|] in
  let cfg = Soundml.Resample.Config.create ~sample_rate ~target:48000 () in
  let stft_cfg = Soundml.Stft.Config.create ~fft_size:1024 () in
  let source =
    Soundml.Pipeline.Format.audio Nx.float32 ~sample_rate ~channels:1
  in
  let front =
    Soundml.Pipeline.( >> )
      (Soundml.Resample.stage cfg)
      (Soundml.Stft.power_stage stft_cfg)
  in
  let identity =
    Soundml.Resample.Config.create ~sample_rate:48000 ~target:48000 ()
  in
  let clip_id = Nx.rand Nx.float32 [|1_000_000|] in
  [ Thumper.bench "stage resample>>stft 44k1-48k f32" (fun () ->
        Soundml.Pipeline.run ~source front clip )
  ; Thumper.bench "direct apply+stft 44k1-48k f32" (fun () ->
        Soundml.Stft.power_spectrum stft_cfg (Soundml.Resample.apply cfg clip) )
  ; Thumper.bench "apply identity 1M f32" (fun () ->
        Soundml.Resample.apply identity clip_id ) ]

let () =
  Nx.Rng.with_key (Nx.Rng.key 42)
  @@ fun () ->
  Thumper.run "soundml"
    ~budgets:
      [ Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05
      ; Thumper.Budget.no_more_alloc_than 0.01 ]
    [ Thumper.group "window" (window_benchmarks ())
    ; Thumper.group "pipeline" (pipeline_benchmarks ())
    ; Thumper.group "stft" (stft_benchmarks ())
    ; Thumper.group "mel" (mel_benchmarks ())
    ; Thumper.group "resample"
        ( resample_apply_benchmarks ()
        @ resample_gemm_benchmarks ()
        @ resample_stream_benchmarks ()
        @ resample_stage_benchmarks ()
        @ resample_config_benchmarks () ) ]
