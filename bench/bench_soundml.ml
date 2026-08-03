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
   mirroring the librosa rows of [bench_soundml.py] one to one. *)

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

let () =
  Nx.Rng.run ~seed:42
  @@ fun () ->
  Thumper.run "soundml"
    ~budgets:
      [ Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05
      ; Thumper.Budget.no_more_alloc_than 0.01 ]
    [ Thumper.group "window" (window_benchmarks ())
    ; Thumper.group "pipeline" (pipeline_benchmarks ())
    ; Thumper.group "stft" (stft_benchmarks ())
    ; Thumper.group "mel" (mel_benchmarks ()) ]
