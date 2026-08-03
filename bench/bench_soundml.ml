(* SoundML benchmarks.

   The pipeline group prices the abstraction: [Pipeline.run] of a three-stage
   stateless toy chain over a one-million-sample float32 tensor, next to the
   hand-written sequence of the same three Nx calls. The two rows must stay
   within a few percent of each other; the committed baseline plus the
   suite-level budgets below turn any drift into a failed build. *)

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
    [Thumper.group "pipeline" (pipeline_benchmarks ())]
