open Windtrap

let () =
  exit
    (run "Pipeline"
       ( Test_law.suite @ Latency_law.suite @ Accept_causal_stream.suite
       @ Test_alloc.suite ) )
