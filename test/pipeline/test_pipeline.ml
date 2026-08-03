open Windtrap

let () =
  run "Pipeline" (Test_law.suite @ Accept_causal_stream.suite @ Test_alloc.suite)
