let () =
  Alcotest.run "pipeline"
    (Test_law.tests @ Accept_causal_stream.tests @ Test_alloc.tests)
