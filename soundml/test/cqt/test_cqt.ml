open Windtrap

let () =
  run "Soundml.Cqt" (Cqt_goldens.suite @ Cqt_props.suite @ Cqt_stream.suite)
