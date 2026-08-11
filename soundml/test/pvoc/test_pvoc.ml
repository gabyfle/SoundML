open Windtrap

let () =
  run "Soundml.Effects"
    (Pvoc_goldens.suite @ Pvoc_locked.suite @ Pvoc_edge.suite)
