open Windtrap

let () =
  exit (run "Soundml.analysis" (Analysis_goldens.suite @ Analysis_law.suite))
