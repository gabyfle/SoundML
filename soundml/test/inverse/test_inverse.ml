open Windtrap

let () =
  exit (run "Soundml.inverse" (Inverse_goldens.suite @ Inverse_law.suite))
