open Windtrap

let () = exit (run "Soundml.delta" (Delta_goldens.suite @ Delta_law.suite))
