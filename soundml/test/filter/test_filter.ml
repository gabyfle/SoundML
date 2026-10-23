open Windtrap

let () = exit (run "Soundml.Filter" (Filter_goldens.suite @ Filter_law.suite))
