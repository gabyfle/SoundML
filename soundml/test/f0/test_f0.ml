open Windtrap

let () = exit (run "Soundml.f0" (F0_goldens.suite @ F0_law.suite))
