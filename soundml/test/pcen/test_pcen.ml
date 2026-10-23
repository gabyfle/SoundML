open Windtrap

let () = exit (run "Soundml.pcen" (Pcen_goldens.suite @ Pcen_law.suite))
