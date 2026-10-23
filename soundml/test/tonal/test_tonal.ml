open Windtrap

let () = exit (run "Soundml.tonal" (Tonal_goldens.suite @ Tonal_law.suite))
