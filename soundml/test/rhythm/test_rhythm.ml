open Windtrap

let () = exit (run "Soundml.rhythm" (Rhythm_goldens.suite @ Rhythm_law.suite))
