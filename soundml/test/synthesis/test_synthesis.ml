open Windtrap

let () =
  exit (run "Soundml.synthesis" (Synthesis_goldens.suite @ Synthesis_law.suite))
