open Windtrap

let () =
  exit
    (run "Soundml.piptrack, pitch_tuning, estimate_tuning"
       (Tuning_goldens.suite @ Tuning_law.suite) )
