open Windtrap

let () =
  exit
    (run "Soundml onset detection"
       (Onset_detect_goldens.suite @ Onset_detect_law.suite) )
