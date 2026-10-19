open Windtrap

let () =
  exit
    (run "features_spectral"
       (Spectral_goldens.suite @ Spectral_props.suite @ Spectral_law.suite) )
