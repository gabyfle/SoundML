open Windtrap

let () =
  exit
    (run "features_energy"
       (Energy_goldens.suite @ Energy_props.suite @ Energy_law.suite) )
