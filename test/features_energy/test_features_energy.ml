open Windtrap

let () =
  run "features_energy"
    (Energy_goldens.suite @ Energy_props.suite @ Energy_law.suite)
