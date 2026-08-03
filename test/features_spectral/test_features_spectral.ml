open Windtrap

let () =
  run "features_spectral"
    (Spectral_goldens.suite @ Spectral_props.suite @ Spectral_law.suite)
