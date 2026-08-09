open Windtrap

let () =
  run "features_onset"
    (Onset_goldens.suite @ Onset_props.suite @ Onset_law.suite)
