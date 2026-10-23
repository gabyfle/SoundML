open Windtrap

let () =
  exit (run "hpss" (Hpss_goldens.suite @ Hpss_props.suite @ Hpss_stage.suite))
