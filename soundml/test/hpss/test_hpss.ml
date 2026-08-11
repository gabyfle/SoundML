open Windtrap

let () = run "hpss" (Hpss_goldens.suite @ Hpss_props.suite)
