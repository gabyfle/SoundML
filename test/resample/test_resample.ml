open Windtrap

let () =
  run "Soundml.Resample"
    (Resample_config.suite @ Resample_kernel.suite @ Resample_law.suite)
