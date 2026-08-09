open Windtrap

let () =
  run "Soundml.Resample"
    ( Resample_config.suite @ Resample_kernel.suite @ Resample_law.suite
    @ Resample_gemm.suite @ Resample_quality.suite @ Resample_alloc.suite
    @ Resample_fft_probe.suite )
