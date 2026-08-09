open Windtrap

let () =
  run "Soundml.Stft"
    (Stft_goldens.suite @ Stft_grid.suite @ Stft_law.suite @ Stft_alloc.suite)
