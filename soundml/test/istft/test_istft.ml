open Windtrap

let () =
  run "Soundml.Stft synthesis"
    (Istft_goldens.suite @ Istft_law.suite @ Gl_goldens.suite @ Gl_law.suite)
