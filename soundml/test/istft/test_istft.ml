open Windtrap

let () =
  exit
    (run "Soundml.Stft synthesis"
       ( Istft_goldens.suite @ Istft_law.suite @ Sistft_law.suite
       @ Sistft_stage.suite @ Gl_goldens.suite @ Gl_law.suite ) )
