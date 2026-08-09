open Windtrap

let () =
  run "Soundml.Mel"
    ( Mel_goldens.suite @ Mel_props.suite @ Mel_law.suite @ Accept_chain.suite
    @ Mel_alloc.suite )
