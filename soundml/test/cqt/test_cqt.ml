open Windtrap

let () =
  exit
    (run "Soundml.Cqt" (Cqt_goldens.suite @ Cqt_props.suite @ Cqt_stream.suite))
