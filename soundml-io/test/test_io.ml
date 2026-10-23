open Windtrap

let () =
  exit
    (run "Soundml_io"
       ( Io_roundtrip.suite @ Io_goldens.suite @ Io_errors.suite
       @ Io_reader.suite @ Io_law.suite @ Io_raw.suite ) )
