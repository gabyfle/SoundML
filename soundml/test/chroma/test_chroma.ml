open Windtrap

let () = exit (run "Soundml.Chroma" (Chroma_goldens.suite @ Chroma_props.suite))
