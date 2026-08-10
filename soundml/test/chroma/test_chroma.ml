open Windtrap

let () = run "Soundml.Chroma" (Chroma_goldens.suite @ Chroma_props.suite)
