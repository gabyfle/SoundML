(* This file must NOT compile: handing an offline pipeline to [Stream.prepare]
   is a type error naming the capability types. The dune rule next to this file
   compiles it, accepts only a failing exit code, and greps the error for
   [offline] and [causal]. *)

let normalize :
    (float array, float array, Soundml.Pipeline.offline) Soundml.Pipeline.t =
  Soundml.Pipeline.offline_only (fun x -> x) ~concat:Array.concat

let bad (source : Soundml.Pipeline.Format.t) =
  Soundml.Pipeline.Stream.prepare normalize ~source ~max_chunk:512
