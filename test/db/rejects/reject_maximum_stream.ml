(* This file must NOT compile: a [Db.stage Maximum] pipeline is offline by the
   GADT argument's own type — the whole-chunk maximum reference needs the whole
   signal — and handing it to [Stream.prepare] is a type error naming the
   capability types. The dune rule next to this file compiles it, accepts only a
   failing exit code, and greps the error for [offline] and [causal]. *)

let referenced_to_peak = Soundml.Db.stage Soundml.Db.Maximum

let bad (source : Soundml.Pipeline.Format.t) =
  Soundml.Pipeline.Stream.prepare referenced_to_peak ~source ~max_chunk:512
