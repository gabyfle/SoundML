(* This file must NOT compile: [Effects.split_stage] takes a fixed reference
   only, and the whole-signal [`Maximum] is not a value of its argument's type.
   The dune rule next to this file compiles it, accepts only a failing exit
   code, and greps the error for the refused tag. *)

let bad () = Soundml.Effects.split_stage ~reference:`Maximum ()
