(* Must-compile companion to the must-not-compile gate in [rejects/].

   COMPILING this module is the test: [Db.stage (Value _)] is an application, so
   the value restriction applies, and the binding must still generalise in its
   capability parameter under the relaxed value restriction — one stage value
   serves BOTH the offline and the streaming driver. Executing it pins that the
   two drivers also agree on the values. *)

open Windtrap
open Soundml

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1

(* the binding is an application: not a syntactic value *)
let p = Db.stage (Db.Value 1.0)

(* first instantiation: offline *)
let p_offline :
    ( (float, Nx.float32_elt) Nx.t
    , (float, Nx.float32_elt) Nx.t
    , Pipeline.offline )
    Pipeline.t =
  p

(* second instantiation of the SAME value: causal *)
let p_stream () = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:16

let suite =
  [ group "capability"
      [ test "one Value stage drives run and Stream" (fun () ->
            let x = Nx.create Nx.float32 [|4|] [|1.; 0.1; 0.01; 10.|] in
            let y_run = Pipeline.run ~source:(source ()) p x in
            let s = p_stream () in
            let pushed = Option.to_list (Pipeline.Stream.push s x) in
            let outs = pushed @ Pipeline.Stream.flush s in
            let y_stream = Nx.concatenate ~axis:0 outs in
            equal ~msg:"same value, two modes"
              (array (float 0.))
              (Nx.to_array y_run) (Nx.to_array y_stream) ;
            ignore p_offline ) ] ]
