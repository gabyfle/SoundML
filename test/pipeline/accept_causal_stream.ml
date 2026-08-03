(* Must-compile companion to the must-not-compile gate in [rejects/].

   COMPILING this module is the test: a composed causal pipeline value, bound at
   a ground element type by an application (so the value restriction applies),
   must generalise in its capability parameter under the relaxed value
   restriction — one value serves BOTH the offline and the streaming driver, in
   either order. If ['k] did not generalise, the first instantiation below would
   pin it and the second would not typecheck. *)

open Soundml

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1

(* the binding is an application: not a syntactic value *)
let p = Pipeline.( >> ) (Toys.gain 2.0) (Toys.block_sum 4)

(* first instantiation: offline *)
let p_offline : (float array, float array, Pipeline.offline) Pipeline.t = p

(* second instantiation of the SAME value: causal *)
let p_stream () = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:16

(* and in the other order *)
let q = Pipeline.( >> ) (Toys.lookahead 2) (Toys.gain 0.5)

let q_stream () = Pipeline.Stream.prepare q ~source:(source ()) ~max_chunk:16

let q_offline : (float array, float array, Pipeline.offline) Pipeline.t = q

(* run and Stream.prepare on one value, both orders, executed *)
let tests =
  [ ( "capability"
    , [ Alcotest.test_case "one causal value drives run and Stream" `Quick
          (fun () ->
            let x = Array.init 10 float_of_int in
            let y_run = Pipeline.run ~source:(source ()) p x in
            let s = p_stream () in
            let pushed = Option.to_list (Pipeline.Stream.push s x) in
            let y_stream = Array.concat (pushed @ Pipeline.Stream.flush s) in
            Alcotest.(check (array (float 0.)))
              "same value, two modes" y_run y_stream ;
            let s = q_stream () in
            let pushed = Option.to_list (Pipeline.Stream.push s x) in
            let z_stream = Array.concat (pushed @ Pipeline.Stream.flush s) in
            let z_run = Pipeline.run ~source:(source ()) q x in
            Alcotest.(check (array (float 0.)))
              "prepare before run" z_run z_stream ;
            ignore p_offline ;
            ignore q_offline ) ] ) ]
