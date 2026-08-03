(* Must-compile capability gate for the composed feature chain.

   COMPILING this module is the test: the chain [Stft.power_stage >> Mel.stage
   >> Db.stage (Value 1.0)] is an application, so the value restriction applies,
   and the binding must still generalise in its capability parameter under the
   relaxed value restriction — one composed value serves BOTH [Pipeline.run] and
   [Pipeline.Stream.prepare]. Executing it pins that the two drivers also agree
   on the values. The [Maximum] variant of the same chain is offline by the
   reference's own GADT index and is already covered by the db suite's
   must-not-compile gate — not duplicated here. *)

open Windtrap
open Soundml

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1

let stft_config = Stft.Config.create ~fft_size:8 ~hop:2 ()

let mel_config = Mel.Config.create ~n_mels:3 ~sample_rate:1000 ~fft_size:8 ()

(* the binding is an application: not a syntactic value *)
let chain :
    ((float, Nx.float32_elt) Nx.t, (float, Nx.float32_elt) Nx.t, 'k) Pipeline.t
    =
  Pipeline.( >> )
    (Pipeline.( >> ) (Stft.power_stage stft_config) (Mel.stage mel_config))
    (Db.stage (Db.Value 1.0))

(* first instantiation: offline *)
let chain_offline :
    ( (float, Nx.float32_elt) Nx.t
    , (float, Nx.float32_elt) Nx.t
    , Pipeline.offline )
    Pipeline.t =
  chain

(* second instantiation of the SAME value: causal streaming *)
let chain_stream () =
  Pipeline.Stream.prepare chain ~source:(source ()) ~max_chunk:16

let suite =
  [ group "capability"
      [ test "the composed chain drives run and Stream" (fun () ->
            let x =
              Nx.create Nx.float32 [|40|]
                (Array.init 40 (fun i -> Float.sin (0.3 *. Float.of_int i)))
            in
            let expected = Pipeline.run ~source:(source ()) chain x in
            let s = chain_stream () in
            (* [@] evaluates its right operand first: sequence the pushes
               explicitly *)
            let pushed =
              List.filter_map (Pipeline.Stream.push s)
                (List.init 5 (fun i -> Nx.shrink [|(i * 8, (i + 1) * 8)|] x))
            in
            let outs = pushed @ Pipeline.Stream.flush s in
            let streamed =
              match outs with
              | [] ->
                  Nx.zeros Nx.float32 [|3; 0|]
              | l ->
                  Nx.concatenate ~axis:(-1) l
            in
            equal ~msg:"same value, two modes"
              (array (float 0.))
              (Nx.to_array expected) (Nx.to_array streamed) ;
            (* the projected decibel values are finite and shaped by the mel
               grid: 3 bands per emitted frame *)
            equal ~msg:"mel bands" int 3 (Nx.dim 0 expected) ;
            ignore chain_offline ) ] ]
