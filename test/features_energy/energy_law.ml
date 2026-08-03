(* The pipeline law for the energy stages:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over randomized partitionings — one-sample chunks, chunks larger than
   frame_length, occasional empty chunks, the empty partition, and inputs
   shorter than the stage latency — for rms_stage (constant-zero borders) and
   zero_crossing_rate_stage (edge-copy borders), across frame geometries
   including non-divisor hops, hop > frame_length, an odd frame length and
   frame_length 1 (no borders at all). The PRNG is seeded so CI is
   deterministic. The comparison is exact for the geometries tested here: at
   these small frame lengths the blocked reduction sums each frame in the same
   order under every partitioning. Larger frames can wobble by an ulp when the
   materialized block extent changes the summation order, so new geometries must
   stay in this regime or compare within a small tolerance instead.

   Also here: agreement between the flat functions and their stages on one
   chunk, the static latency/rate queries, and the capability gate — one stage
   value is 'k-polymorphic and drives Pipeline.run and Pipeline.Stream on the
   same input. *)

open Windtrap
open Soundml

let seed = 0xe4e6

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

(* {2 Partitionings (the seeded pattern of the pipeline law tests)} *)

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (min n 13) in
    s :: random_sizes rng (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

let partitions rng n =
  let named =
    [ ("ones", List.init n (fun _ -> 1))
    ; ("whole", if n = 0 then [] else [n])
    ; ("empty", []) ]
  in
  let named = List.filter (fun (_, s) -> List.fold_left ( + ) 0 s = n) named in
  let random =
    List.init 5 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))
  in
  let random_empty =
    List.init 2 (fun i ->
        ( Printf.sprintf "random-with-empty-chunks-%d" i
        , sprinkle_empty rng (random_sizes rng n) ) )
  in
  named @ random @ random_empty

let chunks_of x sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        let chunk =
          if s = 0 then Nx.zeros (Nx.dtype x) [|0|]
          else Nx.shrink [|(off, off + s)|] x
        in
        chunk :: go (off + s) rest
  in
  go 0 sizes

let stream_outputs p ~source ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

let concat_or empty = function [] -> empty | l -> Nx.concatenate ~axis:(-1) l

(* {2 The law} *)

let law_case name mk_stage =
  test name (fun () ->
      let rng = Random.State.make [|seed|] in
      let src = source () in
      List.iter
        (fun n ->
          let x =
            Nx.create Nx.float32 [|n|]
              (Array.init n (fun i -> Float.sin (0.7 *. Float.of_int i)))
          in
          let p = mk_stage () in
          let expected = Nx.to_array (Pipeline.run ~source:src p x) in
          List.iter
            (fun (pname, sizes) ->
              let max_chunk = List.fold_left max 1 sizes in
              let outs =
                stream_outputs p ~source:src ~max_chunk (chunks_of x sizes)
              in
              let got =
                Nx.to_array (concat_or (Nx.zeros Nx.float32 [|1; 0|]) outs)
              in
              equal
                ~msg:(Printf.sprintf "%s/%s/n=%d" name pname n)
                (array (float 0.))
                expected got )
            (partitions rng n) )
        [61; 17; 2; 0] )

let law_tests =
  [ law_case "rms fl8 hop2" (fun () -> rms_stage ~frame_length:8 ~hop:2 ())
  ; law_case "rms fl8 hop3 (non-divisor)" (fun () ->
        rms_stage ~frame_length:8 ~hop:3 () )
  ; law_case "rms fl4 hop6 (hop past the frame)" (fun () ->
        rms_stage ~frame_length:4 ~hop:6 () )
  ; law_case "rms fl9 hop2 (odd frame)" (fun () ->
        rms_stage ~frame_length:9 ~hop:2 () )
  ; law_case "rms fl1 hop2 (no borders)" (fun () ->
        rms_stage ~frame_length:1 ~hop:2 () )
  ; law_case "zcr fl8 hop2" (fun () ->
        zero_crossing_rate_stage ~frame_length:8 ~hop:2 () )
  ; law_case "zcr fl8 hop3 (non-divisor)" (fun () ->
        zero_crossing_rate_stage ~frame_length:8 ~hop:3 () )
  ; law_case "zcr fl4 hop6 (hop past the frame)" (fun () ->
        zero_crossing_rate_stage ~frame_length:4 ~hop:6 () )
  ; law_case "zcr fl9 hop2 (odd frame)" (fun () ->
        zero_crossing_rate_stage ~frame_length:9 ~hop:2 () )
  ; law_case "zcr fl8 hop2 threshold 0.05" (fun () ->
        zero_crossing_rate_stage ~frame_length:8 ~hop:2 ~threshold:0.05 () ) ]

(* {2 The flat functions are the one-chunk instance} *)

let flat_tests =
  [ test "rms equals its stage on one chunk" (fun () ->
        let x =
          Nx.create Nx.float32 [|47|]
            (Array.init 47 (fun i -> Float.cos (0.3 *. Float.of_int i)))
        in
        let streamed =
          Pipeline.run ~source:(source ())
            (rms_stage ~frame_length:8 ~hop:3 ())
            x
        in
        equal ~msg:"same carry"
          (array (float 0.))
          (Nx.to_array (rms ~frame_length:8 ~hop:3 x))
          (Nx.to_array streamed) )
  ; test "zero_crossing_rate equals its stage on one chunk" (fun () ->
        let x =
          Nx.create Nx.float32 [|47|]
            (Array.init 47 (fun i -> Float.cos (0.3 *. Float.of_int i)))
        in
        let streamed =
          Pipeline.run ~source:(source ())
            (zero_crossing_rate_stage ~frame_length:8 ~hop:3 ())
            x
        in
        equal ~msg:"same carry"
          (array (float 0.))
          (Nx.to_array (zero_crossing_rate ~frame_length:8 ~hop:3 x))
          (Nx.to_array streamed) )
  ; test "an all-empty two-channel stream keeps its leading shape" (fun () ->
        (* the stage only ever sees empty chunks, so its output is the
           witness-built empty chunk — it must carry the stream's leading axes,
           exactly as the flat function's does *)
        let x = Nx.zeros Nx.float32 [|2; 0|] in
        let src =
          Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:2
        in
        let streamed =
          Pipeline.run ~source:src (rms_stage ~frame_length:4 ~hop:2 ()) x
        in
        equal ~msg:"streamed shape" (array int) [|2; 1; 0|] (Nx.shape streamed) ;
        equal ~msg:"flat shape agrees" (array int)
          (Nx.shape (rms ~frame_length:4 ~hop:2 x))
          (Nx.shape streamed) ) ]

(* {2 Static queries} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ test "latency and rate fold exactly" (fun () ->
        equal ~msg:"rms latency is the half frame" rate_t (r 4 1)
          (Pipeline.latency
             ( rms_stage ~frame_length:8 ~hop:2 ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        equal ~msg:"rms rate is 1/hop" rate_t (r 1 2)
          (Pipeline.rate
             ( rms_stage ~frame_length:8 ~hop:2 ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        equal ~msg:"odd frames look 4 ahead too" rate_t (r 4 1)
          (Pipeline.latency
             ( zero_crossing_rate_stage ~frame_length:9 ~hop:3 ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        equal ~msg:"frame_length 1 is latency-free" rate_t (r 0 1)
          (Pipeline.latency
             ( rms_stage ~frame_length:1 ~hop:2 ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ) ]

(* {2 Capability: one causal value, both drivers} *)

(* the binding is an application, so the value restriction applies: ['k] still
   generalises (relaxed value restriction) while the element dtype is fixed
   ground *)
let shared :
    ((float, Nx.float32_elt) Nx.t, (float, Nx.float32_elt) Nx.t, 'k) Pipeline.t
    =
  zero_crossing_rate_stage ~frame_length:8 ~hop:2 ()

(* first instantiation: offline *)
let shared_offline :
    ( (float, Nx.float32_elt) Nx.t
    , (float, Nx.float32_elt) Nx.t
    , Pipeline.offline )
    Pipeline.t =
  shared

let capability_tests =
  [ test "one stage value drives run and Stream" (fun () ->
        let x =
          Nx.create Nx.float32 [|20|]
            (Array.init 20 (fun i -> Float.sin (1.1 *. Float.of_int i)))
        in
        let expected = Pipeline.run ~source:(source ()) shared x in
        (* second instantiation of the SAME value: causal streaming *)
        let s =
          Pipeline.Stream.prepare shared ~source:(source ()) ~max_chunk:20
        in
        (* [@] evaluates its right operand first: sequence push before flush *)
        let pushed = Option.to_list (Pipeline.Stream.push s x) in
        let outs = pushed @ Pipeline.Stream.flush s in
        equal ~msg:"same value, two modes"
          (array (float 0.))
          (Nx.to_array expected)
          (Nx.to_array (concat_or (Nx.zeros Nx.float32 [|1; 0|]) outs)) ;
        ignore shared_offline ) ]

let suite =
  [ group "law" law_tests
  ; group "flat" flat_tests
  ; group "static" static_tests
  ; group "capability" capability_tests ]
