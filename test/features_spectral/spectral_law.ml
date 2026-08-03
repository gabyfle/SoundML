(* The pipeline law for the spectral-shape feature stages:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over randomized partitionings (the seeded pattern of the stft and mel law
   tests). Each stage is memoryless, so alone the law is trivial — pinned
   anyway, over magnitude chunks partitioned along the frame axis — and the
   composed chain [Stft.power_stage ~power:1. >> stage] exercises it behind a
   stateful, rate-changing upstream, where a stage that secretly buffered or
   misread chunk boundaries would break the equality. Also here: the static
   latency/rate queries — the features add no latency and keep the frame
   rate. *)

open Windtrap
open Soundml

let seed = 0x51ab

let sample_rate = 1000

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate ~channels:1

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

(* stft geometry: fft 8, five bins over 0..500 Hz; the feature grid is the FFT
   grid those five bins imply. *)
let stft_config ?(hop = 2) ?(alignment = `Centered) () =
  Stft.Config.create ~alignment ~fft_size:8 ~hop ()

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

(* [chunks_of x sizes] slices [x] along its last axis; empty chunks keep the
   leading shape so multi-axis stages see consistent chunk ranks. *)
let chunks_of x sizes =
  let nd = Nx.ndim x in
  let slice off s =
    if s = 0 then begin
      let shape = Array.copy (Nx.shape x) in
      shape.(nd - 1) <- 0 ;
      Nx.zeros (Nx.dtype x) shape
    end
    else
      Nx.shrink
        (Array.init nd (fun i ->
             if i = nd - 1 then (off, off + s) else (0, Nx.dim i x) ) )
        x
  in
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        slice off s :: go (off + s) rest
  in
  go 0 sizes

let stream_outputs p ~source ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

let concat_or empty = function [] -> empty | l -> Nx.concatenate ~axis:(-1) l

let check_law ~name ~seed p x lengths_last =
  let rng = Random.State.make [|seed|] in
  let src = source () in
  let expected = Nx.to_array (Pipeline.run ~source:src p x) in
  List.iter
    (fun (pname, sizes) ->
      let max_chunk = List.fold_left max 1 sizes in
      let outs = stream_outputs p ~source:src ~max_chunk (chunks_of x sizes) in
      let got = Nx.to_array (concat_or (Nx.zeros Nx.float32 [|0; 0|]) outs) in
      equal
        ~msg:(Printf.sprintf "%s/%s/n=%d" name pname lengths_last)
        (array (float 0.))
        expected got )
    (partitions rng lengths_last)

(* The four stages under test, at the shared grid. *)
let stages () =
  [ ("centroid", spectral_centroid_stage ~sample_rate ())
  ; ("bandwidth", spectral_bandwidth_stage ~p:3. ~sample_rate ())
  ; ("rolloff", spectral_rolloff_stage ~roll_percent:0.6 ~sample_rate ())
  ; ("flatness", spectral_flatness_stage ()) ]

(* {2 The memoryless stages alone, over magnitude chunks} *)

let stage_alone_case (name, p) =
  test (name ^ " stage alone") (fun () ->
      List.iter
        (fun frames ->
          let s =
            Nx.init Nx.float32 [|5; frames|] (fun i ->
                Float.exp
                  (Float.sin (0.7 *. Float.of_int ((i.(0) * frames) + i.(1)))) )
          in
          check_law ~name ~seed p s frames )
        [37; 5; 1; 0] )

(* {2 The composed chain behind a stateful, rate-changing upstream} *)

let chain_case name mk_stft (stage_name, stage) =
  let name = Printf.sprintf "magnitude_stage >> %s %s" stage_name name in
  test name (fun () ->
      List.iter
        (fun n ->
          let x =
            Nx.create Nx.float32 [|n|]
              (Array.init n (fun i -> Float.sin (0.7 *. Float.of_int i)))
          in
          let p =
            Pipeline.( >> ) (Stft.power_stage ~power:1. (mk_stft ())) stage
          in
          check_law ~name ~seed:(seed + 1) p x n )
        [61; 17; 2; 0] )

let law_tests =
  List.map stage_alone_case (stages ())
  @ List.map (chain_case "centered" (fun () -> stft_config ())) (stages ())
  @ [ chain_case "left, non-divisor hop"
        (fun () -> stft_config ~alignment:`Left ~hop:3 ())
        ("centroid", spectral_centroid_stage ~sample_rate ())
    ; chain_case "right reflect (border lookahead)"
        (fun () -> stft_config ~alignment:`Right ~hop:3 ())
        ("centroid", spectral_centroid_stage ~sample_rate ()) ]

(* {2 Static queries} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ test "the features add no latency and keep the frame rate" (fun () ->
        let alone = spectral_flatness_stage () in
        equal ~msg:"stage latency" rate_t (r 0 1)
          (Pipeline.latency
             ( alone
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        equal ~msg:"stage rate" rate_t (r 1 1)
          (Pipeline.rate
             ( spectral_centroid_stage ~sample_rate ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        let chain =
          Pipeline.( >> )
            (Stft.power_stage ~power:1. (stft_config ()))
            (spectral_bandwidth_stage ~sample_rate ())
        in
        equal ~msg:"chain latency is the stft's" rate_t (r 4 1)
          (Pipeline.latency chain) ;
        equal ~msg:"chain rate is the stft's" rate_t (r 1 2)
          (Pipeline.rate chain) ) ]

let suite = [group "law" law_tests; group "static" static_tests]
