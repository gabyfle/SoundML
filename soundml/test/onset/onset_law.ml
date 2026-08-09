(* The pipeline law for the feature stages:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over randomized partitionings (the seeded pattern of the stft and mel law
   tests). onset_strength_stage carries lag frames of state across chunks — a
   kernel that misread a chunk boundary would difference the wrong frames — and
   is exercised alone over spectral chunks partitioned along the frame axis and
   composed behind the full stateful, rate-changing chain [power_stage >>
   Mel.stage >> Db.stage (Value 1.)]. The memoryless spectral_contrast_stage is
   pinned the same way, alone and behind [power_stage ~power:1.]. Also here: the
   static latency/rate queries and the capability gate — one onset stage value
   drives both drivers. *)

open Windtrap
open Soundml

let seed = 0xfea7

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

(* stft geometry: fft 8, bins 5 over 0..500 Hz; contrast geometry: bands [0,
   150], [150, 300], [300, 600] Hz on that grid; mel geometry: 3 bands, every
   filter supported *)
let stft_config ?(hop = 2) ?(alignment = `Centered) () =
  Stft.Config.create ~alignment ~fft_size:8 ~hop ()

let mel_config () = Mel.Config.create ~n_mels:3 ~sample_rate:1000 ~fft_size:8 ()

let contrast_stage ?linear () =
  spectral_contrast_stage (stft_config ()) ~n_bands:2 ~f_min:150. ?linear
    ~sample_rate:1000 ()

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

let check_law ~name ~seed empty p x lengths_last =
  let rng = Random.State.make [|seed|] in
  let src = source () in
  let expected = Nx.to_array (Pipeline.run ~source:src p x) in
  List.iter
    (fun (pname, sizes) ->
      let max_chunk = List.fold_left max 1 sizes in
      let outs = stream_outputs p ~source:src ~max_chunk (chunks_of x sizes) in
      let got = Nx.to_array (concat_or empty outs) in
      equal
        ~msg:(Printf.sprintf "%s/%s/n=%d" name pname lengths_last)
        (array (float 0.))
        expected got )
    (partitions rng lengths_last)

let spectral_chunk frames =
  Nx.init Nx.float32 [|5; frames|] (fun i ->
      Float.exp (Float.sin (0.7 *. Float.of_int ((i.(0) * frames) + i.(1)))) )

let audio_signal n =
  Nx.create Nx.float32 [|n|]
    (Array.init n (fun i -> Float.sin (0.7 *. Float.of_int i)))

(* {2 The stages alone, over spectral chunks} *)

let onset_alone_case name lag =
  test name (fun () ->
      let p = onset_strength_stage ~lag () in
      List.iter
        (fun frames ->
          check_law ~name ~seed
            (Nx.zeros Nx.float32 [|0|])
            p (spectral_chunk frames) frames )
        [37; 5; 1; 0] )

let contrast_alone_case name mk_stage =
  test name (fun () ->
      let p = mk_stage () in
      List.iter
        (fun frames ->
          check_law ~name ~seed
            (Nx.zeros Nx.float32 [|0; 0|])
            p (spectral_chunk frames) frames )
        [37; 5; 1; 0] )

(* {2 The composed chains behind a stateful, rate-changing upstream} *)

let onset_chain_case name mk_stft lag =
  test name (fun () ->
      List.iter
        (fun n ->
          let p =
            Pipeline.( >> )
              (Pipeline.( >> )
                 (Pipeline.( >> )
                    (Stft.power_stage (mk_stft ()))
                    (Mel.stage (mel_config ())) )
                 (Db.stage (Db.Value 1.)) )
              (onset_strength_stage ~lag ())
          in
          check_law ~name ~seed:(seed + 1)
            (Nx.zeros Nx.float32 [|0|])
            p (audio_signal n) n )
        [61; 17; 2; 0] )

let contrast_chain_case name mk_stft =
  test name (fun () ->
      List.iter
        (fun n ->
          let p =
            Pipeline.( >> )
              (Stft.power_stage ~power:1. (mk_stft ()))
              (spectral_contrast_stage (mk_stft ()) ~n_bands:2 ~f_min:150.
                 ~sample_rate:1000 () )
          in
          check_law ~name ~seed:(seed + 2)
            (Nx.zeros Nx.float32 [|0; 0|])
            p (audio_signal n) n )
        [61; 17; 2; 0] )

let law_tests =
  [ onset_alone_case "onset stage alone, lag 1" 1
  ; onset_alone_case "onset stage alone, lag 3" 3
  ; contrast_alone_case "contrast stage alone" (fun () -> contrast_stage ())
  ; contrast_alone_case "contrast stage alone, linear" (fun () ->
        contrast_stage ~linear:true () )
  ; onset_chain_case "power >> mel >> db >> onset centered"
      (fun () -> stft_config ())
      1
  ; onset_chain_case "power >> mel >> db >> onset left, lag 2, non-divisor hop"
      (fun () -> stft_config ~alignment:`Left ~hop:3 ())
      2
  ; contrast_chain_case "magnitude power_stage >> contrast centered" (fun () ->
        stft_config () )
  ; contrast_chain_case
      "magnitude power_stage >> contrast left, non-divisor hop" (fun () ->
        stft_config ~alignment:`Left ~hop:3 () ) ]

(* {2 Static queries} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ test "both stages add no latency and keep the frame rate" (fun () ->
        equal ~msg:"onset latency" rate_t (r 0 1)
          (Pipeline.latency
             ( onset_strength_stage ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        equal ~msg:"onset rate" rate_t (r 1 1)
          (Pipeline.rate
             ( onset_strength_stage ~lag:3 ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        equal ~msg:"contrast latency" rate_t (r 0 1)
          (Pipeline.latency
             ( contrast_stage ()
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        let chain =
          Pipeline.( >> )
            (Pipeline.( >> )
               (Pipeline.( >> )
                  (Stft.power_stage (stft_config ()))
                  (Mel.stage (mel_config ())) )
               (Db.stage (Db.Value 1.)) )
            (onset_strength_stage ())
        in
        equal ~msg:"chain latency is the stft's" rate_t (r 4 1)
          (Pipeline.latency chain) ;
        equal ~msg:"chain rate is the stft's" rate_t (r 1 2)
          (Pipeline.rate chain) ) ]

(* {2 Capability: one causal value, both drivers} *)

(* the binding is an application, so the value restriction applies: ['k] still
   generalises (relaxed value restriction) while the element dtype is fixed
   ground *)
let shared :
    ((float, Nx.float32_elt) Nx.t, (float, Nx.float32_elt) Nx.t, 'k) Pipeline.t
    =
  onset_strength_stage ~lag:2 ()

(* first instantiation: offline *)
let shared_offline :
    ( (float, Nx.float32_elt) Nx.t
    , (float, Nx.float32_elt) Nx.t
    , Pipeline.offline )
    Pipeline.t =
  shared

let capability_tests =
  [ test "one onset stage value drives run and Stream" (fun () ->
        let x = spectral_chunk 20 in
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
          (Nx.to_array (concat_or (Nx.zeros Nx.float32 [|0|]) outs)) ;
        ignore shared_offline ) ]

let suite =
  [ group "law" law_tests
  ; group "static" static_tests
  ; group "capability" capability_tests ]
