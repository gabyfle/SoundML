(* The pipeline law for [Mel.stage]:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over randomized partitionings (the seeded pattern of the stft law tests). The
   stage is memoryless, so on its own the law is trivial — pinned anyway, over
   spectral chunks partitioned along the frame axis — and the composed chain
   [Stft.power_stage >> Mel.stage] exercises it behind a stateful, rate-changing
   upstream, where a stage that secretly buffered or misread chunk boundaries
   would break the equality. Also here: the static latency/rate queries — the
   projection adds no latency and keeps the frame rate. *)

open Windtrap
open Soundml

let seed = 0x3e1

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

(* stft geometry: fft 8, bins 5 over 0..500 Hz; mel geometry: 3 bands on the
   same grid, every filter supported *)
let stft_config ?(hop = 2) ?(alignment = `Centered) () =
  Stft.Config.create ~alignment ~fft_size:8 ~hop ()

let mel_config ?(norm = `Slaney) () =
  Mel.Config.create ~norm ~n_mels:3 ~sample_rate:1000 ~fft_size:8 ()

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

(* {2 The memoryless stage alone, over spectral chunks} *)

let stage_alone_case name mk_mel =
  test name (fun () ->
      let p = Mel.stage (mk_mel ()) in
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

let chain_case name mk_stft mk_mel =
  test name (fun () ->
      List.iter
        (fun n ->
          let x =
            Nx.create Nx.float32 [|n|]
              (Array.init n (fun i -> Float.sin (0.7 *. Float.of_int i)))
          in
          let p =
            Pipeline.( >> )
              (Stft.power_stage (mk_stft ()))
              (Mel.stage (mk_mel ()))
          in
          check_law ~name ~seed:(seed + 1) p x n )
        [61; 17; 2; 0] )

let law_tests =
  [ stage_alone_case "mel stage alone" mel_config
  ; stage_alone_case "mel stage alone, unnormalised" (fun () ->
        mel_config ~norm:`None () )
  ; chain_case "power_stage >> mel centered"
      (fun () -> stft_config ())
      mel_config
  ; chain_case "power_stage >> mel left, non-divisor hop"
      (fun () -> stft_config ~alignment:`Left ~hop:3 ())
      mel_config
  ; chain_case "power_stage >> mel right reflect (border lookahead)"
      (fun () -> stft_config ~alignment:`Right ~hop:3 ())
      mel_config ]

(* {2 Static queries} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ test "the projection adds no latency and keeps the frame rate" (fun () ->
        let alone = Mel.stage (mel_config ()) in
        equal ~msg:"stage latency" rate_t (r 0 1)
          (Pipeline.latency
             ( alone
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        equal ~msg:"stage rate" rate_t (r 1 1)
          (Pipeline.rate
             ( Mel.stage (mel_config ())
               : ( (float, Nx.float32_elt) Nx.t
                 , (float, Nx.float32_elt) Nx.t
                 , Pipeline.offline )
                 Pipeline.t ) ) ;
        let chain =
          Pipeline.( >> )
            (Stft.power_stage (stft_config ()))
            (Mel.stage (mel_config ()))
        in
        equal ~msg:"chain latency is the stft's" rate_t (r 4 1)
          (Pipeline.latency chain) ;
        equal ~msg:"chain rate is the stft's" rate_t (r 1 2)
          (Pipeline.rate chain) )
  ; test "downstream formats pass through unchanged" (fun () ->
        let seen_after_stft = ref None and seen_after_mel = ref None in
        let probe cell =
          Pipeline.kernel
            ~concat:(concat_or (Nx.zeros Nx.float32 [|0; 0|]))
            ~prepare:(fun fmt -> cell := Some fmt)
            ~step:(fun () (c : (float, Nx.float32_elt) Nx.t) -> Some c)
            ()
        in
        let p =
          Pipeline.( >> )
            (Pipeline.( >> )
               (Pipeline.( >> )
                  (Stft.power_stage (stft_config ()))
                  (probe seen_after_stft) )
               (Mel.stage (mel_config ())) )
            (probe seen_after_mel)
        in
        ignore (Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:100) ;
        match (!seen_after_stft, !seen_after_mel) with
        | Some upstream, Some downstream ->
            is_true ~msg:"the mel stage is format-preserving"
              (Pipeline.Format.equal upstream downstream)
        | _ ->
            fail "probe prepares did not run" ) ]

let suite = [group "law" law_tests; group "static" static_tests]
