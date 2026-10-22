(* The pipeline law for [Mel.stage]:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over randomized partitionings (the seeded pattern of the stft law tests). The
   stage is memoryless, so on its own the law is trivial — pinned anyway, over
   spectral chunks partitioned along the frame axis — and the composed chain
   [Stft.power_stage >> Mel.stage] exercises it behind a stateful, rate-changing
   upstream, where a stage that secretly buffered or misread chunk boundaries
   would break the equality. The same holds for [mfcc_stage], alone over mel
   chunks and at the end of [Stft.power_stage >> Mel.stage], and its chain
   agrees bit for bit with the flat [mfcc] wherever the 80 dB clamp, which the
   stage omits, does not bind. Also here: the static latency/rate queries — the
   projection and the cepstrum add no latency and keep the frame rate. *)

open Windtrap
open Soundml

let seed = 0x3e1

let source () = Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

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
      Nx.slice
        (List.init nd (fun i ->
             if i = nd - 1 then Nx.R (off, off + s) else Nx.A ) )
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
        (array float_exact) expected got )
    (partitions rng lengths_last)

(* {2 The memoryless stage alone, over spectral chunks} *)

let spectrum rows frames =
  Nx.init Nx.float32 [|rows; frames|] (fun i ->
      Float.exp (Float.sin (0.7 *. Float.of_int ((i.(0) * frames) + i.(1)))) )

let stage_alone_case name mk_mel =
  test name (fun () ->
      let p = Mel.stage (mk_mel ()) in
      List.iter
        (fun frames -> check_law ~name ~seed p (spectrum 5 frames) frames)
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

(* {2 The cepstral stage, alone and at the end of the chain} *)

(* [wide rows frames] spans 120 dB, with a maximum that moves from frame to
   frame: a stage that clamped each chunk under its own maximum would round the
   quiet bins differently for every partitioning. *)
let wide rows frames =
  Nx.init Nx.float32 [|rows; frames|] (fun i ->
      10. ** -.Float.of_int (((3 * i.(1)) + i.(0)) mod 13) )

let mfcc_alone_case name ?lifter n_mfcc =
  test name (fun () ->
      let p = mfcc_stage (mel_config ()) ~n_mfcc ?lifter () in
      List.iter
        (fun frames ->
          check_law ~name ~seed:(seed + 2) p (spectrum 3 frames) frames ;
          check_law ~name:(name ^ "/wide") ~seed:(seed + 2) p (wide 3 frames)
            frames )
        [37; 5; 1; 0] )

let mfcc_chain_case name mk_stft =
  test name (fun () ->
      List.iter
        (fun n ->
          let x =
            Nx.create Nx.float32 [|n|]
              (Array.init n (fun i -> Float.sin (0.7 *. Float.of_int i)))
          in
          let p =
            Pipeline.( >> )
              (Pipeline.( >> )
                 (Stft.power_stage (mk_stft ()))
                 (Mel.stage (mel_config ())) )
              (mfcc_stage (mel_config ()) ~n_mfcc:3 ())
          in
          check_law ~name ~seed:(seed + 3) p x n )
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
      mel_config
  ; mfcc_alone_case "mfcc stage alone" 3
  ; mfcc_alone_case "mfcc stage alone, two liftered coefficients" ~lifter:22. 2
  ; mfcc_chain_case "power_stage >> mel >> mfcc centered" (fun () ->
        stft_config () )
  ; mfcc_chain_case "power_stage >> mel >> mfcc right, non-divisor hop"
      (fun () -> stft_config ~alignment:`Right ~hop:3 () ) ]

(* {2 The cepstral chain against the flat [mfcc]}

   The stage omits [mfcc]'s 80 dB clamp, so the two agree only where it does not
   bind: each case first checks that the signal's log-mel spans less than 80 dB,
   then demands the same bits. *)

let noise dtype ~seed n =
  let rng = Random.State.make [|seed|] in
  Nx.cast dtype
    (Nx.create Nx.float64 [|n|]
       (Array.init n (fun _ -> Random.State.float rng 2. -. 1.)) )

let mfcc_agreement_case name ?lifter ~n_mfcc ~sample_rate mk_stft mk_mel dtype =
  test name (fun () ->
      let stft = mk_stft () and mel = mk_mel () in
      let x = noise dtype ~seed:(seed + 4) (2 * sample_rate) in
      let db =
        Convert.power_to_db (Nx.cast Nx.float64 (mel_spectrogram stft mel x))
      in
      let span = Nx.item [] (Nx.max db) -. Nx.item [] (Nx.min db) in
      is_true
        ~msg:(Printf.sprintf "the clamp does not bind (%g dB)" span)
        (span < 80.) ;
      let p =
        Pipeline.( >> )
          (Pipeline.( >> ) (Stft.power_stage stft) (Mel.stage mel))
          (mfcc_stage mel ~n_mfcc ?lifter ())
      in
      let source = Pipeline.Format.audio dtype ~sample_rate ~channels:1 in
      let bits t = Array.map Int64.bits_of_float (Nx.to_array t) in
      equal ~msg:"same bits" (array int64)
        (bits (mfcc stft mel ~n_mfcc ?lifter x))
        (bits (Pipeline.run ~source p x)) )

let production_geometry () = Stft.Config.create ~hop:512 ~fft_size:2048 ()

let production_bands () =
  Mel.Config.create ~n_mels:128 ~sample_rate:22050 ~fft_size:2048 ()

let agreement_tests =
  [ mfcc_agreement_case "toy geometry, float32" ~n_mfcc:3 ~sample_rate:1000
      (fun () -> stft_config ())
      mel_config Nx.float32
  ; mfcc_agreement_case "toy geometry, float64, liftered" ~lifter:22. ~n_mfcc:2
      ~sample_rate:1000
      (fun () -> stft_config ())
      mel_config Nx.float64
  ; mfcc_agreement_case "production geometry, float32" ~n_mfcc:20
      ~sample_rate:22050 production_geometry production_bands Nx.float32
  ; mfcc_agreement_case "production geometry, float64" ~n_mfcc:20
      ~sample_rate:22050 production_geometry production_bands Nx.float64 ]

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
  ; test "the cepstrum adds no latency and keeps the frame rate" (fun () ->
        let alone () :
            ( (float, Nx.float32_elt) Nx.t
            , (float, Nx.float32_elt) Nx.t
            , Pipeline.offline )
            Pipeline.t =
          mfcc_stage (mel_config ()) ~n_mfcc:3 ()
        in
        equal ~msg:"stage latency" rate_t (r 0 1) (Pipeline.latency (alone ())) ;
        equal ~msg:"stage rate" rate_t (r 1 1) (Pipeline.rate (alone ())) ;
        let chain =
          Pipeline.( >> )
            (Pipeline.( >> )
               (Stft.power_stage (stft_config ()))
               (Mel.stage (mel_config ())) )
            (mfcc_stage (mel_config ()) ~n_mfcc:3 ())
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

(* {2 Float64 at production size}

   The cases above run at toy sizes, where Nx takes one path for every sum. Past
   a few hundred thousand terms it re-associates its sums by shape and hands
   large products to Accelerate, so a reduction left to Nx agreed with its
   offline form only up to rounding there. These cases run seconds of noise
   through a production geometry at float64 and demand the same bits. *)

let production_rate = 22050

let production_signal () =
  let rng = Random.State.make [|0x64f|] in
  let n = 2 * production_rate in
  Nx.create Nx.float64 [|n|]
    (Array.init n (fun _ -> Random.State.float rng 2. -. 1.))

let bits t = Array.map Int64.bits_of_float (Nx.to_array t)

let production_case name mk =
  test name (fun () ->
      let x = production_signal () in
      let n = Nx.dim 0 x in
      let source =
        Pipeline.Format.audio Nx.float64 ~sample_rate:production_rate
          ~channels:1
      in
      let expected = bits (Pipeline.run ~source (mk ()) x) in
      let rng = Random.State.make [|0x650|] in
      let rec coarse n =
        if n = 0 then []
        else
          let s = 1 + Random.State.int rng (min n 3000) in
          s :: coarse (n - s)
      in
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left max 1 sizes in
          let outs =
            stream_outputs (mk ()) ~source ~max_chunk (chunks_of x sizes)
          in
          equal
            ~msg:(Printf.sprintf "%s/%s" name pname)
            (array int64) expected
            (bits (Nx.concatenate ~axis:(-1) outs)) )
        [ ("512", List.init ((n + 511) / 512) (fun i -> min 512 (n - (i * 512))))
        ; ("random", coarse n) ] )

let production_stft () = Stft.Config.create ~hop:512 ~fft_size:2048 ()

let production_mel () =
  Mel.Config.create ~n_mels:128 ~sample_rate:production_rate ~fft_size:2048 ()

let production_tests =
  [ production_case "power >> mel" (fun () ->
        Pipeline.( >> )
          (Stft.power_stage (production_stft ()))
          (Mel.stage (production_mel ())) )
  ; production_case "power >> mel >> db" (fun () ->
        Pipeline.( >> )
          (Pipeline.( >> )
             (Stft.power_stage (production_stft ()))
             (Mel.stage (production_mel ())) )
          (Db.stage (Db.Value 1.)) )
  ; production_case "power >> mel >> mfcc" (fun () ->
        Pipeline.( >> )
          (Pipeline.( >> )
             (Stft.power_stage (production_stft ()))
             (Mel.stage (production_mel ())) )
          (mfcc_stage (production_mel ()) ~lifter:22. ()) ) ]

let suite =
  [ group "law" law_tests
  ; group "mfcc agreement" agreement_tests
  ; group "production" production_tests
  ; group "static" static_tests ]
