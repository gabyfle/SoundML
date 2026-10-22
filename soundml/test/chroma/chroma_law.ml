(* The pipeline law for [Chroma.stage] and [Chroma.of_cqt_stage]:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over randomized partitionings (the seeded pattern of the stft law tests), for
   each stage alone over spectral chunks and for [Chroma.stage] behind
   [Stft.power_stage], whose stateful framing would expose a stage that misread
   chunk boundaries. Also here: the chain against the flat [chroma_stft] and the
   CQT stage against [Chroma.of_cqt], bit for bit; the same bits at float64 over
   a production geometry, where a projection left to Nx rounded differently with
   the size of its batch; the static latency/rate queries; and the
   construction-time validation. *)

open Windtrap
open Soundml

let seed = 0xc4a

let sample_rate = 22050

let source dtype = Pipeline.Format.audio dtype ~sample_rate ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let stft_config ?(hop = 64) () = Stft.Config.create ~fft_size:256 ~hop ()

let chroma_config ?(fft_size = 256) () =
  Chroma.Config.create ~sample_rate ~fft_size ()

let cqt_config () =
  Cqt.Config.create ~n_bins:84 ~bins_per_octave:12 ~hop:512 ~sample_rate ()

let bits t = Array.map Int64.bits_of_float (Nx.to_array t)

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
    List.init 3 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))
  in
  let random_empty =
    [("random-with-empty-chunks", sprinkle_empty rng (random_sizes rng n))]
  in
  named @ random @ random_empty

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

let check_law ~name ~seed ~empty p x lengths_last =
  let rng = Random.State.make [|seed|] in
  let src = source (Nx.dtype x) in
  let expected = bits (Pipeline.run ~source:src p x) in
  List.iter
    (fun (pname, sizes) ->
      let max_chunk = List.fold_left max 1 sizes in
      let outs = stream_outputs p ~source:src ~max_chunk (chunks_of x sizes) in
      equal
        ~msg:(Printf.sprintf "%s/%s/n=%d" name pname lengths_last)
        (array int64) expected
        (bits (concat_or empty outs)) )
    (partitions rng lengths_last)

let spectrum rows frames =
  Nx.init Nx.float32 [|rows; frames|] (fun i ->
      Float.exp (Float.sin (0.7 *. Float.of_int ((i.(0) * frames) + i.(1)))) )

let audio n =
  Nx.create Nx.float32 [|n|]
    (Array.init n (fun i -> Float.sin (0.37 *. Float.of_int i)))

(* {2 The law} *)

let law_tests =
  [ test "chroma stage alone" (fun () ->
        let p = Chroma.stage (chroma_config ()) in
        List.iter
          (fun frames ->
            check_law ~name:"alone" ~seed
              ~empty:(Nx.zeros Nx.float32 [|12; 0|])
              p (spectrum 129 frames) frames )
          [37; 5; 1; 0] )
  ; test "chroma stage alone, euclidean norm" (fun () ->
        let p = Chroma.stage ~norm:(`P 2.) (chroma_config ()) in
        check_law ~name:"l2" ~seed:(seed + 1)
          ~empty:(Nx.zeros Nx.float32 [|12; 0|])
          p (spectrum 129 23) 23 )
  ; test "power_stage >> chroma stage" (fun () ->
        List.iter
          (fun n ->
            let p =
              Pipeline.( >> )
                (Stft.power_stage (stft_config ()))
                (Chroma.stage (chroma_config ()))
            in
            check_law ~name:"chain" ~seed:(seed + 2)
              ~empty:(Nx.zeros Nx.float32 [|12; 0|])
              p (audio n) n )
          [1000; 257; 2; 0] )
  ; test "of_cqt stage alone" (fun () ->
        let p = Chroma.of_cqt_stage (cqt_config ()) in
        List.iter
          (fun frames ->
            check_law ~name:"cqt" ~seed:(seed + 3)
              ~empty:(Nx.zeros Nx.float32 [|12; 0|])
              p (spectrum 84 frames) frames )
          [41; 3; 1; 0] ) ]

(* {2 The stages against the flat functions} *)

let equivalence_tests =
  [ test "power_stage >> stage runs chroma_stft" (fun () ->
        let check dtype =
          let x = Nx.cast dtype (audio 3001) in
          let p =
            Pipeline.( >> )
              (Stft.power_stage (stft_config ()))
              (Chroma.stage (chroma_config ()))
          in
          equal ~msg:"same bits" (array int64)
            (bits (chroma_stft (stft_config ()) (chroma_config ()) x))
            (bits (Pipeline.run ~source:(source dtype) p x))
        in
        check Nx.float32 ; check Nx.float64 )
  ; test "of_cqt_stage runs of_cqt" (fun () ->
        let m = Nx.cast Nx.float64 (spectrum 84 57) in
        equal ~msg:"same bits" (array int64)
          (bits (Chroma.of_cqt ~norm:(`P 1.) (cqt_config ()) m))
          (bits
             (Pipeline.run ~source:(source Nx.float64)
                (Chroma.of_cqt_stage ~norm:(`P 1.) (cqt_config ()))
                m ) ) ) ]

(* {2 Float64 at production size}

   The cases above run at toy sizes, where Nx takes one path for every sum. Past
   a few hundred thousand terms it re-associates its sums by shape and hands
   large products to Accelerate, so a projection left to Nx agreed with its
   offline form only up to rounding there. This case runs seconds of noise
   through a production geometry at float64 and demands the same bits. *)

let production_tests =
  [ test "power_stage >> chroma stage at float64" (fun () ->
        let rng = Random.State.make [|0x64f|] in
        let n = 2 * sample_rate in
        let x =
          Nx.create Nx.float64 [|n|]
            (Array.init n (fun _ -> Random.State.float rng 2. -. 1.))
        in
        let mk () =
          Pipeline.( >> )
            (Stft.power_stage (Stft.Config.create ~hop:512 ~fft_size:2048 ()))
            (Chroma.stage (chroma_config ~fft_size:2048 ()))
        in
        let src = source Nx.float64 in
        let expected = bits (Pipeline.run ~source:src (mk ()) x) in
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
              stream_outputs (mk ()) ~source:src ~max_chunk (chunks_of x sizes)
            in
            equal ~msg:pname (array int64) expected
              (bits (Nx.concatenate ~axis:(-1) outs)) )
          [ ( "512"
            , List.init ((n + 511) / 512) (fun i -> min 512 (n - (i * 512))) )
          ; ("random", coarse n) ] ) ]

(* {2 Static queries and validation} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ test "the stages add no latency and keep the frame rate" (fun () ->
        let stage = Chroma.stage (chroma_config ()) in
        equal ~msg:"stage latency" rate_t (r 0 1) (Pipeline.latency stage) ;
        equal ~msg:"stage rate" rate_t (r 1 1) (Pipeline.rate stage) ;
        let cqt = Chroma.of_cqt_stage (cqt_config ()) in
        equal ~msg:"cqt stage latency" rate_t (r 0 1) (Pipeline.latency cqt) ;
        equal ~msg:"cqt stage rate" rate_t (r 1 1) (Pipeline.rate cqt) ) ]

let validation_tests =
  [ test "the norm is validated when a stage is built" (fun () ->
        raises ~msg:"stage"
          (Invalid_argument
             "stage: cannot normalise in the -1-norm (the exponent must be \
              finite and positive)" ) (fun () ->
            ignore (Chroma.stage ~norm:(`P (-1.)) (chroma_config ())) ) ;
        raises ~msg:"of_cqt_stage"
          (Invalid_argument
             "of_cqt_stage: cannot normalise in the -1-norm (the exponent must \
              be finite and positive)" ) (fun () ->
            ignore (Chroma.of_cqt_stage ~norm:(`P (-1.)) (cqt_config ())) ) )
  ; test "the resolution is validated when the cqt stage is built" (fun () ->
        raises ~msg:"non-dividing"
          (Invalid_argument
             "of_cqt_stage: cannot fold 12 bins per octave onto 5 chroma bands \
              (bins_per_octave must be an integer multiple of n_chroma)" )
          (fun () -> ignore (Chroma.of_cqt_stage ~n_chroma:5 (cqt_config ())) ) ;
        raises ~msg:"zero bands"
          (Invalid_argument
             "of_cqt_stage: cannot build 0 chroma bands (n_chroma must be at \
              least 1)" ) (fun () ->
            ignore (Chroma.of_cqt_stage ~n_chroma:0 (cqt_config ())) ) )
  ; test "a bins mismatch is reported at the chunk" (fun () ->
        let p = Chroma.stage (chroma_config ()) in
        let s =
          Pipeline.Stream.prepare p ~source:(source Nx.float32) ~max_chunk:4
        in
        raises ~msg:"wrong bin count"
          (Invalid_argument
             "apply: cannot project 100 frequency bins through a matrix built \
              for an FFT of size 256 (129 bins)" ) (fun () ->
            ignore (Pipeline.Stream.push s (spectrum 100 4)) ) ;
        let s =
          Pipeline.Stream.prepare
            (Chroma.of_cqt_stage (cqt_config ()))
            ~source:(source Nx.float32) ~max_chunk:4
        in
        raises ~msg:"wrong constant-Q bin count"
          (Invalid_argument
             "of_cqt_stage: cannot project 72 constant-Q bins through a \
              configuration of 84 bins" ) (fun () ->
            ignore (Pipeline.Stream.push s (spectrum 72 4)) ) ) ]

let suite =
  [ group "law" law_tests
  ; group "equivalence" equivalence_tests
  ; group "production" production_tests
  ; group "static" static_tests
  ; group "validation" validation_tests ]
