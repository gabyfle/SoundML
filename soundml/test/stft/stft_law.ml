(* The pipeline law for the STFT stages:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over randomized partitionings — one-sample chunks, chunks larger than
   fft_size, occasional empty chunks, the empty partition, and inputs shorter
   than the stage latency — for both power_stage (real out) and stage (complex
   out, both witnesses), across alignments and padding modes. The PRNG is seeded
   so CI is deterministic; the comparison is exact, since every partitioning
   routes the same per-frame computation.

   Also here: the capability gate — one power_stage value is 'k-polymorphic and
   drives Pipeline.run and Pipeline.Stream on the same input — and the static
   latency/rate queries. *)

open Windtrap
open Soundml

let seed = 0x57f7

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

let float_law_case name mk_stage =
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
                Nx.to_array (concat_or (Nx.zeros Nx.float32 [|0; 0|]) outs)
              in
              equal
                ~msg:(Printf.sprintf "%s/%s/n=%d" name pname n)
                (array (float 0.))
                expected got )
            (partitions rng n) )
        [61; 17; 2; 0] )

let complex_law_case name cdtype mk_config =
  test name (fun () ->
      let rng = Random.State.make [|seed + 1|] in
      let src = source () in
      List.iter
        (fun n ->
          let x =
            Nx.create Nx.float32 [|n|]
              (Array.init n (fun i -> Float.cos (0.9 *. Float.of_int i)))
          in
          let p = Stft.stage cdtype (mk_config ()) in
          let expected = Nx.to_array (Pipeline.run ~source:src p x) in
          List.iter
            (fun (pname, sizes) ->
              let max_chunk = List.fold_left max 1 sizes in
              let outs =
                stream_outputs p ~source:src ~max_chunk (chunks_of x sizes)
              in
              let got =
                Nx.to_array (concat_or (Nx.zeros cdtype [|0; 0|]) outs)
              in
              equal
                ~msg:(Printf.sprintf "%s/%s/n=%d/count" name pname n)
                int (Array.length expected) (Array.length got) ;
              Array.iteri
                (fun i (e : Complex.t) ->
                  let g = got.(i) in
                  if not (Float.equal e.re g.re && Float.equal e.im g.im) then
                    failf "%s/%s/n=%d: value %d differs" name pname n i )
                expected )
            (partitions rng n) )
        [61; 17; 2; 0] )

let law_tests =
  [ float_law_case "power centered reflect" (fun () ->
        Stft.power_stage (Stft.Config.create ~fft_size:8 ~hop:2 ()) )
  ; float_law_case "power centered edge, non-divisor hop" (fun () ->
        Stft.power_stage (Stft.Config.create ~pad:`Edge ~fft_size:8 ~hop:3 ()) )
  ; float_law_case "power left" (fun () ->
        Stft.power_stage
          (Stft.Config.create ~alignment:`Left ~fft_size:8 ~hop:3 ()) )
  ; float_law_case "power right constant" (fun () ->
        Stft.power_stage
          (Stft.Config.create ~alignment:`Right ~pad:(`Constant 0.) ~fft_size:8
             ~hop:3 () ) )
  ; float_law_case "power right reflect (border lookahead)" (fun () ->
        Stft.power_stage
          (Stft.Config.create ~alignment:`Right ~fft_size:8 ~hop:3 ()) )
  ; float_law_case "power hop larger than fft" (fun () ->
        Stft.power_stage
          (Stft.Config.create ~alignment:`Left ~fft_size:4 ~hop:6 ()) )
  ; float_law_case "power magnitude exponent" (fun () ->
        Stft.power_stage ~power:1. (Stft.Config.create ~fft_size:8 ~hop:2 ()) )
  ; complex_law_case "stage complex64 centered" Nx.complex64 (fun () ->
        Stft.Config.create ~fft_size:8 ~hop:2 () )
  ; complex_law_case "stage complex128 centered constant" Nx.complex128
      (fun () ->
        Stft.Config.create ~pad:(`Constant 0.25) ~fft_size:8 ~hop:5 () ) ]

(* {2 Kernel sequencing} *)

(* Draining consumes the tail: a drained kernel refuses further samples instead
   of silently emitting frames against a cleared state, and reset restores a
   usable kernel. *)
let kernel_tests =
  [ test "step after flush raises until reset" (fun () ->
        let c = Stft.Config.create ~fft_size:16 ~hop:5 () in
        let k =
          Stft.Kernel.prepare Nx.complex128 c Nx.float64 ~channels:1
            ~max_block:64
        in
        let x =
          Nx.create Nx.float64 [|45|]
            (Array.init 45 (fun i -> Float.sin (0.23 *. Float.of_int i)))
        in
        (* [@] evaluates its right operand first: sequence step before flush *)
        let stepped = Stft.Kernel.step k x in
        let drained = Stft.Kernel.flush k in
        is_true ~msg:"step and flush emitted"
          (Option.is_some stepped && Option.is_some drained) ;
        raises_invalid_arg ~msg:"drained step refuses samples"
          "step: cannot feed a drained kernel (flush consumed the tail; reset \
           before reusing)" (fun () -> ignore (Stft.Kernel.step k x) ) ;
        Stft.Kernel.reset k ;
        is_true ~msg:"reset revives the kernel"
          (Option.is_some (Stft.Kernel.step k x)) )
  ; test "zero-size leading axis chunk raises" (fun () ->
        let c = Stft.Config.create ~fft_size:8 ~hop:2 () in
        let k =
          Stft.Kernel.prepare Nx.complex128 c Nx.float64 ~channels:1
            ~max_block:32
        in
        raises_invalid_arg ~msg:"no signals at all"
          "step: cannot analyse a chunk with a zero-size leading axis \
           (channels must be at least 1)" (fun () ->
            ignore (Stft.Kernel.step k (Nx.zeros Nx.float64 [|0; 20|])) ) ) ]

(* {2 Static queries} *)

let r num den = {Pipeline.Rate.num; den}

let static_tests =
  [ test "latency and rate fold exactly" (fun () ->
        let centered = Stft.Config.create ~fft_size:8 ~hop:2 () in
        equal ~msg:"centered latency" rate_t (r 4 1)
          (Pipeline.latency (Stft.power_stage centered)) ;
        equal ~msg:"rate 1/hop" rate_t (r 1 2)
          (Pipeline.rate (Stft.power_stage centered)) ;
        let left = Stft.Config.create ~alignment:`Left ~fft_size:8 ~hop:3 () in
        equal ~msg:"left latency" rate_t (r 0 1)
          (Pipeline.latency (Stft.stage Nx.complex64 left)) ;
        let right =
          Stft.Config.create ~alignment:`Right ~pad:(`Constant 0.) ~fft_size:8
            ~hop:3 ()
        in
        equal ~msg:"right latency" rate_t (r 0 1)
          (Pipeline.latency (Stft.power_stage right)) ;
        let right_reflect =
          Stft.Config.create ~alignment:`Right ~fft_size:8 ~hop:3 ()
        in
        equal ~msg:"right+reflect declares the border lookahead" rate_t (r 7 1)
          (Pipeline.latency (Stft.power_stage right_reflect)) ;
        equal ~msg:"config latency stays geometric" int 0
          (Stft.Config.latency right_reflect) )
  ; test "downstream formats see rate and channels" (fun () ->
        let seen = ref None in
        let probe =
          Pipeline.kernel
            ~concat:(concat_or (Nx.zeros Nx.float32 [|0; 0|]))
            ~prepare:(fun fmt -> seen := Some fmt)
            ~step:(fun () (c : (float, Nx.float32_elt) Nx.t) -> Some c)
            ()
        in
        let p =
          Pipeline.( >> )
            (Stft.power_stage (Stft.Config.create ~fft_size:8 ~hop:2 ()))
            probe
        in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:100 in
        ignore s ;
        match !seen with
        | None ->
            fail "probe prepare did not run"
        | Some fmt ->
            equal ~msg:"frames per second" rate_t (r 500 1)
              (Pipeline.Format.items_per_second fmt) ;
            equal ~msg:"bound covers the latency burst" (option int) (Some 53)
              (Pipeline.Format.max_items fmt) ;
            equal ~msg:"upstream latency in source samples" rate_t (r 4 1)
              (Pipeline.Format.upstream_latency fmt) ) ]

(* {2 Capability: one causal value, both drivers} *)

(* the binding is an application, so the value restriction applies: ['k] still
   generalises (relaxed value restriction) while the element dtype is fixed
   ground *)
let shared :
    ((float, Nx.float32_elt) Nx.t, (float, Nx.float32_elt) Nx.t, 'k) Pipeline.t
    =
  Stft.power_stage (Stft.Config.create ~fft_size:8 ~hop:2 ())

(* first instantiation: offline *)
let shared_offline :
    ( (float, Nx.float32_elt) Nx.t
    , (float, Nx.float32_elt) Nx.t
    , Pipeline.offline )
    Pipeline.t =
  shared

let capability_tests =
  [ test "one power_stage value drives run and Stream" (fun () ->
        let x =
          Nx.create Nx.float32 [|20|]
            (Array.init 20 (fun i -> Float.of_int i /. 7.))
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
          (Nx.to_array (concat_or (Nx.zeros Nx.float32 [|0; 0|]) outs)) ;
        ignore shared_offline )
  ; test "power_stage composes causal downstream" (fun () ->
        let p =
          Pipeline.( >> )
            (Pipeline.stateless (fun t -> Nx.mul_s t 0.5))
            (Stft.power_stage (Stft.Config.create ~fft_size:8 ~hop:2 ()))
        in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:16 in
        let x = Nx.create Nx.float32 [|16|] (Array.init 16 Float.of_int) in
        let pushed = Option.to_list (Pipeline.Stream.push s x) in
        let outs = pushed @ Pipeline.Stream.flush s in
        let streamed = concat_or (Nx.zeros Nx.float32 [|0; 0|]) outs in
        let expected = Pipeline.run ~source:(source ()) p x in
        equal ~msg:"composed law"
          (array (float 0.))
          (Nx.to_array expected) (Nx.to_array streamed) ) ]

let suite =
  [ group "law" law_tests
  ; group "kernel" kernel_tests
  ; group "static" static_tests
  ; group "capability" capability_tests ]
