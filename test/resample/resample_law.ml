(* The pipeline law for the resampler, at bit-equality:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   over seeded adversarial partitionings — one-sample chunks, whole-signal
   chunks, occasional empty chunks, the empty partition, inputs shorter than the
   group delay — and over an explicit max_chunk sweep with uniform chunking. The
   comparison is on IEEE bit patterns, never on tolerances: the C executor
   computes every output sample as one dot product with a fixed summation order,
   so every partitioning must produce identical bits.

   Also here: apply as the one-chunk instance of the kernel (apply equals the
   step/flush fold under every chunking), the identity-rate passthrough, the
   capability polymorphism of the stage, and the composition with a
   rate-sensitive downstream stage. *)

open Windtrap
open Soundml

let seed = 0x9e5a

(* {2 Bit-exact comparison} *)

let bits_of_tensor t =
  Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 t))

let check_bits ~msg expected actual =
  let e = bits_of_tensor expected and a = bits_of_tensor actual in
  equal ~msg:(msg ^ "/shape") (array int) (Nx.shape expected) (Nx.shape actual) ;
  if e <> a then begin
    let i = ref 0 in
    while !i < Array.length e && Int64.equal e.(!i) a.(!i) do
      incr i
    done ;
    failf "%s: bit divergence at flat index %d (%.17g vs %.17g)" msg !i
      (Int64.float_of_bits e.(!i))
      (Int64.float_of_bits a.(!i))
  end

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

let uniform_sizes chunk n =
  let rec go left =
    if left <= 0 then [] else min chunk left :: go (left - chunk)
  in
  go n

let chunks_of x sizes =
  let n = Nx.ndim x in
  let go_slice off s =
    if s = 0 then
      Nx.zeros (Nx.dtype x)
        (Array.append (Array.sub (Nx.shape x) 0 (n - 1)) [|0|])
    else
      Nx.shrink
        (Array.init n (fun i ->
             if i = n - 1 then (off, off + s) else (0, Nx.dim i x) ) )
        x
  in
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        go_slice off s :: go (off + s) rest
  in
  go 0 sizes

let signal dtype n =
  Nx.create dtype [|n|]
    (Array.init n (fun i ->
         Float.sin (0.61 *. Float.of_int i)
         +. (0.25 *. Float.cos (2.3 *. Float.of_int i)) ) )

let stream_outputs p ~source ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source ~max_chunk in
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

let concat_or empty = function [] -> empty | l -> Nx.concatenate ~axis:(-1) l

(* {2 The law: run vs Stream over adversarial partitionings} *)

let law_case name dtype ~sample_rate ~target ~quality ~ns () =
  test name (fun () ->
      let rng = Random.State.make [|seed|] in
      let cfg = Resample.Config.create ~quality ~sample_rate ~target () in
      let p = Resample.stage cfg in
      let src = Pipeline.Format.audio dtype ~sample_rate ~channels:1 in
      List.iter
        (fun n ->
          let x = signal dtype n in
          let expected = Pipeline.run ~source:src p x in
          equal
            ~msg:(Printf.sprintf "%s/n=%d/offline length" name n)
            int
            (Resample.Config.output_frames cfg ~n)
            (Nx.dim (Nx.ndim expected - 1) expected) ;
          List.iter
            (fun (pname, sizes) ->
              let max_chunk = List.fold_left max 1 sizes in
              let outs =
                stream_outputs p ~source:src ~max_chunk (chunks_of x sizes)
              in
              let got = concat_or (Nx.zeros dtype [|0|]) outs in
              check_bits
                ~msg:(Printf.sprintf "%s/%s/n=%d" name pname n)
                expected got )
            (partitions rng n) )
        ns )

(* {2 The max_chunk sweep: same signal, uniform chunkings} *)

let sweep_case ?(chunks = [1; 7; 64; 401; 4096]) name dtype ~sample_rate ~target
    ~quality ~n () =
  test name (fun () ->
      let cfg = Resample.Config.create ~quality ~sample_rate ~target () in
      let p = Resample.stage cfg in
      let src = Pipeline.Format.audio dtype ~sample_rate ~channels:1 in
      let x = signal dtype n in
      let expected = Pipeline.run ~source:src p x in
      List.iter
        (fun max_chunk ->
          let outs =
            stream_outputs p ~source:src ~max_chunk
              (chunks_of x (uniform_sizes max_chunk n))
          in
          let got = concat_or (Nx.zeros dtype [|0|]) outs in
          check_bits
            ~msg:(Printf.sprintf "%s/max_chunk=%d" name max_chunk)
            expected got )
        chunks )

(* {2 apply == the kernel fold, under every chunking} *)

let fold_case name dtype ~channels ~sample_rate ~target ~quality ~n () =
  test name (fun () ->
      let rng = Random.State.make [|seed + 2|] in
      let cfg = Resample.Config.create ~quality ~sample_rate ~target () in
      let x =
        if channels = 1 then signal dtype n
        else
          Nx.create dtype [|channels; n|]
            (Array.init (channels * n) (fun i ->
                 Float.sin
                   ( (0.37 +. (0.11 *. Float.of_int (i / n)))
                   *. Float.of_int (i mod n) ) ) )
      in
      let expected = Resample.apply cfg x in
      List.iter
        (fun (pname, sizes) ->
          let max_block = List.fold_left max 1 sizes in
          let k = Resample.Kernel.prepare cfg dtype ~channels ~max_block in
          let stepped =
            List.filter_map (Resample.Kernel.step k) (chunks_of x sizes)
          in
          let outs = stepped @ Option.to_list (Resample.Kernel.flush k) in
          let empty =
            Nx.zeros dtype (if channels = 1 then [|0|] else [|channels; 0|])
          in
          check_bits
            ~msg:(Printf.sprintf "%s/%s" name pname)
            expected (concat_or empty outs) )
        (partitions rng n) )

(* {2 Identity rate: true passthrough} *)

let identity_tests =
  [ test "apply returns the input itself" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:44100 () in
        let x = signal Nx.float32 64 in
        is_true ~msg:"physical equality" (Resample.apply cfg x == x) )
  ; test "stage forwards bits with latency 0 and rate 1/1" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:8000 ~target:8000 () in
        let p = Resample.stage cfg in
        let rate_t =
          testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()
        in
        equal ~msg:"latency" rate_t
          {Pipeline.Rate.num= 0; den= 1}
          (Pipeline.latency p) ;
        equal ~msg:"rate" rate_t
          {Pipeline.Rate.num= 1; den= 1}
          (Pipeline.rate p) ;
        let src =
          Pipeline.Format.audio Nx.float64 ~sample_rate:8000 ~channels:1
        in
        let x = signal Nx.float64 47 in
        check_bits ~msg:"offline passthrough" x (Pipeline.run ~source:src p x) ;
        let s = Pipeline.Stream.prepare p ~source:src ~max_chunk:16 in
        let pushed =
          List.filter_map (Pipeline.Stream.push s)
            (chunks_of x (uniform_sizes 16 47))
        in
        let outs = pushed @ Pipeline.Stream.flush s in
        check_bits ~msg:"streaming passthrough" x
          (concat_or (Nx.zeros Nx.float64 [|0|]) outs) )
  ; test "identity kernel step copies, never aliases" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:1000 ~target:1000 () in
        let k =
          Resample.Kernel.prepare cfg Nx.float64 ~channels:1 ~max_block:32
        in
        let x = signal Nx.float64 8 in
        match Resample.Kernel.step k x with
        | None ->
            fail "identity step emitted nothing"
        | Some y ->
            is_true ~msg:"fresh tensor" (not (y == x)) ;
            check_bits ~msg:"copied bits" x y ) ]

(* {2 The flat one-liner} *)

let flat_tests =
  [ test "Soundml.resample delegates to apply" (fun () ->
        let x = signal Nx.float64 300 in
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:16000 () in
        check_bits ~msg:"default quality" (Resample.apply cfg x)
          (Soundml.resample ~sample_rate:44100 ~target:16000 x) ;
        let fast =
          Resample.Config.create ~quality:`Fast ~sample_rate:44100 ~target:16000
            ()
        in
        check_bits ~msg:"quality forwards" (Resample.apply fast x)
          (Soundml.resample ~quality:`Fast ~sample_rate:44100 ~target:16000 x) )
  ]

(* {2 Capability: one causal value, both drivers} *)

let shared :
    ((float, Nx.float32_elt) Nx.t, (float, Nx.float32_elt) Nx.t, 'k) Pipeline.t
    =
  Resample.stage
    (Resample.Config.create
       ~quality:(`Custom {Resample.attenuation= 60.; passband= 0.8})
       ~sample_rate:3000 ~target:2000 () )

let shared_offline :
    ( (float, Nx.float32_elt) Nx.t
    , (float, Nx.float32_elt) Nx.t
    , Pipeline.offline )
    Pipeline.t =
  shared

let capability_tests =
  [ test "one stage value drives run and Stream" (fun () ->
        let src =
          Pipeline.Format.audio Nx.float32 ~sample_rate:3000 ~channels:1
        in
        let x = signal Nx.float32 200 in
        let expected = Pipeline.run ~source:src shared x in
        let s = Pipeline.Stream.prepare shared ~source:src ~max_chunk:200 in
        let pushed = Option.to_list (Pipeline.Stream.push s x) in
        let outs = pushed @ Pipeline.Stream.flush s in
        check_bits ~msg:"same value, two modes" expected
          (concat_or (Nx.zeros Nx.float32 [|0|]) outs) ;
        ignore shared_offline )
  ; test "resample composes upstream of the STFT" (fun () ->
        let cfg =
          Resample.Config.create
            ~quality:(`Custom {Resample.attenuation= 60.; passband= 0.8})
            ~sample_rate:1000 ~target:1600 ()
        in
        let p =
          Pipeline.( >> ) (Resample.stage cfg)
            (Stft.power_stage (Stft.Config.create ~fft_size:8 ~hop:3 ()))
        in
        let src =
          Pipeline.Format.audio Nx.float32 ~sample_rate:1000 ~channels:1
        in
        let x = signal Nx.float32 230 in
        let expected = Pipeline.run ~source:src p x in
        List.iter
          (fun max_chunk ->
            let outs =
              stream_outputs p ~source:src ~max_chunk
                (chunks_of x (uniform_sizes max_chunk 230))
            in
            check_bits
              ~msg:(Printf.sprintf "composed law/max_chunk=%d" max_chunk)
              expected
              (concat_or (Nx.zeros Nx.float32 [|0; 0|]) outs) )
          [1; 13; 230] ) ]

(* {2 The suites} *)

let law_tests =
  [ law_case "up 44100->48000 High f64" Nx.float64 ~sample_rate:44100
      ~target:48000 ~quality:`High ~ns:[400; 61; 7; 0] ()
  ; law_case "down 48000->44100 High f32" Nx.float32 ~sample_rate:48000
      ~target:44100 ~quality:`High ~ns:[400; 61; 0] ()
  ; law_case "down 44100->16000 Fast f64" Nx.float64 ~sample_rate:44100
      ~target:16000 ~quality:`Fast ~ns:[500; 90] ()
  ; law_case "up 8000->192000 Fast f32" Nx.float32 ~sample_rate:8000
      ~target:192000 ~quality:`Fast ~ns:[150; 3] ()
  ; law_case "coprime 3->2 custom f64" Nx.float64 ~sample_rate:3 ~target:2
      ~quality:(`Custom {Resample.attenuation= 60.; passband= 0.8})
      ~ns:[200; 29; 1] ()
  ; law_case "integer 44100->22050 Best f64" Nx.float64 ~sample_rate:44100
      ~target:22050 ~quality:`Best ~ns:[300] () ]

let sweep_tests =
  [ sweep_case "sweep 44100->48000 High f64" Nx.float64 ~sample_rate:44100
      ~target:48000 ~quality:`High ~n:1000 ()
  ; sweep_case "sweep 48000->44100 High f32" Nx.float32 ~sample_rate:48000
      ~target:44100 ~quality:`High ~n:1000 () ]

let fold_tests =
  [ fold_case "apply == fold, mono up f64" Nx.float64 ~channels:1
      ~sample_rate:44100 ~target:48000 ~quality:`High ~n:400 ()
  ; fold_case "apply == fold, mono down f32" Nx.float32 ~channels:1
      ~sample_rate:48000 ~target:44100 ~quality:`High ~n:400 ()
  ; fold_case "apply == fold, stereo down f64" Nx.float64 ~channels:2
      ~sample_rate:44100 ~target:16000 ~quality:`Fast ~n:333 () ]

(* {2 The law over cascade plans}

   Wide-ratio conversions run as two chained stages (the planner in
   resample.ml); the composite keeps the law by composition — stage 1 is
   chunk-invariant, so stage 2 sees one sample sequence under every
   partitioning, and stage 2 is invariant to how that sequence is cut. These
   cases pin the composition where it could crack: the adversarial partitioner
   over cascade configs at `High (the shipped spec, both directions and both
   dtypes), one-sample chunks, uniform chunkings walking +-1 around the stage
   rationals' periods, inputs around the composite latency (flush-adjacent
   splits), and the ceil-composition truncation edge — the n = 11 class at 44.1
   -> 16 k, where the raw two-stage stream over-produces by one sample and the
   drain cut restores the exact [ceil (n * L / M)] contract. Every comparison is
   bit-equality against the offline run, whose length law_case asserts against
   [output_frames]. *)

let cascade_law_tests =
  [ law_case "cascade 44100->16000 High f64" Nx.float64 ~sample_rate:44100
      ~target:16000 ~quality:`High ~ns:[1000; 341; 11; 5; 1; 0] ()
  ; law_case "cascade 44100->16000 High f32" Nx.float32 ~sample_rate:44100
      ~target:16000 ~quality:`High ~ns:[1000; 11] ()
  ; law_case "cascade 48000->8000 High f64" Nx.float64 ~sample_rate:48000
      ~target:8000 ~quality:`High ~ns:[2000; 700; 11; 1] ()
  ; law_case "cascade 8000->48000 High f32" Nx.float32 ~sample_rate:8000
      ~target:48000 ~quality:`High ~ns:[500; 113; 11; 1] ()
  ; law_case "cascade 16000->44100 High f64" Nx.float64 ~sample_rate:16000
      ~target:44100 ~quality:`High ~ns:[500; 217; 3] ()
  ; law_case "cascade 22050->44100 Best f64" Nx.float64 ~sample_rate:22050
      ~target:44100 ~quality:`Best ~ns:[400; 11] () ]

let cascade_sweep_tests =
  [ sweep_case
      ~chunks:[1; 2; 3; 440; 441; 442; 4096]
      "cascade sweep 44100->16000 High f64" Nx.float64 ~sample_rate:44100
      ~target:16000 ~quality:`High ~n:2000 ()
  ; sweep_case
      ~chunks:[1; 2; 3; 4; 108; 109; 4096]
      "cascade sweep 8000->48000 High f32" Nx.float32 ~sample_rate:8000
      ~target:48000 ~quality:`High ~n:1500 () ]

let cascade_fold_tests =
  [ fold_case "apply == fold, stereo cascade down f32" Nx.float32 ~channels:2
      ~sample_rate:44100 ~target:16000 ~quality:`High ~n:600 ()
  ; fold_case "apply == fold, mono cascade up f64" Nx.float64 ~channels:1
      ~sample_rate:8000 ~target:48000 ~quality:`High ~n:333 () ]

let cascade_length_tests =
  [ test "cascade totals hit ceil(n * L / M) for every small n" (fun () ->
        (* the drain truncation, walked across every phase alignment: the raw
           two-stage total exceeds the composite ceil by at most one, and the
           excess pattern cycles with n — small n sweeps the whole cycle *)
        List.iter
          (fun (sample_rate, target) ->
            let cfg = Resample.Config.create ~sample_rate ~target () in
            for n = 0 to 130 do
              let y = Resample.apply cfg (signal Nx.float64 n) in
              equal
                ~msg:(Printf.sprintf "%d->%d n=%d" sample_rate target n)
                int
                (Resample.Config.output_frames cfg ~n)
                (Nx.dim (Nx.ndim y - 1) y)
            done )
          [(44100, 16000); (16000, 44100); (48000, 8000); (8000, 48000)] ) ]

let suite =
  [ group "law" law_tests
  ; group "max-chunk sweep" sweep_tests
  ; group "apply-fold" fold_tests
  ; group "cascade-law" cascade_law_tests
  ; group "cascade-sweep" cascade_sweep_tests
  ; group "cascade-fold" cascade_fold_tests
  ; group "cascade-length" cascade_length_tests
  ; group "identity" identity_tests
  ; group "flat" flat_tests
  ; group "capability" capability_tests ]
