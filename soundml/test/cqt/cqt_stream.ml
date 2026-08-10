(* The partition law of the incremental constant-Q kernel, at bit-equality:

   concat (List.filter_map (Kernel.step k) (chunks x)) @ Kernel.flush k = the
   same kernel driven on one chunk

   over a configuration grid that exercises every plan shape — the default
   seven-octave ladder in both dtypes, a non-default resolution with an early
   decimation, the variable-Q ladder, an odd hop (which stops the recursion from
   halving anywhere, so no resampler runs), a deep early decimation at 44.1 kHz,
   the 252-bin chromagram ladder, a single-octave plan and a two-channel stream
   — and over pathological chunkings: one sample at a time, prime chunks, chunks
   that straddle the frame cadence and the latency boundary, one giant chunk
   followed by single samples, interspersed empty chunks, the whole signal.

   Also here: the latency formula against brute-forced constants and against the
   instant each column actually leaves the kernel, the frame-count algebra
   against [Cqt.frames], the drain edges, and the documented agreement between
   the kernel and the offline transform — which is a tolerance, not a law: the
   two paths decimate and project through different arithmetic. *)

open Windtrap
open Soundml

(* {2 A seeded broadband signal} *)

let noise seed n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let signal dtype seed n = Nx.create dtype [|n|] (noise seed n)

(* {2 Bit-exact comparison} *)

let bits_of_tensor t =
  Array.map
    (fun (z : Complex.t) -> (Int64.bits_of_float z.re, Int64.bits_of_float z.im))
    (Nx.to_array t)

let check_bits ~msg expected actual =
  equal ~msg:(msg ^ "/shape") (array int) (Nx.shape expected) (Nx.shape actual) ;
  let e = bits_of_tensor expected and a = bits_of_tensor actual in
  if e <> a then begin
    let i = ref 0 in
    while !i < Array.length e && e.(!i) = a.(!i) do
      incr i
    done ;
    let re, im = a.(!i) and re', im' = e.(!i) in
    failf
      "%s: bit divergence at flat index %d ((%.17g, %.17g) vs (%.17g, %.17g))"
      msg !i (Int64.float_of_bits re) (Int64.float_of_bits im)
      (Int64.float_of_bits re') (Int64.float_of_bits im')
  end

(* {2 Driving the kernel} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let empty_like x =
  let shape = Nx.shape x in
  let lead = Array.sub shape 0 (Array.length shape - 1) in
  Nx.zeros (Nx.dtype x) (Array.append lead [|0|])

let slice x start stop =
  let nd = Nx.ndim x in
  Nx.shrink
    (Array.init nd (fun i ->
         if i = nd - 1 then (start, stop) else (0, Nx.dim i x) ) )
    x

let chunks_of x sizes =
  let rec go off = function
    | [] ->
        []
    | 0 :: rest ->
        empty_like x :: go off rest
    | s :: rest ->
        slice x off (off + s) :: go (off + s) rest
  in
  go 0 sizes

let concat_columns c parts =
  match parts with
  | [] ->
      Nx.zeros Nx.complex128 [|Cqt.Config.n_bins c; 0|]
  | [one] ->
      one
  | many ->
      Nx.concatenate ~axis:(-1) many

let stream c dtype ~channels x sizes =
  let max_block = List.fold_left Stdlib.max 1 sizes in
  let k = Cqt.Kernel.prepare Nx.complex128 c dtype ~channels ~max_block in
  let pushed = List.filter_map (Cqt.Kernel.step k) (chunks_of x sizes) in
  let drained = Option.to_list (Cqt.Kernel.flush k) in
  concat_columns c (pushed @ drained)

let whole c dtype x =
  stream c dtype ~channels:1 x (if last_dim x = 0 then [] else [last_dim x])

(* {2 Chunkings} *)

let uniform size n =
  let rec go left =
    if left <= 0 then [] else Stdlib.min size left :: go (left - size)
  in
  go n

let straddle at n =
  (* two chunks meeting one sample before [at], then the rest: the cadence and
     latency boundaries are crossed inside a chunk rather than between two *)
  if at <= 1 || at >= n - 1 then [n] else [at - 1; 2; n - at - 1]

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 3 = 0 then [0; s] else [s])
    sizes

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (Stdlib.min n 4001) in
    s :: random_sizes rng (n - s)

let chunkings rng c n =
  let hop = Cqt.Config.hop c and latency = Cqt.Config.latency c in
  [ ("ones", List.init n (fun _ -> 1))
  ; ("prime-7", uniform 7 n)
  ; ("prime-1009", uniform 1009 n)
  ; ("hop-1", uniform (hop - 1) n)
  ; ("hop+1", uniform (hop + 1) n)
  ; ("cadence-straddle", straddle (latency + hop) n)
  ; ("latency-straddle", straddle latency n)
  ; ( "giant-then-ones"
    , if n > 1000 then (n - 1000) :: List.init 1000 (fun _ -> 1) else [n] )
  ; ("whole", [n])
  ; ("random-with-empty-chunks", sprinkle_empty rng (random_sizes rng n)) ]

(* {2 The configuration grid}

   One case per plan shape the octave recursion can take. The early-decimation
   count and the hop's parity are what the plan branches on, so the grid pins
   both: [`early = 0`] with six halvings (the default ladder), [`early = 3`] and
   [`early = 5`], and an odd hop, which halves nowhere and therefore runs the
   whole transform without a resampler. *)

type case = Case : string * Cqt.Config.t * (float, 'a) Nx.dtype -> case

let default_config = Cqt.Config.create ~n_bins:84 ~sample_rate:22050 ()

let grid =
  [ Case ("default-84-12-f64", default_config, Nx.float64)
  ; Case ("default-84-12-f32", default_config, Nx.float32)
  ; Case
      ( "bpo24-96-f64"
      , Cqt.Config.create ~n_bins:96 ~bins_per_octave:24 ~sample_rate:22050 ()
      , Nx.float64 )
  ; Case
      ( "vqt-erb-84-12-f32"
      , Cqt.Config.create ~n_bins:84 ~gamma:`Erb ~sample_rate:22050 ()
      , Nx.float32 )
  ; Case
      ( "odd-hop-513-f64"
      , Cqt.Config.create ~n_bins:84 ~hop:513 ~sample_rate:22050 ()
      , Nx.float64 )
  ; Case
      ( "early5-36-12-sr44100-f32"
      , Cqt.Config.create ~n_bins:36 ~hop:2048 ~sample_rate:44100 ()
      , Nx.float32 )
  ; Case
      ( "chroma-252-36-f64"
      , Cqt.Config.create ~n_bins:252 ~bins_per_octave:36 ~sample_rate:22050 ()
      , Nx.float64 )
  ; Case
      ( "single-octave-12-12-f32"
      , Cqt.Config.create ~n_bins:12 ~sample_rate:22050 ()
      , Nx.float32 ) ]

(* {2 The law} *)

let law_tests =
  List.map
    (fun (Case (name, c, dtype)) ->
      test (Printf.sprintf "%s: every chunking emits the same bits" name)
        (fun () ->
          let rng = Random.State.make [|0x5c47|] in
          let n = 2 * Cqt.Config.sample_rate c in
          let x = signal dtype 0x1f35 n in
          let expected = whole c dtype x in
          equal ~msg:(name ^ "/columns") int (Cqt.frames c ~n)
            (last_dim expected) ;
          List.iter
            (fun (partition, sizes) ->
              check_bits
                ~msg:(Printf.sprintf "%s/%s" name partition)
                expected
                (stream c dtype ~channels:1 x sizes) )
            (chunkings rng c n) ) )
    grid

let multichannel_tests =
  [ test "two channels stream as one state" (fun () ->
        let c = default_config in
        let n = 2 * Cqt.Config.sample_rate c in
        let left = noise 0x2a1 n and right = noise 0x9e3 n in
        let x = Nx.create Nx.float64 [|2; n|] (Array.append left right) in
        let expected = stream c Nx.float64 ~channels:2 x [n] in
        equal ~msg:"shape" (array int)
          [|2; Cqt.Config.n_bins c; Cqt.frames c ~n|]
          (Nx.shape expected) ;
        let rng = Random.State.make [|0x5c47|] in
        List.iter
          (fun (partition, sizes) ->
            check_bits ~msg:("2ch/" ^ partition) expected
              (stream c Nx.float64 ~channels:2 x sizes) )
          (List.filter
             (fun (partition, _) -> partition <> "ones")
             (chunkings rng c n) ) ;
        (* each channel is what that channel alone would have produced *)
        List.iteri
          (fun row samples ->
            let alone =
              whole c Nx.float64 (Nx.create Nx.float64 [|n|] samples)
            in
            let taken =
              Nx.reshape (Nx.shape alone)
                (Nx.shrink
                   [| (row, row + 1)
                    ; (0, Nx.dim 1 expected)
                    ; (0, Nx.dim 2 expected) |]
                   expected )
            in
            check_bits ~msg:(Printf.sprintf "2ch/row %d" row) alone taken )
          [left; right] ) ]

(* {2 Latency}

   The seven constants are the formula's own values, brute-forced against the
   offline transform: the smallest [m] such that column [p] of a signal that is
   zero before [m] vanishes identically is [p * hop + latency]. *)

let latency_tests =
  [ test "latency is the plan's largest per-octave lookahead" (fun () ->
        List.iter
          (fun (name, c, expected) ->
            equal ~msg:name int expected (Cqt.Config.latency c) )
          [ ("default 84/12", default_config, 20099)
          ; ( "bpo24 96"
            , Cqt.Config.create ~n_bins:96 ~bins_per_octave:24
                ~sample_rate:22050 ()
            , 28291 )
          ; ( "vqt erb 84/12"
            , Cqt.Config.create ~n_bins:84 ~gamma:`Erb ~sample_rate:22050 ()
            , 12931 )
          ; ( "odd hop 513"
            , Cqt.Config.create ~n_bins:84 ~hop:513 ~sample_rate:22050 ()
            , 8192 )
          ; ( "36/12 sr44100 hop2048"
            , Cqt.Config.create ~n_bins:36 ~hop:2048 ~sample_rate:44100 ()
            , 40387 )
          ; ( "chroma 252/36"
            , Cqt.Config.create ~n_bins:252 ~bins_per_octave:36
                ~sample_rate:22050 ()
            , 44675 )
          ; ( "single octave 12/12"
            , Cqt.Config.create ~n_bins:12 ~sample_rate:22050 ()
            , 20099 ) ] )
  ; test "an odd hop pays the analysis half-window alone" (fun () ->
        (* no octave decimates, so the chain term vanishes and the latency is
           the bottom octave's fft_size / 2 *)
        let c = Cqt.Config.create ~n_bins:84 ~hop:513 ~sample_rate:22050 () in
        equal ~msg:"latency" int 8192 (Cqt.Config.latency c) )
  ; test "column p leaves exactly at p * hop + latency" (fun () ->
        List.iter
          (fun (Case (name, c, dtype)) ->
            let hop = Cqt.Config.hop c and latency = Cqt.Config.latency c in
            List.iter
              (fun p ->
                let closing = (p * hop) + latency in
                let x = signal dtype 0x77c1 closing in
                let k =
                  Cqt.Kernel.prepare Nx.complex128 c dtype ~channels:1
                    ~max_block:closing
                in
                let before =
                  match Cqt.Kernel.step k (slice x 0 (closing - 1)) with
                  | None ->
                      0
                  | Some out ->
                      last_dim out
                in
                equal
                  ~msg:(Printf.sprintf "%s/before column %d" name p)
                  int p before ;
                let at =
                  match Cqt.Kernel.step k (slice x (closing - 1) closing) with
                  | None ->
                      0
                  | Some out ->
                      last_dim out
                in
                equal
                  ~msg:(Printf.sprintf "%s/column %d closes" name p)
                  int 1 at )
              [0; 1] )
          [ Case ("default", default_config, Nx.float64)
          ; Case
              ( "odd hop"
              , Cqt.Config.create ~n_bins:84 ~hop:513 ~sample_rate:22050 ()
              , Nx.float64 )
          ; Case
              ( "early5"
              , Cqt.Config.create ~n_bins:36 ~hop:2048 ~sample_rate:44100 ()
              , Nx.float32 ) ] ) ]

(* {2 The frame grid} *)

let frame_tests =
  [ test "a drained stream emits exactly frames c ~n columns" (fun () ->
        List.iter
          (fun (Case (name, c, dtype)) ->
            let hop = Cqt.Config.hop c and latency = Cqt.Config.latency c in
            List.iter
              (fun n ->
                let x =
                  if n = 0 then Nx.zeros dtype [|0|] else signal dtype 0x3b7 n
                in
                equal
                  ~msg:(Printf.sprintf "%s/n=%d" name n)
                  int (Cqt.frames c ~n)
                  (last_dim (whole c dtype x)) )
              [ 0
              ; 1
              ; hop - 1
              ; hop
              ; latency - 1
              ; latency
              ; latency + 1
              ; latency + (3 * hop)
              ; latency + (3 * hop) + 7 ] )
          grid ) ]

(* {2 Drain edges} *)

let drain_tests =
  [ test "an empty stream drains to nothing" (fun () ->
        let k =
          Cqt.Kernel.prepare Nx.complex128 default_config Nx.float64 ~channels:1
            ~max_block:64
        in
        is_true ~msg:"flush" (Cqt.Kernel.flush k = None) ;
        is_true ~msg:"second flush" (Cqt.Kernel.flush k = None) )
  ; test "a one-sample stream drains to its one column" (fun () ->
        let c = default_config in
        let x = signal Nx.float64 0x11 1 in
        equal ~msg:"columns" int (Cqt.frames c ~n:1)
          (last_dim (whole c Nx.float64 x)) )
  ; test "a drained kernel refuses to be fed" (fun () ->
        let k =
          Cqt.Kernel.prepare Nx.complex128 default_config Nx.float64 ~channels:1
            ~max_block:64
        in
        ignore (Cqt.Kernel.flush k) ;
        raises_invalid_arg ~msg:"step after flush"
          "step: cannot feed a drained kernel (flush consumed the tail; reset \
           before reusing)" (fun () ->
            ignore (Cqt.Kernel.step k (signal Nx.float64 0x11 8)) ) )
  ; test "reset restores the freshly prepared state" (fun () ->
        let c = default_config in
        let n = 25_000 in
        let x = signal Nx.float64 0x4d2 n in
        let k =
          Cqt.Kernel.prepare Nx.complex128 c Nx.float64 ~channels:1 ~max_block:n
        in
        let run () =
          let pushed = Option.to_list (Cqt.Kernel.step k x) in
          let drained = Option.to_list (Cqt.Kernel.flush k) in
          concat_columns c (pushed @ drained)
        in
        let first = run () in
        Cqt.Kernel.reset k ;
        check_bits ~msg:"after reset" first (run ()) ;
        check_bits ~msg:"against a fresh kernel" first (whole c Nx.float64 x) )
  ; test "preparation and feeding validate their arguments" (fun () ->
        let c = default_config in
        raises_invalid_arg ~msg:"channels"
          "prepare: cannot analyse 0 channels (channels must be at least 1)"
          (fun () ->
            ignore
              (Cqt.Kernel.prepare Nx.complex128 c Nx.float64 ~channels:0
                 ~max_block:64 ) ) ;
        raises_invalid_arg ~msg:"max_block"
          "prepare: cannot accept blocks of 0 samples (max_block must be at \
           least 1)" (fun () ->
            ignore
              (Cqt.Kernel.prepare Nx.complex128 c Nx.float64 ~channels:1
                 ~max_block:0 ) ) ;
        raises_invalid_arg ~msg:"dtype"
          "prepare: cannot decimate a float16 signal (the octave recursion \
           resamples, which needs float32 or float64 audio)" (fun () ->
            ignore
              (Cqt.Kernel.prepare Nx.complex128 c Nx.float16 ~channels:1
                 ~max_block:64 ) ) ;
        let k =
          Cqt.Kernel.prepare Nx.complex128 c Nx.float64 ~channels:1
            ~max_block:16
        in
        raises_invalid_arg ~msg:"oversize chunk"
          "step: cannot feed a 17-sample chunk (max_block is 16)" (fun () ->
            ignore (Cqt.Kernel.step k (signal Nx.float64 0x11 17)) ) ;
        raises_invalid_arg ~msg:"rank zero"
          "step: cannot analyse a rank-zero tensor (the time axis must exist)"
          (fun () -> ignore (Cqt.Kernel.step k (Nx.scalar Nx.float64 1.)) ) )
  ; test "a resampler-free plan streams any float dtype" (fun () ->
        (* an odd hop halves nowhere, so no stage constrains the sample type *)
        let c = Cqt.Config.create ~n_bins:12 ~hop:513 ~sample_rate:22050 () in
        let k =
          Cqt.Kernel.prepare Nx.complex64 c Nx.float16 ~channels:1
            ~max_block:1024
        in
        Cqt.Kernel.reset k ) ]

(* {2 Agreement with the offline transform}

   Not a law: {!Cqt.transform} decimates through the resampler's dense offline
   form and projects a whole octave in one matrix product, while the kernel
   decimates through the streaming executor and projects frame by frame. Same
   filter bank, different summation orders. These are the measured bounds,
   relative to the frame peak. *)

let divergence_tests =
  let case (Case (name, c, dtype)) =
    test (Printf.sprintf "%s: the kernel tracks the offline transform" name)
      (fun () ->
        let n = 2 * Cqt.Config.sample_rate c in
        let x = signal dtype 0x1f35 n in
        let expected = Cqt.transform Nx.complex128 c x in
        let got = whole c dtype x in
        equal ~msg:(name ^ "/shape") (array int) (Nx.shape expected)
          (Nx.shape got) ;
        let e = Nx.to_array expected and a = Nx.to_array got in
        let peak =
          Array.fold_left
            (fun acc (z : Complex.t) -> Float.max acc (Complex.norm z))
            0. e
        in
        let worst = ref 0. and where = ref 0 in
        Array.iteri
          (fun i (z : Complex.t) ->
            let d = Complex.norm (Complex.sub z a.(i)) in
            if d > !worst then (
              worst := d ;
              where := i ) )
          e ;
        let bound = match Nx.dtype x with Nx.Float32 -> 2e-6 | _ -> 1e-13 in
        if !worst > bound *. peak then
          failf "%s: %.3e of peak at flat index %d (bound %.1e)" name
            (!worst /. peak) !where bound )
  in
  List.map case grid

let suite =
  [ group "law" law_tests
  ; group "multichannel" multichannel_tests
  ; group "latency" latency_tests
  ; group "frames" frame_tests
  ; group "drain" drain_tests
  ; group "offline agreement" divergence_tests ]
