(* Kernel and offline semantics that the law alone cannot pin:

   - the output-length rule, exact for every face and ratio; - the latency
   probes: the group delay is compensated, so an impulse at input position j
   lands at output position j * L / M, and Config.latency /
   Config.output_latency are exact in both domains; - an independent OCaml
   reference evaluation of the polyphase geometry (the law is self-consistent;
   this pins the indexing against a second implementation); - multi-channel
   planar correctness: channels never mix, leading axes broadcast; - dtype
   behavior: f32 and f64 end-to-end, output dtype = input dtype, the f32 bank is
   the f64 design cast once; - the error matrix of prepare/step/flush and reset
   semantics. *)

open Windtrap
open Soundml

let signal dtype n =
  Nx.create dtype [|n|]
    (Array.init n (fun i -> Float.sin (0.61 *. Float.of_int i)))

let bits t = Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 t))

(* {2 Output length} *)

let ceil_div a b = if a <= 0 then 0 else ((a - 1) / b) + 1

let length_tests =
  [ test "output_frames and apply agree on ceil (n * L / M)" (fun () ->
        List.iter
          (fun (sample_rate, target, l, m) ->
            let cfg = Resample.Config.create ~sample_rate ~target () in
            List.iter
              (fun n ->
                let expected = ceil_div (n * l) m in
                equal
                  ~msg:
                    (Printf.sprintf "output_frames %d->%d n=%d" sample_rate
                       target n )
                  int expected
                  (Resample.Config.output_frames cfg ~n) ;
                let y = Resample.apply cfg (signal Nx.float64 n) in
                equal
                  ~msg:(Printf.sprintf "apply %d->%d n=%d" sample_rate target n)
                  int expected
                  (Nx.dim (Nx.ndim y - 1) y) )
              [0; 1; 2; 3; 17; 100; 147; 160; 1000] )
          [ (44100, 48000, 160, 147)
          ; (48000, 44100, 147, 160)
          ; (44100, 16000, 160, 441)
          ; (44100, 22050, 1, 2)
          ; (22050, 44100, 2, 1)
          ; (3, 2, 2, 3) ] )
  ; test "kernel fold reaches the exact total for short inputs" (fun () ->
        (* inputs far below the group delay still flush to ceil (n * L / M) *)
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        List.iter
          (fun n ->
            let k =
              Resample.Kernel.prepare cfg Nx.float64 ~channels:1 ~max_block:8
            in
            let x = signal Nx.float64 n in
            let stepped = Resample.Kernel.step k x in
            let drained = Resample.Kernel.flush k in
            let total =
              List.fold_left
                (fun acc t -> acc + Nx.dim (Nx.ndim t - 1) t)
                0
                (Option.to_list stepped @ Option.to_list drained)
            in
            equal
              ~msg:(Printf.sprintf "short n=%d" n)
              int
              (Resample.Config.output_frames cfg ~n)
              total )
          [1; 2; 5; 8] ) ]

(* {2 Latency probes} *)

let argmax t =
  let a = Nx.to_array (Nx.cast Nx.float64 t) in
  let best = ref 0 in
  Array.iteri (fun i v -> if Float.abs v > Float.abs a.(!best) then best := i) a ;
  !best

let impulse dtype n j =
  let a = Array.make n 0. in
  a.(j) <- 1. ;
  Nx.create dtype [|n|] a

let latency_tests =
  [ test "an impulse lands at j * L / M — the delay is compensated" (fun () ->
        List.iter
          (fun (sample_rate, target, quality, j, expected) ->
            let cfg = Resample.Config.create ~quality ~sample_rate ~target () in
            let y = Resample.apply cfg (impulse Nx.float64 1000 j) in
            equal
              ~msg:(Printf.sprintf "impulse %d->%d at %d" sample_rate target j)
              int expected (argmax y) )
          [ (44100, 48000, `High, 294, 320) (* 294 * 160 / 147 *)
          ; (44100, 48000, `Fast, 588, 640)
          ; (48000, 44100, `High, 320, 294)
          ; (44100, 22050, `High, 500, 250)
          ; (22050, 44100, `Best, 250, 500) ] )
  ; test "no leading silence: a DC signal is alive at sample zero" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let x = Nx.create Nx.float64 [|500|] (Array.make 500 1.) in
        let y = Nx.to_array (Resample.apply cfg x) in
        (* output 0 reads the window centered on input time 0: alive from the
           first sample (for this upsampling ratio the phase-0 row is
           delta-like, so the edge value sits near unit gain, not at the ideal
           step's 0.5) — never silence, never an over-unity shifted copy *)
        is_true ~msg:"y.(0) alive" (y.(0) > 0.4 && y.(0) < 1.05) ;
        is_true ~msg:"interior at unit gain" (Float.abs (y.(200) -. 1.) < 1e-6) )
  ; test "latency is exact in both domains" (fun () ->
        let rate_t =
          testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()
        in
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        equal ~msg:"input domain" int 95 (Resample.Config.latency cfg) ;
        equal ~msg:"output domain, exact rational" rate_t
          {Pipeline.Rate.num= 15200; den= 147}
          (Resample.Config.output_latency cfg) ;
        equal ~msg:"stage latency in input items" rate_t
          {Pipeline.Rate.num= 95; den= 1}
          (Pipeline.latency (Resample.stage cfg)) ) ]

(* {2 An independent reference evaluation of the geometry} *)

(* [reference cfg x] evaluates the polyphase form directly from the designed
   prototype, in OCaml: output [i] has phase [(i*M) mod L], base [(i*M)/L + K],
   and sums [proto.(p + t*L) * x.(base - t)] over t — the signal is zero outside
   its extent. Summation order differs from the executor's, so agreement is to
   tolerance; what this pins is the indexing. *)
let reference cfg x =
  let l, m =
    let r = Resample.Config.rate cfg in
    (r.Pipeline.Rate.num, r.Pipeline.Rate.den)
  in
  let k = Resample.Config.latency cfg in
  let h = Nx.to_array (Resample.Config.prototype Nx.float64 cfg) in
  let n = Array.length x in
  let total = ceil_div (n * l) m in
  Array.init total (fun i ->
      let t = i * m in
      let p = t mod l and base = (t / l) + k in
      let acc = ref 0. in
      for tt = 0 to 2 * k do
        let hi = p + (tt * l) in
        let xi = base - tt in
        if hi < Array.length h && xi >= 0 && xi < n then
          acc := !acc +. (h.(hi) *. x.(xi))
      done ;
      !acc )

let reference_tests =
  [ test "the executor matches a second implementation of the geometry"
      (fun () ->
        List.iter
          (fun (sample_rate, target) ->
            let cfg =
              Resample.Config.create
                ~quality:(`Custom {Resample.attenuation= 80.; passband= 0.85})
                ~sample_rate ~target ()
            in
            let n = 200 in
            let xa =
              Array.init n (fun i -> Float.cos (0.41 *. Float.of_int i))
            in
            let expected = reference cfg xa in
            let got =
              Nx.to_array (Resample.apply cfg (Nx.create Nx.float64 [|n|] xa))
            in
            equal
              ~msg:(Printf.sprintf "%d->%d length" sample_rate target)
              int (Array.length expected) (Array.length got) ;
            Array.iteri
              (fun i e ->
                if
                  Float.abs (got.(i) -. e) > 1e-10 *. Float.max 1. (Float.abs e)
                then
                  failf "%d->%d: index %d: got %.17g, expected %.17g"
                    sample_rate target i got.(i) e )
              expected )
          [(3, 2); (2, 3); (44100, 48000); (44100, 16000); (5, 5)] ) ]

(* {2 Multi-channel planar correctness} *)

let channel_tests =
  [ test "channels resample independently and never mix" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let n = 300 in
        let a = Array.init n (fun i -> Float.sin (0.3 *. Float.of_int i)) in
        let b = Array.init n (fun i -> Float.cos (1.7 *. Float.of_int i)) in
        let stereo = Nx.create Nx.float64 [|2; n|] (Array.append a b) in
        let y = Resample.apply cfg stereo in
        let ya = Resample.apply cfg (Nx.create Nx.float64 [|n|] a) in
        let yb = Resample.apply cfg (Nx.create Nx.float64 [|n|] b) in
        let out = Nx.dim 1 y in
        equal ~msg:"stereo shape" (array int) [|2; out|] (Nx.shape y) ;
        let row i = Nx.shrink [|(i, i + 1); (0, out)|] y in
        if bits ya <> bits (row 0) then fail "channel 0 diverges" ;
        if bits yb <> bits (row 1) then fail "channel 1 diverges" )
  ; test "leading axes broadcast: a batch is one call" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:22050 ~target:44100 () in
        let n = 120 in
        let clips =
          Array.init (4 * n) (fun i ->
              Float.sin
                ( (0.2 +. (0.1 *. Float.of_int (i / n)))
                *. Float.of_int (i mod n) ) )
        in
        let batch = Nx.create Nx.float64 [|2; 2; n|] clips in
        let y = Resample.apply cfg batch in
        let out = Nx.dim 2 y in
        equal ~msg:"batch shape" (array int) [|2; 2; out|] (Nx.shape y) ;
        (* every clip equals its standalone conversion, bit for bit *)
        for c = 0 to 3 do
          let clip = Nx.create Nx.float64 [|n|] (Array.sub clips (c * n) n) in
          let expected = bits (Resample.apply cfg clip) in
          let got =
            bits
              (Nx.shrink
                 [|(c / 2, (c / 2) + 1); (c mod 2, (c mod 2) + 1); (0, out)|]
                 y )
          in
          if expected <> got then failf "clip %d diverges" c
        done )
  ; test "a mid-stream chunk arriving as a strided view is read correctly"
      (fun () ->
        (* stereo chunks sliced from a larger tensor are not contiguous; the
           kernel must read the logical values, not the raw buffer *)
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let n = 256 in
        let stereo =
          Nx.create Nx.float64 [|2; n|]
            (Array.init (2 * n) (fun i -> Float.sin (0.11 *. Float.of_int i)))
        in
        let expected = bits (Resample.apply cfg stereo) in
        let k =
          Resample.Kernel.prepare cfg Nx.float64 ~channels:2 ~max_block:100
        in
        let chunks =
          [ Nx.shrink [|(0, 2); (0, 100)|] stereo
          ; Nx.shrink [|(0, 2); (100, 177)|] stereo
          ; Nx.shrink [|(0, 2); (177, 256)|] stereo ]
        in
        let stepped = List.filter_map (Resample.Kernel.step k) chunks in
        let outs = stepped @ Option.to_list (Resample.Kernel.flush k) in
        let got = bits (Nx.concatenate ~axis:(-1) outs) in
        if expected <> got then fail "strided chunks diverge from apply" ) ]

(* {2 Dtype behavior} *)

let dtype_tests =
  [ test "output dtype follows the input" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let y32 = Resample.apply cfg (signal Nx.float32 200) in
        let y64 = Resample.apply cfg (signal Nx.float64 200) in
        is_true ~msg:"f32 in, f32 out"
          (match Nx.dtype y32 with Nx.Float32 -> true | _ -> false) ;
        is_true ~msg:"f64 in, f64 out"
          (match Nx.dtype y64 with Nx.Float64 -> true | _ -> false) )
  ; test "the f32 path tracks the f64 design" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let n = 400 in
        let a = Array.init n (fun i -> Float.sin (0.61 *. Float.of_int i)) in
        let y64 =
          Nx.to_array (Resample.apply cfg (Nx.create Nx.float64 [|n|] a))
        in
        let y32 =
          Nx.to_array
            (Nx.cast Nx.float64
               (Resample.apply cfg (Nx.create Nx.float32 [|n|] a)) )
        in
        Array.iteri
          (fun i e ->
            if Float.abs (y32.(i) -. e) > 1e-4 then
              failf "f32 diverges from f64 at %d: %.9g vs %.9g" i y32.(i) e )
          y64 )
  ; test "half floats are refused with a clear message" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        raises_invalid_arg ~msg:"prepare f16"
          "prepare: cannot resample float16 audio (the executor carries \
           float32 and float64)" (fun () ->
            ignore
              (Resample.Kernel.prepare cfg Nx.float16 ~channels:1 ~max_block:8) ) )
  ]

(* {2 The error matrix and sequencing of the kernel} *)

let error_tests =
  [ test "prepare validates channels and max_block" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        raises_invalid_arg ~msg:"channels"
          "prepare: cannot resample 0 channels (channels must be at least 1)"
          (fun () ->
            ignore
              (Resample.Kernel.prepare cfg Nx.float64 ~channels:0 ~max_block:8) ) ;
        raises_invalid_arg ~msg:"max_block"
          "prepare: cannot accept blocks of 0 samples (max_block must be at \
           least 1)" (fun () ->
            ignore
              (Resample.Kernel.prepare cfg Nx.float64 ~channels:1 ~max_block:0) ) )
  ; test "step validates the chunk against the prepared state" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let k =
          Resample.Kernel.prepare cfg Nx.float64 ~channels:1 ~max_block:16
        in
        raises_invalid_arg ~msg:"over-long chunk"
          "step: cannot feed a 17-sample chunk (max_block is 16)" (fun () ->
            ignore (Resample.Kernel.step k (signal Nx.float64 17)) ) ;
        raises_invalid_arg ~msg:"channel mismatch"
          "step: cannot feed 2-channel chunks (the kernel was prepared for 1 \
           channels)" (fun () ->
            ignore (Resample.Kernel.step k (Nx.zeros Nx.float64 [|2; 8|])) ) ;
        raises_invalid_arg ~msg:"rank zero"
          "step: cannot resample a rank-zero tensor (the time axis must exist)"
          (fun () ->
            ignore (Resample.Kernel.step k (Nx.zeros Nx.float64 [||])) ) )
  ; test "draining consumes the tail; reset restores the initial state"
      (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let k =
          Resample.Kernel.prepare cfg Nx.float64 ~channels:1 ~max_block:256
        in
        let x = signal Nx.float64 256 in
        let stepped = Resample.Kernel.step k x in
        let drained = Resample.Kernel.flush k in
        is_true ~msg:"emitted" (Option.is_some stepped && Option.is_some drained) ;
        is_true ~msg:"second flush is None"
          (Option.is_none (Resample.Kernel.flush k)) ;
        raises_invalid_arg ~msg:"drained step refuses samples"
          "step: cannot feed a drained kernel (flush consumed the tail; reset \
           before reusing)" (fun () -> ignore (Resample.Kernel.step k x) ) ;
        Resample.Kernel.reset k ;
        (* the reset kernel replays the same signal to the same bits: the
           history really is zeroed, not merely rewound *)
        let stepped' = Resample.Kernel.step k x in
        let drained' = Resample.Kernel.flush k in
        match (stepped, drained, stepped', drained') with
        | Some a, Some b, Some a', Some b' ->
            if bits a <> bits a' || bits b <> bits b' then
              fail "reset kernel diverges from the fresh kernel"
        | _ ->
            fail "reset kernel emitted a different chunk pattern" )
  ; test "apply rejects rank zero" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        raises_invalid_arg ~msg:"rank zero"
          "apply: cannot resample a rank-zero tensor (the time axis must exist)"
          (fun () -> ignore (Resample.apply cfg (Nx.zeros Nx.float64 [||])) ) )
  ; test "the stage rejects a mismatched source rate at prepare" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let p = Resample.stage cfg in
        let src =
          Pipeline.Format.audio Nx.float64 ~sample_rate:22050 ~channels:1
        in
        raises_invalid_arg ~msg:"format mismatch"
          "prepare: cannot resample a stream at 22050 items/s (the \
           configuration converts from 44100 Hz)" (fun () ->
            ignore (Pipeline.Stream.prepare p ~source:src ~max_chunk:64) ) ) ]

let suite =
  [ group "length" length_tests
  ; group "latency" latency_tests
  ; group "reference" reference_tests
  ; group "channels" channel_tests
  ; group "dtype" dtype_tests
  ; group "errors" error_tests ]
