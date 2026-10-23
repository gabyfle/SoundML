(* The pipeline law for yin_stage:

   yin x = run ~source p x = concat (List.filter_map (Stream.push s) (chunks x)
   @ Stream.flush s)

   exactly, over seeded partitionings — one-sample chunks, the whole signal,
   random sizes, chunks longer than a frame and occasional empty chunks — for
   every padding, uncentered frames, an odd frame length, a hop longer than the
   frame, a batch of channels and both dtypes, including signals shorter than
   the latency. Also here: the latency and rate queries and the emission they
   promise, reset, the invariances of yin and pyin (batching, power-of-two
   scaling), silence, a pure tone, pyin's output invariants, and every
   precondition. *)

open Windtrap
open Soundml

let seed = 0xf0f0

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

(* The law geometry: 1 kHz, periods of 2 to 50 samples, 64-sample frames. *)
let sr = 1000

let fmin = 20.

let fmax = 400.

let source ?(sample_rate = sr) dtype =
  Pipeline.Format.audio dtype ~sample_rate ~channels:1

(* [signal dtype lead n] is a deterministic test signal per channel: two
   partials of a channel-dependent pitch under a slow vibrato, with a gap of
   silence every 97 samples. *)
let signal dtype lead n =
  let planes = Array.fold_left ( * ) 1 lead in
  Nx.create dtype (Array.append lead [|n|])
    (Array.init (planes * n) (fun k ->
         let c = k / n and i = k mod n in
         if i mod 97 < 20 then 0.
         else
           let t = Float.of_int i /. Float.of_int sr in
           let f =
             60. +. (25. *. Float.of_int c) +. (5. *. Float.sin (7. *. t))
           in
           let phase = 2. *. Float.pi *. f *. t in
           Float.sin phase +. (0.3 *. Float.sin ((2. *. phase) +. 0.5)) ) )

let rec random_sizes rng ~most n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (Int.min n most) in
    s :: random_sizes rng ~most (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

let partitions rng n =
  [ ("ones", List.init n (fun _ -> 1))
  ; ("whole", if n = 0 then [] else [n])
  ; ("random-small-0", random_sizes rng ~most:13 n)
  ; ("random-small-1", random_sizes rng ~most:13 n)
  ; ("random-large", random_sizes rng ~most:150 n)
  ; ( "random-with-empty-chunks"
    , sprinkle_empty rng (random_sizes rng ~most:40 n) ) ]

let chunks_of x sizes =
  let nd = Nx.ndim x in
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        Nx.slice
          (List.init nd (fun i ->
               if i = nd - 1 then Nx.R (off, off + s) else Nx.A ) )
          x
        :: go (off + s) rest
  in
  go 0 sizes

let stream_outputs p ~source ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

let flat outs =
  match outs with [] -> [||] | l -> Nx.to_array (Nx.concatenate ~axis:(-1) l)

type config =
  { label: string
  ; frame_length: int
  ; hop: int
  ; center: bool
  ; pad: [`Constant of float | `Edge | `Reflect] }

let base =
  { label= "constant 0"
  ; frame_length= 64
  ; hop= 16
  ; center= true
  ; pad= `Constant 0. }

let configs =
  [ base
  ; {base with label= "edge"; pad= `Edge}
  ; {base with label= "reflect"; pad= `Reflect}
  ; {base with label= "constant 0.5"; pad= `Constant 0.5}
  ; {base with label= "uncentered"; center= false}
  ; {base with label= "odd frame, reflect"; frame_length= 63; pad= `Reflect}
  ; {base with label= "hop longer than the frame"; hop= 80}
  ; { base with
      label= "hop longer than the frame, reflect"
    ; hop= 80
    ; pad= `Reflect } ]

let yin_of c x =
  yin ~frame_length:c.frame_length ~hop:c.hop
    ~alignment:(if c.center then `Centered else `Left)
    ~pad:c.pad ~fmin ~fmax ~sample_rate:sr x

let stage_of c =
  yin_stage ~frame_length:c.frame_length ~hop:c.hop
    ~alignment:(if c.center then `Centered else `Left)
    ~pad:c.pad ~fmin ~fmax ~sample_rate:sr ()

let law_case (type a) (dtype : (float, a) Nx.dtype) ~dname lead n c =
  test
    (Printf.sprintf "%s, %s, [%s] x %d" c.label dname
       (String.concat "; " (List.map string_of_int (Array.to_list lead)))
       n )
    (fun () ->
      let rng = Random.State.make [|seed; n; c.frame_length; c.hop|] in
      let x = signal dtype lead n in
      let expected = Nx.to_array (yin_of c x) in
      let source = source dtype in
      equal ~msg:"run" (array float_exact) expected
        (Nx.to_array (Pipeline.run ~source (stage_of c) x)) ;
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left Int.max 1 sizes in
          let outs =
            stream_outputs (stage_of c) ~source ~max_chunk (chunks_of x sizes)
          in
          equal ~msg:pname (array float_exact) expected (flat outs) )
        (partitions rng n) )

let law_tests =
  List.concat_map
    (fun c ->
      [ law_case Nx.float64 ~dname:"float64" [||] 333 c
      ; law_case Nx.float64 ~dname:"float64" [||] 20 c
      ; law_case Nx.float64 ~dname:"float64" [||] 1 c
      ; law_case Nx.float64 ~dname:"float64" [||] 0 c ] )
    configs
  @ [ law_case Nx.float32 ~dname:"float32" [|2|] 301 base
    ; law_case Nx.float32 ~dname:"float32" [|2|] 301
        {base with label= "reflect"; pad= `Reflect}
    ; law_case Nx.float64 ~dname:"float64" [|2; 2|] 257
        {base with label= "uncentered"; center= false}
    ; law_case Nx.float64 ~dname:"float64" [|3|] 40
        {base with label= "reflect"; pad= `Reflect} ]

(* {1 The stage's declarations} *)

let ceil_div a b = if a <= 0 then 0 else (a + b - 1) / b

(* [emission c n] pushes [n] samples one at a time and checks that after each
   push the stage has emitted at least the frames its latency promises, [ceil
   ((pushed - latency) / hop)], then that the flush brings the total to the
   offline count. *)
let emission c n =
  let p = stage_of c in
  let latency = (Pipeline.latency p).Pipeline.Rate.num in
  let x = signal Nx.float64 [||] n in
  let s = Pipeline.Stream.prepare p ~source:(source Nx.float64) ~max_chunk:1 in
  let emitted = ref 0 in
  for i = 0 to n - 1 do
    ( match Pipeline.Stream.push s (Nx.slice [Nx.R (i, i + 1)] x) with
    | None ->
        ()
    | Some out ->
        emitted := !emitted + Nx.dim 0 out ) ;
    let promised = ceil_div (i + 1 - latency) c.hop in
    is_true
      ~msg:
        (Printf.sprintf "%s: after %d samples, %d frames out of %d promised"
           c.label (i + 1) !emitted promised )
      (!emitted >= promised)
  done ;
  let tail =
    List.fold_left ( + ) 0 (List.map (Nx.dim 0) (Pipeline.Stream.flush s))
  in
  equal ~msg:(c.label ^ ": total") int (Nx.dim 0 (yin_of c x)) (!emitted + tail)

let stage_tests =
  [ test "latency is half a centered frame, a whole uncentered one" (fun () ->
        equal ~msg:"centered" rate_t (r 32 1) (Pipeline.latency (stage_of base)) ;
        equal ~msg:"reflect" rate_t (r 32 1)
          (Pipeline.latency (stage_of {base with pad= `Reflect})) ;
        equal ~msg:"odd" rate_t (r 31 1)
          (Pipeline.latency (stage_of {base with frame_length= 63})) ;
        equal ~msg:"uncentered" rate_t (r 63 1)
          (Pipeline.latency (stage_of {base with center= false})) ;
        equal ~msg:"default" rate_t (r 1024 1)
          (Pipeline.latency
             (yin_stage ~fmin:65. ~fmax:2093. ~sample_rate:22050 ()) ) )
  ; test "one estimate per hop" (fun () ->
        equal ~msg:"rate" rate_t (r 1 16) (Pipeline.rate (stage_of base)) ;
        equal ~msg:"default hop" rate_t (r 1 512)
          (Pipeline.rate
             (yin_stage ~fmin:65. ~fmax:2093. ~sample_rate:22050 ()) ) )
  ; test "the latency's promise holds sample by sample" (fun () ->
        List.iter (fun c -> emission c 300) configs )
  ; test "reset restarts the stream" (fun () ->
        let p = stage_of {base with pad= `Reflect} in
        let x = signal Nx.float64 [||] 200 in
        let s =
          Pipeline.Stream.prepare p ~source:(source Nx.float64) ~max_chunk:200
        in
        let first = Option.get (Pipeline.Stream.push s x) in
        ignore (Pipeline.Stream.flush s) ;
        Pipeline.Stream.reset s ;
        let again = Option.get (Pipeline.Stream.push s x) in
        equal ~msg:"after reset" (array float_exact) (Nx.to_array first)
          (Nx.to_array again) )
  ; test "a source at another rate is refused at prepare" (fun () ->
        raises ~msg:"rate"
          (Invalid_argument
             "yin_stage: cannot estimate the pitch of a stream at 1001 items/s \
              (sample_rate is 1000 Hz)" ) (fun () ->
            ignore
              (Pipeline.Stream.prepare (stage_of base)
                 ~source:(source ~sample_rate:1001 Nx.float64)
                 ~max_chunk:16 ) ) ) ]

(* {1 Invariances and special signals} *)

let same_pyin msg (a : 'a pyin) (b : 'a pyin) =
  equal ~msg:(msg ^ " f0") (array float_exact) (Nx.to_array a.f0)
    (Nx.to_array b.f0) ;
  equal ~msg:(msg ^ " voiced") (array bool) (Nx.to_array a.voiced)
    (Nx.to_array b.voiced) ;
  equal
    ~msg:(msg ^ " voiced_probability")
    (array float_exact)
    (Nx.to_array a.voiced_probability)
    (Nx.to_array b.voiced_probability)

let pyin_of ?fill ?transition_min_prob x =
  pyin ?fill ?transition_min_prob ~frame_length:64 ~hop:16 ~fmin:50. ~fmax:400.
    ~sample_rate:sr ~max_transition_rate:100. x

let row x i = Nx.slice [Nx.I i; Nx.A] x

let property_tests =
  [ test "a batch is its rows, bit for bit" (fun () ->
        let x = signal Nx.float64 [|3|] 400 in
        let y = yin_of base x and p = pyin_of x in
        for i = 0 to 2 do
          equal ~msg:"yin" (array float_exact)
            (Nx.to_array (yin_of base (row x i)))
            (Nx.to_array (row y i)) ;
          let alone = pyin_of (row x i) in
          same_pyin "pyin" alone
            { f0= row p.f0 i
            ; voiced= row p.voiced i
            ; voiced_probability= row p.voiced_probability i }
        done )
  ; test "a power-of-two gain changes no bit" (fun () ->
        let x = signal Nx.float64 [||] 400 in
        let louder = Nx.mul_s x 4. in
        equal ~msg:"yin" (array float_exact)
          (Nx.to_array (yin_of base x))
          (Nx.to_array (yin_of base louder)) ;
        same_pyin "pyin" (pyin_of x) (pyin_of louder) )
  ; test "silence" (fun () ->
        let x = Nx.zeros Nx.float64 [|300|] in
        Array.iter
          (fun v ->
            equal ~msg:"yin of zeros" float_exact
              (Float.of_int sr /. Float.floor (Float.of_int sr /. fmax))
              v )
          (Nx.to_array (yin_of base x)) ;
        let p = pyin_of x in
        Array.iter
          (fun v -> is_true ~msg:"unvoiced" (not v))
          (Nx.to_array p.voiced) ;
        Array.iter
          (fun v -> equal ~msg:"probability" float_exact 0. v)
          (Nx.to_array p.voiced_probability) ;
        Array.iter
          (fun v -> is_true ~msg:"nan" (Float.is_nan v))
          (Nx.to_array p.f0) ;
        Array.iter
          (fun v -> equal ~msg:"decoded" float_exact 50. v)
          (Nx.to_array (pyin_of ~fill:`Decoded x).f0) )
  ; test "a pure tone is found within five cents" (fun () ->
        (* the interpolation runs through the normalized difference, whose
           parabola is skewed: 3.4 cents at worst for this tone *)
        let n = 11025 in
        let x =
          Nx.create Nx.float64 [|n|]
            (Array.init n (fun i ->
                 Float.sin (2. *. Float.pi *. 220. *. Float.of_int i /. 22050.) )
            )
        in
        let f = Nx.to_array (yin ~fmin:65. ~fmax:2093. ~sample_rate:22050 x) in
        (* interior frames: their 2048 samples all lie inside the signal *)
        for p = 2 to Array.length f - 3 do
          let cents = 1200. *. Float.log2 (f.(p) /. 220.) in
          is_true
            ~msg:(Printf.sprintf "frame %d: %.4f cents" p cents)
            (Float.abs cents < 5.)
        done )
  ; test "pyin's outputs agree with each other" (fun () ->
        let x = signal Nx.float64 [|2|] 600 in
        let noise =
          Nx.create Nx.float64 [|600|]
            (Array.init 600 (fun i ->
                 Float.sin (Float.of_int (i * i mod 1009)) ) )
        in
        let x = Nx.concatenate ~axis:0 [x; Nx.reshape [|1; 600|] noise] in
        let p = pyin_of x in
        let f0 = Nx.to_array p.f0
        and voiced = Nx.to_array p.voiced
        and prob = Nx.to_array p.voiced_probability in
        Array.iteri
          (fun i v ->
            equal ~msg:"voiced iff a frequency" bool v
              (not (Float.is_nan f0.(i))) ;
            is_true ~msg:"probability in [0, 1]"
              (prob.(i) >= 0. && prob.(i) <= 1.) )
          voiced ;
        is_true ~msg:"some voiced" (Array.exists Fun.id voiced) ;
        is_true ~msg:"some unvoiced" (Array.exists not voiced) ;
        let zero = Nx.to_array (pyin_of ~fill:(`Value 0.) x).f0 in
        Array.iteri
          (fun i v -> if not v then equal ~msg:"fill" float_exact 0. zero.(i))
          voiced ;
        (* decoded frequencies lie on the grid fmin * 2 ** (k / 120) *)
        Array.iter
          (fun f ->
            let k = Float.round (120. *. Float.log2 (f /. 50.)) in
            equal ~msg:"on the grid" float_exact
              (50. *. Float.pow 2. (k /. 120.))
              f )
          (Nx.to_array (pyin_of ~fill:`Decoded x).f0) ;
        (* the exact search finds the pruned search's path on this signal *)
        same_pyin "exact search" p (pyin_of ~transition_min_prob:0. x) )
  ; test "no frames" (fun () ->
        equal ~msg:"empty" (array int) [|0|]
          (Nx.shape (yin_of base (Nx.zeros Nx.float64 [|0|]))) ;
        equal ~msg:"uncentered, shorter than a frame" (array int) [|2; 0|]
          (Nx.shape
             (yin_of {base with center= false} (Nx.zeros Nx.float64 [|2; 63|])) ) ;
        equal ~msg:"zero channels" (array int) [|0; 7|]
          (Nx.shape (yin_of base (Nx.zeros Nx.float64 [|0; 100|]))) ;
        let p = pyin_of (Nx.zeros Nx.float32 [|3; 0|]) in
        equal ~msg:"pyin empty" (array int) [|3; 0|] (Nx.shape p.f0) ;
        equal ~msg:"pyin empty voiced" (array int) [|3; 0|] (Nx.shape p.voiced) )
  ; test "float32 rounds the float64 result once" (fun () ->
        let x = signal Nx.float32 [||] 400 in
        let wide = Nx.cast Nx.float64 x in
        equal ~msg:"yin" (array float_exact)
          (Nx.to_array (Nx.cast Nx.float32 (yin_of base wide)))
          (Nx.to_array (yin_of base x)) ;
        let p32 = pyin_of x and p64 = pyin_of wide in
        same_pyin "pyin" p32
          { f0= Nx.cast Nx.float32 p64.f0
          ; voiced= p64.voiced
          ; voiced_probability= Nx.cast Nx.float32 p64.voiced_probability } ) ]

(* {1 Preconditions} *)

let x100 = signal Nx.float64 [||] 100

let yin_raises msg message f =
  raises ~msg (Invalid_argument message) (fun () -> ignore (f ()))

let error_tests =
  [ test "yin's parameters are validated" (fun () ->
        yin_raises "sample_rate"
          "yin: cannot estimate pitch at 0 Hz (sample_rate must be at least 1)"
          (fun () -> yin ~fmin ~fmax ~sample_rate:0 x100 ) ;
        yin_raises "frame_length"
          "yin: cannot analyse frames of 0 samples (frame_length must be at \
           least 1)" (fun () ->
            yin ~frame_length:0 ~fmin ~fmax ~sample_rate:sr x100 ) ;
        yin_raises "fmin finite"
          "yin: cannot search between fmin = nan Hz and fmax = 400 Hz (both \
           must be finite)" (fun () ->
            yin ~frame_length:64 ~fmin:Float.nan ~fmax ~sample_rate:sr x100 ) ;
        yin_raises "Nyquist"
          "yin: cannot search up to fmax = 600 Hz at 1000 Hz (fmax must not \
           exceed the Nyquist frequency, 500 Hz)" (fun () ->
            yin ~frame_length:64 ~fmin ~fmax:600. ~sample_rate:sr x100 ) ;
        yin_raises "order"
          "yin: cannot search between fmin = 400 Hz and fmax = 400 Hz (fmin \
           must be below fmax)" (fun () ->
            yin ~frame_length:64 ~fmin:400. ~fmax ~sample_rate:sr x100 ) ;
        yin_raises "positive"
          "yin: cannot search from fmin = -20 Hz (fmin must be positive)"
          (fun () ->
            yin ~frame_length:64 ~fmin:(-20.) ~fmax ~sample_rate:sr x100 ) ;
        yin_raises "a period must fit"
          "yin: cannot fit a period of fmin = 20 Hz in frames of 50 samples at \
           1000 Hz (sample_rate / fmin must be below frame_length - 1: raise \
           fmin above 20.4082 Hz or frame_length to 52)" (fun () ->
            yin ~frame_length:50 ~fmin ~fmax ~sample_rate:sr x100 ) ;
        yin_raises "hop"
          "yin: cannot advance frames by 0 samples (hop must be at least 1)"
          (fun () ->
            yin ~frame_length:64 ~hop:0 ~fmin ~fmax ~sample_rate:sr x100 ) ;
        yin_raises "threshold"
          "yin: cannot select troughs below inf (trough_threshold must be \
           finite)" (fun () ->
            yin ~frame_length:64 ~trough_threshold:Float.infinity ~fmin ~fmax
              ~sample_rate:sr x100 ) ;
        yin_raises "constant"
          "yin: cannot pad with the constant nan (the padding value must be \
           finite)" (fun () ->
            yin ~frame_length:64 ~pad:(`Constant Float.nan) ~fmin ~fmax
              ~sample_rate:sr x100 ) ;
        yin_raises "rank"
          "yin: cannot analyse a rank-zero tensor (the time axis must exist)"
          (fun () ->
            yin ~frame_length:64 ~fmin ~fmax ~sample_rate:sr
              (Nx.scalar Nx.float64 1.) ) ;
        yin_raises "finite"
          "yin: cannot estimate the pitch of a signal holding a non-finite \
           sample" (fun () ->
            yin ~frame_length:64 ~fmin ~fmax ~sample_rate:sr
              (Nx.create Nx.float32 [|3|] [|0.; Float.infinity; 0.|]) ) )
  ; test "the stage validates its parameters and chunks" (fun () ->
        yin_raises "built"
          "yin_stage: cannot advance frames by -1 samples (hop must be at \
           least 1)" (fun () ->
            yin_stage ~frame_length:64 ~hop:(-1) ~fmin ~fmax ~sample_rate:sr () ) ;
        let push chunks =
          stream_outputs (stage_of base) ~source:(source Nx.float64)
            ~max_chunk:100 chunks
        in
        yin_raises "leading shape"
          "yin_stage: cannot continue a stream of leading shape [2] with a \
           chunk of leading shape [3]" (fun () ->
            push [signal Nx.float64 [|2|] 10; signal Nx.float64 [|3|] 10] ) ;
        yin_raises "zero channels"
          "yin_stage: cannot analyse a chunk with a zero-size leading axis \
           (channels must be at least 1)" (fun () ->
            push [Nx.zeros Nx.float64 [|0; 10|]] ) ;
        yin_raises "rank"
          "yin_stage: cannot analyse a rank-zero tensor (the time axis must \
           exist)" (fun () -> push [Nx.scalar Nx.float64 0.] ) ;
        yin_raises "finite"
          "yin_stage: cannot estimate the pitch of a signal holding a \
           non-finite sample" (fun () ->
            push [x100; Nx.create Nx.float64 [|2|] [|Float.nan; 0.|]] ) ;
        yin_raises "drained"
          "yin_stage: cannot feed a drained stream (flush consumed the tail; \
           reset before reusing)" (fun () ->
            let s =
              Pipeline.Stream.prepare (stage_of base)
                ~source:(source Nx.float64) ~max_chunk:100
            in
            ignore (Pipeline.Stream.push s x100) ;
            ignore (Pipeline.Stream.flush s) ;
            Pipeline.Stream.push s x100 ) )
  ; test "pyin's parameters are validated" (fun () ->
        let p = pyin ~frame_length:64 ~fmin:50. ~fmax:400. ~sample_rate:sr in
        yin_raises "thresholds"
          "pyin: cannot use 0 thresholds (n_thresholds must be at least 1)"
          (fun () -> p ~n_thresholds:0 x100 ) ;
        yin_raises "beta"
          "pyin: cannot use the first beta parameter = 0 (the first beta \
           parameter must be finite and positive)" (fun () ->
            p ~beta_parameters:(0., 18.) x100 ) ;
        yin_raises "beta, second"
          "pyin: cannot use the second beta parameter = inf (the second beta \
           parameter must be finite and positive)" (fun () ->
            p ~beta_parameters:(2., Float.infinity) x100 ) ;
        yin_raises "boltzmann"
          "pyin: cannot use boltzmann_parameter = 0 (boltzmann_parameter must \
           be finite and positive)" (fun () -> p ~boltzmann_parameter:0. x100 ) ;
        yin_raises "resolution"
          "pyin: cannot use resolution = -0.1 (resolution must be finite and \
           positive)" (fun () -> p ~resolution:(-0.1) x100 ) ;
        yin_raises "transition rate"
          "pyin: cannot use max_transition_rate = -1 (max_transition_rate must \
           be finite and non-negative)" (fun () ->
            p ~max_transition_rate:(-1.) x100 ) ;
        yin_raises "switch"
          "pyin: cannot use switch_prob = 1.5 (switch_prob must lie in [0, 1])"
          (fun () -> p ~switch_prob:1.5 x100 ) ;
        yin_raises "no trough"
          "pyin: cannot use no_trough_prob = nan (no_trough_prob must lie in \
           [0, 1])" (fun () -> p ~no_trough_prob:Float.nan x100 ) ;
        yin_raises "pruning"
          "pyin: cannot use transition_min_prob = -0.001 (transition_min_prob \
           must be finite and non-negative)" (fun () ->
            p ~transition_min_prob:(-1e-3) x100 ) ;
        yin_raises "grid"
          "pyin: cannot decode 1 pitch bin between fmin = 100 Hz and fmax = \
           100.5 Hz at a resolution of 0.1 semitones (widen the range or \
           refine the resolution to reach two)" (fun () ->
            pyin ~frame_length:64 ~fmin:100. ~fmax:100.5 ~sample_rate:sr x100 ) ;
        yin_raises "band"
          "pyin: cannot spread pitch transitions over 381 bins of 279 (lower \
           max_transition_rate or hop, or widen the pitch range)" (fun () ->
            pyin ~frame_length:512 ~hop:700 ~fmin:80. ~fmax:400.
              ~sample_rate:8000 x100 ) ;
        yin_raises "pruned away"
          "pyin: cannot decode with transition_min_prob = 0.5 (no transition \
           into state 0 reaches it; lower it)" (fun () ->
            pyin ~frame_length:64 ~transition_min_prob:0.5 ~fmin:50. ~fmax:400.
              ~sample_rate:sr x100 ) ;
        yin_raises "shared"
          "pyin: cannot search up to fmax = 600 Hz at 1000 Hz (fmax must not \
           exceed the Nyquist frequency, 500 Hz)" (fun () ->
            pyin ~frame_length:64 ~fmin ~fmax:600. ~sample_rate:sr x100 ) ;
        yin_raises "finite"
          "pyin: cannot estimate the pitch of a signal holding a non-finite \
           sample" (fun () ->
            pyin ~frame_length:64 ~fmin ~fmax ~sample_rate:sr
              (Nx.create Nx.float64 [|2|] [|0.; Float.neg_infinity|]) ) ) ]

let suite =
  [ group "law" law_tests
  ; group "stage" stage_tests
  ; group "properties" property_tests
  ; group "errors" error_tests ]
