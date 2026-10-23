(* The three modifications as pipeline stages.

   - the laws, bit for bit: {!Effects.phase_vocoder_stage},
   {!Effects.time_stretch_stage} and {!Effects.pitch_shift_stage}, run on a
   whole input or streamed over any partitioning of it — one item at a time,
   chunks shorter than a hop and than a frame, prime runs, a giant chunk then
   ones, random runs — give exactly the offline function at the double the
   rational rate names, at both dtypes and in both phase modes. The grid crosses
   rates above one, below one and non-dyadic, the geometries of the synthesis
   suite, and the cell where librosa's [arange] lands on the frame count.

   - the lengths: the stretch ends at [round (n / rate)] whether that is short
   of the synthesis' default length or past it, and the pitch shift at [n].

   - the declared lookahead is the lookahead of the chain each stage drives, and
   brute force checks the emission against it; the length bound never holds a
   sample back under centered frames.

   - the threaded format and every chunk's fit in its bound, flush included;
   reset, drain and the validation messages.

   - every committed golden vector, replayed through the stages at the parity
   suite's tolerances, and bit for bit against the offline functions. *)

open Windtrap
open Soundml

let lcg ?(seed = 20250803) n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let ceil_div a b = (a + b - 1) / b

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
    t

(* Comparison is on the bits: the law is exactness, not closeness. *)
let bits t = Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 t))

let complex_bits t =
  Array.append
    (Array.map Int64.bits_of_float (Nx.to_array (Nx.real Nx.float64 t)))
    (Array.map Int64.bits_of_float (Nx.to_array (Nx.imag Nx.float64 t)))

let bit_array = array int64

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = {Pipeline.Rate.num; den}

let quotient (q : Pipeline.Rate.t) = Float.of_int q.num /. Float.of_int q.den

let rate_name (q : Pipeline.Rate.t) = Printf.sprintf "%d/%d" q.num q.den

let source dtype ~channels =
  Pipeline.Format.audio dtype ~sample_rate:22050 ~channels

(* [round_half_even x], the rounding the stretched length uses. *)
let round_half_even x =
  if Float.abs x >= 0x1p52 then x
  else if x >= 0. then x +. 0x1p52 -. 0x1p52
  else x -. 0x1p52 +. 0x1p52

let stretched_length n q =
  Float.to_int (round_half_even (Float.of_int n /. quotient q))

(* {1 Partitionings} *)

let partitions ?(ones = true) ?(short = []) ~limit n =
  let rng = Random.State.make [|0x5eed; n; limit|] in
  let rec random_sizes n =
    if n = 0 then []
    else
      let s = 1 + Random.State.int rng (Int.min n limit) in
      s :: random_sizes (n - s)
  in
  let rec cycle pattern sizes acc left =
    if left <= 0 then List.rev acc
    else
      match sizes with
      | [] ->
          cycle pattern pattern acc left
      | s :: rest ->
          let s = Int.min s left in
          cycle pattern rest (s :: acc) (left - s)
  in
  let pieces pattern = cycle pattern pattern [] n in
  [ ("whole", [n])
  ; ("straddle", pieces [3; 1; 4; 1; 5; 9; 2; 6])
  ; ("with-empties", 0 :: pieces [2; 0; 0; 5; 1; 0]) ]
  @ (if ones then [("ones", List.init n (fun _ -> 1))] else [])
  @ (if n > 97 then [("prime-97", pieces [97])] else [])
  @ (if n > 5 then [("giant-then-ones", [n - 5; 1; 1; 1; 1; 1])] else [])
  @ List.map (fun s -> (Printf.sprintf "runs-of-%d" s, pieces [s])) short
  @ List.init 3 (fun i -> (Printf.sprintf "random-%d" i, random_sizes n))

(* [probe seen ~concat] passes chunks through and records its input format: the
   format the stage under test threads downstream. *)
let probe seen ~concat =
  Pipeline.kernel ~concat
    ~prepare:(fun fmt -> seen := Some fmt)
    ~step:(fun () c -> Some c)
    ()

(* [stream p ~source ~concat ~msg x sizes] streams [x] through [p] in chunks of
   the given sizes along its last axis and is everything emitted, flush
   included; it fails unless every chunk fits the bound [p] threads. *)
let stream p ~source ~concat ~msg x sizes =
  let seen = ref None in
  let s =
    Pipeline.Stream.prepare
      Pipeline.(p >> probe seen ~concat)
      ~source
      ~max_chunk:(List.fold_left Int.max 1 sizes)
  in
  let bound =
    match !seen with
    | Some fmt ->
        Option.get (Pipeline.Format.max_items fmt)
    | None ->
        fail "the probe was not prepared"
  in
  let check where out =
    if last_dim out > bound then
      failf "%s: a %s chunk of %d exceeds the threaded bound %d" msg where
        (last_dim out) bound
  in
  let outs = ref [] and pos = ref 0 in
  List.iter
    (fun size ->
      let chunk = shrink_last x !pos (!pos + size) in
      pos := !pos + size ;
      match Pipeline.Stream.push s chunk with
      | None ->
          ()
      | Some out ->
          check "push" out ;
          outs := out :: !outs )
    sizes ;
  List.iter
    (fun out ->
      check "flush" out ;
      outs := out :: !outs )
    (Pipeline.Stream.flush s) ;
  equal
    ~msg:(msg ^ ": a second flush is empty")
    int 0
    (List.length (Pipeline.Stream.flush s)) ;
  concat (List.rev !outs)

let concat_or_empty empty = function
  | [] ->
      empty
  | parts ->
      Nx.concatenate ~axis:(-1) parts

(* {1 The vocoder} *)

let vocoder_configs =
  [ ("fft256 hop64", Stft.Config.create ~fft_size:256 ~hop:64 ())
  ; ("fft64 hop16", Stft.Config.create ~fft_size:64 ~hop:16 ())
  ; ("fft31 hop5", Stft.Config.create ~fft_size:31 ~hop:5 ())
  ; ("fft32 hop7", Stft.Config.create ~fft_size:32 ~hop:7 ()) ]

(* Above one, below one, the identity, dyadic and not, terms that are not small,
   and [7/10], at which 21 frames put [arange]'s last position on the frame
   count itself. *)
let vocoder_rates =
  [ r 1 2
  ; r 3 4
  ; r 1 1
  ; r 5 4
  ; r 137 100
  ; r 2 1
  ; r 7 10
  ; r 277 349
  ; r 349 277
  ; r 37 10
  ; r 1 3 ]

(* [spectrum shape] is a spectrum no signal need have produced, so the law is
   not restricted to consistent frames. *)
let spectrum shape =
  let n = Array.fold_left ( * ) 1 shape in
  let v = lcg (2 * n) in
  Nx.complex Nx.complex128
    ~re:(Nx.create Nx.float64 shape (Array.sub v 0 n))
    ~im:(Nx.create Nx.float64 shape (Array.sub v n n))

let frame_source = source Nx.float64 ~channels:1

let vocoder_law : type c. (Complex.t, c) Nx.dtype -> string -> unit =
 fun cdtype dname ->
  List.iter
    (fun (cname, c) ->
      let bins = Stft.Config.bins c in
      List.iter
        (fun q ->
          List.iter
            (fun (shape, frames) ->
              let z = Nx.cast cdtype (spectrum shape) in
              List.iter
                (fun phase ->
                  let label =
                    Printf.sprintf "%s %s %s frames=%d %s rank %d" cname dname
                      (rate_name q) frames
                      ( match phase with
                      | `Independent ->
                          "independent"
                      | `Locked ->
                          "locked" )
                      (Array.length shape)
                  in
                  let expected =
                    complex_bits
                      (Effects.phase_vocoder ~phase c ~rate:(quotient q) z)
                  in
                  let stage = Effects.phase_vocoder_stage ~phase c ~rate:q in
                  equal ~msg:(label ^ ": run") bit_array expected
                    (complex_bits (Pipeline.run ~source:frame_source stage z)) ;
                  let lead = Array.sub shape 0 (Array.length shape - 1) in
                  let empty = Nx.zeros cdtype (Array.append lead [|0|]) in
                  List.iter
                    (fun (pname, sizes) ->
                      let msg = Printf.sprintf "%s: %s" label pname in
                      equal ~msg bit_array expected
                        (complex_bits
                           (stream stage ~source:frame_source
                              ~concat:(concat_or_empty empty) ~msg z sizes ) ) )
                    (partitions ~limit:5 frames) )
                [`Independent; `Locked] )
            ( List.map (fun m -> ([|bins; m|], m)) [0; 1; 2; 3; 21; 40]
            @ [([|2; 3; bins; 9|], 9)] ) )
        vocoder_rates )
    vocoder_configs

(* After [R] frames the vocoder has emitted every output frame whose position
   lies below [R - 1], and at least the guarantee of its one-frame latency. *)
let emission_profile c q frames =
  let z = spectrum [|Stft.Config.bins c; frames|] in
  let s =
    Pipeline.Stream.prepare
      (Effects.phase_vocoder_stage c ~rate:q)
      ~source:frame_source ~max_chunk:1
  in
  let emitted = ref 0 in
  for p = 0 to frames - 1 do
    ( match Pipeline.Stream.push s (shrink_last z p (p + 1)) with
    | None ->
        ()
    | Some out ->
        emitted := !emitted + last_dim out ) ;
    let received = p + 1 in
    let expected = ref 0 in
    while Float.of_int !expected *. quotient q < Float.of_int (received - 1) do
      incr expected
    done ;
    equal
      ~msg:(Printf.sprintf "%s after %d frames" (rate_name q) received)
      int !expected !emitted ;
    let guarantee =
      if received <= 1 then 0
      else (((received - 1) * q.den) + q.num - 1) / q.num
    in
    if !emitted < guarantee then
      failf "%s after %d frames: %d emitted, under the guarantee %d"
        (rate_name q) received !emitted guarantee
  done ;
  let tail =
    List.fold_left (fun a out -> a + last_dim out) 0 (Pipeline.Stream.flush s)
  in
  equal
    ~msg:(Printf.sprintf "%s: the total" (rate_name q))
    int
    (Float.to_int (Float.ceil (Float.of_int frames /. quotient q)))
    (!emitted + tail)

let vocoder_tests =
  [ test "streamed and run, the stage is phase_vocoder (complex128)" (fun () ->
        vocoder_law Nx.complex128 "c128" )
  ; test "streamed and run, the stage is phase_vocoder (complex64)" (fun () ->
        vocoder_law Nx.complex64 "c64" )
  ; test "21 frames at 7/10 extrapolate the last position by one frame"
      (fun () ->
        (* librosa raises there: [arange (0, 21, 0.7)] ends on 21.0 *)
        let c = Stft.Config.create ~fft_size:256 ~hop:64 () in
        let q = r 7 10 in
        equal ~msg:"the last position" float_exact 21.
          (Float.of_int 30 *. quotient q) ;
        let z = spectrum [|Stft.Config.bins c; 21|] in
        let y = Effects.phase_vocoder c ~rate:(quotient q) z in
        equal ~msg:"31 output frames" int 31 (last_dim y) ;
        let m t f =
          Nx.to_array
            (Nx.magnitude Nx.float64 (Nx.slice [Nx.A; Nx.R (f, f + 1)] t))
        in
        let last = m y 30 and a = m z 20 and b = m z 19 in
        Array.iteri
          (fun k v ->
            let line = (2. *. a.(k)) -. b.(k) in
            if Float.abs (Float.abs line -. v) > 1e-12 then
              failf "bin %d: %g, not |2 |X20| - |X19|| = %g" k v
                (Float.abs line) )
          last ;
        equal ~msg:"streamed one frame at a time" bit_array (complex_bits y)
          (complex_bits
             (stream
                (Effects.phase_vocoder_stage c ~rate:q)
                ~source:frame_source ~concat:(concat_or_empty y) ~msg:"7/10" z
                (List.init 21 (fun _ -> 1)) ) ) )
  ; test "after R frames, every position below R - 1 is out" (fun () ->
        List.iter
          (fun q -> emission_profile (snd (List.hd vocoder_configs)) q 40)
          vocoder_rates )
  ; test "rate and latency" (fun () ->
        List.iter
          (fun (name, c) ->
            List.iter
              (fun q ->
                let stage = Effects.phase_vocoder_stage c ~rate:q in
                equal
                  ~msg:(Printf.sprintf "%s %s: rate" name (rate_name q))
                  rate_t (r q.den q.num) (Pipeline.rate stage) ;
                equal
                  ~msg:(Printf.sprintf "%s %s: one frame" name (rate_name q))
                  rate_t (r 1 1) (Pipeline.latency stage) ;
                equal
                  ~msg:(Printf.sprintf "%s %s: one hop" name (rate_name q))
                  rate_t
                  (r
                     ( (Pipeline.latency (Stft.stage Nx.complex128 c)).num
                     + Stft.Config.hop c )
                     1 )
                  (Pipeline.latency
                     Pipeline.(Stft.stage Nx.complex128 c >> stage) ) )
              vocoder_rates )
          vocoder_configs )
  ; test "reset replays the stream" (fun () ->
        let c = Stft.Config.create ~fft_size:64 ~hop:16 () in
        let z = spectrum [|Stft.Config.bins c; 30|] in
        List.iter
          (fun phase ->
            let s =
              Pipeline.Stream.prepare
                (Effects.phase_vocoder_stage ~phase c ~rate:(r 3 4))
                ~source:frame_source ~max_chunk:7
            in
            let run () =
              let outs = ref [] in
              List.iter
                (fun (a, b) ->
                  match Pipeline.Stream.push s (shrink_last z a b) with
                  | None ->
                      ()
                  | Some y ->
                      outs := y :: !outs )
                [(0, 7); (7, 9); (9, 16); (16, 23); (23, 30)] ;
              List.iter (fun y -> outs := y :: !outs) (Pipeline.Stream.flush s) ;
              complex_bits (Nx.concatenate ~axis:(-1) (List.rev !outs))
            in
            ignore (Pipeline.Stream.push s (shrink_last z 0 5)) ;
            Pipeline.Stream.reset s ;
            let first = run () in
            Pipeline.Stream.reset s ;
            equal ~msg:"twice" bit_array first (run ()) ;
            equal ~msg:"and the offline function" bit_array first
              (complex_bits (Effects.phase_vocoder ~phase c ~rate:0.75 z)) )
          [`Independent; `Locked] )
  ; test "the vocoder stage validates its rate and its chunks" (fun () ->
        let c = Stft.Config.create ~fft_size:256 ~hop:64 () in
        raises ~msg:"zero numerator"
          (Invalid_argument
             "phase_vocoder_stage: cannot stretch by a rate of 0/1 (both terms \
              must be at least 1)" ) (fun () ->
            ignore (Effects.phase_vocoder_stage c ~rate:(r 0 1)) ) ;
        raises ~msg:"negative denominator"
          (Invalid_argument
             "phase_vocoder_stage: cannot stretch by a rate of 3/-2 (both \
              terms must be at least 1)" ) (fun () ->
            ignore (Effects.phase_vocoder_stage c ~rate:(r 3 (-2))) ) ;
        let prepared () =
          Pipeline.Stream.prepare
            (Effects.phase_vocoder_stage c ~rate:(r 5 4))
            ~source:frame_source ~max_chunk:8
        in
        raises ~msg:"rank one"
          (Invalid_argument
             "phase_vocoder_stage: cannot vocode a rank-1 tensor (the bin and \
              frame axes must exist)" ) (fun () ->
            ignore
              (Pipeline.Stream.push (prepared ())
                 (Nx.zeros Nx.complex128 [|4|]) ) ) ;
        raises ~msg:"wrong bin count"
          (Invalid_argument
             "phase_vocoder_stage: cannot vocode 64 frequency bins of a \
              256-point transform (the bin axis must hold fft_size / 2 + 1 = \
              129 values)" ) (fun () ->
            ignore
              (Pipeline.Stream.push (prepared ())
                 (Nx.zeros Nx.complex128 [|64; 3|]) ) ) ;
        let s = prepared () in
        ignore (Pipeline.Stream.push s (Nx.zeros Nx.complex128 [|2; 129; 3|])) ;
        raises ~msg:"a new leading shape"
          (Invalid_argument
             "phase_vocoder_stage: cannot continue a stream of leading shape \
              [2] with a chunk of leading shape [3]" ) (fun () ->
            ignore
              (Pipeline.Stream.push s (Nx.zeros Nx.complex128 [|3; 129; 3|])) ) ;
        ignore (Pipeline.Stream.flush s) ;
        raises ~msg:"after the flush"
          (Invalid_argument
             "phase_vocoder_stage: cannot feed a drained stream (flush \
              consumed the tail; reset before reusing)" ) (fun () ->
            ignore
              (Pipeline.Stream.push s (Nx.zeros Nx.complex128 [|2; 129; 3|])) ) ;
        Pipeline.Stream.reset s ;
        is_some ~msg:"reset revives it"
          (Pipeline.Stream.push s (Nx.zeros Nx.complex128 [|3; 129; 3|])) )
  ; test "a stream of no signals vocodes to no signals" (fun () ->
        let c = Stft.Config.create ~fft_size:64 ~hop:16 () in
        let z = Nx.zeros Nx.complex128 [|0; Stft.Config.bins c; 12|] in
        let y = Effects.phase_vocoder c ~rate:0.75 z in
        let streamed =
          stream
            (Effects.phase_vocoder_stage c ~rate:(r 3 4))
            ~source:frame_source ~concat:(concat_or_empty y) ~msg:"empty batch"
            z [5; 7]
        in
        equal ~msg:"shape" (array int) (Nx.shape y) (Nx.shape streamed) ) ]

(* {1 Time stretching} *)

let stretch_configs =
  [ ( "fft256 hop64 constant"
    , Stft.Config.create ~fft_size:256 ~hop:64 ~pad:(`Constant 0.) () )
  ; ("fft2048 hop512", Stft.Config.create ~fft_size:2048 ~hop:512 ())
  ; ("fft64 hop16", Stft.Config.create ~fft_size:64 ~hop:16 ())
  ; ("fft31 hop5", Stft.Config.create ~fft_size:31 ~hop:5 ())
  ; ( "left fft16 hop5 win12"
    , Stft.Config.create ~alignment:`Left ~fft_size:16 ~hop:5 ~win_length:12 ()
    )
  ; ( "right fft9 hop4 win7"
    , Stft.Config.create ~alignment:`Right ~fft_size:9 ~hop:4 ~win_length:7 ()
    ) ]

let stretch_rates = [r 2 1; r 137 100; r 37 10; r 1 2; r 3 4; r 7 10; r 277 349]

(* The dtype-polymorphic driver: the record makes one call site serve both
   storage widths. *)
type 'a signal = {dtype: (float, 'a) Nx.dtype; name: string}

let f64 = {dtype= Nx.float64; name= "f64"}

let f32 = {dtype= Nx.float32; name= "f32"}

let stretch_law : type a. a signal -> unit =
 fun {dtype; name} ->
  List.iter
    (fun (cname, c) ->
      let short = if Stft.Config.fft_size c >= 2048 then [100] else [3; 7] in
      List.iter
        (fun n ->
          let x = Nx.create dtype [|n|] (lcg n) in
          List.iter
            (fun q ->
              List.iter
                (fun phase ->
                  let label =
                    Printf.sprintf "%s %s n=%d %s %s" cname name n (rate_name q)
                      ( match phase with
                      | `Independent ->
                          "independent"
                      | `Locked ->
                          "locked" )
                  in
                  let offline =
                    Effects.time_stretch ~phase c ~rate:(quotient q) x
                  in
                  equal ~msg:(label ^ ": length") int (stretched_length n q)
                    (last_dim offline) ;
                  let expected = bits offline in
                  let stage = Effects.time_stretch_stage ~phase c ~rate:q in
                  let source = source dtype ~channels:1 in
                  equal ~msg:(label ^ ": run") bit_array expected
                    (bits (Pipeline.run ~source stage x)) ;
                  let empty = Nx.zeros dtype [|0|] in
                  List.iter
                    (fun (pname, sizes) ->
                      let msg = Printf.sprintf "%s: %s" label pname in
                      equal ~msg bit_array expected
                        (bits
                           (stream stage ~source ~concat:(concat_or_empty empty)
                              ~msg x sizes ) ) )
                    (partitions ~short ~limit:37 n) )
                [`Independent; `Locked] )
            stretch_rates )
        [0; 1; 5; 127; 128; 1000; 1300] )
    stretch_configs

(* [default_length c ~frames] is the length the synthesis totals over [frames]
   frames, which the stretched length may fall short of or pass. *)
let default_length c ~frames =
  if frames = 0 then 0
  else
    let fft = Stft.Config.fft_size c and hop = Stft.Config.hop c in
    let left, right =
      match Stft.Config.alignment c with
      | `Centered ->
          (fft / 2, fft / 2)
      | `Left ->
          (0, 0)
      | `Right ->
          (fft - 1, 0)
    in
    ((frames - 1) * hop) + fft - left - right

(* [chain dtype c q] is the composition the stretch drives. *)
let chain dtype c q =
  Pipeline.(
    Stft.stage Nx.complex128 c
    >> Effects.phase_vocoder_stage c ~rate:q
    >> Stft.synthesis_stage dtype c )

(* [one_at_a_time p ~source x] pushes [x] one sample at a time through [p] and
   is the number of samples emitted after each push. *)
let one_at_a_time p ~source x =
  let s = Pipeline.Stream.prepare p ~source ~max_chunk:1 in
  let n = last_dim x in
  let emitted = ref 0 in
  let counts =
    Array.init n (fun i ->
        ( match Pipeline.Stream.push s (shrink_last x i (i + 1)) with
        | None ->
            ()
        | Some y ->
            emitted := !emitted + last_dim y ) ;
        !emitted )
  in
  ignore (Pipeline.Stream.flush s) ;
  counts

let lookahead_configs =
  List.filter (fun (_, c) -> Stft.Config.fft_size c <= 256) stretch_configs
  @ [ ( "right constant fft16 hop4"
      , Stft.Config.create ~alignment:`Right ~pad:(`Constant 0.) ~fft_size:16
          ~hop:4 () )
    ; ("centered fft8 hop5 tail-held", Stft.Config.create ~fft_size:8 ~hop:5 ())
    ; ( "centered rectangular fft8 hop8"
      , Stft.Config.create ~window:Window.Rectangular ~fft_size:8 ~hop:8 () )
    ; ( "right rectangular fft8 hop8"
      , Stft.Config.create ~alignment:`Right ~window:Window.Rectangular
          ~fft_size:8 ~hop:8 () ) ]

let stretch_tests =
  [ test "streamed and run, the stage is time_stretch (float64)" (fun () ->
        stretch_law f64 )
  ; test "streamed and run, the stage is time_stretch (float32)" (fun () ->
        stretch_law f32 )
  ; test "the grid ends both short of and past the synthesis length" (fun () ->
        (* past it, the stream carries on into the positions the last frame
           reaches beyond the trailing trim; short of it, it stops early *)
        let past = ref 0 and short = ref 0 in
        List.iter
          (fun (_, c) ->
            List.iter
              (fun n ->
                List.iter
                  (fun q ->
                    let frames = Stft.frames c ~n in
                    let vocoded =
                      if frames = 0 then 0
                      else
                        Float.to_int
                          (Float.ceil (Float.of_int frames /. quotient q))
                    in
                    let t = default_length c ~frames:vocoded in
                    let l = stretched_length n q in
                    if l > t then incr past else if l < t then incr short )
                  stretch_rates )
              [0; 1; 5; 127; 128; 1000; 1300] )
          stretch_configs ;
        is_true ~msg:(Printf.sprintf "%d cells past" !past) (!past > 50) ;
        is_true ~msg:(Printf.sprintf "%d cells short" !short) (!short > 50) )
  ; test "a batch streams as its rows" (fun () ->
        let c = Stft.Config.create ~fft_size:64 ~hop:16 () in
        let x = Nx.create Nx.float64 [|2; 3; 300|] (lcg 1800) in
        let source = source Nx.float64 ~channels:6 in
        List.iter
          (fun q ->
            let expected =
              Effects.time_stretch ~phase:`Locked c ~rate:(quotient q) x
            in
            let stage = Effects.time_stretch_stage ~phase:`Locked c ~rate:q in
            List.iter
              (fun (pname, sizes) ->
                equal
                  ~msg:(Printf.sprintf "%s %s" (rate_name q) pname)
                  bit_array (bits expected)
                  (bits
                     (stream stage ~source
                        ~concat:
                          (concat_or_empty (Nx.zeros Nx.float64 [|2; 3; 0|]))
                        ~msg:pname x sizes ) ) )
              (partitions ~short:[7] ~limit:37 300) )
          [r 137 100; r 7 10] )
  ; test "musical material streams bit for bit" (fun () ->
        let c = Stft.Config.create ~fft_size:2048 ~hop:512 () in
        let source = source Nx.float64 ~channels:1 in
        let n = Pvoc_signals.sample_rate / 2 in
        let runs size =
          List.init (ceil_div n size) (fun i -> Int.min size (n - (i * size)))
        in
        List.iter
          (fun (sname, x) ->
            List.iter
              (fun (q, phase) ->
                let expected =
                  bits (Effects.time_stretch ~phase c ~rate:(quotient q) x)
                in
                let stage = Effects.time_stretch_stage ~phase c ~rate:q in
                List.iter
                  (fun (pname, sizes) ->
                    let msg =
                      Printf.sprintf "%s %s %s" sname (rate_name q) pname
                    in
                    equal ~msg bit_array expected
                      (bits
                         (stream stage ~source
                            ~concat:
                              (concat_or_empty (Nx.zeros Nx.float64 [|0|]))
                            ~msg x sizes ) ) )
                  [ ("whole", [n])
                  ; ("runs-of-100", runs 100)
                  ; ("runs-of-1500", runs 1500) ] )
              [ (r 3 4, `Independent)
              ; (r 137 100, `Locked)
              ; (r 2 1, `Independent) ] )
          [ ("sweep", Pvoc_signals.sweep n)
          ; ("harmonics", Pvoc_signals.harmonics n) ] )
  ; test "the lookahead is the lookahead of the chain it drives" (fun () ->
        List.iter
          (fun (name, c) ->
            List.iter
              (fun (q : Pipeline.Rate.t) ->
                let lst = (Pipeline.latency (Stft.stage Nx.complex128 c)).num in
                let ls = Stft.Config.synthesis_latency c in
                let stage = Effects.time_stretch_stage c ~rate:q in
                let msg = Printf.sprintf "%s %s" name (rate_name q) in
                equal ~msg rate_t
                  (Pipeline.latency (chain Nx.float64 c q))
                  (Pipeline.latency stage) ;
                (* lst + hop + ls * num / den, exactly *)
                equal ~msg:(msg ^ ": in closed form") rate_t
                  { Pipeline.Rate.num=
                      ((lst + Stft.Config.hop c) * q.den) + (ls * q.num)
                  ; den= q.den }
                  (Pipeline.latency stage) ;
                equal ~msg:(msg ^ ": rate") rate_t (r q.den q.num)
                  (Pipeline.rate stage) )
              stretch_rates )
          stretch_configs ;
        equal ~msg:"2048/512 at rate 1: 1024 + 512 + 1024" rate_t (r 2560 1)
          (Pipeline.latency
             (Effects.time_stretch_stage
                (Stft.Config.create ~fft_size:2048 ~hop:512 ())
                ~rate:(r 1 1) ) ) )
  ; test "brute force: the declaration bounds the deficit" (fun () ->
        (* after [n] samples at least [ceil ((n - lookahead) * den / num)] are
           out, under every alignment but [`Left], whose analysis declares no
           latency although its first frame needs a whole frame of samples *)
        let x = Nx.create Nx.float64 [|300|] (lcg 300) in
        let source = source Nx.float64 ~channels:1 in
        List.iter
          (fun (name, c) ->
            if Stft.Config.alignment c <> `Left then
              List.iter
                (fun (q : Pipeline.Rate.t) ->
                  let stage = Effects.time_stretch_stage c ~rate:q in
                  let la = Pipeline.latency stage in
                  Array.iteri
                    (fun i emitted ->
                      let numer = (((i + 1) * la.den) - la.num) * q.den
                      and denom = la.den * q.num in
                      let guarantee =
                        if numer <= 0 then 0 else (numer + denom - 1) / denom
                      in
                      if emitted < guarantee then
                        failf "%s %s after %d samples: %d out, %d guaranteed"
                          name (rate_name q) (i + 1) emitted guarantee )
                    (one_at_a_time stage ~source x) )
                stretch_rates )
          lookahead_configs )
  ; test "the length bound holds nothing back under centered frames" (fun () ->
        (* the stage emits what the chain emits, push for push, and parts from
           it only at the flush, where it ends at the stretched length; under
           [`Left] the chain runs ahead of the length the input maps to, and the
           stage holds back to it *)
        let x = Nx.create Nx.float64 [|300|] (lcg 300) in
        let source = source Nx.float64 ~channels:1 in
        List.iter
          (fun (name, c) ->
            List.iter
              (fun q ->
                let stage =
                  one_at_a_time (Effects.time_stretch_stage c ~rate:q) ~source x
                in
                let msg = Printf.sprintf "%s %s" name (rate_name q) in
                Array.iteri
                  (fun i emitted ->
                    let bound = stretched_length (i + 1) q in
                    if emitted > bound then
                      failf "%s: %d out after %d samples, past %d" msg emitted
                        (i + 1) bound )
                  stage ;
                if Stft.Config.alignment c = `Centered then
                  equal ~msg (array int)
                    (one_at_a_time (chain Nx.float64 c q) ~source x)
                    stage )
              stretch_rates )
          lookahead_configs ;
        List.iter
          (fun (name, c) ->
            let q = r 37 10 in
            let stage =
              one_at_a_time (Effects.time_stretch_stage c ~rate:q) ~source x
            in
            let chained = one_at_a_time (chain Nx.float64 c q) ~source x in
            is_true
              ~msg:("the bound binds under " ^ name)
              (Array.exists2 (fun a b -> a < b) stage chained) )
          [ ( "left fft16 hop5"
            , Stft.Config.create ~alignment:`Left ~fft_size:16 ~hop:5 () )
          ; ( "right rectangular fft8 hop8"
            , Stft.Config.create ~alignment:`Right ~window:Window.Rectangular
                ~fft_size:8 ~hop:8 () ) ] )
  ; test "the threaded format" (fun () ->
        let c = Stft.Config.create ~fft_size:256 ~hop:64 () in
        let seen = ref None in
        let p =
          Pipeline.(
            Effects.time_stretch_stage c ~rate:(r 137 100)
            >> probe seen ~concat:(concat_or_empty (Nx.zeros Nx.float32 [|0|])) )
        in
        ignore
          (Pipeline.Stream.prepare p
             ~source:(source Nx.float32 ~channels:1)
             ~max_chunk:64 ) ;
        match !seen with
        | None ->
            fail "the probe was not prepared"
        | Some fmt ->
            equal ~msg:"items per second" rate_t
              (r (22050 * 100) 137)
              (Pipeline.Format.items_per_second fmt) ;
            is_true ~msg:"float32 samples"
              (Pipeline.Format.equal_dtype
                 (Pipeline.Format.dtype fmt)
                 (Pipeline.Format.Dtype Nx.float32) ) ;
            equal ~msg:"upstream latency" rate_t
              (Pipeline.latency
                 (Effects.time_stretch_stage c ~rate:(r 137 100)) )
              (Pipeline.Format.upstream_latency fmt) )
  ; test "empty pushes and an empty stream" (fun () ->
        let c = Stft.Config.create ~fft_size:64 ~hop:16 () in
        let x = Nx.create Nx.float64 [|2; 200|] (lcg 400) in
        let source = source Nx.float64 ~channels:2 in
        let stage = Effects.time_stretch_stage c ~rate:(r 3 4) in
        let empty = Nx.zeros Nx.float64 [|2; 0|] in
        equal ~msg:"empty chunks change nothing" bit_array
          (bits (Effects.time_stretch c ~rate:0.75 x))
          (bits
             (stream stage ~source ~concat:(concat_or_empty empty)
                ~msg:"with empties" x
                [0; 13; 0; 0; 100; 0; 87; 0] ) ) ;
        let s = Pipeline.Stream.prepare stage ~source ~max_chunk:8 in
        equal ~msg:"no push, nothing to flush" int 0
          (List.length (Pipeline.Stream.flush s)) ;
        equal ~msg:"an empty signal runs to an empty signal" (array int) [|2; 0|]
          (Nx.shape (Pipeline.run ~source stage empty)) )
  ; test "the stretch stage validates its rate, geometry and chunks" (fun () ->
        let c = Stft.Config.create ~fft_size:64 ~hop:16 () in
        raises ~msg:"zero numerator"
          (Invalid_argument
             "time_stretch_stage: cannot stretch by a rate of 0/1 (both terms \
              must be at least 1)" ) (fun () ->
            ignore (Effects.time_stretch_stage c ~rate:(r 0 1)) ) ;
        raises ~msg:"no overlap"
          (Invalid_argument
             "time_stretch_stage: cannot invert a 64-point window advanced by \
              64 samples inside a 64-point frame (the overlap-added squared \
              window must stay above 1e-10 of its largest value at every \
              position)" ) (fun () ->
            ignore
              (Effects.time_stretch_stage
                 (Stft.Config.create ~fft_size:64 ~hop:64 ())
                 ~rate:(r 5 4) ) ) ;
        let prepared () =
          Pipeline.Stream.prepare
            (Effects.time_stretch_stage c ~rate:(r 5 4))
            ~source:(source Nx.float64 ~channels:2)
            ~max_chunk:64
        in
        raises ~msg:"rank zero"
          (Invalid_argument
             "time_stretch_stage: cannot process a rank-zero tensor (the time \
              axis must exist)" ) (fun () ->
            ignore
              (Pipeline.Stream.push (prepared ()) (Nx.zeros Nx.float64 [||])) ) ;
        raises ~msg:"no signals"
          (Invalid_argument
             "time_stretch_stage: cannot stretch a chunk with a zero-size \
              leading axis (channels must be at least 1)" ) (fun () ->
            ignore
              (Pipeline.Stream.push (prepared ())
                 (Nx.zeros Nx.float64 [|0; 8|]) ) ) ;
        let s = prepared () in
        ignore (Pipeline.Stream.push s (Nx.zeros Nx.float64 [|2; 40|])) ;
        raises ~msg:"a new leading shape"
          (Invalid_argument
             "time_stretch_stage: cannot continue a stream of leading shape \
              [2] with a chunk of leading shape [3]" ) (fun () ->
            ignore (Pipeline.Stream.push s (Nx.zeros Nx.float64 [|3; 40|])) ) ;
        ignore (Pipeline.Stream.flush s) ;
        raises ~msg:"after the flush"
          (Invalid_argument
             "time_stretch_stage: cannot feed a drained stream (flush consumed \
              the tail; reset before reusing)" ) (fun () ->
            ignore (Pipeline.Stream.push s (Nx.zeros Nx.float64 [|2; 40|])) ) )
  ; test "reset replays the stretch" (fun () ->
        let c = Stft.Config.create ~alignment:`Left ~fft_size:16 ~hop:5 () in
        let x = Nx.create Nx.float64 [|200|] (lcg 200) in
        let s =
          Pipeline.Stream.prepare
            (Effects.time_stretch_stage c ~rate:(r 37 10))
            ~source:(source Nx.float64 ~channels:1)
            ~max_chunk:9
        in
        let run () =
          let outs = ref [] and pos = ref 0 in
          while !pos < 200 do
            let stop = Int.min 200 (!pos + 9) in
            ( match Pipeline.Stream.push s (shrink_last x !pos stop) with
            | None ->
                ()
            | Some y ->
                outs := y :: !outs ) ;
            pos := stop
          done ;
          List.iter (fun y -> outs := y :: !outs) (Pipeline.Stream.flush s) ;
          bits (Nx.concatenate ~axis:(-1) (List.rev !outs))
        in
        ignore (Pipeline.Stream.push s (shrink_last x 0 9)) ;
        Pipeline.Stream.reset s ;
        let first = run () in
        equal ~msg:"the offline function" bit_array
          (bits (Effects.time_stretch c ~rate:3.7 x))
          first ;
        Pipeline.Stream.reset s ;
        equal ~msg:"after reset" bit_array first (run ()) ) ]

(* {1 Pitch shifting} *)

let pitch_ratios =
  [r 2 1; r 1 2; Effects.semitones 4.; Effects.semitones (-6.); r 3 2; r 1 1]

let pitch_configs =
  [ ("fft2048 hop512", Stft.Config.create ~fft_size:2048 ~hop:512 ())
  ; ( "fft256 hop64 constant"
    , Stft.Config.create ~fft_size:256 ~hop:64 ~pad:(`Constant 0.) () ) ]

let pitch_law : type a. a signal -> unit =
 fun {dtype; name} ->
  List.iter
    (fun (cname, c) ->
      List.iter
        (fun n ->
          let x = Nx.create dtype [|n|] (lcg n) in
          List.iter
            (fun ratio ->
              List.iter
                (fun (quality, qname) ->
                  let label =
                    Printf.sprintf "%s %s n=%d %s %s" cname name n
                      (rate_name ratio) qname
                  in
                  let offline = Effects.pitch_shift ~quality c ~ratio x in
                  equal ~msg:(label ^ ": length") int n (last_dim offline) ;
                  let expected = bits offline in
                  let stage = Effects.pitch_shift_stage ~quality c ~ratio in
                  let source = source dtype ~channels:1 in
                  equal ~msg:(label ^ ": run") bit_array expected
                    (bits (Pipeline.run ~source stage x)) ;
                  List.iter
                    (fun (pname, sizes) ->
                      let msg = Printf.sprintf "%s: %s" label pname in
                      equal ~msg bit_array expected
                        (bits
                           (stream stage ~source
                              ~concat:(concat_or_empty (Nx.zeros dtype [|0|]))
                              ~msg x sizes ) ) )
                    (partitions ~short:[100] ~limit:37 n) )
                [(`High, "high"); (`Fast, "fast")] )
            pitch_ratios )
        [37; 1000; 4000] )
    pitch_configs

let pitch_tests =
  [ test "streamed and run, the stage is pitch_shift (float64)" (fun () ->
        pitch_law f64 )
  ; test "streamed and run, the stage is pitch_shift (float32)" (fun () ->
        pitch_law f32 )
  ; test "locked phases shift bit for bit too" (fun () ->
        let c = Stft.Config.create ~fft_size:256 ~hop:64 () in
        let x = Pvoc_signals.harmonics 4000 in
        let source = source Nx.float64 ~channels:1 in
        List.iter
          (fun ratio ->
            let expected =
              bits (Effects.pitch_shift ~phase:`Locked c ~ratio x)
            in
            let stage = Effects.pitch_shift_stage ~phase:`Locked c ~ratio in
            List.iter
              (fun (pname, sizes) ->
                equal
                  ~msg:(Printf.sprintf "%s %s" (rate_name ratio) pname)
                  bit_array expected
                  (bits
                     (stream stage ~source
                        ~concat:(concat_or_empty (Nx.zeros Nx.float64 [|0|]))
                        ~msg:pname x sizes ) ) )
              (partitions ~ones:false ~short:[100] ~limit:37 4000) )
          [Effects.semitones 4.; r 1 2] )
  ; test "the lookahead is the chain's, rounded up to a sample" (fun () ->
        List.iter
          (fun (name, c) ->
            List.iter
              (fun (ratio : Pipeline.Rate.t) ->
                let resampler =
                  Resample.Config.create ~sample_rate:ratio.num
                    ~target:ratio.den ()
                in
                let composed =
                  Pipeline.latency
                    Pipeline.(
                      Effects.time_stretch_stage c ~rate:(r ratio.den ratio.num)
                      >> Resample.stage resampler )
                in
                let declared =
                  Pipeline.latency (Effects.pitch_shift_stage c ~ratio)
                in
                let msg = Printf.sprintf "%s %s" name (rate_name ratio) in
                equal ~msg:(msg ^ ": a whole sample") int 1 declared.den ;
                (* composed <= declared < composed + 1 *)
                is_true
                  ~msg:
                    (Format.asprintf "%s: %a covers %a by less than a sample"
                       msg Pipeline.Rate.pp declared Pipeline.Rate.pp composed )
                  ( declared.num * composed.den >= composed.num
                  && declared.num * composed.den < composed.num + composed.den
                  ) ;
                let lst = (Pipeline.latency (Stft.stage Nx.complex128 c)).num in
                let k = Resample.Config.latency resampler in
                let ls = Stft.Config.synthesis_latency c in
                equal ~msg:(msg ^ ": in closed form") int
                  ( lst + Stft.Config.hop c
                  + ((((ls + k) * ratio.den) + ratio.num - 1) / ratio.num) )
                  declared.num ;
                equal ~msg:(msg ^ ": rate") rate_t (r 1 1)
                  (Pipeline.rate (Effects.pitch_shift_stage c ~ratio)) )
              pitch_ratios )
          pitch_configs )
  ; test "the pitch stage validates its ratio, geometry and format" (fun () ->
        let c = Stft.Config.create ~fft_size:256 ~hop:64 () in
        raises ~msg:"zero numerator"
          (Invalid_argument
             "pitch_shift_stage: cannot shift by a frequency ratio of 0/1 \
              (both terms must be at least 1)" ) (fun () ->
            ignore (Effects.pitch_shift_stage c ~ratio:(r 0 1)) ) ;
        raises ~msg:"no overlap"
          (Invalid_argument
             "pitch_shift_stage: cannot invert a 64-point window advanced by \
              64 samples inside a 64-point frame (the overlap-added squared \
              window must stay above 1e-10 of its largest value at every \
              position)" ) (fun () ->
            ignore
              (Effects.pitch_shift_stage
                 (Stft.Config.create ~fft_size:64 ~hop:64 ())
                 ~ratio:(r 3 2) ) ) ;
        raises ~msg:"float16"
          (Invalid_argument
             "pitch_shift_stage: cannot shift float16 audio (the resampler \
              carries float32 and float64)" ) (fun () ->
            ignore
              (Pipeline.Stream.prepare
                 (Effects.pitch_shift_stage c ~ratio:(r 3 2))
                 ~source:(source Nx.float16 ~channels:1)
                 ~max_chunk:64 ) ) ;
        let s =
          Pipeline.Stream.prepare
            (Effects.pitch_shift_stage c ~ratio:(r 3 2))
            ~source:(source Nx.float64 ~channels:1)
            ~max_chunk:64
        in
        ignore (Pipeline.Stream.push s (Nx.zeros Nx.float64 [|64|])) ;
        ignore (Pipeline.Stream.flush s) ;
        raises ~msg:"after the flush"
          (Invalid_argument
             "pitch_shift_stage: cannot feed a drained stream (flush consumed \
              the tail; reset before reusing)" ) (fun () ->
            ignore (Pipeline.Stream.push s (Nx.zeros Nx.float64 [|64|])) ) ) ]

(* {1 The goldens, through the stages}

   Every committed vector, replayed through the stage at the rational its rate
   names, run whole and streamed over one random partition, at the parity
   suite's tolerances — and, because the law makes them the same computation,
   bit for bit against the offline function the vectors already pin. *)

let rational_of_rate rate =
  match
    List.find_opt
      (fun q -> Float.equal (quotient q) rate)
      [r 137 100; r 5 4; r 3 4; r 1 2; r 2 1; r 1 1]
  with
  | Some q ->
      q
  | None ->
      failf "no rational for the rate %.17g" rate

let golden_check : type a.
       a signal
    -> string
    -> Tutils.Golden.case
    -> (unit -> ((float, a) Nx.t, (float, a) Nx.t, Pipeline.causal) Pipeline.t)
    -> ((float, a) Nx.t -> (float, a) Nx.t)
    -> float option
    -> unit =
 fun {dtype; _} stem case stage offline fraction ->
  let shape = Pvoc_goldens.shape case in
  let x = Nx.create dtype shape (lcg (Array.fold_left ( * ) 1 shape)) in
  let channels = match shape with [|c; _|] -> c | _ -> 1 in
  let source = source dtype ~channels in
  let expected = bits (offline x) in
  let atol, rtol =
    match (fraction, Tutils.Golden.string_param case "dtype") with
    | Some f, _ ->
        ( f
          *. Array.fold_left
               (fun a v -> Float.max a (Float.abs v))
               0. case.values
        , 0. )
    | None, "float64" ->
        (Pvoc_goldens.float64_atol, 0.)
    | None, _ ->
        (Tutils.float32_atol, Tutils.float32_rtol)
  in
  let n = Tutils.Golden.int_param case "n" in
  let rng = Random.State.make [|Hashtbl.hash case.name|] in
  let rec sizes left =
    if left = 0 then []
    else
      let s = 1 + Random.State.int rng (Int.min left 61) in
      s :: sizes (left - s)
  in
  let lead = Array.sub shape 0 (Array.length shape - 1) in
  List.iter
    (fun (how, got) ->
      let msg = Printf.sprintf "%s %s %s" stem case.name how in
      Tutils.check_close ~rtol ~atol ~shape:case.shape ~msg
        ~expected:case.values got ;
      equal ~msg:(msg ^ ": the offline bits") bit_array expected (bits got) )
    [ ("run", Pipeline.run ~source (stage ()) x)
    ; ( "streamed"
      , stream (stage ()) ~source
          ~concat:(concat_or_empty (Nx.zeros dtype (Array.append lead [|0|])))
          ~msg:case.name x (sizes n) ) ]

let golden_case stem (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let c = Pvoc_goldens.config case in
      let phase = if stem = "locked" then `Locked else `Independent in
      let replay (type a) (signal : a signal) =
        match stem with
        | "pitch" ->
            let ratio =
              r
                (Tutils.Golden.int_param case "num")
                (Tutils.Golden.int_param case "den")
            in
            golden_check signal stem case
              (fun () -> Effects.pitch_shift_stage c ~ratio)
              (fun x -> Effects.pitch_shift c ~ratio x)
              (Some Pvoc_goldens.pitch_fraction)
        | _ ->
            let q =
              if stem = "pitchstretch" then
                r
                  (Tutils.Golden.int_param case "den")
                  (Tutils.Golden.int_param case "num")
              else rational_of_rate (Tutils.Golden.float_param case "rate")
            in
            equal ~msg:"the rational names the vector's rate" float_exact
              (Tutils.Golden.float_param case "rate")
              (quotient q) ;
            golden_check signal stem case
              (fun () -> Effects.time_stretch_stage ~phase c ~rate:q)
              (fun x -> Effects.time_stretch ~phase c ~rate:(quotient q) x)
              None
      in
      match Tutils.Golden.string_param case "dtype" with
      | "float64" ->
          replay f64
      | "float32" ->
          replay f32
      | other ->
          failf "golden case %s: unknown dtype %s" case.name other )

let golden_tests =
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      group ("stage-golden-" ^ stem) (List.map (golden_case stem) file.cases) )
    (Tutils.Golden.load_dir
       ~filter:(fun name ->
         List.mem name
           ["stretch.json"; "pitchstretch.json"; "pitch.json"; "locked.json"] )
       "vectors" )

let suite =
  [ group "vocoder-stage" vocoder_tests
  ; group "time-stretch-stage" stretch_tests
  ; group "pitch-shift-stage" pitch_tests ]
  @ golden_tests
