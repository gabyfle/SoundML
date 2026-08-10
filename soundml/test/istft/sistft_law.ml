(* The laws of Soundml.Stft.Synthesis, anchored on the offline synthesis:

   - partition invariance, bit for bit. For every chunking of the frame sequence
   — one frame at a time, prime runs, a giant chunk then ones, runs that
   straddle the hop-block boundary — concatenating the emitted chunks and the
   drain gives the same bits as the one-batch instance. The kernel's state after
   any number of frames is a function of the frames alone (the last [ceil
   (fft_size / hop) - 1] windowed frames, the frame count, the head trim still
   owed and the held tail), and every fold it performs is per position and in a
   fixed order, so nothing about where the chunk boundaries fell can reach the
   arithmetic.

   - the one-batch instance is [invert] at its default length, bit for bit: the
   same windowed frames, the same tap-ascending overlap-add, the same three
   envelope regions and the same trim. That is the anchor the streaming kernel
   inherits every golden vector and every reconstruction bound through, so this
   file states no reconstruction property of its own — it states that the two
   computations are the same computation.

   - the emitted total is [invert]'s default length exactly, over every geometry
   in the grid, including the ones where the trailing trim swallows the drain.

   - the lookahead is [Config.synthesis_latency]: measured against the naive
   rate map, the kernel's largest output deficit is that many samples, and the
   first sample appears once frame [synthesis_latency / hop] has been consumed —
   the completion law of the interface, brute-forced.

   - the invertibility criterion is checked once, at [prepare], and the shape,
   drain and dtype contracts fire before any transform runs.

   The grid crosses non-divisible hops, windows shorter than the transform, odd
   transform sizes, all three alignments, both window normalisations that change
   the analysis window, and the three geometries where one hop reaches past the
   trailing trim ([hop + right_width > fft_size]) — the only ones where the
   kernel holds a tail back that the trim then discards. *)

open Windtrap
open Soundml

(* The generator's LCG, reproduced with the same integer arithmetic. *)
let lcg n =
  let state = ref 20250803 in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.shrink
    (Array.init nd (fun i ->
         if i = nd - 1 then (start, stop) else (0, Nx.dim i t) ) )
    t

let concat_last parts = Nx.concatenate ~axis:(-1) parts

let batch_shape t =
  let s = Nx.shape t in
  Array.sub s 0 (Array.length s - 2)

(* Comparison is on the bits: the law is exactness, not closeness. *)
let bits t = Array.map Int64.bits_of_float (Nx.to_array t)

let bit_array = array int64

let left_width fft = function
  | `Centered ->
      fft / 2
  | `Left ->
      0
  | `Right ->
      fft - 1

let right_width fft = function `Centered -> fft / 2 | `Left | `Right -> 0

(* {1 The configuration grid} *)

let grid =
  [ ("centered fft8 hop2", Stft.Config.create ~fft_size:8 ~hop:2 ())
  ; ( "centered fft8 hop3 non-divisible"
    , Stft.Config.create ~fft_size:8 ~hop:3 () )
  ; ("centered fft9 hop2 odd", Stft.Config.create ~fft_size:9 ~hop:2 ())
  ; ("centered fft9 hop4 odd", Stft.Config.create ~fft_size:9 ~hop:4 ())
  ; ("left fft8 hop3", Stft.Config.create ~alignment:`Left ~fft_size:8 ~hop:3 ())
  ; ( "left fft16 hop4 win10"
    , Stft.Config.create ~alignment:`Left ~fft_size:16 ~hop:4 ~win_length:10 ()
    )
  ; ( "right fft8 hop3"
    , Stft.Config.create ~alignment:`Right ~fft_size:8 ~hop:3 () )
  ; ( "right fft9 hop4 win7"
    , Stft.Config.create ~alignment:`Right ~fft_size:9 ~hop:4 ~win_length:7 ()
    )
  ; ( "centered fft16 hop5 win12"
    , Stft.Config.create ~fft_size:16 ~hop:5 ~win_length:12 () )
  ; ( "centered fft32 hop7 win21"
    , Stft.Config.create ~fft_size:32 ~hop:7 ~win_length:21 () )
  ; ("centered fft8 hop5 tail-held", Stft.Config.create ~fft_size:8 ~hop:5 ())
  ; ("centered fft8 hop6 tail-held", Stft.Config.create ~fft_size:8 ~hop:6 ())
  ; ( "centered fft16 hop11 tail-held"
    , Stft.Config.create ~fft_size:16 ~hop:11 () )
  ; ( "centered fft16 hop4 magnitude"
    , Stft.Config.create ~fft_size:16 ~hop:4 ~scale:`Magnitude () )
  ; ( "centered fft16 hop4 hamming"
    , Stft.Config.create ~window:Window.Hamming ~fft_size:16 ~hop:4 () ) ]

(* {1 Chunkings of the frame sequence} *)

let chunkings total =
  let rec pieces sizes acc left =
    if left <= 0 then List.rev acc
    else
      match sizes with
      | [] ->
          List.rev (left :: acc)
      | s :: rest ->
          let s = Stdlib.min s left in
          pieces rest (s :: acc) (left - s)
  in
  [ ("whole", [total])
  ; ("ones", pieces (List.init total (fun _ -> 1)) [] total)
  ; ("primes", pieces [2; 3; 5; 7; 11; 13; 2; 3; 5; 7] [] total)
  ; ( "giant-then-ones"
    , pieces (Stdlib.max 1 (total - 3) :: List.init 8 (fun _ -> 1)) [] total )
  ; ("irregular", pieces [1; 4; 1; 1; 9; 2; 1; 6; 3; 1; 1; 1] [] total)
  ; ("straddle", pieces [3; 1; 4; 1; 5; 9; 2; 6] [] total) ]

(* [drive dtype c z sizes] feeds [z] to a fresh kernel in the given frame
   batches and is everything the kernel emitted, drain included. *)
let drive : type a c.
       (float, a) Nx.dtype
    -> Stft.Config.t
    -> (Complex.t, c) Nx.t
    -> int list
    -> (float, a) Nx.t =
 fun dtype c z sizes ->
  let k =
    Stft.Synthesis.prepare dtype c (Nx.dtype z) ~channels:1
      ~max_block:(List.fold_left Stdlib.max 1 sizes)
  in
  let outs = ref [] and pos = ref 0 in
  List.iter
    (fun s ->
      let chunk = shrink_last z !pos (!pos + s) in
      pos := !pos + s ;
      match Stft.Synthesis.step k chunk with
      | None ->
          ()
      | Some out ->
          outs := out :: !outs )
    sizes ;
  ( match Stft.Synthesis.flush k with
  | None ->
      ()
  | Some out ->
      outs := out :: !outs ) ;
  match List.rev !outs with
  | [] ->
      Nx.zeros dtype (Array.append (batch_shape z) [|0|])
  | parts ->
      concat_last parts

(* [check_spectrum] is the whole law on one spectrum: every chunking agrees with
   the one-batch instance, which agrees with [invert] at its default length. *)
let check_spectrum : type a c.
       (float, a) Nx.dtype
    -> Stft.Config.t
    -> string
    -> (Complex.t, c) Nx.t
    -> unit =
 fun dtype c label z ->
  let total = last_dim z in
  let offline = Stft.invert dtype c z in
  let one_shot = drive dtype c z [total] in
  equal
    ~msg:(Printf.sprintf "%s: one batch is invert" label)
    bit_array (bits offline) (bits one_shot) ;
  equal
    ~msg:(Printf.sprintf "%s: one batch has invert's length" label)
    int (last_dim offline) (last_dim one_shot) ;
  List.iter
    (fun (cname, sizes) ->
      let got = drive dtype c z sizes in
      equal
        ~msg:(Printf.sprintf "%s: chunking %s" label cname)
        bit_array (bits one_shot) (bits got) )
    (chunkings total)

(* The spectra: transforms of LCG signals at both dtype pairings, spectra
   invented frame by frame (which no signal need have produced, so the law is
   not restricted to consistent frames), and one batched leading axis. *)
let synthetic c frames =
  let bins = Stft.Config.bins c in
  let v = lcg (2 * bins * frames) in
  Nx.init Nx.complex128 [|bins; frames|] (fun ix ->
      { Complex.re= v.((ix.(0) * frames) + ix.(1))
      ; im= v.((bins * frames) + (ix.(0) * frames) + ix.(1)) } )

let law_case (name, c) =
  test name (fun () ->
      List.iter
        (fun n ->
          let signal = lcg n in
          let x64 = Nx.create Nx.float64 [|n|] signal in
          check_spectrum Nx.float64 c
            (Printf.sprintf "f64/n=%d" n)
            (Stft.transform Nx.complex128 c x64) ;
          let x32 = Nx.create Nx.float32 [|n|] signal in
          check_spectrum Nx.float32 c
            (Printf.sprintf "f32/n=%d" n)
            (Stft.transform Nx.complex64 c x32) )
        [1; 5; 17; 61] ;
      List.iter
        (fun frames ->
          let z = synthetic c frames in
          check_spectrum Nx.float64 c
            (Printf.sprintf "synthetic/f64/frames=%d" frames)
            z ;
          check_spectrum Nx.float32 c
            (Printf.sprintf "synthetic/f32/frames=%d" frames)
            z )
        [1; 5; 17; 61] ;
      let batched = Nx.create Nx.float64 [|2; 40|] (lcg 80) in
      check_spectrum Nx.float64 c "batch [2;40]"
        (Stft.transform Nx.complex128 c batched) )

let law_tests = List.map law_case grid

(* {1 The emitted total is invert's default length} *)

let length_tests =
  [ test "the stream totals invert's default length" (fun () ->
        List.iter
          (fun (name, c) ->
            List.iter
              (fun frames ->
                let z = synthetic c frames in
                equal
                  ~msg:(Printf.sprintf "%s frames=%d" name frames)
                  int
                  (last_dim (Stft.invert Nx.float64 c z))
                  (last_dim
                     (drive Nx.float64 c z (List.init frames (fun _ -> 1))) ) )
              [1; 2; 3; 7; 40] )
          grid )
  ; test "an empty spectrum emits nothing" (fun () ->
        List.iter
          (fun (name, c) ->
            let z = Nx.zeros Nx.complex128 [|Stft.Config.bins c; 0|] in
            let k =
              Stft.Synthesis.prepare Nx.float64 c Nx.complex128 ~channels:1
                ~max_block:4
            in
            is_none
              ~msg:(name ^ ": no frames, no samples")
              (Stft.Synthesis.step k z) ;
            is_none ~msg:(name ^ ": nothing to drain") (Stft.Synthesis.flush k) ;
            equal ~msg:(name ^ ": invert agrees") int 0
              (last_dim (Stft.invert Nx.float64 c z)) )
          grid ) ]

(* {1 The lookahead, brute-forced}

   One frame per step, measuring what the kernel has released against the naive
   rate map [frames * hop]. *)

let deficit_profile c frames =
  let z = synthetic c frames in
  let k =
    Stft.Synthesis.prepare Nx.float64 c Nx.complex128 ~channels:1 ~max_block:1
  in
  let emitted = ref 0 and worst = ref 0 and first = ref (-1) in
  for p = 0 to frames - 1 do
    ( match Stft.Synthesis.step k (shrink_last z p (p + 1)) with
    | None ->
        ()
    | Some out ->
        emitted := !emitted + last_dim out ) ;
    if !emitted > 0 && !first < 0 then first := p ;
    let d = ((p + 1) * Stft.Config.hop c) - !emitted in
    if d > !worst then worst := d
  done ;
  (!worst, !first)

let latency_tests =
  [ test "the largest deficit is Config.synthesis_latency" (fun () ->
        List.iter
          (fun (name, c) ->
            let worst, _ = deficit_profile c 40 in
            equal ~msg:name int (Stft.Config.synthesis_latency c) worst )
          grid )
  ; test "the first sample lands where the completion law says" (fun () ->
        List.iter
          (fun (name, c) ->
            let _, first = deficit_profile c 40 in
            equal ~msg:name int
              (Stft.Config.synthesis_latency c / Stft.Config.hop c)
              first )
          grid )
  ; test "synthesis_latency is the two trims of the geometry" (fun () ->
        List.iter
          (fun (name, c) ->
            let fft = Stft.Config.fft_size c and hop = Stft.Config.hop c in
            let alignment = Stft.Config.alignment c in
            equal ~msg:name int
              ( left_width fft alignment
              + Stdlib.max 0 (hop + right_width fft alignment - fft) )
              (Stft.Config.synthesis_latency c) )
          grid ;
        equal ~msg:"left holds nothing back" int 0
          (Stft.Config.synthesis_latency
             (Stft.Config.create ~alignment:`Left ~fft_size:16 ~hop:4 ()) ) ;
        equal ~msg:"right holds the whole head extension" int 15
          (Stft.Config.synthesis_latency
             (Stft.Config.create ~alignment:`Right ~fft_size:16 ~hop:4 ()) ) ;
        equal ~msg:"a hop past the trailing trim holds the overshoot" int 11
          (Stft.Config.synthesis_latency
             (Stft.Config.create ~fft_size:16 ~hop:11 ()) ) ) ]

(* {1 Contracts} *)

let c8 = Stft.Config.create ~fft_size:8 ~hop:2 ()

let prepare8 () =
  Stft.Synthesis.prepare Nx.float64 c8 Nx.complex128 ~channels:1 ~max_block:8

let frames8 n = synthetic c8 n

let contract_tests =
  [ test "prepare rejects a geometry that determines no signal" (fun () ->
        raises_invalid_arg ~msg:"a hop wider than the frame"
          "prepare: cannot invert a 8-point window advanced by 9 samples \
           inside a 8-point frame (the overlap-added squared window must stay \
           above 1e-10 of its largest value at every position)" (fun () ->
            Stft.Synthesis.prepare Nx.float64
              (Stft.Config.create ~fft_size:8 ~hop:9 ())
              Nx.complex128 ~channels:1 ~max_block:4 ) ;
        raises_invalid_arg ~msg:"a periodic Hann advanced by its own length"
          "prepare: cannot invert a 8-point window advanced by 8 samples \
           inside a 8-point frame (the overlap-added squared window must stay \
           above 1e-10 of its largest value at every position)" (fun () ->
            Stft.Synthesis.prepare Nx.float64
              (Stft.Config.create ~fft_size:8 ~hop:8 ())
              Nx.complex128 ~channels:1 ~max_block:4 ) ;
        (* the same geometry with a window that does overlap-add is accepted *)
        no_raise ~msg:"a rectangular window at the frame boundary" (fun () ->
            Stft.Synthesis.prepare Nx.float64
              (Stft.Config.create ~window:Window.Rectangular ~fft_size:8 ~hop:8
                 () )
              Nx.complex128 ~channels:1 ~max_block:4 ) )
  ; test "prepare validates its allocation parameters" (fun () ->
        raises_invalid_arg ~msg:"channels"
          "prepare: cannot synthesise 0 channels (channels must be at least 1)"
          (fun () ->
            Stft.Synthesis.prepare Nx.float64 c8 Nx.complex128 ~channels:0
              ~max_block:4 ) ;
        raises_invalid_arg ~msg:"max_block"
          "prepare: cannot accept blocks of 0 frames (max_block must be at \
           least 1)" (fun () ->
            Stft.Synthesis.prepare Nx.float64 c8 Nx.complex128 ~channels:1
              ~max_block:0 ) )
  ; test "a drained kernel refuses frames until reset" (fun () ->
        let k = prepare8 () in
        let z = frames8 6 in
        is_some ~msg:"step emitted" (Stft.Synthesis.step k z) ;
        is_some ~msg:"flush emitted" (Stft.Synthesis.flush k) ;
        is_none ~msg:"a second flush is empty" (Stft.Synthesis.flush k) ;
        raises_invalid_arg ~msg:"drained step refuses frames"
          "step: cannot feed a drained kernel (flush consumed the tail; reset \
           before reusing)" (fun () -> Stft.Synthesis.step k z ) ;
        Stft.Synthesis.reset k ;
        is_some ~msg:"reset revives the kernel" (Stft.Synthesis.step k z) )
  ; test "reset replays the stream from scratch" (fun () ->
        let k = prepare8 () in
        let z = frames8 9 in
        let run () =
          let outs = ref [] in
          for p = 0 to 8 do
            match Stft.Synthesis.step k (shrink_last z p (p + 1)) with
            | None ->
                ()
            | Some out ->
                outs := out :: !outs
          done ;
          ( match Stft.Synthesis.flush k with
          | None ->
              ()
          | Some out ->
              outs := out :: !outs ) ;
          concat_last (List.rev !outs)
        in
        let first = bits (run ()) in
        Stft.Synthesis.reset k ;
        equal ~msg:"post-reset stream" bit_array first (bits (run ())) )
  ; test "the frame shape is checked before any transform" (fun () ->
        let k = prepare8 () in
        raises_invalid_arg ~msg:"rank below two"
          "step: cannot invert a rank-1 tensor (the bin and frame axes must \
           exist)" (fun () ->
            Stft.Synthesis.step k (Nx.zeros Nx.complex128 [|5|]) ) ;
        raises_invalid_arg ~msg:"wrong bin axis"
          "step: cannot invert 4 frequency bins of a 8-point transform (the \
           bin axis must hold fft_size / 2 + 1 = 5 values)" (fun () ->
            Stft.Synthesis.step k (Nx.zeros Nx.complex128 [|4; 3|]) ) ;
        raises_invalid_arg ~msg:"zero-size leading axis"
          "step: cannot synthesise frames with a zero-size leading axis \
           (channels must be at least 1)" (fun () ->
            Stft.Synthesis.step k (Nx.zeros Nx.complex128 [|0; 5; 3|]) ) )
  ; test "the emitted chunks own their samples" (fun () ->
        (* [step] borrows its frames: overwriting the caller's buffer after the
           call must not move a sample that has already been emitted *)
        let k = prepare8 () in
        let z = frames8 6 in
        let buffer = Nx.copy z in
        let out =
          match Stft.Synthesis.step k buffer with
          | Some out ->
              out
          | None ->
              fail "step must emit"
        in
        let snapshot = bits out in
        Nx.blit (Nx.mul_s z Complex.{re= -3.; im= 0.}) buffer ;
        equal ~msg:"the emitted chunk survives buffer reuse" bit_array snapshot
          (bits out) ) ]

let suite =
  [ group "synthesis-kernel-law" law_tests
  ; group "synthesis-kernel-length" length_tests
  ; group "synthesis-kernel-latency" latency_tests
  ; group "synthesis-kernel-contracts" contract_tests ]
