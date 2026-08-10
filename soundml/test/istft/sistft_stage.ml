(* The round trip as a pipeline: [Stft.stage c >> Stft.synthesis_stage c].

   - the law, bit for bit: streaming the chain over any partitioning of the
   signal — one sample at a time included — gives exactly [invert (transform
   x)], the offline round trip at its default length. The analysis kernel's own
   law puts the frames of the stream on the grid [transform] uses, and the
   synthesis kernel's law turns any chunking of those frames into the same
   samples, so the composition inherits both.

   - the reported lookahead is the sum of the two declarations, [latency (stage
   c) + Config.synthesis_latency c] source samples, one made on the input side
   of the analysis and one on the output side of the synthesis. Brute force
   measures what the chain actually does — when its first sample appears and how
   far its emission ever trails the input — and the two are compared position by
   position, including where the declaration is a bound rather than an equality
   and where [`Left] under-reports, which the interface states.

   - the chain is the identity on the interior of the round trip, the positions
   the analysis windows fully cover, to the bound the offline suite proves.

   The signal grid is the configuration grid of {!Sistft_law}, driven at n =
   400. *)

open Windtrap
open Soundml

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

let concat_last = function
  | [] ->
      Nx.zeros Nx.float64 [|0|]
  | parts ->
      Nx.concatenate ~axis:(-1) parts

let bits t = Array.map Int64.bits_of_float (Nx.to_array t)

let bit_array = array int64

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

let r num den = {Pipeline.Rate.num; den}

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:1000 ~channels:1

let grid = Sistft_law.grid

let round_trip c =
  Pipeline.(Stft.stage Nx.complex128 c >> Stft.synthesis_stage Nx.float64 c)

let n_signal = 400

let signal = Nx.create Nx.float64 [|n_signal|] (lcg n_signal)

let offline c = Stft.invert Nx.float64 c (Stft.transform Nx.complex128 c signal)

(* {1 Partitionings of the signal} *)

let partitions n =
  let rng = Random.State.make [|0x157f|] in
  let rec random_sizes n =
    if n = 0 then []
    else
      let s = 1 + Random.State.int rng (Stdlib.min n 37) in
      s :: random_sizes (n - s)
  in
  let rec cycle sizes acc left =
    if left <= 0 then List.rev acc
    else
      match sizes with
      | [] ->
          cycle [3; 1; 4; 1; 5; 9; 2; 6] acc left
      | s :: rest ->
          let s = Stdlib.min s left in
          cycle rest (s :: acc) (left - s)
  in
  [ ("ones", List.init n (fun _ -> 1))
  ; ("whole", [n])
  ; ("straddle", cycle [] [] n)
  ; ("giant-then-ones", [n - 5; 1; 1; 1; 1; 1])
  ; ("prime-97", cycle [97] [] n) ]
  @ List.init 3 (fun i -> (Printf.sprintf "random-%d" i, random_sizes n))

let stream_chain c sizes =
  let s =
    Pipeline.Stream.prepare (round_trip c) ~source:(source ())
      ~max_chunk:(List.fold_left Stdlib.max 1 sizes)
  in
  let outs = ref [] and pos = ref 0 in
  List.iter
    (fun size ->
      let chunk = shrink_last signal !pos (!pos + size) in
      pos := !pos + size ;
      match Pipeline.Stream.push s chunk with
      | None ->
          ()
      | Some out ->
          outs := out :: !outs )
    sizes ;
  List.iter (fun out -> outs := out :: !outs) (Pipeline.Stream.flush s) ;
  concat_last (List.rev !outs)

let law_case (name, c) =
  test name (fun () ->
      let expected = bits (offline c) in
      equal ~msg:"run is the offline round trip" bit_array expected
        (bits (Pipeline.run ~source:(source ()) (round_trip c) signal)) ;
      List.iter
        (fun (pname, sizes) ->
          equal
            ~msg:(Printf.sprintf "streamed %s" pname)
            bit_array expected
            (bits (stream_chain c sizes)) )
        (partitions n_signal) )

let law_tests = List.map law_case grid

(* {1 The declared lookahead}

   [install_threshold] is the raw samples the analysis buffers before its left
   border is computable: reflection reads [left + 1] of them, a constant or edge
   extension only the first. The analysis stage declares the larger of that
   reach and the geometric lookahead; the synthesis stage declares its two
   trims; the composition adds them. *)

let install_threshold c =
  match Stft.Config.pad c with
  | `Reflect ->
      Sistft_law.left_width (Stft.Config.fft_size c) (Stft.Config.alignment c)
      + 1
  | `Constant _ | `Edge ->
      1

let declared c =
  Stdlib.max (Stft.Config.latency c) (install_threshold c - 1)
  + Stft.Config.synthesis_latency c

(* The first input sample after which the chain has emitted anything: frame [p]
   releases padded positions up to [p * hop + min (hop, fft_size - R)], so the
   head trim [L] is paid at the first [p] whose release passes it, and that
   frame needs [p * hop + fft_size - L] samples — unless the border itself needs
   more. *)
let first_emission c =
  let fft = Stft.Config.fft_size c and hop = Stft.Config.hop c in
  let alignment = Stft.Config.alignment c in
  let l = Sistft_law.left_width fft alignment in
  let e = Stdlib.min hop (fft - Sistft_law.right_width fft alignment) in
  let p = ref 0 in
  while (!p * hop) + e <= l do
    incr p
  done ;
  Stdlib.max (install_threshold c) ((!p * hop) + fft - l)

let measure c =
  let s =
    Pipeline.Stream.prepare (round_trip c) ~source:(source ()) ~max_chunk:1
  in
  let emitted = ref 0 and first = ref (-1) and worst = ref 0 in
  for i = 0 to n_signal - 1 do
    ( match Pipeline.Stream.push s (shrink_last signal i (i + 1)) with
    | None ->
        ()
    | Some out ->
        emitted := !emitted + last_dim out ) ;
    if !emitted > 0 && !first < 0 then first := i + 1 ;
    if i + 1 - !emitted > !worst then worst := i + 1 - !emitted
  done ;
  ignore (Pipeline.Stream.flush s) ;
  (!first, !worst)

let latency_tests =
  [ test "the composition reports the sum of the two declarations" (fun () ->
        List.iter
          (fun (name, c) ->
            equal ~msg:name rate_t
              (r (declared c) 1)
              (Pipeline.latency (round_trip c)) ;
            equal ~msg:(name ^ ": rate") rate_t (r 1 1)
              (Pipeline.rate (round_trip c)) )
          grid )
  ; test "each alignment reports what its geometry declares" (fun () ->
        let centered_even = Stft.Config.create ~fft_size:16 ~hop:4 () in
        equal ~msg:"centered even: fft_size + 0" rate_t (r 16 1)
          (Pipeline.latency (round_trip centered_even)) ;
        let centered_odd = Stft.Config.create ~fft_size:9 ~hop:4 () in
        equal ~msg:"centered odd: fft_size - 1" rate_t (r 8 1)
          (Pipeline.latency (round_trip centered_odd)) ;
        let left = Stft.Config.create ~alignment:`Left ~fft_size:16 ~hop:4 () in
        equal ~msg:"left declares nothing, at either end" rate_t (r 0 1)
          (Pipeline.latency (round_trip left)) ;
        let right_causal =
          Stft.Config.create ~alignment:`Right ~pad:(`Constant 0.) ~fft_size:16
            ~hop:4 ()
        in
        equal ~msg:"right with a causal border: fft_size - 1" rate_t (r 15 1)
          (Pipeline.latency (round_trip right_causal)) ;
        let right_reflect =
          Stft.Config.create ~alignment:`Right ~fft_size:16 ~hop:4 ()
        in
        equal ~msg:"right with a reflecting border: 2 * fft_size - 2" rate_t
          (r 30 1)
          (Pipeline.latency (round_trip right_reflect)) )
  ; test "brute force: the first sample lands where the formula says" (fun () ->
        List.iter
          (fun (name, c) ->
            let first, _ = measure c in
            equal ~msg:name int (first_emission c) first )
          grid ;
        List.iter
          (fun (name, c) ->
            let first, _ = measure c in
            equal ~msg:name int (first_emission c) first )
          [ ( "right constant fft16 hop4"
            , Stft.Config.create ~alignment:`Right ~pad:(`Constant 0.)
                ~fft_size:16 ~hop:4 () )
          ; ( "right edge fft9 hop2"
            , Stft.Config.create ~alignment:`Right ~pad:`Edge ~fft_size:9 ~hop:2
                () ) ] )
  ; test "brute force: the deficit the chain really carries" (fun () ->
        (* the physical truth, independent of what is declared: the round trip
           trails its input by [fft_size - 1] samples plus whatever overshoot of
           one hop past the trailing trim the synthesis holds *)
        List.iter
          (fun (name, c) ->
            let fft = Stft.Config.fft_size c and hop = Stft.Config.hop c in
            let alignment = Stft.Config.alignment c in
            let overshoot =
              Stdlib.max 0 (hop + Sistft_law.right_width fft alignment - fft)
            in
            let _, worst = measure c in
            equal ~msg:name int (fft - 1 + overshoot) worst )
          grid )
  ; test "the declaration bounds the deficit, or is documented not to"
      (fun () ->
        List.iter
          (fun (name, c) ->
            let _, worst = measure c in
            match Stft.Config.alignment c with
            | `Left ->
                (* the one under-report, inherited from Config.latency *)
                equal ~msg:(name ^ ": left declares zero") int 0 (declared c) ;
                is_true
                  ~msg:(name ^ ": and the truth is fft_size - 1")
                  (worst = Stft.Config.fft_size c - 1)
            | `Centered | `Right ->
                is_true
                  ~msg:
                    (Printf.sprintf "%s: declared %d covers the deficit %d" name
                       (declared c) worst )
                  (declared c >= worst) )
          grid ) ]

(* {1 Threading and the drain bound} *)

let probe seen =
  Pipeline.kernel
    ~concat:(function
      | [] ->
          Nx.zeros Nx.float64 [|0|]
      | parts ->
          Nx.concatenate ~axis:(-1) parts )
    ~prepare:(fun fmt -> seen := Some fmt)
    ~step:(fun () (c : (float, Nx.float64_elt) Nx.t) -> Some c)
    ()

let threading_tests =
  [ test "the threaded format carries the synthesis dtype and rate" (fun () ->
        let seen = ref None in
        let c = Stft.Config.create ~fft_size:16 ~hop:4 () in
        let p = Pipeline.(round_trip c >> probe seen) in
        ignore (Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:100) ;
        match !seen with
        | None ->
            fail "probe prepare did not run"
        | Some fmt ->
            equal ~msg:"back at the source rate" rate_t (r 1000 1)
              (Pipeline.Format.items_per_second fmt) ;
            is_true ~msg:"float64 samples"
              (Pipeline.Format.equal_dtype
                 (Pipeline.Format.dtype fmt)
                 (Pipeline.Format.Dtype Nx.float64) ) ;
            equal ~msg:"upstream latency in source samples" rate_t
              (r (declared c) 1)
              (Pipeline.Format.upstream_latency fmt) )
  ; test "every chunk fits the threaded bound, drain included" (fun () ->
        List.iter
          (fun (name, c) ->
            let seen = ref None in
            let p = Pipeline.(round_trip c >> probe seen) in
            let s =
              Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:7
            in
            let bound =
              match !seen with
              | Some fmt ->
                  Option.get (Pipeline.Format.max_items fmt)
              | None ->
                  fail "probe prepare did not run"
            in
            let check where out =
              is_true
                ~msg:
                  (Printf.sprintf "%s: %s chunk of %d fits the bound %d" name
                     where (last_dim out) bound )
                (last_dim out <= bound)
            in
            let pos = ref 0 in
            while !pos < n_signal do
              let stop = Stdlib.min n_signal (!pos + 7) in
              ( match Pipeline.Stream.push s (shrink_last signal !pos stop) with
              | None ->
                  ()
              | Some out ->
                  check "push" out ) ;
              pos := stop
            done ;
            List.iter (check "flush") (Pipeline.Stream.flush s) )
          grid )
  ; test "reset replays the chain from scratch" (fun () ->
        let c = Stft.Config.create ~fft_size:16 ~hop:5 ~win_length:12 () in
        let s =
          Pipeline.Stream.prepare (round_trip c) ~source:(source ())
            ~max_chunk:64
        in
        ignore (Pipeline.Stream.push s (shrink_last signal 0 33)) ;
        Pipeline.Stream.reset s ;
        let outs = ref [] and pos = ref 0 in
        while !pos < n_signal do
          let stop = Stdlib.min n_signal (!pos + 64) in
          ( match Pipeline.Stream.push s (shrink_last signal !pos stop) with
          | None ->
              ()
          | Some out ->
              outs := out :: !outs ) ;
          pos := stop
        done ;
        List.iter (fun out -> outs := out :: !outs) (Pipeline.Stream.flush s) ;
        equal ~msg:"post-reset round trip" bit_array
          (bits (offline c))
          (bits (concat_last (List.rev !outs))) )
  ; test "an uninvertible geometry is rejected at the stage" (fun () ->
        raises_invalid_arg ~msg:"a periodic Hann advanced by its own length"
          "synthesis_stage: cannot invert a 8-point window advanced by 8 \
           samples inside a 8-point frame (the overlap-added squared window \
           must stay above 1e-10 of its largest value at every position)"
          (fun () ->
            ignore
              (Stft.synthesis_stage Nx.float64
                 (Stft.Config.create ~fft_size:8 ~hop:8 ()) ) ) ) ]

(* {1 The identity on the interior}

   The round trip returns the signal wherever the frame pattern covers it: the
   padded positions from [fft_size - hop] up to [frames * hop], which in signal
   coordinates start at [max 0 (fft_size - hop - L)] and end at [frames * hop -
   L]. Outside it the reconstruction is the ill-conditioned edge {!Stft.invert}
   documents, and the offline suite carries the bound this one inherits. *)

let interior_tests =
  [ test "the streamed chain is the identity on the interior" (fun () ->
        List.iter
          (fun (name, c) ->
            let fft = Stft.Config.fft_size c and hop = Stft.Config.hop c in
            let l = Sistft_law.left_width fft (Stft.Config.alignment c) in
            let frames = Stft.frames c ~n:n_signal in
            let lo = Stdlib.max 0 (fft - hop - l) in
            let got = stream_chain c [n_signal] in
            let hi = Stdlib.min (last_dim got) ((frames * hop) - l) in
            if hi > lo then
              Tutils.check_close ~rtol:0. ~atol:1e-12 ~msg:name
                ~expected:(Array.sub (Nx.to_array signal) lo (hi - lo))
                (Nx.shrink [|(lo, hi)|] got) )
          grid ) ]

let suite =
  [ group "round-trip-chain-law" law_tests
  ; group "round-trip-chain-latency" latency_tests
  ; group "round-trip-chain-threading" threading_tests
  ; group "round-trip-chain-interior" interior_tests ]
