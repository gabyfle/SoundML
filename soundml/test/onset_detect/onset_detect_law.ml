(* The pipeline law for peak_pick_stage:

   Array.concat (List.filter_map (Stream.push s) chunks @ Stream.flush s) =
   peak_pick x

   exactly, over seeded partitionings — one-frame chunks, the whole envelope,
   random sizes and occasional empty chunks — at both dtypes, over envelopes
   with ties, NaN and infinities, under parameter sets from no lookahead to
   windows wider than the envelope, with every peak arriving on the push that
   completes its windows. Also here: librosa's greedy loop transcribed as an
   oracle for the picker, the properties of the selections, the detector against
   its parts, backtracking, the window defaults, the edges, and every error. *)

open Windtrap
open Soundml

let seed = 0x0d5e7

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

let frames_t = array int

type windows =
  { pre_max: int
  ; post_max: int
  ; pre_avg: int
  ; post_avg: int
  ; delta: float
  ; wait: int }

let windows (pre_max, post_max, pre_avg, post_avg, delta, wait) =
  {pre_max; post_max; pre_avg; post_avg; delta; wait}

let pick ?selection w x =
  peak_pick ?selection ~pre_max:w.pre_max ~post_max:w.post_max
    ~pre_avg:w.pre_avg ~post_avg:w.post_avg ~delta:w.delta ~wait:w.wait x

let stage w =
  peak_pick_stage ~pre_max:w.pre_max ~post_max:w.post_max ~pre_avg:w.pre_avg
    ~post_avg:w.post_avg ~delta:w.delta ~wait:w.wait

let lookahead w = Int.max w.post_max w.post_avg - 1

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let tensor dtype values = Nx.create dtype [|Array.length values|] values

(* {1 Envelopes and partitionings} *)

(* [values rng kind n] is [n] envelope values: continuous, quarters (full of
   exact ties), or small integers among which NaN and infinities. *)
let values rng kind n =
  Array.init n (fun _ ->
      match kind with
      | `Continuous ->
          Random.State.float rng 1.
      | `Quarters ->
          Float.of_int (Random.State.int rng 5) /. 4.
      | `Special -> (
        match Random.State.int rng 12 with
        | 0 ->
            Float.nan
        | 1 ->
            Float.infinity
        | 2 ->
            Float.neg_infinity
        | _ ->
            Float.of_int (Random.State.int rng 4) ) )

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (Int.min n 7) in
    s :: random_sizes rng (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

let partitions rng n =
  [("ones", List.init n (fun _ -> 1)); ("whole", if n = 0 then [] else [n])]
  @ List.init 3 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))
  @ List.init 2 (fun i ->
      ( Printf.sprintf "random-with-empty-chunks-%d" i
      , 0 :: sprinkle_empty rng (random_sizes rng n) ) )

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

(* [stream p chunks] pushes [chunks] and flushes, and is every peak in order
   with where it arrived: [`Push (before, after)], the frames seen before and
   after the push that emitted it, or [`Flush]. A push emitting an empty array
   fails the test. *)
let stream p chunks =
  let max_chunk = List.fold_left (fun m c -> Int.max m (last_dim c)) 1 chunks in
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk in
  let seen = ref 0 and out = ref [] in
  List.iter
    (fun c ->
      let before = !seen in
      seen := before + last_dim c ;
      match Pipeline.Stream.push s c with
      | None ->
          ()
      | Some [||] ->
          failf "a push emitted an empty array"
      | Some a ->
          Array.iter (fun f -> out := (f, `Push (before, !seen)) :: !out) a )
    chunks ;
  List.iter
    (Array.iter (fun f -> out := (f, `Flush) :: !out))
    (Pipeline.Stream.flush s) ;
  List.rev !out

(* [check_arrivals ~msg ~ahead ~n arrived] checks that each peak [f] arrived on
   the push that brought the frames seen to [f + ahead + 1], the first that
   completes its windows, or at the flush when the [n] frames never did. *)
let check_arrivals ~msg ~ahead ~n arrived =
  List.iter
    (fun (f, where) ->
      match where with
      | `Push (before, after) ->
          if not (before < f + ahead + 1 && f + ahead + 1 <= after) then
            failf "%s: peak %d arrived on the push from %d to %d frames" msg f
              before after
      | `Flush ->
          if f + ahead + 1 <= n then
            failf "%s: peak %d waited for the flush of %d frames" msg f n )
    arrived

(* {1 librosa's greedy loop}

   [__peak_pick_greedy] transcribed: frame 0 tested alone, then a scan stepping
   one frame past a failure and [wait + 1] frames past a peak, over numba's
   [np.max] (the first NaN it meets, else the largest value) and [np.mean] (zero
   plus the terms in order, over their count). The library keeps the candidates
   past the wait instead; the two must agree everywhere. *)

let np_max v lo hi =
  let rec go i m =
    if i >= hi then m
    else
      let a = v.(i) in
      if Float.is_nan a then a else go (i + 1) (if a > m then a else m)
  in
  if Float.is_nan v.(lo) then v.(lo) else go (lo + 1) v.(lo)

let np_mean v lo hi =
  let s = ref 0. in
  for i = lo to hi - 1 do
    s := !s +. v.(i)
  done ;
  !s /. Float.of_int (hi - lo)

let librosa_greedy w v =
  let len = Array.length v in
  let peaks = Array.make len false in
  if len > 0 then begin
    peaks.(0) <-
      v.(0) >= np_max v 0 (Int.min w.post_max len)
      && v.(0) >= np_mean v 0 (Int.min w.post_avg len) +. w.delta ;
    let n = ref (if peaks.(0) then w.wait + 1 else 1) in
    while !n < len do
      let maxn =
        np_max v (Int.max 0 (!n - w.pre_max)) (Int.min (!n + w.post_max) len)
      in
      peaks.(!n) <- v.(!n) = maxn ;
      if not peaks.(!n) then incr n
      else begin
        let avgn =
          np_mean v (Int.max 0 (!n - w.pre_avg)) (Int.min (!n + w.post_avg) len)
        in
        peaks.(!n) <- v.(!n) >= avgn +. w.delta ;
        if not peaks.(!n) then incr n else n := !n + w.wait + 1
      end
    done
  end ;
  let out = ref [] in
  for n = len - 1 downto 0 do
    if peaks.(n) then out := n :: !out
  done ;
  Array.of_list !out

(* [candidate w v n] is whether frame [n] of [v] (no NaN) is a local maximum
   clearing the threshold. *)
let candidate w v n =
  let len = Array.length v in
  let top =
    np_max v (Int.max 0 (n - w.pre_max)) (Int.min (n + w.post_max) len)
  in
  v.(n) = top
  && v.(n)
     >= np_mean v (Int.max 0 (n - w.pre_avg)) (Int.min (n + w.post_avg) len)
        +. w.delta

(* {1 The stage law} *)

let parameter_sets =
  [ ("online, no lookahead", (3, 1, 10, 1, 0.1, 3))
  ; ("librosa at 22050 Hz / 512", (1, 1, 4, 5, 0.07, 1))
  ; ("librosa at 16000 Hz / 160", (3, 1, 10, 11, 0.07, 3))
  ; ("librosa's docstring", (3, 3, 3, 5, 0.5, 10))
  ; ("the frame alone", (0, 1, 0, 1, 0., 0))
  ; ("lookahead on the maximum", (2, 7, 1, 3, 0., 2))
  ; ("windows wider than most envelopes", (50, 40, 60, 30, 0., 0)) ]

let kinds =
  [ (`Continuous, "continuous")
  ; (`Quarters, "quarters")
  ; (`Special, "NaN and infinities") ]

let lengths = [0; 1; 2; 5; 17; 64; 150]

let leads = [[||]; [|1|]; [|1; 1|]]

let law_case (type a) (dtype : (float, a) Nx.dtype) dname (pname, params)
    (kind, kname) =
  test (Printf.sprintf "%s, %s, %s" pname kname dname) (fun () ->
      let w = windows params in
      let p = stage w in
      let ahead = lookahead w in
      let rng = Random.State.make [|seed; ahead; w.pre_max; w.wait|] in
      List.iter
        (fun n ->
          List.iter
            (fun lead ->
              let v = values rng kind n in
              let x = Nx.create dtype (Array.append lead [|n|]) v in
              let image = Nx.to_array (Nx.cast Nx.float64 x) in
              let msg what =
                Printf.sprintf "%s, n %d, rank %d" what n (Nx.ndim x)
              in
              let expected = pick w x in
              equal ~msg:(msg "librosa's loop") frames_t
                (librosa_greedy w image) expected ;
              equal ~msg:(msg "run") frames_t expected
                (Pipeline.run ~source:(source ()) p x) ;
              List.iter
                (fun (part, sizes) ->
                  let arrived = stream p (chunks_of x sizes) in
                  equal ~msg:(msg part) frames_t expected
                    (Array.of_list (List.map fst arrived)) ;
                  check_arrivals ~msg:(msg part) ~ahead ~n arrived )
                (partitions rng n) )
            leads )
        lengths )

let law_tests =
  List.concat_map
    (fun params ->
      List.concat_map
        (fun kind ->
          [ law_case Nx.float64 "float64" params kind
          ; law_case Nx.float32 "float32" params kind ] )
        kinds )
    parameter_sets

let stage_tests =
  [ test "latency is max post_max post_avg - 1 frames, at the frame rate"
      (fun () ->
        let latency params = Pipeline.latency (stage (windows params)) in
        equal ~msg:"online" rate_t (r 0 1) (latency (3, 1, 10, 1, 0.1, 3)) ;
        equal ~msg:"librosa" rate_t (r 4 1) (latency (1, 1, 4, 5, 0.07, 1)) ;
        equal ~msg:"maximum" rate_t (r 6 1) (latency (2, 7, 1, 3, 0., 2)) ;
        equal ~msg:"rate" rate_t (r 1 1)
          (Pipeline.rate (stage (windows (1, 1, 4, 5, 0.07, 1)))) )
  ; test "reset restarts the frame count and the wait" (fun () ->
        let w = windows (1, 1, 2, 2, 0., 3) in
        let x = tensor Nx.float64 [|0.; 1.; 0.; 0.; 2.; 0.; 1.; 0.|] in
        let s =
          Pipeline.Stream.prepare (stage w) ~source:(source ()) ~max_chunk:8
        in
        let run () =
          let pushed = Option.to_list (Pipeline.Stream.push s x) in
          Array.concat (pushed @ Pipeline.Stream.flush s)
        in
        let first = run () in
        equal ~msg:"offline" frames_t (pick w x) first ;
        Pipeline.Stream.reset s ;
        equal ~msg:"after reset" frames_t first (run ()) )
  ; test "Böck et al.'s online picker decides each frame as it arrives"
      (fun () ->
        let w = windows (3, 1, 10, 1, 0.1, 3) in
        let x = tensor Nx.float32 [|0.; 0.5; 0.1; 0.; 0.; 0.9; 0.2; 0.1|] in
        let arrived =
          stream (stage w) (chunks_of x (List.init 8 (fun _ -> 1)))
        in
        equal ~msg:"peaks" frames_t (pick w x)
          (Array.of_list (List.map fst arrived)) ;
        List.iter
          (fun (f, where) ->
            match where with
            | `Push (_, after) ->
                equal ~msg:"arrival" int (f + 1) after
            | `Flush ->
                failf "peak %d waited for the flush" f )
          arrived ) ]

(* {1 The selections} *)

let selection_tests =
  let w = windows (2, 2, 3, 3, 0.1, 2) in
  let trials f =
    let rng = Random.State.make [|seed; 7|] in
    for trial = 0 to 199 do
      let n = Random.State.int rng 60 in
      f trial (values rng `Quarters n)
    done
  in
  let sum v peaks = Array.fold_left (fun s p -> s +. v.(p)) 0. peaks in
  let spaced msg peaks =
    Array.iteri
      (fun i p ->
        if i > 0 && p - peaks.(i - 1) <= w.wait then
          failf "%s: peaks %d and %d within the wait" msg peaks.(i - 1) p )
      peaks
  in
  [ test "greedy peaks are candidates, ascending, spaced past the wait"
      (fun () ->
        trials (fun trial v ->
            let peaks = pick w (tensor Nx.float64 v) in
            let msg = Printf.sprintf "trial %d" trial in
            spaced msg peaks ;
            Array.iter
              (fun p ->
                if not (candidate w v p) then
                  failf "%s: %d is no candidate" msg p )
              peaks ;
            (* every candidate the wait does not exclude is a peak *)
            let next = ref 0 in
            Array.iteri
              (fun n _ ->
                if n >= !next && candidate w v n then begin
                  if not (Array.mem n peaks) then failf "%s: %d skipped" msg n ;
                  next := n + w.wait + 1
                end )
              v ) )
  ; test "dp_count finds the most peaks, dp_value the largest sum" (fun () ->
        trials (fun trial v ->
            let x = tensor Nx.float64 v in
            let msg = Printf.sprintf "trial %d" trial in
            let greedy = pick w x in
            let count = pick ~selection:`Dp_count w x in
            let value = pick ~selection:`Dp_value w x in
            spaced (msg ^ " dp_count") count ;
            spaced (msg ^ " dp_value") value ;
            Array.iter
              (fun p ->
                if not (candidate w v p) then
                  failf "%s: %d is no candidate" msg p )
              (Array.append count value) ;
            if Array.length count < Array.length greedy then
              failf "%s: dp_count found fewer peaks than greedy" msg ;
            (* quarters: the sums are exact *)
            if sum v value < sum v greedy || sum v value < sum v count then
              failf "%s: dp_value found a smaller sum" msg ) )
  ; test "all frames candidates: greedy takes the earliest, dp the latest"
      (fun () ->
        let x = tensor Nx.float64 (Array.make 4 0.5) in
        let w = windows (1, 1, 1, 1, 0., 1) in
        equal ~msg:"greedy" frames_t [|0; 2|] (pick w x) ;
        equal ~msg:"dp_count" frames_t [|1; 3|] (pick ~selection:`Dp_count w x) ;
        equal ~msg:"dp_value" frames_t [|1; 3|] (pick ~selection:`Dp_value w x) )
  ; test "float32 envelopes are picked as their float64 image" (fun () ->
        let rng = Random.State.make [|seed; 32|] in
        List.iter
          (fun selection ->
            for _ = 1 to 50 do
              let x32 = tensor Nx.float32 (values rng `Continuous 80) in
              let x64 = Nx.cast Nx.float64 x32 in
              let w = windows (1, 1, 4, 5, 0.07, 1) in
              equal ~msg:"image" frames_t (pick ~selection w x64)
                (pick ~selection w x32)
            done )
          [`Greedy; `Dp_count; `Dp_value] ) ]

(* {1 The detector against its parts} *)

(* [defaults ~sample_rate ~hop] is librosa's windows in integer division. *)
let defaults ~sample_rate ~hop =
  let thirty = 3 * sample_rate / 100 / hop
  and hundred = sample_rate / 10 / hop in
  windows (thirty, 1, hundred, hundred + 1, 0.07, thirty)

let normalised v =
  let low = Array.fold_left Float.min v.(0) v in
  let y = Array.map (fun a -> a -. low) v in
  let d = Array.fold_left Float.max y.(0) y +. Float.min_float in
  Array.map (fun a -> a /. d) y

let geometries =
  List.concat_map
    (fun sample_rate ->
      List.map
        (fun hop -> (sample_rate, hop))
        [64; 128; 160; 256; 441; 480; 512; 1024] )
    [8000; 11025; 16000; 22050; 32000; 44100; 48000; 96000]

let spiky rng n =
  Array.init n (fun _ ->
      let a = Random.State.float rng 1. in
      a *. a *. a *. 3. )

let detector_tests =
  [ test "the windows default to librosa's durations in whole frames" (fun () ->
        List.iter
          (fun ((sample_rate, hop), (pre_max, pre_avg)) ->
            let w = defaults ~sample_rate ~hop in
            equal ~msg:"pre_max" int pre_max w.pre_max ;
            equal ~msg:"pre_avg" int pre_avg w.pre_avg )
          [ ((22050, 512), (1, 4))
          ; ((16000, 160), (3, 10))
          ; ((44100, 441), (3, 10))
          ; ((48000, 480), (3, 10))
          ; ((8000, 128), (1, 6))
          ; ((8000, 2400), (0, 0)) ] ;
        let rng = Random.State.make [|seed; 11|] in
        List.iter
          (fun (sample_rate, hop) ->
            let env = tensor Nx.float64 (spiky rng 300) in
            let msg = Printf.sprintf "%d Hz, hop %d" sample_rate hop in
            let w = defaults ~sample_rate ~hop in
            equal ~msg frames_t (pick w env)
              (onset_detect_of_envelope ~normalize:false ~hop ~sample_rate env) ;
            equal ~msg:(msg ^ ", normalised") frames_t
              (pick w (tensor Nx.float64 (normalised (Nx.to_array env))))
              (onset_detect_of_envelope ~hop ~sample_rate env) )
          geometries )
  ; test "backtracking composes the picker with onset_backtrack" (fun () ->
        let rng = Random.State.make [|seed; 13|] in
        for _ = 1 to 40 do
          let env = tensor Nx.float32 (spiky rng 200) in
          let energy = tensor Nx.float64 (spiky rng 230) in
          let picks normalize =
            onset_detect_of_envelope ~normalize ~sample_rate:16000 ~hop:160 env
          in
          equal ~msg:"energy" frames_t
            (onset_backtrack (picks true) energy)
            (onset_detect_of_envelope ~backtrack:(`Energy energy)
               ~sample_rate:16000 ~hop:160 env ) ;
          equal ~msg:"raw envelope" frames_t
            (onset_backtrack (picks false) env)
            (onset_detect_of_envelope ~normalize:false ~backtrack:`Envelope
               ~sample_rate:16000 ~hop:160 env ) ;
          equal ~msg:"normalised envelope" frames_t
            (onset_backtrack (picks true)
               (tensor Nx.float64
                  (normalised (Nx.to_array (Nx.cast Nx.float64 env))) ) )
            (onset_detect_of_envelope ~backtrack:`Envelope ~sample_rate:16000
               ~hop:160 env )
        done )
  ; test "scaling by a power of two leaves the normalised onsets" (fun () ->
        let rng = Random.State.make [|seed; 17|] in
        for _ = 1 to 40 do
          let v = spiky rng 150 in
          let detect scale =
            onset_detect_of_envelope ~sample_rate:22050
              (tensor Nx.float64 (Array.map (fun a -> a *. scale) v))
          in
          equal ~msg:"x4" frames_t (detect 1.) (detect 4.) ;
          equal ~msg:"/8" frames_t (detect 1.) (detect 0.125)
        done )
  ; test "from audio, the detector is the envelope's" (fun () ->
        let rng = Random.State.make [|seed; 19|] in
        List.iter
          (fun (sample_rate, fft_size, hop, n_mels) ->
            let stft =
              Stft.Config.create ~fft_size ~hop ~pad:(`Constant 0.) ()
            in
            let mel = Mel.Config.create ~n_mels ~sample_rate ~fft_size () in
            let n = sample_rate / 2 in
            let y =
              Array.init n (fun i ->
                  let burst =
                    if i mod (sample_rate / 8) < 300 then 0.5 else 0.01
                  in
                  burst *. (Random.State.float rng 2. -. 1.) )
            in
            let check (type a) (dtype : (float, a) Nx.dtype) =
              let x = tensor dtype y in
              let env = onset_strength stft mel x in
              equal ~msg:"defaults" frames_t
                (onset_detect_of_envelope ~hop ~sample_rate env)
                (onset_detect stft mel x) ;
              equal ~msg:"options" frames_t
                (onset_detect_of_envelope ~hop ~backtrack:`Envelope
                   ~selection:`Dp_count ~wait:2 ~delta:0.1 ~sample_rate env )
                (onset_detect stft mel ~backtrack:`Envelope ~selection:`Dp_count
                   ~wait:2 ~delta:0.1 x ) ;
              equal ~msg:"one signal of rank two" frames_t
                (onset_detect stft mel x)
                (onset_detect stft mel (Nx.reshape [|1; n|] x)) ;
              if Array.length (onset_detect stft mel x) = 0 then
                failf "no onset in the bursts"
            in
            check Nx.float64 ; check Nx.float32 )
          [(22050, 2048, 512, 128); (16000, 512, 160, 40)] ) ]

(* {1 Backtracking} *)

let backtrack_tests =
  let is_minimum e i =
    let m = Array.length e in
    i = 0 || (i >= 1 && i <= m - 2 && e.(i) <= e.(i - 1) && e.(i) < e.(i + 1))
  in
  [ test "each event moves to the latest minimum at or before it" (fun () ->
        let rng = Random.State.make [|seed; 23|] in
        for trial = 0 to 199 do
          let m = Random.State.int rng 40 in
          let e =
            values rng (if trial mod 2 = 0 then `Quarters else `Continuous) m
          in
          let events =
            Array.init (Random.State.int rng 12) (fun _ ->
                Random.State.int rng (m + 5) )
          in
          let got = onset_backtrack events (tensor Nx.float64 e) in
          equal ~msg:"length" int (Array.length events) (Array.length got) ;
          Array.iteri
            (fun i k ->
              let b = got.(i) in
              let msg = Printf.sprintf "trial %d, event %d" trial k in
              if b > k || not (is_minimum e b) then failf "%s: %d" msg b ;
              for j = b + 1 to Int.min k (m - 1) do
                if is_minimum e j then failf "%s: minimum %d after %d" msg j b
              done )
            events ;
          equal ~msg:"idempotent" frames_t got
            (onset_backtrack got (tensor Nx.float64 e)) ;
          equal ~msg:"float32" frames_t
            (onset_backtrack events (Nx.cast Nx.float64 (tensor Nx.float32 e)))
            (onset_backtrack events (tensor Nx.float32 e))
        done )
  ; test "sorted events stay sorted; events past the end take the last minimum"
      (fun () ->
        let e =
          tensor Nx.float64 [|5.; 3.; 4.; 2.; 2.; 6.; 1.; 0.; 0.; 3.; 7.|]
        in
        equal ~msg:"every frame" frames_t
          [|0; 1; 1; 1; 4; 4; 4; 4; 8; 8; 8; 8; 8|]
          (onset_backtrack (Array.init 13 Fun.id) e) ;
        equal ~msg:"order kept" frames_t [|8; 1; 4; 1|]
          (onset_backtrack [|10; 2; 5; 2|] e) ;
        equal ~msg:"rms output [1; frames]" frames_t [|8; 1; 4; 1|]
          (onset_backtrack [|10; 2; 5; 2|] (Nx.reshape [|1; 11|] e)) ) ]

(* {1 Edges} *)

let edge_tests =
  let detect ?normalize ?delta values =
    onset_detect_of_envelope ?normalize ?delta ~sample_rate:22050
      (tensor Nx.float64 values)
  in
  [ test "an envelope with nothing to pick has no onset" (fun () ->
        equal ~msg:"empty" frames_t [||] (detect [||]) ;
        equal ~msg:"empty, raw" frames_t [||] (detect ~normalize:false [||]) ;
        equal ~msg:"zeros" frames_t [||] (detect (Array.make 20 0.)) ;
        equal ~msg:"zeros, raw, delta 0" frames_t [||]
          (detect ~normalize:false ~delta:0. (Array.make 20 0.)) ;
        equal ~msg:"constant" frames_t [||] (detect (Array.make 20 2.5)) ;
        equal ~msg:"NaN" frames_t [||] (detect [|0.; 1.; Float.nan; 0.; 2.|]) ;
        equal ~msg:"infinity, raw" frames_t [||]
          (detect ~normalize:false [|0.; Float.infinity; 0.; 2.|]) ;
        equal ~msg:"constant, raw, delta 0: every frame past the wait" frames_t
          [|0; 2; 4|]
          (detect ~normalize:false ~delta:0. (Array.make 6 2.5)) )
  ; test "no onset to backtrack gives no frames" (fun () ->
        equal ~msg:"backtrack" frames_t [||]
          (onset_detect_of_envelope ~delta:5. ~backtrack:`Envelope
             ~sample_rate:22050
             (tensor Nx.float64 [|0.; 1.; 0.; 2.; 0.|]) ) ;
        equal ~msg:"no events" frames_t [||]
          (onset_backtrack [||] (tensor Nx.float64 [|1.; 0.; 1.|])) )
  ; test "peak_pick keeps the reference's NaN and infinities" (fun () ->
        let w = windows (1, 1, 1, 1, 0., 0) in
        let x =
          tensor Nx.float64 [|0.; 1.; Float.nan; 0.; 2.; 0.; Float.infinity; 0.|]
        in
        (* a NaN rejects every frame whose maximum's window holds it, and the
           infinite frame equals its maximum and its mean *)
        equal ~msg:"greedy" frames_t [|0; 1; 4; 6|] (pick w x) ;
        equal ~msg:"empty" frames_t [||] (pick w (tensor Nx.float64 [||])) )
  ; test "windows of any size do not overflow" (fun () ->
        let x = tensor Nx.float64 [|0.; 3.; 1.; 3.; 0.; 2.|] in
        let huge = windows (max_int, max_int, max_int, max_int, 0., max_int) in
        let whole = windows (6, 6, 6, 6, 0., 6) in
        equal ~msg:"greedy" frames_t (pick whole x) (pick huge x) ;
        equal ~msg:"dp_count" frames_t
          (pick ~selection:`Dp_count whole x)
          (pick ~selection:`Dp_count huge x) ;
        equal ~msg:"first peak only" frames_t [|1|] (pick huge x) ) ]

(* {1 Errors} *)

let error_tests =
  let x = tensor Nx.float64 [|0.; 1.; 0.|] in
  let w = windows (1, 1, 1, 1, 0., 0) in
  let raises_msg msg expected f = raises ~msg (Invalid_argument expected) f in
  [ test "peak_pick validates its windows" (fun () ->
        raises_msg "pre_max"
          "peak_pick: cannot look back -1 frames for the local maximum \
           (pre_max must be non-negative)" (fun () ->
            ignore (pick {w with pre_max= -1} x) ) ;
        raises_msg "post_max"
          "peak_pick: cannot end the local maximum's window 0 frames after its \
           frame (post_max must be at least 1)" (fun () ->
            ignore (pick {w with post_max= 0} x) ) ;
        raises_msg "pre_avg"
          "peak_pick: cannot look back -2 frames for the local mean (pre_avg \
           must be non-negative)" (fun () ->
            ignore (pick {w with pre_avg= -2} x) ) ;
        raises_msg "post_avg"
          "peak_pick: cannot end the local mean's window 0 frames after its \
           frame (post_avg must be at least 1)" (fun () ->
            ignore (pick {w with post_avg= 0} x) ) ;
        raises_msg "delta"
          "peak_pick: cannot threshold at -0.5 above the local mean (delta \
           must be finite and non-negative)" (fun () ->
            ignore (pick {w with delta= -0.5} x) ) ;
        raises_msg "delta nan"
          "peak_pick: cannot threshold at nan above the local mean (delta must \
           be finite and non-negative)" (fun () ->
            ignore (pick {w with delta= Float.nan} x) ) ;
        raises_msg "wait"
          "peak_pick: cannot wait -1 frames after a peak (wait must be \
           non-negative)" (fun () -> ignore (pick {w with wait= -1} x) ) )
  ; test "peak_pick reads one signal of float32 or float64" (fun () ->
        raises_msg "rank"
          "peak_pick: cannot pick peaks in a rank-zero tensor (the frame axis \
           must exist)" (fun () -> ignore (pick w (Nx.scalar Nx.float64 1.)) ) ;
        raises_msg "batch"
          "peak_pick: cannot read a tensor of leading shape [2] as one signal \
           (every leading axis must have size 1)" (fun () ->
            ignore (pick w (Nx.zeros Nx.float64 [|2; 3|])) ) ;
        raises_msg "dtype"
          "peak_pick: cannot pick peaks in this dtype (float32 and float64 \
           only)" (fun () -> ignore (pick w (Nx.zeros Nx.float16 [|3|])) ) )
  ; test "peak_pick_stage validates when built and at each chunk" (fun () ->
        raises_msg "built"
          "peak_pick_stage: cannot wait -3 frames after a peak (wait must be \
           non-negative)" (fun () -> ignore (stage {w with wait= -3}) ) ;
        let s =
          Pipeline.Stream.prepare (stage w) ~source:(source ()) ~max_chunk:4
        in
        raises_msg "chunk"
          "peak_pick_stage: cannot read a tensor of leading shape [1; 2] as \
           one signal (every leading axis must have size 1)" (fun () ->
            ignore (Pipeline.Stream.push s (Nx.zeros Nx.float64 [|1; 2; 4|])) ) ;
        raises_msg "rank"
          "peak_pick_stage: cannot pick peaks in a rank-zero tensor (the frame \
           axis must exist)" (fun () ->
            ignore (Pipeline.Stream.push s (Nx.scalar Nx.float64 1.)) ) )
  ; test "the detector validates its rate, hop, windows and curves" (fun () ->
        let env = tensor Nx.float64 [|0.; 1.; 0.; 2.; 0.|] in
        raises_msg "sample_rate"
          "onset_detect_of_envelope: cannot derive the windows at 0 Hz \
           (sample_rate must be at least 1)" (fun () ->
            ignore (onset_detect_of_envelope ~sample_rate:0 env) ) ;
        raises_msg "hop"
          "onset_detect_of_envelope: cannot advance frames by 0 samples (hop \
           must be at least 1)" (fun () ->
            ignore (onset_detect_of_envelope ~hop:0 ~sample_rate:22050 env) ) ;
        raises_msg "window"
          "onset_detect_of_envelope: cannot end the local mean's window 0 \
           frames after its frame (post_avg must be at least 1)" (fun () ->
            ignore (onset_detect_of_envelope ~post_avg:0 ~sample_rate:22050 env) ) ;
        raises_msg "validated even with nothing to pick"
          "onset_detect_of_envelope: cannot wait -1 frames after a peak (wait \
           must be non-negative)" (fun () ->
            ignore
              (onset_detect_of_envelope ~wait:(-1) ~sample_rate:22050
                 (tensor Nx.float64 [||]) ) ) ;
        raises_msg "energy"
          "onset_detect_of_envelope: cannot backtrack along a rank-zero tensor \
           (the frame axis must exist)" (fun () ->
            ignore
              (onset_detect_of_envelope
                 ~backtrack:(`Energy (Nx.scalar Nx.float32 1.))
                 ~sample_rate:22050 env ) ) ;
        raises_msg "envelope batch"
          "onset_detect_of_envelope: cannot read a tensor of leading shape [2; \
           1] as one signal (every leading axis must have size 1)" (fun () ->
            ignore
              (onset_detect_of_envelope ~sample_rate:22050
                 (Nx.zeros Nx.float64 [|2; 1; 5|]) ) ) )
  ; test "onset_detect validates before any analysis" (fun () ->
        let stft =
          Stft.Config.create ~fft_size:512 ~hop:128 ~pad:(`Constant 0.) ()
        in
        let mel =
          Mel.Config.create ~n_mels:40 ~sample_rate:16000 ~fft_size:512 ()
        in
        let other =
          Mel.Config.create ~n_mels:40 ~sample_rate:16000 ~fft_size:1024 ()
        in
        let audio = Nx.zeros Nx.float64 [|4000|] in
        raises_msg "sizes"
          "onset_detect: cannot project a 512-point STFT through a filterbank \
           built for an FFT of size 1024 (the two configurations must agree on \
           fft_size)" (fun () -> ignore (onset_detect stft other audio) ) ;
        raises_msg "rank"
          "onset_detect: cannot analyse a rank-zero tensor (the time axis must \
           exist)" (fun () ->
            ignore (onset_detect stft mel (Nx.scalar Nx.float64 0.)) ) ;
        raises_msg "stereo"
          "onset_detect: cannot read a tensor of leading shape [2] as one \
           signal (every leading axis must have size 1)" (fun () ->
            ignore (onset_detect stft mel (Nx.zeros Nx.float64 [|2; 4000|])) ) ;
        raises_msg "windows"
          "onset_detect: cannot look back -1 frames for the local maximum \
           (pre_max must be non-negative)" (fun () ->
            ignore (onset_detect stft mel ~pre_max:(-1) audio) ) )
  ; test "onset_backtrack validates its events and energy" (fun () ->
        let e = tensor Nx.float64 [|1.; 0.; 1.|] in
        raises_msg "event"
          "onset_backtrack: cannot backtrack an event at frame -1 (events must \
           be non-negative)" (fun () -> ignore (onset_backtrack [|2; -1|] e) ) ;
        raises_msg "rank"
          "onset_backtrack: cannot backtrack along a rank-zero tensor (the \
           frame axis must exist)" (fun () ->
            ignore (onset_backtrack [|0|] (Nx.scalar Nx.float64 1.)) ) ;
        raises_msg "batch"
          "onset_backtrack: cannot read a tensor of leading shape [3] as one \
           signal (every leading axis must have size 1)" (fun () ->
            ignore (onset_backtrack [|0|] (Nx.zeros Nx.float64 [|3; 3|])) ) ) ]

let suite =
  [ group "stage law" law_tests
  ; group "stage" stage_tests
  ; group "selections" selection_tests
  ; group "detector" detector_tests
  ; group "backtracking" backtrack_tests
  ; group "edges" edge_tests
  ; group "errors" error_tests ]
