(* Laws without an oracle.

   autocorrelate: a shorter max_size keeps exactly the first lags, a row is the
   same bits alone or in a batch, float32 is the double lags rounded once, and
   the lags agree with the direct sums.

   lpc: a known AR(2) is recovered, the degenerate signals give their exact
   filters, short signals extend by zeros, a row is the same bits alone or in a
   batch, and float32 is the double coefficients rounded once.

   poly_features: the pipeline law for poly_features_stage —

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   exactly, over seeded partitionings (one-frame chunks, the whole spectrogram,
   random sizes and occasional empty chunks), at even and odd FFT sizes and both
   dtypes, and the chain with Stft.power_stage against the audio face; the faces
   against each other; per-frame independence; exact polynomials and means.

   And the errors of every precondition. *)

open Windtrap
open Soundml

let seed = 0xa7a1

let lcg_values ?(seed = 20261103) n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let noise ?seed dtype shape =
  Nx.create dtype shape (lcg_values ?seed (Array.fold_left ( * ) 1 shape))

let bits t = Nx.to_array (Nx.cast Nx.float64 t)

let same ~msg a b = equal ~msg (array float_exact) (bits a) (bits b)

let row t i =
  let nd = Nx.ndim t in
  Nx.slice (List.init nd (fun d -> if d = 0 then Nx.I i else Nx.A)) t

let last_range t lo hi =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun d -> if d = nd - 1 then Nx.R (lo, hi) else Nx.A))
    t

(* {1 autocorrelate} *)

(* [direct x] is every lag of the rank-one [x] as a direct double sum in index
   order. *)
let direct x =
  let v = bits x in
  let n = Array.length v in
  Array.init n (fun k ->
      let s = ref 0. in
      for t = 0 to n - 1 - k do
        s := !s +. (v.(t) *. v.(t + k))
      done ;
      !s )

(* Measured worst disagreement between the transform and the direct sums over
   the signals below: 1.3e-15 of lag 0, on the tone, whose direct sums
   accumulate the larger share of it; the bound is ten times that. *)
let direct_tol = 1.3e-14

let autocorrelate_tests =
  [ test "a shorter max_size keeps exactly the first lags" (fun () ->
        List.iter
          (fun n ->
            let x = noise Nx.float64 [|2; n|] in
            let full = autocorrelate x in
            List.iter
              (fun m ->
                same
                  ~msg:(Printf.sprintf "n %d, max_size %d" n m)
                  (last_range full 0 (Int.min m n))
                  (autocorrelate ~max_size:m x) )
              [1; 2; n / 3; n - 1; n; n + 5] ;
            let x32 = noise Nx.float32 [|n|] in
            same
              ~msg:(Printf.sprintf "float32, n %d" n)
              (last_range (autocorrelate x32) 0 (n / 2))
              (autocorrelate ~max_size:(n / 2) x32) )
          [7; 64; 1009] )
  ; test "a row is the same bits alone or in a batch" (fun () ->
        let x = noise Nx.float64 [|2; 3; 257|] in
        let batch = autocorrelate ~max_size:40 x in
        for i = 0 to 1 do
          for j = 0 to 2 do
            same
              ~msg:(Printf.sprintf "row %d %d" i j)
              (row (row batch i) j)
              (autocorrelate ~max_size:40 (row (row x i) j))
          done
        done )
  ; test "float32 is the double lags rounded once" (fun () ->
        let x = noise Nx.float32 [|3; 500|] in
        same ~msg:"float32"
          (Nx.cast Nx.float32 (autocorrelate (Nx.cast Nx.float64 x)))
          (autocorrelate x) )
  ; test "the lags are the direct sums" (fun () ->
        let tone =
          Nx.create Nx.float64 [|600|]
            (Array.init 600 (fun t ->
                 Float.sin (2. *. Float.pi *. 440. *. Float.of_int t /. 8000.) )
            )
        in
        List.iter
          (fun (name, x) ->
            let expected = direct x in
            let got = bits (autocorrelate x) in
            let r0 = expected.(0) in
            let worst = ref 0. in
            Array.iteri
              (fun k e ->
                worst := Float.max !worst (Float.abs (got.(k) -. e) /. r0) )
              expected ;
            at_most ~msg:name float_exact ~than:direct_tol !worst )
          [ ("noise", noise Nx.float64 [|777|])
          ; ("tone", tone)
          ; ("one sample", Nx.create Nx.float64 [|1|] [|0.75|]) ] )
  ; test "empty signals" (fun () ->
        equal ~msg:"no samples" (array int) [|3; 0|]
          (Nx.shape (autocorrelate (Nx.zeros Nx.float64 [|3; 0|]))) ;
        equal ~msg:"no samples, max_size" (array int) [|0|]
          (Nx.shape (autocorrelate ~max_size:4 (Nx.zeros Nx.float32 [|0|]))) ;
        equal ~msg:"no rows" (array int) [|0; 5|]
          (Nx.shape (autocorrelate (Nx.zeros Nx.float64 [|0; 5|]))) )
  ; test "errors" (fun () ->
        raises ~msg:"max_size 0"
          (Invalid_argument
             "autocorrelate: cannot keep 0 lags (max_size must be at least 1)"
          ) (fun () -> autocorrelate ~max_size:0 (noise Nx.float64 [|8|]) ) ;
        raises ~msg:"max_size -1"
          (Invalid_argument
             "autocorrelate: cannot keep -1 lags (max_size must be at least 1)"
          ) (fun () -> autocorrelate ~max_size:(-1) (noise Nx.float64 [|8|]) ) ;
        raises ~msg:"rank 0"
          (Invalid_argument
             "autocorrelate: cannot correlate a rank-zero tensor (the time \
              axis must exist)" ) (fun () ->
            autocorrelate (Nx.scalar Nx.float64 1.) ) ) ]

(* {1 lpc} *)

(* [ar2 n] is [n] samples of the AR(2) process [y t = 1.6 y (t - 1) - 0.9 y (t -
   2) + e t], driven by the LCG, after 500 samples of burn-in; each product is
   bound first, so the samples are the same bits on every platform. *)
let ar2 n =
  let e = lcg_values ~seed:77 (n + 500) in
  let y = Array.make (n + 500) 0. in
  for t = 0 to n + 499 do
    let y1 = if t >= 1 then y.(t - 1) else 0. in
    let y2 = if t >= 2 then y.(t - 2) else 0. in
    let p1 = 1.6 *. y1 in
    let p2 = 0.9 *. y2 in
    y.(t) <- p1 -. p2 +. e.(t)
  done ;
  Nx.create Nx.float64 [|n|] (Array.sub y 500 n)

let lpc_tests =
  [ test "an AR(2) process is recovered" (fun () ->
        let a = bits (lpc ~order:2 (ar2 65536)) in
        equal ~msg:"a 0" float_exact 1. a.(0) ;
        equal ~msg:"a 1" (float 0.02) (-1.6) a.(1) ;
        equal ~msg:"a 2" (float 0.02) 0.9 a.(2) ;
        (* Beyond the process's order, the extra coefficients vanish. *)
        let a = bits (lpc ~order:6 (ar2 65536)) in
        for k = 3 to 6 do
          equal ~msg:(Printf.sprintf "a %d" k) (float 0.02) 0. a.(k)
        done )
  ; test "degenerate signals" (fun () ->
        let lpc4 values =
          bits (lpc ~order:4 (Nx.create Nx.float64 [|16|] values))
        in
        equal ~msg:"zeros" (array float_exact) [|1.; 0.; 0.; 0.; 0.|]
          (lpc4 (Array.make 16 0.)) ;
        equal ~msg:"ones" (array float_exact) [|1.; -1.; 0.; 0.; 0.|]
          (lpc4 (Array.make 16 1.)) ;
        equal ~msg:"alternating" (array float_exact) [|1.; 1.; 0.; 0.; 0.|]
          (lpc4 (Array.init 16 (fun t -> if t mod 2 = 0 then 1. else -1.))) ;
        equal ~msg:"no rows" (array int) [|0; 4|]
          (Nx.shape (lpc ~order:3 (Nx.zeros Nx.float32 [|0; 5|]))) )
  ; test "a short signal extends by zeros" (fun () ->
        let x = noise Nx.float64 [|3; 4|] in
        for n = 0 to 4 do
          let xn = last_range x 0 n in
          let base =
            if n >= 2 then bits (lpc ~order:(n - 1) xn)
            else Array.concat (List.init 3 (fun _ -> [|1.|]))
          in
          let keep = Int.max 1 n in
          List.iter
            (fun order ->
              let got = bits (lpc ~order xn) in
              let expected =
                Array.init
                  (3 * (order + 1))
                  (fun i ->
                    let r = i / (order + 1) and k = i mod (order + 1) in
                    if k < keep then base.((r * keep) + k) else 0. )
              in
              equal
                ~msg:(Printf.sprintf "n %d, order %d" n order)
                (array float_exact) expected got )
            [Int.max 1 n; n + 3]
        done )
  ; test "a row is the same bits alone or in a batch" (fun () ->
        let x = noise Nx.float64 [|2; 3; 300|] in
        let batch = lpc ~order:9 x in
        for i = 0 to 1 do
          for j = 0 to 2 do
            same
              ~msg:(Printf.sprintf "row %d %d" i j)
              (row (row batch i) j)
              (lpc ~order:9 (row (row x i) j))
          done
        done )
  ; test "float32 is the double coefficients rounded once" (fun () ->
        let x = Nx.cast Nx.float32 (ar2 2048) in
        same ~msg:"float32"
          (Nx.cast Nx.float32 (lpc ~order:8 (Nx.cast Nx.float64 x)))
          (lpc ~order:8 x) )
  ; test "errors" (fun () ->
        let x = noise Nx.float64 [|64|] in
        raises ~msg:"order 0"
          (Invalid_argument
             "lpc: cannot fit a predictor of order 0 (order must be at least 1)"
          ) (fun () -> lpc ~order:0 x ) ;
        raises ~msg:"order -2"
          (Invalid_argument
             "lpc: cannot fit a predictor of order -2 (order must be at least \
              1)" ) (fun () -> lpc ~order:(-2) x ) ;
        raises ~msg:"rank 0"
          (Invalid_argument
             "lpc: cannot analyse a rank-zero tensor (the time axis must exist)"
          ) (fun () -> lpc ~order:2 (Nx.scalar Nx.float64 1.) ) ;
        let non_finite v =
          let a = lcg_values 64 in
          a.(17) <- v ;
          Nx.create Nx.float32 [|2; 32|] a
        in
        let message =
          Invalid_argument
            "lpc: cannot fit a signal holding NaN or infinite samples (every \
             sample must be finite)"
        in
        raises ~msg:"NaN" message (fun () ->
            lpc ~order:2 (non_finite Float.nan) ) ;
        raises ~msg:"inf" message (fun () ->
            lpc ~order:2 (non_finite Float.infinity) ) ;
        raises ~msg:"-inf" message (fun () ->
            lpc ~order:2 (non_finite Float.neg_infinity) ) ;
        let overflow order =
          Invalid_argument
            (Printf.sprintf
               "lpc: cannot fit an order-%d predictor to this signal (a \
                coefficient overflowed: the signal is too ill-conditioned for \
                the order, or its energy or coefficients exceed the range of \
                the dtype)"
               order )
        in
        (* The squares of samples near 1e200 overflow the energy. *)
        raises ~msg:"energy overflow" (overflow 2) (fun () ->
            lpc ~order:2 (Nx.mul_s (noise Nx.float64 [|64|]) 1e200) ) ;
        (* A noiseless sinusoid fitted far beyond its two poles diverges. *)
        let tone =
          Nx.create Nx.float64 [|256|]
            (Array.init 256 (fun t ->
                 Float.cos (2. *. Float.pi *. Float.of_int t /. 8.) ) )
        in
        raises ~msg:"divergence" (overflow 200) (fun () -> lpc ~order:200 tone) )
  ]

(* {1 poly_features} *)

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:22050 ~channels:1

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
  let named =
    [("ones", List.init n (fun _ -> 1)); ("whole", if n = 0 then [] else [n])]
  in
  let random =
    List.init 3 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))
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
        last_range x off (off + s) :: go (off + s) rest
  in
  go 0 sizes

let stream_outputs p ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

let config fft_size =
  Stft.Config.create ~fft_size ~hop:(fft_size / 4) ~alignment:`Centered
    ~pad:(`Constant 0.) ()

let law_case (type a) ~name (dtype : (float, a) Nx.dtype) ~fft_size ~order shape
    =
  test name (fun () ->
      let rng = Random.State.make [|seed; fft_size; order|] in
      let c = config fft_size in
      let x = noise dtype shape in
      let expected = poly_features c ~order ~sample_rate:22050 x in
      let s = Stft.power_spectrum ~power:1. c x in
      let frames = Nx.dim (Nx.ndim s - 1) s in
      let p = poly_features_stage c ~order ~sample_rate:22050 () in
      same ~msg:"run" expected (Pipeline.run ~source:(source ()) p s) ;
      same ~msg:"chain" expected
        (Pipeline.run ~source:(source ())
           Pipeline.(Stft.power_stage ~power:1. c >> p)
           x ) ;
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left Int.max 1 sizes in
          match stream_outputs p ~max_chunk (chunks_of s sizes) with
          | [] ->
              failf "%s: no output" pname
          | outs ->
              same ~msg:pname expected (Nx.concatenate ~axis:(-1) outs) )
        (partitions rng frames) ;
      (* The chain behind the stateful, rate-changing STFT, fed raw audio. *)
      let chain = Pipeline.(Stft.power_stage ~power:1. c >> p) in
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left Int.max 1 sizes in
          match stream_outputs chain ~max_chunk (chunks_of x sizes) with
          | [] ->
              failf "chain, %s: no output" pname
          | outs ->
              same ~msg:("chain, " ^ pname) expected
                (Nx.concatenate ~axis:(-1) outs) )
        (partitions rng (Nx.dim (Nx.ndim x - 1) x)) )

let stage_tests =
  [ law_case ~name:"float64, fft 128, order 1, [600]" Nx.float64 ~fft_size:128
      ~order:1 [|600|]
  ; law_case ~name:"float32, fft 128, order 3, [2; 500]" Nx.float32
      ~fft_size:128 ~order:3 [|2; 500|]
  ; law_case ~name:"float64, odd fft 127, order 2, [2; 1; 450]" Nx.float64
      ~fft_size:127 ~order:2 [|2; 1; 450|]
  ; law_case ~name:"float32, odd fft 63, order 0, [300]" Nx.float32 ~fft_size:63
      ~order:0 [|300|]
  ; test "reset restarts nothing: the stage is memoryless" (fun () ->
        let c = config 64 in
        let p = poly_features_stage c ~order:2 ~sample_rate:16000 () in
        let st = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:8 in
        let s = Nx.abs (noise Nx.float64 [|33; 8|]) in
        let first = Option.get (Pipeline.Stream.push st s) in
        ignore (Pipeline.Stream.push st s) ;
        Pipeline.Stream.reset st ;
        let again = Option.get (Pipeline.Stream.push st s) in
        same ~msg:"after reset" first again )
  ; test "an empty chunk maps to an empty chunk" (fun () ->
        let c = config 64 in
        let p = poly_features_stage c ~order:2 ~sample_rate:16000 () in
        let st = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:8 in
        let out = Pipeline.Stream.push st (Nx.zeros Nx.float32 [|2; 33; 0|]) in
        equal ~msg:"shape" (array int) [|2; 3; 0|] (Nx.shape (Option.get out)) )
  ]

(* [polynomial coeffs grid frames] is the [[bins; frames]] spectrogram whose
   frame [t] is the polynomial [coeffs] (highest degree first) over [grid],
   scaled by [t + 1]. *)
let polynomial coeffs grid frames =
  let bins = Array.length grid in
  Nx.create Nx.float64 [|bins; frames|]
    (Array.init (bins * frames) (fun i ->
         let f = grid.(i / frames) and t = i mod frames in
         let v = Array.fold_left (fun acc c -> (acc *. f) +. c) 0. coeffs in
         v *. Float.of_int (t + 1) ) )

let fft_grid ~sample_rate ~fft_size bins =
  let step =
    1.0 /. (Float.of_int fft_size *. (1.0 /. Float.of_int sample_rate))
  in
  Array.init bins (fun k -> Float.of_int k *. step)

let poly_tests =
  [ test "the audio face is the spectrogram face over its magnitudes" (fun () ->
        List.iter
          (fun (fft_size, order) ->
            let c = config fft_size in
            let x = noise Nx.float32 [|2; 700|] in
            same
              ~msg:(Printf.sprintf "fft %d, order %d" fft_size order)
              (poly_features c ~order ~sample_rate:22050 x)
              (poly_features_of_spectrogram ~order ~sample_rate:22050
                 (Stft.power_spectrum ~power:1. c x) ) )
          [(128, 0); (128, 1); (256, 3)] ;
        (* An odd FFT size fits on its own grid, which freqs passes. *)
        let c = config 127 in
        let x = noise Nx.float64 [|700|] in
        let grid = fft_grid ~sample_rate:22050 ~fft_size:127 64 in
        same ~msg:"odd fft 127"
          (poly_features c ~order:2 ~sample_rate:22050 x)
          (poly_features_of_spectrogram ~order:2
             ~freqs:(Nx.create Nx.float64 [|64|] grid)
             ~sample_rate:22050
             (Stft.power_spectrum ~power:1. c x) ) )
  ; test "a frame is the same bits alone or among others" (fun () ->
        let s = Nx.abs (noise Nx.float64 [|2; 3; 65; 9|]) in
        let all = poly_features_of_spectrogram ~order:3 ~sample_rate:8000 s in
        same ~msg:"one clip"
          (row (row all 1) 2)
          (poly_features_of_spectrogram ~order:3 ~sample_rate:8000
             (row (row s 1) 2) ) ;
        for t = 0 to 8 do
          same
            ~msg:(Printf.sprintf "frame %d" t)
            (last_range all t (t + 1))
            (poly_features_of_spectrogram ~order:3 ~sample_rate:8000
               (last_range s t (t + 1)) )
        done )
  ; test "a non-finite value spoils its own frame only" (fun () ->
        let v = Array.map Float.abs (lcg_values (33 * 4)) in
        let clean = Nx.create Nx.float64 [|33; 4|] (Array.copy v) in
        v.((10 * 4) + 2) <- Float.nan ;
        let dirty = Nx.create Nx.float64 [|33; 4|] v in
        let fit s = bits (poly_features_of_spectrogram ~sample_rate:8000 s) in
        let a = fit clean and b = fit dirty in
        Array.iteri
          (fun i e ->
            if i mod 4 = 2 then
              is_true ~msg:(Printf.sprintf "NaN %d" i) (Float.is_nan b.(i))
            else equal ~msg:(Printf.sprintf "%d" i) float_exact e b.(i) )
          a )
  ; test "an exact polynomial is recovered" (fun () ->
        let grid = fft_grid ~sample_rate:22050 ~fft_size:512 257 in
        let coeffs = [|3e-8; -2e-4; 0.5; 7.|] in
        let s = polynomial coeffs grid 3 in
        List.iter
          (fun order ->
            let got =
              bits (poly_features_of_spectrogram ~order ~sample_rate:22050 s)
            in
            for t = 0 to 2 do
              let scale = Float.of_int (t + 1) in
              for i = 0 to order do
                (* Rows above the polynomial's degree fit zero. *)
                let expected =
                  if i < order - 3 then 0.
                  else coeffs.(i - (order - 3)) *. scale
                in
                let tol =
                  if expected = 0. then 1e-12 *. Float.abs coeffs.(0)
                  else 1e-9 *. Float.abs expected
                in
                equal
                  ~msg:(Printf.sprintf "order %d, row %d, frame %d" order i t)
                  (float tol) expected
                  got.((i * 3) + t)
              done
            done )
          [3; 4; 5] )
  ; test "order 0 is the frame mean" (fun () ->
        let s = Nx.abs (noise Nx.float64 [|100; 6|]) in
        let got =
          bits (poly_features_of_spectrogram ~order:0 ~sample_rate:8000 s)
        in
        let v = bits s in
        for t = 0 to 5 do
          let sum = ref 0. in
          for k = 0 to 99 do
            sum := !sum +. v.((k * 6) + t)
          done ;
          equal
            ~msg:(Printf.sprintf "frame %d" t)
            (float 1e-15) (!sum /. 100.) got.(t)
        done )
  ; test "a flat frame's order-0 fit is its level, at any bin count" (fun () ->
        (* Every bin weighs [1 / bins] at order 0. Summed plainly, over the
           decomposition or over the frame, the rounding of [bins] near-equal
           terms accumulates the same way at every step — [2.5e-13] of the level
           over 8939 bins; the compensated sums leave about an ulp. *)
        List.iter
          (fun bins ->
            List.iter
              (fun level ->
                let s = Nx.full Nx.float64 [|bins; 2|] level in
                let got =
                  bits
                    (poly_features_of_spectrogram ~order:0 ~sample_rate:22050 s)
                in
                Array.iteri
                  (fun t v ->
                    equal
                      ~msg:
                        (Printf.sprintf "%d bins, level %g, frame %d" bins level
                           t )
                      (float (2. *. epsilon_float *. level))
                      level v )
                  got )
              [1.; 0.1; 3e5] )
          [257; 513; 1089; 2049; 4097; 8939] )
  ; test "a fit with fewer frequencies than coefficients interpolates"
      (fun () ->
        (* Three bins, order 3, and repeated frequencies: rank-deficient fits
           take the minimum-norm solution, which still passes through every
           point it can. *)
        let check name freqs s =
          let bins = Array.length freqs in
          let frames = Nx.dim 1 s in
          let got =
            bits
              (poly_features_of_spectrogram ~order:3
                 ~freqs:(Nx.create Nx.float64 [|bins|] freqs)
                 ~sample_rate:8000 s )
          in
          let v = bits s in
          for k = 0 to bins - 1 do
            for t = 0 to frames - 1 do
              let fit = ref 0. in
              for i = 0 to 3 do
                fit := (!fit *. freqs.(k)) +. got.((i * frames) + t)
              done ;
              equal
                ~msg:(Printf.sprintf "%s, bin %d, frame %d" name k t)
                (float 1e-9)
                v.((k * frames) + t)
                !fit
            done
          done
        in
        check "three bins" [|0.; 2000.; 4000.|]
          (Nx.abs (noise Nx.float64 [|3; 4|])) ;
        let s = Nx.abs (noise Nx.float64 [|3; 4|]) in
        let v = bits s in
        (* bins 0 and 1, then 2 and 3, share a frequency and a value *)
        let paired =
          Nx.create Nx.float64 [|6; 4|]
            (Array.init 24 (fun i -> v.((i / 8 * 4) + (i mod 4))))
        in
        check "paired" [|100.; 100.; 900.; 900.; 1500.; 1500.|] paired )
  ; test "empty spectrograms" (fun () ->
        equal ~msg:"no frames" (array int) [|2; 4; 0|]
          (Nx.shape
             (poly_features_of_spectrogram ~order:3 ~sample_rate:8000
                (Nx.zeros Nx.float64 [|2; 33; 0|]) ) ) ;
        equal ~msg:"no clips" (array int) [|0; 2; 5|]
          (Nx.shape
             (poly_features_of_spectrogram ~sample_rate:8000
                (Nx.zeros Nx.float32 [|0; 33; 5|]) ) ) ;
        equal ~msg:"no samples" (array int) [|2; 0|]
          (Nx.shape
             (poly_features (config 64) ~sample_rate:8000
                (Nx.zeros Nx.float64 [|0|]) ) ) )
  ; test "errors" (fun () ->
        let s = Nx.abs (noise Nx.float64 [|33; 4|]) in
        let c = config 64 in
        raises ~msg:"order"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot fit a polynomial of order \
              -1 (order must be at least 0)" ) (fun () ->
            poly_features_of_spectrogram ~order:(-1) ~sample_rate:8000 s ) ;
        raises ~msg:"order, audio"
          (Invalid_argument
             "poly_features: cannot fit a polynomial of order -1 (order must \
              be at least 0)" ) (fun () ->
            poly_features c ~order:(-1) ~sample_rate:8000
              (noise Nx.float64 [|100|]) ) ;
        raises ~msg:"order, stage"
          (Invalid_argument
             "poly_features_stage: cannot fit a polynomial of order -1 (order \
              must be at least 0)" ) (fun () ->
            poly_features_stage c ~order:(-1) ~sample_rate:8000 () ) ;
        raises ~msg:"sample rate"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot use a sample rate of 0 Hz \
              (sample_rate must be at least 1)" ) (fun () ->
            poly_features_of_spectrogram ~sample_rate:0 s ) ;
        raises ~msg:"sample rate, stage"
          (Invalid_argument
             "poly_features_stage: cannot use a sample rate of 0 Hz \
              (sample_rate must be at least 1)" ) (fun () ->
            poly_features_stage c ~sample_rate:0 () ) ;
        raises ~msg:"rank"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot fit a rank-1 tensor (a \
              spectrogram is [...; bins; frames])" ) (fun () ->
            poly_features_of_spectrogram ~sample_rate:8000
              (noise Nx.float64 [|33|]) ) ;
        raises ~msg:"rank, audio"
          (Invalid_argument
             "poly_features: cannot analyse a rank-zero tensor (the time axis \
              must exist)" ) (fun () ->
            poly_features c ~sample_rate:8000 (Nx.scalar Nx.float64 1.) ) ;
        raises ~msg:"one bin"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot derive bin frequencies for \
              a 1-bin spectrogram (the implied FFT size is 0; pass freqs \
              explicitly)" ) (fun () ->
            poly_features_of_spectrogram ~sample_rate:8000
              (Nx.ones Nx.float64 [|1; 4|]) ) ;
        raises ~msg:"freqs length"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot pair 32 bin frequencies \
              with 33 bins (freqs holds one frequency per bin)" ) (fun () ->
            poly_features_of_spectrogram
              ~freqs:(Nx.ones Nx.float64 [|32|])
              ~sample_rate:8000 s ) ;
        raises ~msg:"freqs rank"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot use a rank-2 freqs tensor \
              (freqs is rank-one, one frequency per bin)" ) (fun () ->
            poly_features_of_spectrogram
              ~freqs:(Nx.ones Nx.float64 [|33; 4|])
              ~sample_rate:8000 s ) ;
        raises ~msg:"freqs NaN"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot fit over a non-finite bin \
              frequency (freqs must be finite)" ) (fun () ->
            poly_features_of_spectrogram
              ~freqs:
                (Nx.create Nx.float64 [|33|]
                   (Array.init 33 (fun k ->
                        if k = 5 then Float.nan else Float.of_int k ) ) )
              ~sample_rate:8000 s ) ;
        raises ~msg:"freqs all zero"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot fit a polynomial of order 1 \
              over these bin frequencies (a power of the frequencies vanishes \
              at every bin or its squares overflow double precision)" )
          (fun () ->
            poly_features_of_spectrogram
              ~freqs:(Nx.zeros Nx.float64 [|33|])
              ~sample_rate:8000 s ) ;
        raises ~msg:"freqs overflow"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot fit a polynomial of order 3 \
              over these bin frequencies (a power of the frequencies vanishes \
              at every bin or its squares overflow double precision)" )
          (fun () ->
            poly_features_of_spectrogram ~order:3
              ~freqs:(Nx.full Nx.float64 [|33|] 1e120)
              ~sample_rate:8000 s ) ;
        raises ~msg:"no bins"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot fit a polynomial over no \
              bins (the spectrogram needs at least one bin)" ) (fun () ->
            poly_features_of_spectrogram
              ~freqs:(Nx.zeros Nx.float64 [|0|])
              ~sample_rate:8000
              (Nx.zeros Nx.float64 [|0; 3|]) ) ;
        raises ~msg:"no bins, implied grid"
          (Invalid_argument
             "poly_features_of_spectrogram: cannot fit a polynomial over no \
              bins (the spectrogram needs at least one bin)" ) (fun () ->
            poly_features_of_spectrogram ~sample_rate:8000
              (Nx.zeros Nx.float64 [|2; 0; 3|]) ) ;
        let c2048 =
          Stft.Config.create ~fft_size:2048 ~hop:512 ~pad:(`Constant 0.) ()
        in
        raises ~msg:"overflow, audio"
          (Invalid_argument
             "poly_features: cannot fit a polynomial of order 38 over these \
              bin frequencies (a power of the frequencies vanishes at every \
              bin or its squares overflow double precision)" ) (fun () ->
            poly_features c2048 ~order:38 ~sample_rate:22050
              (noise Nx.float64 [|4096|]) ) ;
        raises ~msg:"overflow, stage"
          (Invalid_argument
             "poly_features_stage: cannot fit a polynomial of order 38 over \
              these bin frequencies (a power of the frequencies vanishes at \
              every bin or its squares overflow double precision)" ) (fun () ->
            poly_features_stage c2048 ~order:38 ~sample_rate:22050 () ) ;
        let p = poly_features_stage c ~sample_rate:8000 () in
        raises ~msg:"stage bins"
          (Invalid_argument
             "poly_features_stage: cannot fit 32 bins with a stage built for 33"
          ) (fun () ->
            stream_outputs p ~max_chunk:4 [Nx.zeros Nx.float64 [|32; 4|]] ) ;
        raises ~msg:"stage rank"
          (Invalid_argument
             "poly_features_stage: cannot fit a rank-1 chunk (the stage needs \
              [...; bins; frames])" ) (fun () ->
            stream_outputs p ~max_chunk:4 [Nx.zeros Nx.float64 [|33|]] ) ) ]

let suite =
  [ group "autocorrelate" autocorrelate_tests
  ; group "lpc" lpc_tests
  ; group "poly_features stage" stage_tests
  ; group "poly_features" poly_tests ]
