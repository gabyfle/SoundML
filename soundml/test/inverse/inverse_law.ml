(* Properties of the feature inversions, checked without an oracle.

   mfcc_to_mel: the cepstral round trip through mfcc_stage (exact at n_mfcc =
   n_mels), the reference as an exact factor, the float32 face as the rounding
   of the float64 one, and the stage against the flat function on every
   partitioning.

   mel_to_stft, the parity contract's own terms: re-projection of mel
   spectrograms of signals at several geometries (a rank-deficient filterbank
   among them), the optimality conditions of non-negative least squares on
   frames that some spectrum fits and on frames none fits, a fit never worse
   than librosa's start, a tonal envelope whose start goes negative fitted
   exactly, the start returned where it already fits, scale equivariance, exact
   zeros where no filter reaches, and independence: a frame alone, a clip of a
   batch, every partitioning of the stage and of the chain behind
   mfcc_to_mel_stage give the same bits.

   mel_to_audio and mfcc_to_audio: their compositions, bit for bit. Every
   Stft.Config passes its padding explicitly. *)

open Windtrap
open Soundml

let seed = 0x1a7e5e

(* Bounds, each about ten times the worst value measured here (in brackets). The
   cepstral round trip through mfcc_stage, relative to the largest coefficient
   [9.5e-16], and the complete inverse at n_mfcc = n_mels, relative to each mel
   power [6.9e-15]. *)
let round_trip_bound = 1e-14

let complete_bound = 7e-14

(* mel_to_stft on mel spectrograms of signals, which some spectrum fits: the
   largest relative 2-norm distance between a frame and its re-projection
   [1.6e-14], and the worst violation of the optimality conditions relative to
   the frame's largest entry of |A^T m| [3.9e-15]. *)
let reprojection_bound = 2e-13

let fitted_optimality_bound = 4e-14

(* On the targets below, which no non-negative spectrum fits, the dual grows as
   1 / lambda and cancels: the optimality residual, against the same per-frame
   scale, sits between 5e-12 and 1.6e-9 on the full-rank filterbanks and reaches
   1.6e-8 on the rank-deficient one [1.6e-8]. Other targets reach further (the
   documentation states the range measured over a wider set). *)
let unfitted_optimality_bound = 2e-7

(* A start that fits, against the spectrum it came from, relative to its peak
   [3.5e-15]; and the power spectrum of a scaled input against the scaled power
   spectrum, relative to its peak [5.6e-16]. *)
let start_bound = 4e-14

let scale_bound = 6e-15

(* {1 Signals and helpers} *)

let lcg ?(seed = 20261101) n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let sawtooth ~sample_rate n =
  Array.init n (fun i ->
      let phase = Float.of_int i *. 220. /. Float.of_int sample_rate in
      0.3 *. (2. *. (phase -. Float.round phase)) )

let tones ~sample_rate n =
  Array.init n (fun i ->
      let t = Float.of_int i /. Float.of_int sample_rate in
      (0.2 *. Float.sin (2. *. Float.pi *. 440. *. t))
      +. (0.1 *. Float.sin (2. *. Float.pi *. 1234.5 *. t)) )

let signals ~sample_rate n =
  [ ("lcg", Array.map (fun v -> 0.1 *. v) (lcg n))
  ; ("sawtooth", sawtooth ~sample_rate n)
  ; ("tones", tones ~sample_rate n) ]

let stft_config fft_size =
  Stft.Config.create ~pad:(`Constant 0.) ~hop:(fft_size / 4) ~fft_size ()

(* The geometries: librosa's default, an HTK filterbank without normalisation, a
   band that leaves bins unreached at both ends, and a filterbank of rank 109
   out of 128 (narrow low filters sharing bins). *)
let default_mel () =
  Mel.Config.create ~n_mels:128 ~sample_rate:22050 ~fft_size:2048 ()

let geometries () =
  [ ("22050/2048/128", default_mel (), 22050)
  ; ( "16000/400/40 htk"
    , Mel.Config.create ~scale:`Htk ~norm:`None ~n_mels:40 ~sample_rate:16000
        ~fft_size:400 ()
    , 16000 )
  ; ( "16000/512/40 300-7600 Hz"
    , Mel.Config.create ~f_min:300. ~f_max:7600. ~n_mels:40 ~sample_rate:16000
        ~fft_size:512 ()
    , 16000 )
  ; ( "22050/512/128 rank-deficient"
    , Mel.Config.create ~n_mels:128 ~sample_rate:22050 ~fft_size:512 ()
    , 22050 ) ]

let mel_of mel ?power y =
  let stft = stft_config (Mel.Config.fft_size mel) in
  mel_spectrogram stft mel ?power (Nx.create Nx.float64 [|Array.length y|] y)

let bits x = Nx.to_array (Nx.cast Nx.float64 x)

let same_bits ~msg a b =
  equal ~msg (array int) (Nx.shape a) (Nx.shape b) ;
  equal ~msg (array float_exact) (bits a) (bits b)

let within ~msg bound value =
  if not (value <= bound) then failf "%s: %g exceeds %g" msg value bound

let frames x = Nx.dim (Nx.ndim x - 1) x

(* [frame_error m r] is the largest relative 2-norm distance between the frames
   of two [[bands; frames]] tensors, over frames where [m] is not zero. *)
let frame_error m r =
  let a = bits m and b = bits r in
  let t_count = frames m in
  let bands = Array.length a / t_count in
  let worst = ref 0. in
  for t = 0 to t_count - 1 do
    let num = ref 0. and den = ref 0. in
    for i = 0 to bands - 1 do
      let x = a.((i * t_count) + t) in
      let d = b.((i * t_count) + t) -. x in
      num := !num +. (d *. d) ;
      den := !den +. (x *. x)
    done ;
    if !den > 0. then worst := Float.max !worst (Float.sqrt (!num /. !den))
  done ;
  !worst

let power_of ~power s = if Float.equal power 2. then Nx.mul s s else s

(* [kkt mel ~power m s] is the worst violation, over the frames of the [[bands;
   frames]] tensor [m], of the optimality conditions of non-negative least
   squares by [x = s ** power] — the gradient [A^T (A x - m)] is zero where [x >
   0] and non-negative where [x = 0] — relative to the frame's own scale, the
   largest entry of [|A^T m|] in that frame. Frames of scale zero have nothing
   to violate. *)
let kkt mel ~power m s =
  let a = Mel.filterbank Nx.float64 mel in
  let m = Nx.cast Nx.float64 m in
  let x = power_of ~power (Nx.cast Nx.float64 s) in
  let at = Nx.transpose a in
  let g = bits (Nx.matmul at (Nx.sub (Nx.matmul a x) m)) in
  let atm = bits (Nx.matmul at m) and xs = bits x in
  let t_count = frames m in
  let worst = ref 0. in
  for t = 0 to t_count - 1 do
    let scale = ref 0. and violation = ref 0. in
    for j = 0 to (Array.length g / t_count) - 1 do
      let i = (j * t_count) + t in
      let v = if xs.(i) > 0. then Float.abs g.(i) else -.g.(i) in
      scale := Float.max !scale (Float.abs atm.(i)) ;
      violation := Float.max !violation v
    done ;
    if !scale > 0. then worst := Float.max !worst (!violation /. !scale)
  done ;
  !worst

(* [librosa_pinv a] is the pseudo-inverse librosa starts from, scipy's pinv at
   rtol = 1e-15, which drops the singular values at most 1e-15 of the largest:
   Nx.pinv scales its rtol by the larger dimension. *)
let librosa_pinv a =
  Nx.pinv ~rtol:(1e-15 /. Float.of_int (Int.max (Nx.dim 0 a) (Nx.dim 1 a))) a

(* [fit mel m x] is each frame's [1/2 |A x - m|^2]. *)
let fit mel m x =
  let a = Mel.filterbank Nx.float64 mel in
  let r = Nx.sub (Nx.matmul a x) m in
  bits (Nx.mul_s (Nx.sum ~axes:[0] (Nx.mul r r)) 0.5)

(* {1 Streaming helpers} *)

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (min n 7) in
    s :: random_sizes rng (n - s)

let partitions rng n =
  [ ("ones", List.init n (fun _ -> 1))
  ; ("whole", [n])
  ; ("odd", random_sizes rng n)
  ; ( "with empty chunks"
    , List.concat_map
        (fun s -> if Random.State.bool rng then [0; s] else [s])
        (random_sizes rng n) ) ]

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

let streamed p x sizes =
  let max_chunk = List.fold_left Int.max 1 sizes in
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk in
  let pushed = List.filter_map (Pipeline.Stream.push s) (chunks_of x sizes) in
  let outs = pushed @ Pipeline.Stream.flush s in
  Nx.concatenate ~axis:(-1) outs

let partition_law ~name p offline x =
  let rng = Random.State.make [|seed|] in
  List.iter
    (fun (pname, sizes) ->
      same_bits ~msg:(name ^ ", " ^ pname) offline (streamed p x sizes) )
    (partitions rng (frames x))

(* {1 mfcc_to_mel} *)

let mel40 () = Mel.Config.create ~n_mels:40 ~sample_rate:22050 ~fft_size:2048 ()

let noise_mel mel = mel_of mel (Array.map (fun v -> 0.1 *. v) (lcg 8192))

let cepstrum mel ~n_mfcc ?lifter m =
  Pipeline.run ~source:(source ()) (mfcc_stage mel ~n_mfcc ?lifter ()) m

let cepstral_tests =
  [ test "the round trip through mfcc_stage returns the cepstrum" (fun () ->
        let mel = mel40 () in
        let m = noise_mel mel in
        List.iter
          (fun (n_mfcc, lifter) ->
            let c = cepstrum mel ~n_mfcc ~lifter m in
            let back =
              cepstrum mel ~n_mfcc ~lifter (mfcc_to_mel mel ~lifter c)
            in
            let peak = Nx.item [] (Nx.max (Nx.abs c)) in
            let err = Nx.item [] (Nx.max (Nx.abs (Nx.sub back c))) /. peak in
            within
              ~msg:(Printf.sprintf "n_mfcc %d, lifter %g" n_mfcc lifter)
              round_trip_bound err )
          [(13, 0.); (13, 22.); (40, 0.); (40, 22.)] )
  ; test "all the coefficients invert exactly" (fun () ->
        let mel = mel40 () in
        let m = noise_mel mel in
        List.iter
          (fun lifter ->
            let c = cepstrum mel ~n_mfcc:40 ~lifter m in
            let back = mfcc_to_mel mel ~lifter c in
            let err = Nx.item [] (Nx.max (Nx.div (Nx.abs (Nx.sub back m)) m)) in
            within ~msg:(Printf.sprintf "lifter %g" lifter) complete_bound err )
          [0.; 22.] )
  ; test "the reference is an exact factor" (fun () ->
        let mel = mel40 () in
        let c = cepstrum mel ~n_mfcc:20 (noise_mel mel) in
        same_bits ~msg:"reference 2.5"
          (Nx.mul_s (mfcc_to_mel mel c) 2.5)
          (mfcc_to_mel mel ~reference:2.5 c) )
  ; test "float32 is the rounding of float64" (fun () ->
        let mel = mel40 () in
        let c = Nx.cast Nx.float32 (cepstrum mel ~n_mfcc:20 (noise_mel mel)) in
        same_bits ~msg:"float32"
          (Nx.cast Nx.float32
             (mfcc_to_mel mel ~lifter:22. (Nx.cast Nx.float64 c)) )
          (mfcc_to_mel mel ~lifter:22. c) )
  ; test "a frame alone, a clip of a batch, every partitioning" (fun () ->
        let mel = mel40 () in
        let c = cepstrum mel ~n_mfcc:13 (noise_mel mel) in
        let whole = mfcc_to_mel mel ~lifter:22. c in
        same_bits ~msg:"frame 5 alone"
          (Nx.slice [Nx.A; Nx.R (5, 6)] whole)
          (mfcc_to_mel mel ~lifter:22. (Nx.slice [Nx.A; Nx.R (5, 6)] c)) ;
        let batch = Nx.stack ~axis:0 [c; Nx.mul_s c 0.5] in
        same_bits ~msg:"batch, second clip"
          (mfcc_to_mel mel ~lifter:22. (Nx.mul_s c 0.5))
          (Nx.slice [Nx.I 1] (mfcc_to_mel mel ~lifter:22. batch)) ;
        partition_law ~name:"stage"
          (mfcc_to_mel_stage mel ~n_mfcc:13 ~lifter:22. ())
          whole c )
  ; test "no frames, no clips" (fun () ->
        let mel = mel40 () in
        equal ~msg:"no frames" (array int) [|40; 0|]
          (Nx.shape (mfcc_to_mel mel (Nx.zeros Nx.float64 [|13; 0|]))) ;
        equal ~msg:"no clips" (array int) [|0; 40; 3|]
          (Nx.shape (mfcc_to_mel mel (Nx.zeros Nx.float32 [|0; 13; 3|]))) ) ]

(* {1 mel_to_stft} *)

let reprojection_tests =
  List.concat_map
    (fun (gname, mel, sample_rate) ->
      List.map
        (fun power ->
          test
            (Printf.sprintf "%s, power %g: re-projection and optimality" gname
               power ) (fun () ->
              List.iter
                (fun (sname, y) ->
                  let m = mel_of mel ~power y in
                  let s = mel_to_stft mel ~power m in
                  let back = Mel.apply mel (power_of ~power s) in
                  within
                    ~msg:(gname ^ ", " ^ sname ^ ", re-projection")
                    reprojection_bound (frame_error m back) ;
                  within
                    ~msg:(gname ^ ", " ^ sname ^ ", optimality")
                    fitted_optimality_bound (kkt mel ~power m s) )
                (signals ~sample_rate (sample_rate / 4)) ) )
        [2.; 1.] )
    (geometries ())

(* Targets no non-negative spectrum fits: isolated band peaks over a floor, a
   band an order of magnitude below its neighbours, and |LCG| values. *)
let unfittable bands =
  let peaks = Array.make (bands * 3) 1e-3 in
  peaks.((bands / 3 * 3) + 0) <- 1. ;
  List.iter
    (fun i -> peaks.((i * 3) + 1) <- 1.)
    [bands / 8; bands / 2; 3 * bands / 4] ;
  for i = 0 to bands - 1 do
    peaks.((i * 3) + 2) <- (if i = bands / 4 then 1e-2 else 1.)
  done ;
  [ ("peaks", Nx.create Nx.float64 [|bands; 3|] peaks)
  ; ( "|lcg|"
    , Nx.create Nx.float64 [|bands; 4|]
        (Array.map Float.abs (lcg ~seed:7 (bands * 4))) ) ]

let optimality_tests =
  List.map
    (fun (gname, mel, _) ->
      test (gname ^ ": unfittable targets") (fun () ->
          let a = Mel.filterbank Nx.float64 mel in
          let start m =
            let x0 = Nx.matmul (librosa_pinv a) m in
            Nx.maximum x0 (Nx.zeros_like x0)
          in
          List.iter
            (fun (tname, m) ->
              let s = mel_to_stft mel m in
              within
                ~msg:(gname ^ ", " ^ tname ^ ", optimality")
                unfitted_optimality_bound (kkt mel ~power:2. m s) ;
              let ours = fit mel m (Nx.mul s s)
              and theirs = fit mel m (start m) in
              Array.iteri
                (fun t f ->
                  if not (f <= theirs.(t) *. (1. +. 1e-9)) then
                    failf "%s, %s, frame %d: fit %g above the start's %g" gname
                      tname t f theirs.(t) )
                ours )
            (unfittable (Mel.Config.n_mels mel)) ) )
    (geometries ())

let mel_to_stft_tests =
  [ test "a start that fits is returned" (fun () ->
        (* m = A (A^T w): the pseudo-inverse returns A^T w, non-negative *)
        let mel = default_mel () in
        let a = Mel.filterbank Nx.float64 mel in
        let w =
          Nx.create Nx.float64 [|128; 3|]
            (Array.map (fun v -> 1.5 +. v) (lcg ~seed:3 (128 * 3)))
        in
        let x = Nx.matmul (Nx.transpose a) w in
        let m = Nx.matmul a x in
        let peak = Nx.item [] (Nx.max x) in
        let err s = Nx.item [] (Nx.max (Nx.abs (Nx.sub s x))) /. peak in
        within ~msg:"power 1" start_bound (err (mel_to_stft mel ~power:1. m)) ;
        let s = mel_to_stft mel m in
        within ~msg:"power 2" start_bound (err (Nx.mul s s)) )
  ; test "scale equivariance" (fun () ->
        let mel = default_mel () in
        let m = mel_of mel (tones ~sample_rate:22050 8192) in
        let s = mel_to_stft mel m in
        List.iter
          (fun k ->
            (* bit for bit under a power of four, whose root is a power of
               two *)
            same_bits
              ~msg:(Printf.sprintf "4 ** %d" k)
              (Nx.mul_s s (Float.ldexp 1. k))
              (mel_to_stft mel (Nx.mul_s m (Float.ldexp 1. (2 * k)))) )
          [-20; 7] ;
        (* elsewhere to rounding, in the power domain: the square root amplifies
           a power's rounding by the inverse of its magnitude *)
        List.iter
          (fun c ->
            let got = mel_to_stft mel (Nx.mul_s m c) in
            let want = Nx.mul_s (Nx.mul s s) c in
            let err =
              Nx.item [] (Nx.max (Nx.abs (Nx.sub (Nx.mul got got) want)))
              /. Nx.item [] (Nx.max want)
            in
            within ~msg:(Printf.sprintf "scale %g" c) scale_bound err )
          [1e-6; 1e6] )
  ; test "unreached bins are exactly zero, the rest finite and non-negative"
      (fun () ->
        List.iter
          (fun (gname, mel, sample_rate) ->
            let a = bits (Mel.filterbank Nx.float64 mel) in
            let bins = Mel.Config.bins mel and bands = Mel.Config.n_mels mel in
            let m = mel_of mel (lcg (sample_rate / 8)) in
            let s = bits (mel_to_stft mel m) in
            let t_count = frames m in
            for j = 0 to bins - 1 do
              let reached = ref false in
              for i = 0 to bands - 1 do
                if a.((i * bins) + j) <> 0. then reached := true
              done ;
              for t = 0 to t_count - 1 do
                let v = s.((j * t_count) + t) in
                if not !reached then
                  equal
                    ~msg:(Printf.sprintf "%s, bin %d" gname j)
                    float_exact 0. v
                else if not (Float.is_finite v && v >= 0.) then
                  failf "%s, bin %d, frame %d: %g" gname j t v
              done
            done )
          (geometries ()) )
  ; test "zero and negative frames give the zero spectrum" (fun () ->
        let mel = default_mel () in
        let m =
          Nx.create Nx.float64 [|128; 2|]
            (Array.init 256 (fun i ->
                 if i mod 2 = 0 then 0. else -1. -. Float.of_int i ) )
        in
        equal ~msg:"zero" (array float_exact)
          (Array.make (1025 * 2) 0.)
          (bits (mel_to_stft mel m)) )
  ; test "no frames, no clips" (fun () ->
        let mel = mel40 () in
        equal ~msg:"no frames" (array int) [|1025; 0|]
          (Nx.shape (mel_to_stft mel (Nx.zeros Nx.float64 [|40; 0|]))) ;
        equal ~msg:"no clips" (array int) [|0; 1025; 3|]
          (Nx.shape (mel_to_stft mel (Nx.zeros Nx.float32 [|0; 40; 3|]))) )
  ; test "float32 is the rounding of float64" (fun () ->
        let mel = mel40 () in
        let m = Nx.cast Nx.float32 (noise_mel mel) in
        List.iter
          (fun power ->
            same_bits
              ~msg:(Printf.sprintf "power %g" power)
              (Nx.cast Nx.float32
                 (mel_to_stft mel ~power (Nx.cast Nx.float64 m)) )
              (mel_to_stft mel ~power m) )
          [2.; 1.; 0.5] )
  ; test "a frame alone, a clip of a batch, every partitioning" (fun () ->
        let mel = default_mel () in
        let m = mel_of mel (sawtooth ~sample_rate:22050 8192) in
        let whole = mel_to_stft mel m in
        same_bits ~msg:"frame 7 alone"
          (Nx.slice [Nx.A; Nx.R (7, 8)] whole)
          (mel_to_stft mel (Nx.slice [Nx.A; Nx.R (7, 8)] m)) ;
        let other = mel_of mel (lcg 8192) in
        let batch = Nx.stack ~axis:0 [other; m] in
        same_bits ~msg:"batch, second clip" whole
          (Nx.slice [Nx.I 1] (mel_to_stft mel batch)) ;
        partition_law ~name:"stage" (mel_to_stft_stage mel ()) whole m )
  ; test "a tonal envelope reaches a negative start and is fitted" (fun () ->
        (* The envelope mfcc_to_mel finds in 20 coefficients of a 440 Hz sine is
           smooth, yet on 22050/2048/64 its pseudo-inverse goes negative on most
           frames, so the result there is not librosa's start: the parity
           contract's second case. Some non-negative spectrum fits the envelope,
           and the solve finds it. *)
        let mel =
          Mel.Config.create ~n_mels:64 ~sample_rate:22050 ~fft_size:2048 ()
        in
        let y =
          Array.init 8192 (fun i ->
              0.5
              *. Float.sin (2. *. Float.pi *. 440. *. Float.of_int i /. 22050.) )
        in
        let m = mfcc_to_mel mel (cepstrum mel ~n_mfcc:20 (mel_of mel y)) in
        let a = Mel.filterbank Nx.float64 mel in
        let reached =
          bits (Nx.sum ~axes:[0] a) |> Array.map (fun v -> v > 0.)
        in
        let start = bits (Nx.matmul (librosa_pinv a) m) in
        let t_count = frames m in
        let negative = ref 0 in
        for t = 0 to t_count - 1 do
          let hi = ref 0. and lo = ref 0. in
          Array.iteri
            (fun j r ->
              if r then begin
                let v = start.((j * t_count) + t) in
                hi := Float.max !hi v ;
                lo := Float.min !lo v
              end )
            reached ;
          if !lo < -1e-6 *. !hi then incr negative
        done ;
        if not (2 * !negative > t_count) then
          failf "%d of %d frames have a negative start" !negative t_count ;
        let s = mel_to_stft mel m in
        within ~msg:"re-projection" reprojection_bound
          (frame_error m (Mel.apply mel (Nx.mul s s))) ;
        within ~msg:"optimality" fitted_optimality_bound (kkt mel ~power:2. m s) )
  ; test "the cepstral chain streams" (fun () ->
        let mel = default_mel () in
        let c = cepstrum mel ~n_mfcc:20 (noise_mel mel) in
        partition_law ~name:"chain"
          Pipeline.(mfcc_to_mel_stage mel () >> mel_to_stft_stage mel ())
          (mel_to_stft mel (mfcc_to_mel mel c))
          c ) ]

(* {1 Back to audio} *)

let audio_tests =
  [ test "mel_to_audio is Griffin-Lim of mel_to_stft" (fun () ->
        let mel =
          Mel.Config.create ~n_mels:64 ~sample_rate:22050 ~fft_size:512 ()
        in
        let stft = stft_config 512 in
        let m = mel_of mel (tones ~sample_rate:22050 4096) in
        same_bits ~msg:"defaults"
          (Stft.griffin_lim stft (mel_to_stft mel m))
          (mel_to_audio stft mel m) ;
        same_bits ~msg:"power 1, 3 iterations, length 3000"
          (Stft.griffin_lim ~n_iter:3 ~length:3000 stft
             (mel_to_stft mel ~power:1. m) )
          (mel_to_audio stft mel ~power:1. ~n_iter:3 ~length:3000 m) )
  ; test "mfcc_to_audio is mel_to_audio of mfcc_to_mel" (fun () ->
        let mel =
          Mel.Config.create ~n_mels:64 ~sample_rate:22050 ~fft_size:512 ()
        in
        let stft = stft_config 512 in
        let c =
          Nx.cast Nx.float32
            (cepstrum mel ~n_mfcc:13 ~lifter:22.
               (mel_of mel (sawtooth ~sample_rate:22050 4096)) )
        in
        same_bits ~msg:"float32"
          (mel_to_audio stft mel ~n_iter:4 (mfcc_to_mel mel ~lifter:22. c))
          (mfcc_to_audio stft mel ~lifter:22. ~n_iter:4 c) ) ]

(* {1 Errors} *)

let error_tests =
  let mel = mel40 () in
  let c = Nx.ones Nx.float64 [|13; 4|] in
  let m = Nx.ones Nx.float64 [|40; 4|] in
  [ test "mfcc_to_mel validates its parameters and input" (fun () ->
        raises ~msg:"rank"
          (Invalid_argument
             "mfcc_to_mel: cannot invert a rank-1 tensor (the input must be \
              [...; n_mfcc; frames])" ) (fun () ->
            mfcc_to_mel mel (Nx.ones Nx.float64 [|13|]) ) ;
        raises ~msg:"n_mfcc"
          (Invalid_argument
             "mfcc_to_mel: cannot invert 41 cepstral coefficients into 40 mel \
              bands (n_mfcc must lie in [1, n_mels])" ) (fun () ->
            mfcc_to_mel mel (Nx.ones Nx.float64 [|41; 4|]) ) ;
        raises ~msg:"lifter"
          (Invalid_argument
             "mfcc_to_mel: cannot lifter with a coefficient of -1 (lifter must \
              be finite and non-negative)" ) (fun () ->
            mfcc_to_mel mel ~lifter:(-1.) c ) ;
        raises ~msg:"lifter nan"
          (Invalid_argument
             "mfcc_to_mel: cannot lifter with a coefficient of nan (lifter \
              must be finite and non-negative)" ) (fun () ->
            mfcc_to_mel mel ~lifter:Float.nan c ) ;
        raises ~msg:"vanishing weight"
          (Invalid_argument
             "mfcc_to_mel: cannot undo the lifter 2 at coefficient 2 (its \
              weight 1 + (lifter / 2) sin (pi (k + 1) / lifter) vanishes)" )
          (fun () -> mfcc_to_mel mel ~lifter:2. c ) ;
        raises ~msg:"reference"
          (Invalid_argument
             "mfcc_to_mel: cannot scale by a reference of 0 (reference must be \
              finite and positive)" ) (fun () ->
            mfcc_to_mel mel ~reference:0. c ) ;
        (* two coefficients keep the lifter 2 weights clear of zero *)
        equal ~msg:"lifter 2, two coefficients" (array int) [|40; 4|]
          (Nx.shape (mfcc_to_mel mel ~lifter:2. (Nx.ones Nx.float64 [|2; 4|]))) )
  ; test "mfcc_to_mel_stage validates when built and at the chunk" (fun () ->
        raises ~msg:"n_mfcc"
          (Invalid_argument
             "mfcc_to_mel_stage: cannot invert 0 cepstral coefficients into 40 \
              mel bands (n_mfcc must lie in [1, n_mels])" ) (fun () ->
            mfcc_to_mel_stage mel ~n_mfcc:0 () ) ;
        raises ~msg:"chunk"
          (Invalid_argument
             "mfcc_to_mel_stage: cannot invert 13 cepstral coefficients with a \
              stage built for 20" ) (fun () ->
            Pipeline.run ~source:(source ()) (mfcc_to_mel_stage mel ()) c ) ;
        raises ~msg:"rank"
          (Invalid_argument
             "mfcc_to_mel_stage: cannot invert a rank-1 chunk (the stage needs \
              [...; n_mfcc; frames])" ) (fun () ->
            Pipeline.run ~source:(source ()) (mfcc_to_mel_stage mel ())
              (Nx.ones Nx.float64 [|20|]) ) )
  ; test "mel_to_stft validates its parameters and input" (fun () ->
        raises ~msg:"power"
          (Invalid_argument
             "mel_to_stft: cannot use power = 0 (power must be finite and \
              positive)" ) (fun () -> mel_to_stft mel ~power:0. m ) ;
        raises ~msg:"rank"
          (Invalid_argument
             "mel_to_stft: cannot invert a rank-1 tensor (the input must be \
              [...; n_mels; frames])" ) (fun () ->
            mel_to_stft mel (Nx.ones Nx.float64 [|40|]) ) ;
        raises ~msg:"bands"
          (Invalid_argument
             "mel_to_stft: cannot invert 39 mel bands with a filterbank of 40"
          ) (fun () -> mel_to_stft mel (Nx.ones Nx.float64 [|39; 4|]) ) ;
        let bad v =
          Nx.create Nx.float64 [|40; 2|]
            (Array.init 80 (fun i -> if i = 33 then v else 1.))
        in
        List.iter
          (fun v ->
            raises ~msg:(Printf.sprintf "%g" v)
              (Invalid_argument
                 "mel_to_stft: cannot invert a mel spectrogram holding NaN or \
                  infinite values" ) (fun () -> mel_to_stft mel (bad v) ) )
          [Float.nan; Float.infinity] )
  ; test "mel_to_stft_stage validates when built and at the chunk" (fun () ->
        raises ~msg:"power"
          (Invalid_argument
             "mel_to_stft_stage: cannot use power = nan (power must be finite \
              and positive)" ) (fun () ->
            mel_to_stft_stage mel ~power:Float.nan () ) ;
        raises ~msg:"bands"
          (Invalid_argument
             "mel_to_stft_stage: cannot invert 13 mel bands with a stage built \
              for 40" ) (fun () ->
            Pipeline.run ~source:(source ()) (mel_to_stft_stage mel ()) c ) ;
        raises ~msg:"rank"
          (Invalid_argument
             "mel_to_stft_stage: cannot invert a rank-1 chunk (the stage needs \
              [...; n_mels; frames])" ) (fun () ->
            Pipeline.run ~source:(source ()) (mel_to_stft_stage mel ())
              (Nx.ones Nx.float64 [|40|]) ) ;
        raises ~msg:"nan"
          (Invalid_argument
             "mel_to_stft_stage: cannot invert a chunk holding NaN or infinite \
              values" ) (fun () ->
            Pipeline.run ~source:(source ()) (mel_to_stft_stage mel ())
              (Nx.full Nx.float64 [|40; 2|] Float.nan) ) )
  ; test "the audio inversions need agreeing configurations" (fun () ->
        let stft = stft_config 1024 in
        raises ~msg:"mel_to_audio"
          (Invalid_argument
             "mel_to_audio: cannot synthesise a 1024-point STFT from a \
              filterbank built for an FFT of size 2048 (the two configurations \
              must agree on fft_size)" ) (fun () -> mel_to_audio stft mel m ) ;
        raises ~msg:"mfcc_to_audio"
          (Invalid_argument
             "mfcc_to_audio: cannot synthesise a 1024-point STFT from a \
              filterbank built for an FFT of size 2048 (the two configurations \
              must agree on fft_size)" ) (fun () -> mfcc_to_audio stft mel c ) ;
        raises ~msg:"mel_to_audio, input"
          (Invalid_argument
             "mel_to_audio: cannot invert 13 mel bands with a filterbank of 40"
          ) (fun () -> mel_to_audio (stft_config 2048) mel c ) ;
        raises ~msg:"mfcc_to_audio, input"
          (Invalid_argument
             "mfcc_to_audio: cannot scale by a reference of -1 (reference must \
              be finite and positive)" ) (fun () ->
            mfcc_to_audio (stft_config 2048) mel ~reference:(-1.) c ) ;
        raises ~msg:"mfcc_to_audio, NaN cepstrum"
          (Invalid_argument
             "mfcc_to_audio: cannot invert a mel spectrogram holding NaN or \
              infinite values" ) (fun () ->
            mfcc_to_audio (stft_config 2048) mel
              (Nx.full Nx.float64 [|13; 4|] Float.nan) ) )
  ; test "the audio inversions check Griffin-Lim's terms before solving"
      (fun () ->
        (* reported under the function's own name, which Stft.griffin_lim,
           reached only after the solve, would not use *)
        let stft = stft_config 2048 in
        raises ~msg:"n_iter"
          (Invalid_argument
             "mel_to_audio: cannot run 0 iterations (n_iter must be at least 1)"
          ) (fun () -> mel_to_audio stft mel ~n_iter:0 m ) ;
        raises ~msg:"length"
          (Invalid_argument
             "mfcc_to_audio: cannot synthesise a signal of length -1 (length \
              must be non-negative)" ) (fun () ->
            mfcc_to_audio stft mel ~length:(-1) c ) ;
        raises ~msg:"invertible"
          (Invalid_argument
             "mel_to_audio: cannot invert a 2048-point window advanced by 2048 \
              samples inside a 2048-point frame (the overlap-added squared \
              window must stay above 1e-10 of its largest value at every \
              position)" ) (fun () ->
            mel_to_audio
              (Stft.Config.create ~pad:(`Constant 0.) ~hop:2048 ~fft_size:2048
                 () )
              mel m ) ;
        raises ~msg:"power"
          (Invalid_argument
             "mfcc_to_audio: cannot use power = -2 (power must be finite and \
              positive)" ) (fun () -> mfcc_to_audio stft mel ~power:(-2.) c ) )
  ]

let suite =
  [ group "mfcc_to_mel" cepstral_tests
  ; group "re-projection" reprojection_tests
  ; group "unfittable" optimality_tests
  ; group "mel_to_stft" mel_to_stft_tests
  ; group "audio" audio_tests
  ; group "errors" error_tests ]
