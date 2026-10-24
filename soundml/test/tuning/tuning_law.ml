(* The laws of the tuning module, checked without an oracle:

   - the pipeline law for piptrack_stage — every partitioning of a spectrogram
   (one-frame chunks, the whole, random sizes, occasional empty chunks) gives
   the bits of piptrack on the whole, and Stft.power_stage ~power:1. c >>
   piptrack_stage streams piptrack of the magnitude spectrogram; - piptrack's
   own structure — frames and channels are independent, a power of two scales
   the magnitudes and nothing else, every detection is a peak of the definition
   and every peak of the definition is detected, the top bin of an odd transform
   can peak and the Nyquist bin cannot; - pitch_tuning on synthesised ladders,
   skipped entries and empty sets; - estimate_tuning's agreement with its
   spectrogram face, its pooling, its scale invariance, and its accuracy on
   tones within the interpolation bias; - and the errors. *)

open Windtrap
open Soundml

let seed = 0x7e11e

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

let rec random_sizes rng ~max n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (Int.min n max) in
    s :: random_sizes rng ~max (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

let partitions ?(max = 7) rng n =
  let named =
    [("ones", List.init n (fun _ -> 1)); ("whole", if n = 0 then [] else [n])]
  in
  let random =
    List.init 4 (fun i ->
        (Printf.sprintf "random-%d" i, random_sizes rng ~max n) )
  in
  let random_empty =
    List.init 2 (fun i ->
        ( Printf.sprintf "random-with-empty-chunks-%d" i
        , sprinkle_empty rng (random_sizes rng ~max n) ) )
  in
  named @ random @ random_empty

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

let concat_pairs = function
  | [] ->
      None
  | l ->
      Some
        ( Nx.concatenate ~axis:(-1) (List.map fst l)
        , Nx.concatenate ~axis:(-1) (List.map snd l) )

let bits x = Nx.to_array (Nx.cast Nx.float64 x)

let equal_pair ~msg (ep, em) (gp, gm) =
  equal ~msg:(msg ^ ": shape") (array int) (Nx.shape ep) (Nx.shape gp) ;
  equal ~msg:(msg ^ ": pitches") (array float_exact) (bits ep) (bits gp) ;
  equal ~msg:(msg ^ ": magnitudes") (array float_exact) (bits em) (bits gm)

let lcg n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

(* A magnitude spectrogram with peaks, plateaus and zeros: |LCG| values
   quantized to 1/64, so equal neighbours and exact zeros occur. *)
let spectrogram dtype shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create dtype shape
    (Array.map (fun v -> Float.floor (Float.abs v *. 64.) /. 64.) (lcg n))

let tone ?(n = 22050) ?(sample_rate = 22050) partials =
  let sr = Float.of_int sample_rate in
  Array.init n (fun i ->
      let t = Float.of_int i /. sr in
      List.fold_left
        (fun y (f, a) ->
          let v = a *. Float.sin (2. *. Float.pi *. f *. t) in
          y +. v )
        0. partials )

let cents f c = f *. Float.pow 2. (c /. 1200.)

let analysis ?(fft_size = 2048) ?hop () =
  Stft.Config.create ~fft_size ?hop ~pad:(`Constant 0.) ()

(* {1 The stage} *)

let stage_law (type a) ~name (dtype : (float, a) Nx.dtype) ?threshold ?reference
    ?fft_size ?fmax ~sample_rate shape =
  test name (fun () ->
      let rng = Random.State.make [|seed; Array.length shape|] in
      let x = spectrogram dtype shape in
      let expected =
        piptrack ?threshold ?reference ?fft_size ?fmax ~sample_rate x
      in
      let p =
        piptrack_stage ?threshold ?reference ?fft_size ?fmax ~sample_rate ()
      in
      equal_pair ~msg:"run" expected (Pipeline.run ~source:(source ()) p x) ;
      let frames = shape.(Array.length shape - 1) in
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left Int.max 1 sizes in
          match
            concat_pairs
              (stream_outputs p ~source:(source ()) ~max_chunk
                 (chunks_of x sizes) )
          with
          | None ->
              failf "%s: no output" pname
          | Some got ->
              equal_pair ~msg:pname expected got )
        (partitions rng frames) )

let stage_tests =
  [ stage_law ~name:"float64, [129; 37], defaults" Nx.float64 ~sample_rate:22050
      [|129; 37|]
  ; stage_law ~name:"float32, [2; 65; 23], threshold 0.3" Nx.float32
      ~threshold:0.3 ~sample_rate:8000 [|2; 65; 23|]
  ; stage_law ~name:"float64, [64; 19], fixed reference, odd FFT" Nx.float64
      ~reference:(`Value 0.5) ~fft_size:127 ~fmax:1e9 ~sample_rate:16000
      [|64; 19|]
  ; test "power_stage >> piptrack_stage streams piptrack" (fun () ->
        let rng = Random.State.make [|seed; 2|] in
        let sample_rate = 16000 in
        let c = analysis ~fft_size:512 ~hop:128 () in
        let x =
          Nx.create Nx.float64 [|6000|]
            (tone ~n:6000 ~sample_rate
               [(cents 440. 23., 0.5); (cents 659.255 (-31.), 0.3)] )
        in
        let expected =
          piptrack ~sample_rate (Stft.power_spectrum ~power:1. c x)
        in
        let p =
          Pipeline.(
            Stft.power_stage ~power:1. c >> piptrack_stage ~sample_rate () )
        in
        let source =
          Pipeline.Format.audio Nx.float64 ~sample_rate ~channels:1
        in
        equal_pair ~msg:"run" expected (Pipeline.run ~source p x) ;
        List.iter
          (fun (pname, sizes) ->
            let max_chunk = List.fold_left Int.max 1 sizes in
            match
              concat_pairs
                (stream_outputs p ~source ~max_chunk (chunks_of x sizes))
            with
            | None ->
                failf "%s: no output" pname
            | Some got ->
                equal_pair ~msg:pname expected got )
          ( ("333", random_sizes (Random.State.make [|0|]) ~max:333 6000)
          :: List.filter
               (fun (n, _) -> n <> "ones")
               (partitions ~max:700 rng 6000) ) )
  ; test "no latency, frame rate kept" (fun () ->
        let p = piptrack_stage ~sample_rate:22050 () in
        equal ~msg:"latency" rate_t (r 0 1) (Pipeline.latency p) ;
        equal ~msg:"rate" rate_t (r 1 1) (Pipeline.rate p) )
  ; test "an empty chunk maps to an empty pair" (fun () ->
        let p, m =
          Pipeline.run ~source:(source ())
            (piptrack_stage ~sample_rate:22050 ())
            (Nx.zeros Nx.float64 [|129; 0|])
        in
        equal ~msg:"pitches" (array int) [|129; 0|] (Nx.shape p) ;
        equal ~msg:"magnitudes" (array int) [|129; 0|] (Nx.shape m) ) ]

(* {1 piptrack} *)

(* [peaks_by_definition ~threshold ~reference ~fmin ~fmax ~fft_size ~sample_rate
   col] is the bins of the frame [col] that librosa's definition detects,
   evaluated naively: the thresholded frame, its local maxima, and the band in
   rfftfreq arithmetic. *)
let peaks_by_definition ~threshold ~reference ~fmin ~fmax ~fft_size ~sample_rate
    col =
  let n = Array.length col in
  let rv =
    match reference with
    | `Value v ->
        v
    | `Maximum ->
        threshold *. Array.fold_left Float.max 0. col
  in
  let z = Array.map (fun v -> if v > rv then v else 0.) col in
  let fmin = Float.max fmin 0.
  and fmax = Float.min fmax (Float.of_int sample_rate /. 2.) in
  let step =
    1.0 /. (Float.of_int fft_size *. (1.0 /. Float.of_int sample_rate))
  in
  List.filter
    (fun k ->
      let f = Float.of_int k *. step in
      let peak =
        if k = n - 1 then z.(k) > z.(k - 1)
        else z.(k) > z.(k - 1) && z.(k) >= z.(k + 1)
      in
      peak && fmin <= f && f < fmax )
    (List.init (n - 1) (fun k -> k + 1))

let definition_law ~name ?(threshold = 0.1) ?(reference = `Maximum)
    ?(fmin = 150.) ?(fmax = 4000.) ?fft_size ~sample_rate shape =
  test name (fun () ->
      let s = spectrogram Nx.float64 shape in
      let bins = shape.(0) and frames = shape.(1) in
      let n = Option.value fft_size ~default:(2 * (bins - 1)) in
      let pitches, mags =
        piptrack ~threshold ~reference ~fmin ~fmax ?fft_size ~sample_rate s
      in
      let sv = Nx.to_array s
      and pv = Nx.to_array pitches
      and mv = Nx.to_array mags in
      let width = Float.of_int sample_rate /. Float.of_int n in
      for t = 0 to frames - 1 do
        let col = Array.init bins (fun k -> sv.((k * frames) + t)) in
        let expected =
          peaks_by_definition ~threshold ~reference ~fmin ~fmax ~fft_size:n
            ~sample_rate col
        in
        let got =
          List.filter
            (fun k -> pv.((k * frames) + t) <> 0.)
            (List.init bins Fun.id)
        in
        equal
          ~msg:(Printf.sprintf "frame %d: support" t)
          (list int) expected got ;
        List.iter
          (fun k ->
            let i = (k * frames) + t in
            let centre = Float.of_int k *. width in
            if not (Float.abs (pv.(i) -. centre) < width) then
              failf "frame %d, bin %d: pitch %g more than a bin from %g" t k
                pv.(i) centre ;
            if not (mv.(i) >= sv.(i)) then
              failf "frame %d, bin %d: magnitude %g below the bin's %g" t k
                mv.(i) sv.(i) )
          got ;
        Array.iteri
          (fun k _ ->
            let i = (k * frames) + t in
            if pv.(i) = 0. <> (mv.(i) = 0.) then
              failf "frame %d, bin %d: pitch and magnitude supports differ" t k )
          col
      done )

let piptrack_tests =
  [ definition_law ~name:"detections are the definition's, defaults"
      ~sample_rate:22050 [|129; 31|]
  ; definition_law ~name:"detections are the definition's, threshold 0"
      ~threshold:0. ~fmin:0. ~fmax:1e9 ~sample_rate:8000 [|65; 17|]
  ; definition_law ~name:"detections are the definition's, fixed reference"
      ~reference:(`Value 0.6) ~fmin:(-3.) ~fmax:3000. ~sample_rate:16000
      [|128; 13|] ~fft_size:255
  ; definition_law ~name:"detections are the definition's, empty band"
      ~fmin:3000. ~fmax:2000. ~sample_rate:22050 [|65; 9|]
  ; test "frames and channels are independent" (fun () ->
        let x = spectrogram Nx.float64 [|2; 3; 65; 21|] in
        let p, m = piptrack ~sample_rate:8000 x in
        for i = 0 to 1 do
          for j = 0 to 2 do
            let one = Nx.slice [Nx.I i; Nx.I j; Nx.A; Nx.A] in
            equal_pair
              ~msg:(Printf.sprintf "channel %d.%d" i j)
              (piptrack ~sample_rate:8000 (one x))
              (one p, one m)
          done
        done ;
        let frames = Nx.slice [Nx.A; Nx.A; Nx.A; Nx.R (5, 12)] in
        equal_pair ~msg:"frames 5..12"
          (piptrack ~sample_rate:8000 (frames x))
          (frames p, frames m) )
  ; test "a power of two scales the magnitudes alone" (fun () ->
        let x = spectrogram Nx.float64 [|129; 23|] in
        let scaled = Nx.mul_s x 8. in
        let check ~msg reference reference' =
          let p, m = piptrack ~reference ~sample_rate:22050 x in
          let p', m' =
            piptrack ~reference:reference' ~sample_rate:22050 scaled
          in
          equal ~msg:(msg ^ ": pitches") (array float_exact) (bits p) (bits p') ;
          equal ~msg:(msg ^ ": magnitudes") (array float_exact)
            (bits (Nx.mul_s m 8.))
            (bits m')
        in
        check ~msg:"maximum" `Maximum `Maximum ;
        check ~msg:"value" (`Value 0.4) (`Value 3.2) )
  ; test
      "the top bin of an odd transform can peak; at 8 kHz, the Nyquist bin of \
       a 64-point one lies outside the band" (fun () ->
        (* a ramp: every frame rises to its top bin *)
        let bins = 33 in
        let x =
          Nx.create Nx.float64 [|bins; 2|]
            (Array.init (2 * bins) (fun i -> Float.of_int ((i / 2) + 1)))
        in
        let top = Nx.slice [Nx.I (bins - 1); Nx.A] in
        let p, m =
          piptrack ~threshold:0. ~fmax:1e9 ~fft_size:65 ~sample_rate:8000 x
        in
        equal ~msg:"odd: pitch" (array float_exact)
          [|32. *. 8000. /. 65.; 32. *. 8000. /. 65.|]
          (bits (top p)) ;
        equal ~msg:"odd: magnitude" (array float_exact) [|33.; 33.|]
          (bits (top m)) ;
        let p, _ = piptrack ~threshold:0. ~fmax:1e9 ~sample_rate:8000 x in
        equal ~msg:"even: Nyquist" (array float_exact) [|0.; 0.|] (bits (top p)) ;
        equal ~msg:"even: no peak at all" (array float_exact)
          (Array.make (2 * bins) 0.)
          (bits p) )
  ; test "no frames, or a zero-size axis, maps to empty tensors" (fun () ->
        List.iter
          (fun shape ->
            let p, m =
              piptrack ~sample_rate:22050 (Nx.zeros Nx.float32 shape)
            in
            equal ~msg:"pitches" (array int) shape (Nx.shape p) ;
            equal ~msg:"magnitudes" (array int) shape (Nx.shape m) )
          [[|129; 0|]; [|0; 129; 5|]] )
  ; test "silence has no peak" (fun () ->
        let p, m =
          piptrack ~sample_rate:22050 (Nx.zeros Nx.float64 [|129; 4|])
        in
        equal ~msg:"pitches" (array float_exact)
          (Array.make (129 * 4) 0.)
          (bits p) ;
        equal ~msg:"magnitudes" (array float_exact)
          (Array.make (129 * 4) 0.)
          (bits m) )
  ; test "float16 is the double result rounded once" (fun () ->
        let x = Nx.cast Nx.float16 (spectrogram Nx.float64 [|65; 9|]) in
        let p, m = piptrack ~sample_rate:8000 x in
        let p', m' = piptrack ~sample_rate:8000 (Nx.cast Nx.float64 x) in
        equal_pair ~msg:"float16"
          (Nx.cast Nx.float16 p', Nx.cast Nx.float16 m')
          (p, m) ) ]

(* {1 pitch_tuning} *)

let ladder ?(bins_per_octave = 12) f0 c count =
  let bpo = Float.of_int bins_per_octave in
  Array.init count (fun i ->
      f0 *. Float.pow 2. ((Float.of_int i /. bpo) +. (c /. (100. *. bpo))) )

let pt ?resolution ?bins_per_octave values =
  pitch_tuning ?resolution ?bins_per_octave
    (Nx.create Nx.float64 [|Array.length values|] values)

let pitch_tuning_tests =
  [ test "mid-bin ladders give the left edge of their bin" (fun () ->
        List.iter
          (fun (c, expected) ->
            equal
              ~msg:(Printf.sprintf "%g cents" c)
              (float 1e-12) expected
              (pt (ladder 55. c 30)) )
          [ (-49.5, -0.5)
          ; (-33.5, -0.34)
          ; (-12.5, -0.13)
          ; (-0.5, -0.01)
          ; (0.5, 0.)
          ; (18.5, 0.18)
          ; (49.5, 0.49) ] ;
        equal ~msg:"0.3: four bins" (float 1e-12) 0.
          (pt ~resolution:0.3 (ladder 55. 10.5 30)) ;
        equal ~msg:"36 per octave" (float 1e-12) (-0.17)
          (pt ~bins_per_octave:36 (ladder ~bins_per_octave:36 55. (-16.5) 40)) )
  ; test "NaN, zero and negative frequencies are skipped" (fun () ->
        let base = ladder 110. 21.5 9 in
        equal ~msg:"skipped" float_exact (pt base)
          (pt (Array.concat [[|Float.nan; 0.; -440.|]; base; [|-0.|]])) )
  ; test "empty and infinite sets" (fun () ->
        equal ~msg:"empty" float_exact 0. (pt [||]) ;
        equal ~msg:"non-positive" float_exact 0. (pt [|0.; -1.; Float.nan|]) ;
        (* librosa's own result: an infinity passes the positivity test but has
           no residual, so every bin is empty and the first edge wins *)
        equal ~msg:"infinite" float_exact (-0.5) (pt [|Float.infinity|]) )
  ; test "any shape and float dtype, widened exactly" (fun () ->
        let base = ladder 98. (-27.5) 12 in
        let f32 =
          Array.map (fun v -> Int32.float_of_bits (Int32.bits_of_float v)) base
        in
        equal ~msg:"float32" float_exact (pt f32)
          (pitch_tuning (Nx.create Nx.float32 [|12|] f32)) ;
        equal ~msg:"rank 2" float_exact (pt base)
          (pitch_tuning (Nx.create Nx.float64 [|3; 4|] base)) )
  ; test "the most populated bin wins, the lowest on a tie" (fun () ->
        equal ~msg:"majority" (float 1e-12) 0.12
          (pt (Array.append (ladder 220. 12.5 7) (ladder 55. (-30.5) 5))) ;
        equal ~msg:"tie" (float 1e-12) (-0.31)
          (pt (Array.append (ladder 220. 30.5 4) (ladder 55. (-30.5) 4))) ) ]

(* {1 estimate_tuning} *)

let a4 ?n c =
  Nx.create Nx.float64
    [|Option.value n ~default:22050|]
    (tone ?n [(cents 440. c, 0.5)])

let estimate_tests =
  [ test "A4 tones are estimated within the interpolation bias" (fun () ->
        (* the estimate is a left edge, up to one 0.01 bin below the tone, and
           carries the parabola's bias on a Hann analysis: over 27 detunings
           spread from -49.5 to 47.5 cents the error stays within 0.025, so 0.04
           bounds both with room *)
        List.iter
          (fun c ->
            let e = estimate_tuning (analysis ()) ~sample_rate:22050 (a4 c) in
            if not (Float.abs (e -. (c /. 100.)) <= 0.04) then
              failf "%g cents estimated at %g" c e )
          [-49.5; -41.5; -29.5; -21.5; -12.5; 0.5; 3.5; 14.5; 22.5; 36.5; 47.5] )
  ; test "the audio face is the spectrogram face of the magnitudes" (fun () ->
        let c = analysis ~fft_size:1024 ~hop:256 () in
        List.iter
          (fun x ->
            equal ~msg:"estimate" float_exact
              (estimate_tuning_of_spectrogram ~fft_size:1024 ~sample_rate:22050
                 (Stft.power_spectrum ~power:1. c x) )
              (estimate_tuning c ~sample_rate:22050 x) )
          [a4 13.5; a4 (-36.5); Nx.create Nx.float64 [|9000|] (lcg 9000)] )
  ; test "leading axes pool into one estimate" (fun () ->
        let c = analysis () in
        let s1 = Stft.power_spectrum ~power:1. c (a4 23.5)
        and s2 =
          Stft.power_spectrum ~power:1. c
            (Nx.create Nx.float64 [|22050|]
               (tone [(cents 329.628 (-14.5), 0.3)]) )
        in
        equal ~msg:"pooled" float_exact
          (estimate_tuning_of_spectrogram ~sample_rate:22050
             (Nx.concatenate ~axis:(-1) [s1; s2]) )
          (estimate_tuning_of_spectrogram ~sample_rate:22050
             (Nx.stack [s1; s2]) ) )
  ; test "a power of two changes nothing" (fun () ->
        let c = analysis () in
        let x = a4 (-21.5) in
        equal ~msg:"scaled" float_exact
          (estimate_tuning c ~sample_rate:22050 x)
          (estimate_tuning c ~sample_rate:22050 (Nx.mul_s x 4.)) )
  ; test "float32 audio is the float64 audio of the same samples" (fun () ->
        let c = analysis () in
        let x32 = Nx.cast Nx.float32 (a4 31.5) in
        equal ~msg:"float32" float_exact
          (estimate_tuning c ~sample_rate:22050 (Nx.cast Nx.float64 x32))
          (estimate_tuning c ~sample_rate:22050 x32) )
  ; test "silence and empty audio have tuning zero" (fun () ->
        let c = analysis () in
        equal ~msg:"zeros" float_exact 0.
          (estimate_tuning c ~sample_rate:22050 (Nx.zeros Nx.float64 [|4096|])) ;
        equal ~msg:"empty" float_exact 0.
          (estimate_tuning c ~sample_rate:22050 (Nx.zeros Nx.float64 [|0|])) ;
        equal ~msg:"empty spectrogram" float_exact 0.
          (estimate_tuning_of_spectrogram ~sample_rate:22050
             (Nx.zeros Nx.float32 [|1025; 0|]) ) ) ]

(* {1 Errors} *)

let spec = spectrogram Nx.float64 [|65; 4|]

let error_tests =
  [ test "piptrack's parameters are validated" (fun () ->
        raises ~msg:"sample_rate"
          (Invalid_argument
             "piptrack: cannot track pitches at 0 Hz (sample_rate must be at \
              least 1)" ) (fun () -> ignore (piptrack ~sample_rate:0 spec) ) ;
        raises ~msg:"fmin"
          (Invalid_argument
             "piptrack: cannot bound the band at fmin = nan (fmin must be a \
              number)" ) (fun () ->
            ignore (piptrack ~fmin:Float.nan ~sample_rate:8000 spec) ) ;
        raises ~msg:"fmax"
          (Invalid_argument
             "piptrack_stage: cannot bound the band at fmax = nan (fmax must \
              be a number)" ) (fun () ->
            ignore (piptrack_stage ~fmax:Float.nan ~sample_rate:8000 ()) ) ;
        raises ~msg:"threshold"
          (Invalid_argument
             "piptrack: cannot use threshold = -0.1 (threshold must be finite \
              and non-negative)" ) (fun () ->
            ignore (piptrack ~threshold:(-0.1) ~sample_rate:8000 spec) ) ;
        raises ~msg:"threshold inf"
          (Invalid_argument
             "piptrack_stage: cannot use threshold = inf (threshold must be \
              finite and non-negative)" ) (fun () ->
            ignore
              (piptrack_stage ~threshold:Float.infinity ~sample_rate:8000 ()) ) ;
        raises ~msg:"reference"
          (Invalid_argument
             "piptrack: cannot compare magnitudes with `Value -1 (a fixed \
              reference must be finite and non-negative)" ) (fun () ->
            ignore (piptrack ~reference:(`Value (-1.)) ~sample_rate:8000 spec) ) ;
        raises ~msg:"fft_size"
          (Invalid_argument
             "piptrack: cannot track pitches with an FFT of size 1 (fft_size \
              must be at least 2)" ) (fun () ->
            ignore (piptrack ~fft_size:1 ~sample_rate:8000 spec) ) )
  ; test "piptrack's spectrogram is validated" (fun () ->
        raises ~msg:"rank"
          (Invalid_argument
             "piptrack: cannot analyse a rank-1 tensor (a spectrogram is [...; \
              bins; frames])" ) (fun () ->
            ignore (piptrack ~sample_rate:8000 (Nx.ones Nx.float64 [|9|])) ) ;
        raises ~msg:"one bin"
          (Invalid_argument
             "piptrack: cannot track pitches in a 1-bin spectrogram (a \
              spectrogram has at least 2 bins)" ) (fun () ->
            ignore (piptrack ~sample_rate:8000 (Nx.ones Nx.float64 [|1; 9|])) ) ;
        raises ~msg:"fft_size pairing"
          (Invalid_argument
             "piptrack: cannot track 65 bins with an FFT of size 130 (fft_size \
              / 2 + 1 must be the bin count)" ) (fun () ->
            ignore (piptrack ~fft_size:130 ~sample_rate:8000 spec) ) ;
        let negative =
          Invalid_argument
            "piptrack: cannot analyse a spectrogram with negative or NaN \
             values (a magnitude spectrogram is non-negative)"
        in
        raises ~msg:"negative" negative (fun () ->
            ignore (piptrack ~sample_rate:8000 (Nx.neg spec)) ) ;
        raises ~msg:"nan" negative (fun () ->
            ignore
              (piptrack ~sample_rate:8000
                 (Nx.create Nx.float32 [|2; 2|] [|1.; Float.nan; 0.; 1.|]) ) ) )
  ; test "the stage checks each chunk" (fun () ->
        raises ~msg:"negative chunk"
          (Invalid_argument
             "piptrack_stage: cannot analyse a spectrogram with negative or \
              NaN values (a magnitude spectrogram is non-negative)" ) (fun () ->
            ignore
              (stream_outputs
                 (piptrack_stage ~sample_rate:8000 ())
                 ~source:(source ()) ~max_chunk:4
                 [spec; Nx.neg spec] ) ) )
  ; test "pitch_tuning's parameters are validated" (fun () ->
        let f = Nx.create Nx.float64 [|1|] [|440.|] in
        List.iter
          (fun resolution ->
            raises
              ~msg:(Printf.sprintf "resolution %g" resolution)
              (Invalid_argument
                 (Printf.sprintf
                    "pitch_tuning: cannot estimate the tuning at a resolution \
                     of %g (resolution must lie in [1e-9, 1))"
                    resolution ) )
              (fun () -> ignore (pitch_tuning ~resolution f)) )
          [0.; 1.; 1e-10; Float.nan; -0.01] ;
        raises ~msg:"bins_per_octave"
          (Invalid_argument
             "pitch_tuning: cannot divide the octave into 0 bins \
              (bins_per_octave must be at least 1)" ) (fun () ->
            ignore (pitch_tuning ~bins_per_octave:0 f) ) )
  ; test "estimate_tuning's inputs are validated" (fun () ->
        raises ~msg:"rank zero"
          (Invalid_argument
             "estimate_tuning: cannot analyse a rank-zero tensor (the time \
              axis must exist)" ) (fun () ->
            ignore
              (estimate_tuning (analysis ()) ~sample_rate:22050
                 (Nx.scalar Nx.float64 1.) ) ) ;
        raises ~msg:"resolution"
          (Invalid_argument
             "estimate_tuning: cannot estimate the tuning at a resolution of 2 \
              (resolution must lie in [1e-9, 1))" ) (fun () ->
            ignore
              (estimate_tuning (analysis ()) ~resolution:2. ~sample_rate:22050
                 (a4 0.) ) ) ;
        raises ~msg:"negative spectrogram"
          (Invalid_argument
             "estimate_tuning_of_spectrogram: cannot analyse a spectrogram \
              with negative or NaN values (a magnitude spectrogram is \
              non-negative)" ) (fun () ->
            ignore
              (estimate_tuning_of_spectrogram ~sample_rate:8000 (Nx.neg spec)) ) ;
        raises ~msg:"bins_per_octave"
          (Invalid_argument
             "estimate_tuning_of_spectrogram: cannot divide the octave into -3 \
              bins (bins_per_octave must be at least 1)" ) (fun () ->
            ignore
              (estimate_tuning_of_spectrogram ~bins_per_octave:(-3)
                 ~sample_rate:8000 spec ) ) ) ]

let suite =
  [ group "stage" stage_tests
  ; group "piptrack" piptrack_tests
  ; group "pitch_tuning" pitch_tuning_tests
  ; group "estimate_tuning" estimate_tests
  ; group "errors" error_tests ]
