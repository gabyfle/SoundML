(* Laws of the synthetic signals and the mu-law, checked bit for bit where they
   are exact:

   - a tone is computed from each sample's index, so a shorter tone is a prefix
   of a longer one, whatever the split of the work across domains; - every
   float32 or narrower tone, chirp and continuous mu-law value is the float64
   value rounded once; - a chirp holds exactly the samples asked for (librosa
   sometimes drops one); - a click track is the sequential placement loop,
   rounding after every click, at every split of the track across domains, and
   its positions agree whether given as times, frames or samples; - the mu-law
   round trip, the codes' range and order, and the stages, which stream the flat
   functions bit for bit on every partitioning.

   Also here: every precondition's error. *)

open Windtrap
open Soundml

let seed = 0x5ce7e

let values t = Nx.to_array (Nx.cast Nx.float64 t)

(* [same ~msg a b]: [a] and [b] hold the same bits (float_exact tells -0 from
   0). *)
let same ~msg a b =
  equal ~msg (array int) (Nx.shape b) (Nx.shape a) ;
  equal ~msg (array float_exact) (values a) (values b)

let invalid ~msg substring f =
  raises_match ~msg (Exn.invalid_arg ~substring) (fun () -> ignore (f ()))

let prefix t m = Nx.slice [Nx.R (0, m)] t

(* {1 Tones} *)

(* Long enough for the work to split into several ranges across domains. *)
let long = (3 * 16384) + 17

let tone_tests =
  [ test "a shorter tone is a prefix of a longer one" (fun () ->
        let check (type a) (dtype : (float, a) Nx.dtype) name =
          let whole = tone dtype ~sample_rate:22050 ~length:long 997.3 in
          List.iter
            (fun m ->
              same
                ~msg:(Printf.sprintf "%s, %d samples" name m)
                (prefix whole m)
                (tone dtype ~sample_rate:22050 ~length:m 997.3) )
            [0; 1; 4095; 4096; 4097; 10_000; long]
        in
        check Nx.float64 "float64" ; check Nx.float32 "float32" )
  ; test "a duration is its samples truncated toward zero" (fun () ->
        (* 0.7 * 44100 = 30869.999999999996 *)
        same ~msg:"0.7 s at 44.1 kHz"
          (tone Nx.float64 ~sample_rate:44100 ~length:30869 440.)
          (tone Nx.float64 ~sample_rate:44100 ~duration:0.7 440.) ;
        same ~msg:"the length wins"
          (tone Nx.float64 ~sample_rate:22050 ~length:7 440.)
          (tone Nx.float64 ~sample_rate:22050 ~length:7 ~duration:1. 440.) )
  ; test "the default phase is -pi / 2, a sine" (fun () ->
        let t = tone Nx.float64 ~sample_rate:8000 ~length:100 250. in
        same ~msg:"default"
          (tone Nx.float64 ~sample_rate:8000 ~length:100
             ~phase:(-.Float.pi *. 0.5) 250. )
          t ;
        equal ~msg:"starts at zero" (float 1e-16) 0. (values t).(0) )
  ; test "narrower tones are the float64 tone rounded once" (fun () ->
        let t64 = tone Nx.float64 ~sample_rate:22050 ~length:long 440. in
        same ~msg:"float32" (Nx.cast Nx.float32 t64)
          (tone Nx.float32 ~sample_rate:22050 ~length:long 440.) ;
        same ~msg:"float16" (Nx.cast Nx.float16 t64)
          (tone Nx.float16 ~sample_rate:22050 ~length:long 440.) ;
        same ~msg:"bfloat16" (Nx.cast Nx.bfloat16 t64)
          (tone Nx.bfloat16 ~sample_rate:22050 ~length:long 440.) )
  ; test "a tone of no sample is empty" (fun () ->
        equal ~msg:"length 0" (array int) [|0|]
          (Nx.shape (tone Nx.float32 ~sample_rate:22050 ~length:0 440.)) ;
        equal ~msg:"duration below a sample" (array int) [|0|]
          (Nx.shape (tone Nx.float64 ~sample_rate:22050 ~duration:1e-5 440.)) )
  ]

(* {1 Chirps} *)

let chirp_tests =
  [ test "a chirp holds exactly the samples asked for" (fun () ->
        (* librosa returns one fewer for 7, 14, 27, 28, ... at 48 kHz *)
        List.iter
          (fun sample_rate ->
            for l = 1 to 200 do
              List.iter
                (fun linear ->
                  equal
                    ~msg:(Printf.sprintf "%d samples at %d Hz" l sample_rate)
                    (array int) [|l|]
                    (Nx.shape
                       (chirp Nx.float64 ~length:l ~linear ~sample_rate
                          ~fmin:110. ~fmax:880. () ) ) )
                [false; true]
            done )
          [48000; 22050] ;
        equal ~msg:"357 at 22.05 kHz" (array int) [|357|]
          (Nx.shape
             (chirp Nx.float32 ~length:357 ~sample_rate:22050 ~fmin:110.
                ~fmax:880. () ) ) )
  ; test "a duration is its samples truncated toward zero" (fun () ->
        equal ~msg:"0.2 s at 22.05 kHz" (array int) [|4410|]
          (Nx.shape
             (chirp Nx.float64 ~duration:0.2 ~sample_rate:22050 ~fmin:7040.
                ~fmax:110. () ) ) ;
        same ~msg:"the length wins"
          (chirp Nx.float64 ~length:300 ~sample_rate:22050 ~fmin:110. ~fmax:880.
             () )
          (chirp Nx.float64 ~length:300 ~duration:1. ~sample_rate:22050
             ~fmin:110. ~fmax:880. () ) )
  ; test "narrower chirps are the float64 chirp rounded once" (fun () ->
        List.iter
          (fun (name, linear, fmin, fmax) ->
            let run dtype =
              chirp dtype ~length:long ~linear ~sample_rate:22050 ~fmin ~fmax ()
            in
            let c64 = run Nx.float64 in
            same ~msg:(name ^ ", float32") (Nx.cast Nx.float32 c64)
              (run Nx.float32) ;
            same ~msg:(name ^ ", float16") (Nx.cast Nx.float16 c64)
              (run Nx.float16) )
          [ ("exponential", false, 110., 7040.)
          ; ("linear", true, 7040., 0.)
          ; ("equal", false, 440., 440.) ] )
  ; test "a chirp of no sample is empty" (fun () ->
        List.iter
          (fun linear ->
            equal ~msg:"length 0" (array int) [|0|]
              (Nx.shape
                 (chirp Nx.float64 ~length:0 ~linear ~sample_rate:22050
                    ~fmin:110. ~fmax:880. () ) ) ;
            equal ~msg:"duration 0" (array int) [|0|]
              (Nx.shape
                 (chirp Nx.float64 ~duration:0. ~linear ~sample_rate:22050
                    ~fmin:110. ~fmax:880. () ) ) )
          [false; true] ) ]

(* {1 Click tracks} *)

let lcg_values n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let stereo_click dtype = Nx.create dtype [|2; 64|] (lcg_values 128)

(* [to_f32 v] is [v] rounded to float32. *)
let to_f32 v = Int32.float_of_bits (Int32.bits_of_float v)

(* [placement ~round ~rows ~m wave positions length] is librosa's placement loop
   written out: a zero track of [rows] rows, each click of [m] samples added in
   the order given over [\[p, min (p + m, length))], every sum passed through
   [round]. *)
let placement ~round ~rows ~m wave positions length =
  let track = Array.make (rows * length) 0. in
  Array.iter
    (fun p ->
      if p < length then
        for r = 0 to rows - 1 do
          for j = 0 to Int.min m (length - p) - 1 do
            let i = (r * length) + p + j in
            track.(i) <- round (track.(i) +. wave.((r * m) + j))
          done
        done )
    positions ;
  track

let clicks_tests =
  let sr = 22050 in
  let default_click dtype = clicks dtype ~sample_rate:sr (`Samples [|0|]) in
  [ test "a lone click is the click, shifted" (fun () ->
        let check (type a) (dtype : (float, a) Nx.dtype) name =
          let c = default_click dtype in
          equal ~msg:name (array int) [|2205|] (Nx.shape c) ;
          same
            ~msg:(name ^ ", ending at the length")
            c
            (clicks dtype ~length:2205 ~sample_rate:sr (`Samples [|0|])) ;
          same ~msg:(name ^ ", shifted")
            (Nx.concatenate ~axis:0 [Nx.zeros dtype [|333|]; c])
            (clicks dtype ~sample_rate:sr (`Samples [|333|]))
        in
        check Nx.float64 "float64" ; check Nx.float32 "float32" )
  ; test "times, frames and samples place clicks alike" (fun () ->
        let by_samples =
          clicks Nx.float32 ~sample_rate:sr (`Samples [|0; 1536; 3584|])
        in
        same ~msg:"frames" by_samples
          (clicks Nx.float32 ~hop:512 ~sample_rate:sr (`Frames [|0; 3; 7|])) ;
        same ~msg:"times" by_samples
          (clicks Nx.float32 ~sample_rate:sr
             (`Times
                (Array.map
                   (fun k -> (Float.of_int k +. 0.5) /. Float.of_int sr)
                   [|0; 1536; 3584|] ) ) ) ;
        (* truncated toward zero, as librosa converts times *)
        same ~msg:"a time within a sample before zero"
          (default_click Nx.float32)
          (clicks Nx.float32 ~sample_rate:sr
             (`Times [|-0.4 /. Float.of_int sr|]) ) )
  ; test "separate clicks are shifted copies, in any order" (fun () ->
        let check (type a) (dtype : (float, a) Nx.dtype) name =
          let click = stereo_click dtype in
          let place p =
            clicks dtype ~click ~length:400 ~sample_rate:sr (`Samples p)
          in
          let sum =
            Nx.add (Nx.add (place [|0|]) (place [|100|])) (place [|300|])
          in
          List.iter
            (fun p -> same ~msg:(name ^ ", order") sum (place p))
            [[|0; 100; 300|]; [|300; 0; 100|]; [|100; 300; 0|]]
        in
        check Nx.float64 "float64" ; check Nx.float32 "float32" )
  ; test "the track is the placement loop, rounding after every click"
      (fun () ->
        (* overlapping, unsorted and repeated positions, the last ones cut by
           the end of a track long enough to split across domains *)
        let length = (5 * 32768) + 7 in
        let rng = Random.State.make [|seed|] in
        let positions =
          Array.append
            (Array.init 300 (fun _ -> Random.State.int rng length))
            [|length - 1; 17; 17; length - 1000|]
        in
        let wave = values (default_click Nx.float64) in
        let check (type a) (dtype : (float, a) Nx.dtype) name round =
          equal ~msg:name (array float_exact)
            (placement ~round ~rows:1 ~m:2205 wave positions length)
            (values (clicks dtype ~length ~sample_rate:sr (`Samples positions)))
        in
        check Nx.float64 "float64" Fun.id ;
        check Nx.float32 "float32" to_f32 ;
        let click = stereo_click Nx.float64 in
        equal ~msg:"stereo" (array float_exact)
          (placement ~round:Fun.id ~rows:2 ~m:64 (values click) positions length)
          (values
             (clicks Nx.float64 ~click ~length ~sample_rate:sr
                (`Samples positions) ) ) )
  ; test "narrower tracks sum in double and round once" (fun () ->
        let p = `Samples [|0; 441; 1102; 441|] in
        same ~msg:"float16"
          (Nx.cast Nx.float16 (clicks Nx.float64 ~sample_rate:sr p))
          (clicks Nx.float16 ~sample_rate:sr p) )
  ; test "positions at or past the length are dropped" (fun () ->
        let click = stereo_click Nx.float64 in
        same ~msg:"dropped"
          (clicks Nx.float64 ~click ~length:100 ~sample_rate:sr (`Samples [|50|]))
          (clicks Nx.float64 ~click ~length:100 ~sample_rate:sr
             (`Samples [|50; 100; 250|]) ) ;
        same ~msg:"no position"
          (Nx.zeros Nx.float32 [|10|])
          (clicks Nx.float32 ~length:10 ~sample_rate:sr (`Times [||])) ;
        (* int (22050 * 1e-5) = 0: an empty click *)
        same ~msg:"empty click"
          (Nx.zeros Nx.float64 [|5|])
          (clicks Nx.float64 ~click_duration:1e-5 ~sample_rate:sr
             (`Samples [|5; 2|]) ) ) ]

(* {1 Mu-law companding} *)

let grid n =
  Array.init n (fun i -> (2. *. Float.of_int i /. Float.of_int (n - 1)) -. 1.)

let mu_tests =
  let x64 = Nx.create Nx.float64 [|4001|] (grid 4001) in
  [ test "the continuous round trip" (fun () ->
        (* measured: 8.9e-16 at mu = 255, 5.6e-16 at mu = 15 *)
        List.iter
          (fun mu ->
            let back =
              values
                (mu_expand ~mu ~quantize:false
                   (mu_compress ~mu ~quantize:false x64) )
            in
            Array.iteri
              (fun i v ->
                equal
                  ~msg:(Printf.sprintf "mu %g, %g" mu v)
                  (float 1e-14) v back.(i) )
              (values x64) )
          [255.; 15.] )
  ; test "codes are whole, ordered and in range" (fun () ->
        List.iter
          (fun mu ->
            let codes = values (mu_compress ~mu x64) in
            let levels = Float.to_int (1. +. mu) in
            let offset = Float.to_int (mu +. 1.) / 2 in
            let msg = Printf.sprintf "mu %g" mu in
            Array.iteri
              (fun i c ->
                is_true ~msg:(msg ^ ", whole") (Float.is_integer c) ;
                if i > 0 then
                  is_true ~msg:(msg ^ ", ordered") (codes.(i - 1) <= c) )
              codes ;
            equal ~msg:(msg ^ ", at -1") float_exact (Float.of_int (-offset))
              codes.(0) ;
            equal ~msg:(msg ^ ", at 1") float_exact
              (Float.of_int (levels - 1 - offset))
              codes.(Array.length codes - 1) )
          [255.; 15.; 100.5; 1.; 254.; 65535.] )
  ; test "the expansion is odd" (fun () ->
        let codes =
          Nx.create Nx.float64 [|128|]
            (Array.init 128 (fun i -> Float.of_int (i + 1)))
        in
        same ~msg:"codes" (Nx.neg (mu_expand codes)) (mu_expand (Nx.neg codes)) ;
        let x =
          Nx.create Nx.float64 [|100|]
            (Array.map (fun v -> (v +. 1.01) /. 2.02) (lcg_values 100))
        in
        same ~msg:"continuous"
          (Nx.neg (mu_expand ~quantize:false x))
          (mu_expand ~quantize:false (Nx.neg x)) )
  ; test "narrower dtypes are the float64 values rounded once" (fun () ->
        let x32 = Nx.create Nx.float32 [|3; 501|] (lcg_values 1503) in
        let wide = Nx.cast Nx.float64 x32 in
        List.iter
          (fun quantize ->
            let msg = if quantize then "codes" else "continuous" in
            same
              ~msg:("compress float32, " ^ msg)
              (Nx.cast Nx.float32 (mu_compress ~quantize wide))
              (mu_compress ~quantize x32) ;
            let x16 = Nx.cast Nx.float16 x32 in
            same
              ~msg:("compress float16, " ^ msg)
              (Nx.cast Nx.float16
                 (mu_compress ~quantize (Nx.cast Nx.float64 x16)) )
              (mu_compress ~quantize x16) )
          [false; true] ;
        same ~msg:"expand float32"
          (Nx.cast Nx.float32 (mu_expand ~quantize:false wide))
          (mu_expand ~quantize:false x32) ;
        let codes =
          Nx.create Nx.bfloat16 [|257|]
            (Array.init 257 (fun i -> Float.of_int (i - 128)))
        in
        same ~msg:"expand bfloat16 codes"
          (Nx.cast Nx.bfloat16 (mu_expand (Nx.cast Nx.float64 codes)))
          (mu_expand codes) ) ]

(* {1 Stages} *)

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (Int.min n 97) in
    s :: random_sizes rng (n - s)

let partitions rng n =
  [ ("whole", [n])
  ; ("ones", List.init n (fun _ -> 1))
  ; ( "odd"
    , List.init (n / 7) (fun _ -> 7) @ if n mod 7 = 0 then [] else [n mod 7] )
  ; ("random", random_sizes rng n)
  ; ( "random with empty chunks"
    , List.concat_map
        (fun s -> if Random.State.int rng 3 = 0 then [0; s] else [s])
        (random_sizes rng n) ) ]

let chunks_of x sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        Nx.slice [Nx.A; Nx.R (off, off + s)] x :: go (off + s) rest
  in
  go 0 sizes

let streamed p ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk in
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

let stage_law (type a) ~name (dtype : (float, a) Nx.dtype) ~stage ~flat x =
  test name (fun () ->
      let rng = Random.State.make [|seed|] in
      let x = Nx.cast dtype x in
      let expected = flat x in
      same ~msg:"run" expected (Pipeline.run ~source:(source ()) stage x) ;
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left Int.max 1 sizes in
          let got =
            Nx.concatenate ~axis:(-1)
              (streamed stage ~max_chunk (chunks_of x sizes))
          in
          same ~msg:pname expected got )
        (partitions rng (Nx.dim 1 x)) )

let stage_tests =
  let x = Nx.create Nx.float64 [|2; 301|] (lcg_values 602) in
  let codes =
    Nx.create Nx.float64 [|2; 301|]
      (Array.map (fun v -> Float.round (v *. 128.)) (lcg_values 602))
  in
  [ stage_law ~name:"mu_compress_stage, codes, float32" Nx.float32
      ~stage:(mu_compress_stage ())
      ~flat:(fun x -> mu_compress x)
      x
  ; stage_law ~name:"mu_compress_stage, continuous, float64" Nx.float64
      ~stage:(mu_compress_stage ~mu:15. ~quantize:false ())
      ~flat:(fun x -> mu_compress ~mu:15. ~quantize:false x)
      x
  ; stage_law ~name:"mu_expand_stage, codes, float64" Nx.float64
      ~stage:(mu_expand_stage ())
      ~flat:(fun x -> mu_expand x)
      codes
  ; stage_law ~name:"mu_expand_stage, continuous, float32" Nx.float32
      ~stage:(mu_expand_stage ~quantize:false ())
      ~flat:(fun x -> mu_expand ~quantize:false x)
      x
  ; test "no latency, rate kept" (fun () ->
        let r num den = Pipeline.Rate.{num; den} in
        equal ~msg:"compress latency" rate_t (r 0 1)
          (Pipeline.latency (mu_compress_stage ())) ;
        equal ~msg:"expand rate" rate_t (r 1 1)
          (Pipeline.rate (mu_expand_stage ())) )
  ; test "a chunk out of range raises at that chunk" (fun () ->
        let s =
          Pipeline.Stream.prepare (mu_compress_stage ()) ~source:(source ())
            ~max_chunk:4
        in
        ignore
          (Pipeline.Stream.push s (Nx.create Nx.float64 [|2|] [|0.5; -1.|])) ;
        raises ~msg:"push"
          (Invalid_argument
             "mu_compress_stage: cannot compress 1.5 (every sample must lie in \
              [-1, 1])" ) (fun () ->
            Pipeline.Stream.push s (Nx.create Nx.float64 [|2|] [|0.; 1.5|]) ) )
  ]

(* {1 Errors} *)

let error_tests =
  let sr = 22050 in
  let x = Nx.create Nx.float64 [|3|] [|0.; 0.5; -1.|] in
  [ test "tone" (fun () ->
        raises ~msg:"sample rate"
          (Invalid_argument
             "tone: cannot synthesize at 0 Hz (sample_rate must be at least 1)"
          ) (fun () -> tone Nx.float64 ~sample_rate:0 ~length:4 440. ) ;
        raises ~msg:"no length"
          (Invalid_argument
             "tone: cannot size the signal (a length or a duration must be \
              given)" ) (fun () -> tone Nx.float64 ~sample_rate:sr 440. ) ;
        invalid ~msg:"negative length" "length must be non-negative" (fun () ->
            tone Nx.float64 ~sample_rate:sr ~length:(-1) 440. ) ;
        List.iter
          (fun d ->
            invalid ~msg:"duration" "duration must be finite and non-negative"
              (fun () -> tone Nx.float64 ~sample_rate:sr ~duration:d 440. ) )
          [-1.; Float.nan; Float.infinity] ;
        invalid ~msg:"huge duration" "2^62 samples" (fun () ->
            tone Nx.float64 ~sample_rate:sr ~duration:1e300 440. ) ;
        invalid ~msg:"frequency" "frequency must be finite" (fun () ->
            tone Nx.float64 ~sample_rate:sr ~length:4 Float.nan ) ;
        invalid ~msg:"phase" "phase must be finite" (fun () ->
            tone Nx.float64 ~sample_rate:sr ~length:4 ~phase:Float.infinity 440. ) )
  ; test "chirp" (fun () ->
        let run ?length ?duration ?linear ?phase ?(sample_rate = sr) fmin fmax
            () =
          chirp Nx.float64 ?length ?duration ?linear ?phase ~sample_rate ~fmin
            ~fmax ()
        in
        invalid ~msg:"sample rate" "sample_rate must be at least 1"
          (run ~sample_rate:0 ~length:4 110. 880.) ;
        invalid ~msg:"no length" "a length or a duration must be given"
          (run 110. 880.) ;
        invalid ~msg:"negative length" "length must be non-negative"
          (run ~length:(-3) 110. 880.) ;
        invalid ~msg:"duration" "duration must be finite"
          (run ~duration:Float.nan 110. 880.) ;
        invalid ~msg:"fmin" "fmin must be finite" (run ~length:4 Float.nan 880.) ;
        invalid ~msg:"fmax" "fmax must be finite"
          (run ~length:4 ~linear:true 110. Float.infinity) ;
        invalid ~msg:"phase" "phase must be finite"
          (run ~length:4 ~phase:Float.nan 110. 880.) ;
        raises ~msg:"exponential from 0"
          (Invalid_argument
             "chirp: cannot sweep exponentially from 0 Hz to 880 Hz (fmin and \
              fmax must be nonzero and of the same sign)" )
          (run ~length:4 0. 880.) ;
        invalid ~msg:"opposite signs" "same sign" (run ~length:4 (-110.) 880.) ;
        (* scipy's product underflows to zero, and so does the library's *)
        invalid ~msg:"underflow" "same sign" (run ~length:4 1e-200 1e-200) ;
        equal ~msg:"linear from 0" (array int) [|4|]
          (Nx.shape (run ~length:4 ~linear:true 0. 880. ())) )
  ; test "clicks" (fun () ->
        let run ?hop ?click ?click_frequency ?click_duration ?length
            ?(sample_rate = sr) p () =
          clicks Nx.float64 ?hop ?click ?click_frequency ?click_duration ?length
            ~sample_rate p
        in
        invalid ~msg:"sample rate" "sample_rate must be at least 1"
          (run ~sample_rate:0 (`Samples [|0|])) ;
        invalid ~msg:"hop" "hop must be at least 1" (run ~hop:0 (`Frames [|0|])) ;
        List.iter
          (fun v ->
            invalid ~msg:"click frequency"
              "click_frequency must be finite and positive"
              (run ~click_frequency:v (`Samples [|0|])) ;
            invalid ~msg:"click duration"
              "click_duration must be finite and positive"
              (run ~click_duration:v (`Samples [|0|])) )
          [0.; -1.; Float.nan; Float.infinity] ;
        invalid ~msg:"huge click" "2^62 samples"
          (run ~click_duration:1e300 (`Samples [|0|])) ;
        invalid ~msg:"length" "length must be at least 1"
          (run ~length:0 (`Samples [|0|])) ;
        raises ~msg:"nothing to size"
          (Invalid_argument
             "clicks: cannot size a click track without clicks (a length or at \
              least one position must be given)" )
          (run (`Frames [||])) ;
        invalid ~msg:"nan time" "times must be finite"
          (run (`Times [|0.; Float.nan|])) ;
        invalid ~msg:"time before the first sample" "first sample"
          (run (`Times [|-1. /. Float.of_int sr|])) ;
        invalid ~msg:"negative frame" "frames must be non-negative"
          (run (`Frames [|2; -1|])) ;
        invalid ~msg:"negative sample" "first sample" (run (`Samples [|-5|])) ;
        invalid ~msg:"rank-zero click" "rank-zero click"
          (run ~click:(Nx.scalar Nx.float64 1.) (`Samples [|0|])) ;
        invalid ~msg:"non-finite click" "non-finite"
          (run
             ~click:(Nx.create Nx.float64 [|3|] [|0.; Float.nan; 1.|])
             (`Samples [|0|]) ) )
  ; test "mu-law" (fun () ->
        List.iter
          (fun mu ->
            invalid ~msg:"mu" "mu must be finite and positive" (fun () ->
                mu_compress ~mu x ) ;
            invalid ~msg:"mu" "mu must be finite and positive" (fun () ->
                mu_expand ~mu ~quantize:false x ) ;
            invalid ~msg:"stage mu" "mu must be finite and positive" (fun () ->
                mu_expand_stage ~mu () ) )
          [0.; -1.; Float.nan; Float.infinity] ;
        raises ~msg:"quantized mu below 1"
          (Invalid_argument
             "mu_compress: cannot quantize with mu = 0.5 (quantize needs mu in \
              [1, 2^53))" ) (fun () -> mu_compress ~mu:0.5 x ) ;
        invalid ~msg:"quantized huge mu" "quantize needs mu" (fun () ->
            mu_expand ~mu:0x1p53 x ) ;
        invalid ~msg:"stage quantized mu" "quantize needs mu" (fun () ->
            mu_compress_stage ~mu:0.5 () ) ;
        equal ~msg:"unquantized mu below 1" (array int) [|3|]
          (Nx.shape (mu_compress ~mu:0.5 ~quantize:false x)) ;
        List.iter
          (fun v ->
            invalid ~msg:"compress range" "every sample must lie in [-1, 1]"
              (fun () -> mu_compress (Nx.create Nx.float32 [|2|] [|0.; v|]) ) ;
            invalid ~msg:"expand range" "every value must lie in [-1, 1]"
              (fun () ->
                mu_expand ~quantize:false (Nx.create Nx.float64 [|2|] [|0.; v|]) ) )
          [1.5; -1.0000001; Float.nan; Float.neg_infinity] ;
        raises ~msg:"code 129"
          (Invalid_argument
             "mu_expand: cannot expand the code 129 (codes must lie in [-(1 + \
              mu) / 2, (1 + mu) / 2])" ) (fun () ->
            mu_expand (Nx.create Nx.float64 [|2|] [|128.; 129.|]) ) ;
        invalid ~msg:"nan code" "codes must lie" (fun () ->
            mu_expand (Nx.create Nx.float64 [|1|] [|Float.nan|]) ) ;
        equal ~msg:"code 128 and -128" (array int) [|2|]
          (Nx.shape (mu_expand (Nx.create Nx.float64 [|2|] [|128.; -128.|]))) )
  ]

let suite =
  [ group "tone" tone_tests
  ; group "chirp" chirp_tests
  ; group "clicks" clicks_tests
  ; group "mu-law" mu_tests
  ; group "stages" stage_tests
  ; group "errors" error_tests ]
