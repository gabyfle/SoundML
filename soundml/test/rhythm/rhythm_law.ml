(* The pipeline laws for tempogram_stage and fourier_tempogram_stage:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   exactly, and equal to the flat function, over seeded partitionings —
   one-value chunks, the whole envelope, random sizes and occasional empty
   chunks — for both alignments, odd and even windows, envelopes shorter than a
   window and batches. Also here: the latency queries, reset and the drained
   stream; the properties the goldens do not pin (the float32 path is the
   float64 path on the same values, rounded; scaling an envelope by a power of
   two moves no beat and no pulse; the zero lag and the zero tempo; the shape of
   a pulse); and every precondition. *)

open Windtrap
open Soundml

let seed = 0x7e3b0

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (min n 7) in
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

let stream_outputs p ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

(* [envelope dtype shape] is a non-negative onset-like envelope: a floor with a
   pulse train, both ends nonzero so that both ramps are read. *)
let envelope dtype shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create dtype shape
    (Array.init n (fun i ->
         let v =
           0.05 *. Float.abs (Float.sin (0.37 *. Float.of_int (i * i mod 211)))
         in
         if i mod 9 = 4 then v +. 1. else v +. 0.01 ) )

let bits t = Nx.to_array (Nx.cast Nx.float64 t)

let joined ~empty = function [] -> empty | l -> Nx.concatenate ~axis:(-1) l

(* {1 Tempogram stage} *)

let tempogram_law (type a) ~name (dtype : (float, a) Nx.dtype) ~win ~alignment
    ?(norm = `Inf) shape =
  test name (fun () ->
      let rng =
        Random.State.make [|seed; win; Array.fold_left ( + ) 0 shape|]
      in
      let x = envelope dtype shape in
      let expected = tempogram ~win_length:win ~alignment ~norm x in
      let p = tempogram_stage ~win_length:win ~alignment ~norm () in
      equal ~msg:"shape" (array int) (Nx.shape expected)
        (Nx.shape (Pipeline.run ~source:(source ()) p x)) ;
      equal ~msg:"run" (array float_exact) (bits expected)
        (bits (Pipeline.run ~source:(source ()) p x)) ;
      let frames = shape.(Array.length shape - 1) in
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left max 1 sizes in
          let outs = stream_outputs p ~max_chunk (chunks_of x sizes) in
          let got = joined ~empty:(Nx.zeros dtype (Nx.shape expected)) outs in
          equal ~msg:(pname ^ " shape") (array int) (Nx.shape expected)
            (Nx.shape got) ;
          equal ~msg:pname (array float_exact) (bits expected) (bits got) )
        (partitions rng frames) )

let law_tests =
  [ tempogram_law ~name:"float64, win 8, centred, 40 values" Nx.float64 ~win:8
      ~alignment:`Centered [|40|]
  ; tempogram_law ~name:"float32, win 7, centred, 30 values" Nx.float32 ~win:7
      ~alignment:`Centered [|30|]
  ; tempogram_law ~name:"float64, win 8, centred, 3 values" Nx.float64 ~win:8
      ~alignment:`Centered [|3|]
  ; tempogram_law ~name:"float64, win 8, centred, 1 value" Nx.float64 ~win:8
      ~alignment:`Centered [|1|]
  ; tempogram_law ~name:"float64, win 8, centred, empty" Nx.float64 ~win:8
      ~alignment:`Centered [|0|]
  ; tempogram_law ~name:"float64, win 1, centred" Nx.float64 ~win:1
      ~alignment:`Centered [|12|]
  ; tempogram_law ~name:"float64, win 2, centred" Nx.float64 ~win:2
      ~alignment:`Centered [|12|]
  ; tempogram_law ~name:"float64, win 3, centred, norm None" Nx.float64 ~win:3
      ~alignment:`Centered ~norm:`None [|12|]
  ; tempogram_law ~name:"float64, win 16, centred, [2; 3; 50], norm 2"
      Nx.float64 ~win:16 ~alignment:`Centered ~norm:(`P 2.) [|2; 3; 50|]
  ; tempogram_law ~name:"float64, win 64, centred, 150 values" Nx.float64
      ~win:64 ~alignment:`Centered [|150|]
  ; tempogram_law ~name:"float64, win 8, left, 40 values" Nx.float64 ~win:8
      ~alignment:`Left [|40|]
  ; tempogram_law ~name:"float32, win 9, left, [2; 30]" Nx.float32 ~win:9
      ~alignment:`Left [|2; 30|]
  ; tempogram_law ~name:"float64, win 8, left, exactly one frame" Nx.float64
      ~win:8 ~alignment:`Left [|8|] ]

let long_law =
  test "float64, win 384, centred, 500 values, whole and random" (fun () ->
      let x = envelope Nx.float64 [|500|] in
      let expected = bits (tempogram x) in
      let p = tempogram_stage () in
      let rng = Random.State.make [|seed; 384|] in
      List.iter
        (fun sizes ->
          let max_chunk = List.fold_left max 1 sizes in
          let got =
            bits
              (Nx.concatenate ~axis:(-1)
                 (stream_outputs p ~max_chunk (chunks_of x sizes)) )
          in
          equal ~msg:"partition" (array float_exact) expected got )
        [[500]; [191; 1; 308]; random_sizes rng 500; [250; 250]] )

let stage_tests =
  [ test "latency" (fun () ->
        let lat alignment win =
          Pipeline.latency (tempogram_stage ~win_length:win ~alignment ())
        in
        equal ~msg:"centred 384" rate_t (r 191 1) (lat `Centered 384) ;
        equal ~msg:"centred 383" rate_t (r 191 1) (lat `Centered 383) ;
        equal ~msg:"centred 2" rate_t (r 0 1) (lat `Centered 2) ;
        equal ~msg:"centred 1" rate_t (r 0 1) (lat `Centered 1) ;
        equal ~msg:"left 384" rate_t (r 383 1) (lat `Left 384) ;
        equal ~msg:"rate" rate_t (r 1 1) (Pipeline.rate (tempogram_stage ())) )
  ; test "a push emits a frame once its last value arrives" (fun () ->
        let p = tempogram_stage ~win_length:8 () in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:4 in
        let x = envelope Nx.float64 [|6|] in
        (* latency 3: the first three values complete no frame *)
        is_none ~msg:"three values"
          (Pipeline.Stream.push s (Nx.slice [Nx.R (0, 3)] x)) ;
        let out =
          require_some (Pipeline.Stream.push s (Nx.slice [Nx.R (3, 6)] x))
        in
        equal ~msg:"three frames" (array int) [|8; 3|] (Nx.shape out) ;
        match Pipeline.Stream.flush s with
        | [tail] ->
            equal ~msg:"flushed frames" (array int) [|8; 3|] (Nx.shape tail)
        | l ->
            failf "flush emitted %d chunks" (List.length l) )
  ; test "reset restarts the stream" (fun () ->
        let p = tempogram_stage ~win_length:8 () in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:20 in
        let x = envelope Nx.float64 [|20|] in
        let once () =
          let a = Option.to_list (Pipeline.Stream.push s x) in
          bits (Nx.concatenate ~axis:(-1) (a @ Pipeline.Stream.flush s))
        in
        let first = once () in
        Pipeline.Stream.reset s ;
        equal ~msg:"after reset" (array float_exact) first (once ()) )
  ; test "a drained stream refuses chunks" (fun () ->
        let p = tempogram_stage ~win_length:8 () in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:20 in
        ignore (Pipeline.Stream.push s (envelope Nx.float64 [|5|])) ;
        ignore (Pipeline.Stream.flush s) ;
        equal ~msg:"second flush" int 0 (List.length (Pipeline.Stream.flush s)) ;
        raises
          (Invalid_argument
             "tempogram_stage: cannot feed a drained stream (flush consumed \
              the tail; reset before reusing)" ) (fun () ->
            Pipeline.Stream.push s (envelope Nx.float64 [|5|]) ) )
  ; test "a short left-aligned stream raises at the flush" (fun () ->
        let p = tempogram_stage ~win_length:8 ~alignment:`Left () in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:20 in
        ignore (Pipeline.Stream.push s (envelope Nx.float64 [|5|])) ;
        raises
          (Invalid_argument
             "tempogram_stage: cannot frame 5 onset values with a window of 8 \
              (`Left alignment needs at least win_length values)" ) (fun () ->
            Pipeline.Stream.flush s ) )
  ; test "the leading shape is fixed by the first chunk" (fun () ->
        let p = tempogram_stage ~win_length:8 () in
        let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:20 in
        ignore (Pipeline.Stream.push s (envelope Nx.float64 [|2; 5|])) ;
        raises
          (Invalid_argument
             "tempogram_stage: cannot continue a stream of leading shape [2] \
              with a chunk of leading shape [3]" ) (fun () ->
            Pipeline.Stream.push s (envelope Nx.float64 [|3; 5|]) ) ) ]

(* {1 Fourier tempogram stage} *)

let fourier_law ~name ~win ~alignment shape =
  test name (fun () ->
      let rng = Random.State.make [|seed; win; 17|] in
      let x = envelope Nx.float64 shape in
      let parts z =
        let z = Nx.cast Nx.complex128 z in
        Array.append (bits (Nx.real Nx.float64 z)) (bits (Nx.imag Nx.float64 z))
      in
      let expected =
        fourier_tempogram Nx.complex128 ~win_length:win ~alignment x
      in
      let p =
        fourier_tempogram_stage Nx.complex128 ~win_length:win ~alignment ()
      in
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left max 1 sizes in
          let outs = stream_outputs p ~max_chunk (chunks_of x sizes) in
          let got = Nx.concatenate ~axis:(-1) outs in
          equal ~msg:(pname ^ " shape") (array int) (Nx.shape expected)
            (Nx.shape got) ;
          equal ~msg:pname (array float_exact) (parts expected) (parts got) )
        (List.filter
           (fun (_, s) -> s <> [])
           (partitions rng shape.(Array.length shape - 1)) ) )

let fourier_tests =
  [ fourier_law ~name:"win 16, centred, 60 values" ~win:16 ~alignment:`Centered
      [|60|]
  ; fourier_law ~name:"win 15, centred, [2; 40]" ~win:15 ~alignment:`Centered
      [|2; 40|]
  ; fourier_law ~name:"win 16, left, 60 values" ~win:16 ~alignment:`Left [|60|]
  ; test "latency" (fun () ->
        equal ~msg:"centred" rate_t (r 192 1)
          (Pipeline.latency (fourier_tempogram_stage Nx.complex128 ())) ;
        equal ~msg:"left" rate_t (r 0 1)
          (Pipeline.latency
             (fourier_tempogram_stage Nx.complex128 ~alignment:`Left ()) ) ) ]

(* {1 Properties} *)

let quantized n = Nx.cast Nx.float32 (envelope Nx.float64 [|n|])

let property_tests =
  [ test "float32 is the float64 path on the same values, rounded" (fun () ->
        let x32 = quantized 400 in
        let x64 = Nx.cast Nx.float64 x32 in
        let same msg a b =
          equal ~msg (array float_exact) (bits (Nx.cast Nx.float32 a)) (bits b)
        in
        same "tempogram"
          (tempogram ~win_length:64 x64)
          (tempogram ~win_length:64 x32) ;
        same "tempo"
          (tempo ~aggregate:`None ~sample_rate:22050 x64)
          (tempo ~aggregate:`None ~sample_rate:22050 x32) ;
        same "plp" (plp ~sample_rate:22050 x64) (plp ~sample_rate:22050 x32) ;
        let bpm64, beats64 = beat_track ~sample_rate:22050 x64 in
        let bpm32, beats32 = beat_track ~sample_rate:22050 x32 in
        equal ~msg:"beat_track tempo" float_exact bpm64 bpm32 ;
        equal ~msg:"beat_track beats" (array int) beats64 beats32 )
  ; test "zero lag and zero tempo" (fun () ->
        let f =
          Nx.to_array (tempo_frequencies Nx.float64 ~sample_rate:22050 8)
        in
        equal ~msg:"zero lag" float_exact Float.infinity f.(0) ;
        equal ~msg:"lag 1" (float 1e-12) 2583.984375 f.(1) ;
        let f =
          Nx.to_array
            (fourier_tempo_frequencies Nx.float64 ~sample_rate:22050 384)
        in
        equal ~msg:"bins" int 193 (Array.length f) ;
        equal ~msg:"zero tempo" float_exact 0. f.(0) )
  ; test "the inf norm reads one at the zero lag" (fun () ->
        let g = tempogram ~win_length:16 (envelope Nx.float64 [|40|]) in
        let g = Nx.to_array g in
        for t = 0 to 39 do
          equal ~msg:(string_of_int t) (float 1e-15) 1. g.(t)
        done ;
        let z =
          Nx.to_array (tempogram ~win_length:16 (Nx.zeros Nx.float64 [|10|]))
        in
        Array.iter (fun v -> equal ~msg:"silence" float_exact 0. v) z )
  ; test "without a norm, the zero lag is the frame's energy" (fun () ->
        let x = envelope Nx.float64 [|30|] in
        let g =
          Nx.to_array (tempogram ~win_length:8 ~alignment:`Left ~norm:`None x)
        in
        let w = Nx.to_array (Window.make Nx.float64 Window.Hann 8) in
        let v = Nx.to_array x in
        let frames = 30 - 8 + 1 in
        for t = 0 to frames - 1 do
          let e = ref 0. in
          for i = 0 to 7 do
            let f = v.(t + i) *. w.(i) in
            let sq = f *. f in
            e := !e +. sq
          done ;
          equal ~msg:(string_of_int t) float_exact !e g.(t)
        done )
  ; test "a power of two moves no beat" (fun () ->
        let x = envelope Nx.float64 [|500|] in
        let x2 = Nx.mul_s x 2. in
        let bpm, beats = beat_track ~sample_rate:22050 x in
        let bpm2, beats2 = beat_track ~sample_rate:22050 x2 in
        equal ~msg:"tempo" float_exact bpm bpm2 ;
        equal ~msg:"beats" (array int) beats beats2 ;
        is_true ~msg:"some beats" (Array.length beats > 10) ;
        Array.iteri
          (fun i b ->
            is_true ~msg:"inside" (b >= 0 && b < 500) ;
            if i > 0 then is_true ~msg:"increasing" (b > beats.(i - 1)) )
          beats )
  ; test "a given tempo is returned as given" (fun () ->
        let bpm, _ =
          beat_track ~bpm:97.5 ~sample_rate:22050 (envelope Nx.float64 [|300|])
        in
        equal ~msg:"bpm" float_exact 97.5 bpm )
  ; test "an envelope of zeros has no beat" (fun () ->
        let bpm, beats =
          beat_track ~bpm:120. ~sample_rate:22050 (Nx.zeros Nx.float32 [|50|])
        in
        equal ~msg:"tempo" float_exact 0. bpm ;
        equal ~msg:"beats" (array int) [||] beats )
  ; test "a power of two moves no unit pulse" (fun () ->
        let x = envelope Nx.float64 [|500|] in
        equal ~msg:"pulse" (array float_exact)
          (bits (plp ~kernels:`Unit ~sample_rate:22050 x))
          (bits (plp ~kernels:`Unit ~sample_rate:22050 (Nx.mul_s x 2.))) )
  ; test "a pulse lies in [0, 1] and peaks at one" (fun () ->
        List.iter
          (fun kernels ->
            let p =
              bits
                (plp ~kernels ~sample_rate:22050
                   (envelope Nx.float64 [|2; 300|]) )
            in
            equal ~msg:"length" int 600 (Array.length p) ;
            Array.iter (fun v -> is_true ~msg:"range" (v >= 0. && v <= 1.)) p ;
            List.iter
              (fun row ->
                let m = ref 0. in
                for t = 0 to 299 do
                  m := Float.max !m p.((row * 300) + t)
                done ;
                equal ~msg:"peak" float_exact 1. !m )
              [0; 1] )
          [`Reference; `Unit] )
  ; test "empty envelopes" (fun () ->
        let e = Nx.zeros Nx.float64 [|2; 0|] in
        equal ~msg:"tempogram" (array int) [|2; 384; 0|]
          (Nx.shape (tempogram e)) ;
        equal ~msg:"tempo" (array int) [|2; 0|]
          (Nx.shape (tempo ~aggregate:`None ~sample_rate:22050 e)) ;
        equal ~msg:"plp" (array int) [|2; 0|]
          (Nx.shape (plp ~sample_rate:22050 e)) ) ]

(* {1 Preconditions} *)

let invalid substring f =
  raises_match (Exn.invalid_arg ~substring) (fun () -> ignore (f ()))

let x = envelope Nx.float64 [|100|]

let error_tests =
  [ test "frequencies" (fun () ->
        raises
          (Invalid_argument
             "tempo_frequencies: cannot describe a tempogram of 0 lag bins (at \
              least one is needed)" ) (fun () ->
            tempo_frequencies Nx.float64 ~sample_rate:22050 0 ) ;
        invalid "hop must be at least 1" (fun () ->
            tempo_frequencies Nx.float64 ~hop:0 ~sample_rate:22050 4 ) ;
        invalid "sample_rate must be at least 1" (fun () ->
            fourier_tempo_frequencies Nx.float64 ~sample_rate:0 4 ) ;
        invalid "win_length must be at least 1" (fun () ->
            fourier_tempo_frequencies Nx.float64 ~sample_rate:22050 0 ) )
  ; test "tempograms" (fun () ->
        raises
          (Invalid_argument
             "tempogram: cannot analyse over a window of 0 frames (win_length \
              must be at least 1)" ) (fun () -> tempogram ~win_length:0 x ) ;
        invalid "the exponent must be finite and positive" (fun () ->
            tempogram ~norm:(`P 0.) x ) ;
        invalid "the exponent must be finite and positive" (fun () ->
            tempogram_stage ~norm:(`P Float.nan) () ) ;
        invalid "win_length must be at least 1" (fun () ->
            tempogram_stage ~win_length:0 () ) ;
        raises
          (Invalid_argument
             "tempogram: cannot frame 100 onset values with a window of 384 \
              (`Left alignment needs at least win_length values)" ) (fun () ->
            tempogram ~alignment:`Left x ) ;
        raises
          (Invalid_argument
             "tempogram: cannot analyse a rank-zero tensor (the frame axis \
              must exist)" ) (fun () -> tempogram (Nx.scalar Nx.float64 1.) ) ;
        raises
          (Invalid_argument
             "tempogram: cannot analyse an onset envelope holding a non-finite \
              value (every value must be finite)" ) (fun () ->
            tempogram (Nx.create Nx.float64 [|3|] [|1.; Float.nan; 2.|]) ) ;
        invalid "float32 and float64 only" (fun () ->
            tempogram (Nx.create Nx.float16 [|3|] [|1.; 0.; 2.|]) ) ;
        invalid "win_length must be at least 1" (fun () ->
            fourier_tempogram Nx.complex128 ~win_length:0 x ) ;
        invalid "every value must be finite" (fun () ->
            fourier_tempogram Nx.complex128
              (Nx.create Nx.float64 [|3|] [|1.; Float.infinity; 2.|]) ) ;
        invalid "win_length must be at least 1" (fun () ->
            fourier_tempogram_stage Nx.complex128 ~win_length:0 () ) )
  ; test "tempo" (fun () ->
        raises
          (Invalid_argument
             "tempo: cannot use start_bpm = 0 (start_bpm must be finite and \
              positive)" ) (fun () -> tempo ~start_bpm:0. ~sample_rate:22050 x ) ;
        invalid "std_bpm must be finite and positive" (fun () ->
            tempo ~std_bpm:0. ~sample_rate:22050 x ) ;
        invalid "ac_size must be finite and positive" (fun () ->
            tempo ~ac_size:Float.nan ~sample_rate:22050 x ) ;
        raises
          (Invalid_argument
             "tempo: cannot autocorrelate over 0.01 seconds at 22050 Hz and a \
              hop of 512 (ac_size * sample_rate must span at least one hop)" )
          (fun () -> tempo ~ac_size:0.01 ~sample_rate:22050 x ) ;
        invalid "tempo_max must be positive" (fun () ->
            tempo ~tempo_max:0. ~sample_rate:22050 x ) ;
        invalid "hop must be at least 1" (fun () ->
            tempo ~hop:0 ~sample_rate:22050 x ) ;
        raises
          (Invalid_argument
             "tempo: cannot aggregate the tempogram of an empty onset envelope"
          ) (fun () -> tempo ~sample_rate:22050 (Nx.zeros Nx.float64 [|0|]) ) )
  ; test "beat_track" (fun () ->
        raises
          (Invalid_argument
             "beat_track: cannot use tightness = 0 (tightness must be finite \
              and positive)" ) (fun () ->
            beat_track ~tightness:0. ~sample_rate:22050 x ) ;
        raises
          (Invalid_argument
             "beat_track: cannot use tightness = 1e+39 (tightness must be \
              finite in single precision)" ) (fun () ->
            beat_track ~tightness:1e39 ~sample_rate:22050 x ) ;
        invalid "bpm must be finite and positive" (fun () ->
            beat_track ~bpm:(-1.) ~sample_rate:22050 x ) ;
        invalid "start_bpm must be finite and positive" (fun () ->
            beat_track ~start_bpm:Float.infinity ~sample_rate:22050 x ) ;
        raises
          (Invalid_argument
             "beat_track: cannot read a tensor of leading shape [2] as one \
              onset envelope (every leading axis must have size 1)" ) (fun () ->
            beat_track ~sample_rate:22050 (envelope Nx.float64 [|2; 50|]) ) ;
        equal ~msg:"a [1; 1; frames] envelope is one envelope"
          (pair float_exact (array int))
          (beat_track ~sample_rate:22050 (envelope Nx.float64 [|50|]))
          (beat_track ~sample_rate:22050 (envelope Nx.float64 [|1; 1; 50|])) ;
        raises
          (Invalid_argument
             "beat_track: cannot track beats in 1 onset frames (at least 2 are \
              needed)" ) (fun () ->
            beat_track ~sample_rate:22050 (Nx.ones Nx.float64 [|1|]) ) ;
        raises
          (Invalid_argument
             "beat_track: cannot track a tempo of 1e+06 BPM at 43.0664 frames \
              per second (a beat must span at least one frame)" ) (fun () ->
            beat_track ~bpm:1e6 ~sample_rate:22050 x ) ;
        invalid "every value must be finite" (fun () ->
            beat_track ~sample_rate:22050
              (Nx.create Nx.float64 [|3|] [|1.; Float.nan; 2.|]) ) )
  ; test "plp" (fun () ->
        raises
          (Invalid_argument
             "plp: cannot search tempi in [30, 30] BPM (tempo_max must exceed \
              tempo_min)" ) (fun () -> plp ~tempo_max:30. ~sample_rate:22050 x ) ;
        invalid "tempo_min must be finite and non-negative" (fun () ->
            plp ~tempo_min:(-1.) ~sample_rate:22050 x ) ;
        invalid "tempo_max must exceed tempo_min" (fun () ->
            plp ~tempo_max:Float.nan ~sample_rate:22050 x ) ;
        invalid "win_length must be at least 1" (fun () ->
            plp ~win_length:0 ~sample_rate:22050 x ) ;
        invalid "sample_rate must be at least 1" (fun () ->
            plp ~sample_rate:0 x ) ) ]

let suite =
  [ group "tempogram stage law" (long_law :: law_tests)
  ; group "tempogram stage" stage_tests
  ; group "fourier tempogram stage" fourier_tests
  ; group "properties" property_tests
  ; group "errors" error_tests ]
