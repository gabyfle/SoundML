(* The pipeline laws for chroma_cens_stage and tonnetz_stage:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s)

   exactly, and both equal the flat functions, over seeded partitionings —
   one-frame chunks, the whole sequence, fixed sizes around the window, random
   sizes and occasional empty chunks — for odd and even smoothing lengths,
   windows whose endpoints are dropped or kept, no smoothing, batches and silent
   stretches. Also here: the composition identities with the chromagram, the
   streamed compositions over constant-Q magnitudes, the properties the
   quantiser and the projection promise, the latency queries, and every
   error. *)

open Windtrap
open Soundml

let seed = 0x7042a1

let source dtype = Pipeline.Format.audio dtype ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (Int.min n 7) in
    s :: random_sizes rng (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

let fixed k n =
  if n = 0 then []
  else List.init ((n + k - 1) / k) (fun i -> Int.min k (n - (i * k)))

let partitions rng n =
  let named =
    [ ("ones", List.init n (fun _ -> 1))
    ; ("whole", if n = 0 then [] else [n])
    ; ("empty-first-and-last", [0] @ (if n = 0 then [] else [n]) @ [0]) ]
  in
  let sized =
    List.map (fun k -> (Printf.sprintf "fixed-%d" k, fixed k n)) [2; 7; 41; 333]
  in
  let random =
    List.init 3 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))
  in
  let random_empty =
    List.init 2 (fun i ->
        ( Printf.sprintf "random-with-empty-chunks-%d" i
        , sprinkle_empty rng (random_sizes rng n) ) )
  in
  named @ sized @ random @ random_empty

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

(* [chromagram dtype lead frames] is a non-negative [[lead...; 12; frames]]
   chromagram, |LCG| values with every frame [t] where [t mod 37] lies in [20,
   32) silent: stretches longer than the shorter windows, so frames whose whole
   window is silent occur too. *)
let chromagram dtype lead frames =
  let shape = Array.append lead [|12; frames|] in
  let values = Tonal_inputs.chroma "lcg" shape in
  Array.iteri
    (fun i _ ->
      let t = i mod frames in
      if t mod 37 >= 20 && t mod 37 < 32 then values.(i) <- 0. )
    values ;
  Nx.create dtype shape values

let same ~msg a b =
  equal ~msg (array float_exact) (Nx.to_array a) (Nx.to_array b)

let concat_outputs = function
  | [] ->
      [||]
  | l ->
      Nx.to_array (Nx.concatenate ~axis:(-1) l)

(* [law ~name ~dtype ~flat ~stage ~lead frame_counts] checks [stage ()] against
   [flat] on [dtype] chromagrams of every frame count, offline and over every
   partitioning. *)
let law ~name ~dtype ~flat ~stage ~lead frame_counts =
  test name (fun () ->
      let rng = Random.State.make [|seed; Hashtbl.hash name|] in
      List.iter
        (fun frames ->
          let x = chromagram dtype lead frames in
          let expected = Nx.to_array (flat x) in
          let p = stage () in
          equal
            ~msg:(Printf.sprintf "run n=%d" frames)
            (array float_exact) expected
            (Nx.to_array (Pipeline.run ~source:(source dtype) p x)) ;
          List.iter
            (fun (pname, sizes) ->
              let max_chunk = List.fold_left Int.max 1 sizes in
              let outs =
                stream_outputs p ~source:(source dtype) ~max_chunk
                  (chunks_of x sizes)
              in
              equal
                ~msg:(Printf.sprintf "%s n=%d" pname frames)
                (array float_exact) expected (concat_outputs outs) )
            (partitions rng frames) )
        frame_counts )

let frame_counts = [0; 1; 5; 20; 21; 41; 100; 333]

let cens_law ?smoothing ?smoothing_window ?norm ~dtype ~lead name =
  law ~name ~dtype ~lead frame_counts
    ~flat:(fun x -> chroma_cens_of_chroma ?smoothing ?smoothing_window ?norm x)
    ~stage:(fun () -> chroma_cens_stage ?smoothing ?smoothing_window ?norm ())

let f64 = Nx.float64

let law_tests =
  [ cens_law ~dtype:f64 ~lead:[||]
      "cens, defaults (hann over 41 frames), [12; frames]"
  ; cens_law ~dtype:f64 ~lead:[|2|] "cens, defaults, [2; 12; frames]"
  ; cens_law ~dtype:Nx.float32 ~lead:[||] "cens, defaults, float32"
  ; cens_law ~dtype:f64 ~smoothing:(Some 40) ~lead:[||]
      "cens, even smoothing 40"
  ; cens_law ~dtype:f64 ~smoothing:(Some 9) ~norm:`Inf ~lead:[|3|]
      "cens, smoothing 9, inf norm, [3; 12; frames]"
  ; cens_law ~dtype:f64 ~smoothing:(Some 2) ~lead:[||] "cens, smoothing 2"
  ; cens_law ~dtype:f64 ~smoothing:(Some 1) ~lead:[||]
      "cens, smoothing 1 (identity taps)"
  ; cens_law ~dtype:f64 ~smoothing:None ~norm:`None ~lead:[|2|]
      "cens, unsmoothed"
  ; cens_law ~dtype:f64 ~smoothing_window:Window.Hamming ~lead:[||]
      "cens, hamming (endpoints kept)"
  ; cens_law ~dtype:f64 ~smoothing:(Some 8) ~smoothing_window:Window.Blackman
      ~lead:[||] "cens, blackman 8 (endpoints dropped)"
  ; law ~name:"tonnetz, [12; frames]" ~dtype:f64 ~lead:[||] frame_counts
      ~flat:tonnetz_of_chroma ~stage:tonnetz_stage
  ; law ~name:"tonnetz, [2; 12; frames], float32" ~dtype:Nx.float32 ~lead:[|2|]
      frame_counts ~flat:tonnetz_of_chroma ~stage:tonnetz_stage ]

(* {1 Compositions} *)

let cqt =
  Cqt.Config.create ~hop:512 ~pad:(`Constant 0.) ~n_bins:84 ~sample_rate:22050
    ()

let audio dtype channels n =
  let y = Tonal_inputs.signal "chords" n in
  Nx.create dtype [|channels; n|]
    (Array.init (channels * n) (fun i ->
         let c = i / n in
         if c = 0 then y.(i mod n) else 0.5 *. y.((i + 4096) mod n) ) )

let composition_tests =
  [ test "chroma_cens is CENS of the unnormalised constant-Q chromagram"
      (fun () ->
        let x = audio Nx.float64 2 8192 in
        same ~msg:"defaults" (chroma_cens cqt x)
          (chroma_cens_of_chroma (chroma_cqt cqt ~norm:`None x)) ;
        let x32 = audio Nx.float32 1 8192 in
        same ~msg:"float32, smoothing 9, l1"
          (chroma_cens cqt ~smoothing:(Some 9) ~norm:(`P 1.) x32)
          (chroma_cens_of_chroma ~smoothing:(Some 9) ~norm:(`P 1.)
             (chroma_cqt cqt ~norm:`None x32) ) )
  ; test "tonnetz is the centroid of the default constant-Q chromagram"
      (fun () ->
        let x = audio Nx.float64 2 8192 in
        same ~msg:"float64" (tonnetz cqt x)
          (tonnetz_of_chroma (chroma_cqt cqt x)) ;
        let x24 = audio Nx.float32 1 8192 in
        let cqt24 =
          Cqt.Config.create ~bins_per_octave:24 ~hop:512 ~pad:(`Constant 0.)
            ~n_bins:168 ~sample_rate:22050 ()
        in
        same ~msg:"float32, 24 bands"
          (tonnetz cqt24 ~n_chroma:24 x24)
          (tonnetz_of_chroma (chroma_cqt cqt24 ~n_chroma:24 x24)) )
  ; test
      "streamed over constant-Q magnitudes, the chains total the flat functions"
      (fun () ->
        let x = audio Nx.float64 1 16384 in
        let m = Cqt.power_spectrum ~power:1. cqt x in
        let frames = Nx.dim (Nx.ndim m - 1) m in
        let cens =
          Pipeline.(Chroma.of_cqt_stage ~norm:`None cqt >> chroma_cens_stage ())
        in
        let ton = Pipeline.(Chroma.of_cqt_stage cqt >> tonnetz_stage ()) in
        List.iter
          (fun sizes ->
            let max_chunk = List.fold_left Int.max 1 sizes in
            equal ~msg:"cens" (array float_exact)
              (Nx.to_array (chroma_cens cqt x))
              (concat_outputs
                 (stream_outputs cens ~source:(source Nx.float64) ~max_chunk
                    (chunks_of m sizes) ) ) ;
            equal ~msg:"tonnetz" (array float_exact)
              (Nx.to_array (tonnetz cqt x))
              (concat_outputs
                 (stream_outputs ton ~source:(source Nx.float64) ~max_chunk
                    (chunks_of m sizes) ) ) )
          [[frames]; fixed 1 frames; fixed 5 frames] ) ]

(* {1 Properties} *)

(* [column pairs] is a one-frame chromagram with the given (class, weight)
   entries. *)
let column pairs =
  let v = Array.make 12 0. in
  List.iter (fun (k, w) -> v.(k) <- w) pairs ;
  Nx.create Nx.float64 [|12; 1|] v

let property_tests =
  [ test "the comparison is strict: shares on a threshold take the lower level"
      (fun () ->
        let levels pairs =
          Nx.to_array
            (chroma_cens_of_chroma ~smoothing:None ~norm:`None (column pairs))
        in
        (* shares of 2/5 and 1/5 *)
        let l = levels [(0, 2.); (4, 1.); (7, 1.); (11, 1.)] in
        equal ~msg:"0.4" float_exact 0.75 l.(0) ;
        equal ~msg:"0.2" float_exact 0.5 l.(4) ;
        (* shares of 1/10 *)
        let l = levels (List.init 10 (fun k -> (k, 1.))) in
        equal ~msg:"0.1" float_exact 0.25 l.(3) ;
        (* a share of 9/20 and eleven of 1/20 *)
        let l = levels ((0, 9.) :: List.init 11 (fun k -> (k + 1, 1.))) in
        equal ~msg:"0.45" float_exact 1.0 l.(0) ;
        equal ~msg:"0.05" float_exact 0. l.(5) )
  ; test "unsmoothed levels are multiples of a quarter; frames are unit or zero"
      (fun () ->
        let x = chromagram Nx.float64 [||] 120 in
        Array.iter
          (fun v ->
            is_true ~msg:"level"
              (List.exists (Float.equal v) [0.; 0.25; 0.5; 0.75; 1.]) )
          (Nx.to_array (chroma_cens_of_chroma ~smoothing:None ~norm:`None x)) ;
        let y = Nx.to_array (chroma_cens_of_chroma x) in
        for t = 0 to 119 do
          let s = ref 0. in
          for k = 0 to 11 do
            s := !s +. (y.((k * 120) + t) *. y.((k * 120) + t))
          done ;
          is_true
            ~msg:(Printf.sprintf "frame %d norm %g" t !s)
            (!s = 0. || Float.abs (!s -. 1.) < 1e-14)
        done )
  ; test
      "silence: zero frames give zero centroids, and zero CENS beyond the \
       window's reach" (fun () ->
        let x = chromagram Nx.float64 [||] 100 in
        let t = Nx.to_array (tonnetz_of_chroma x) in
        let c = Nx.to_array (chroma_cens_of_chroma ~smoothing:(Some 9) x) in
        (* frames 20..31 and 57..68 are silent; Hann over 9 frames reads 4
           frames on either side *)
        for f = 0 to 99 do
          let silent = f mod 37 >= 20 && f mod 37 < 32 in
          if silent then
            for p = 0 to 5 do
              equal ~msg:"tonnetz" float_exact 0. t.((p * 100) + f)
            done ;
          if f mod 37 >= 24 && f mod 37 < 28 then
            for k = 0 to 11 do
              equal ~msg:"cens" float_exact 0. c.((k * 100) + f)
            done
        done ;
        let z = Nx.zeros Nx.float64 [|12; 30|] in
        equal ~msg:"all-zero cens" (array float_exact) (Array.make 360 0.)
          (Nx.to_array (chroma_cens_of_chroma z)) )
  ; test "batch independence and power-of-two scale invariance, bit for bit"
      (fun () ->
        let x = chromagram Nx.float64 [|3|] 80 in
        let row i = Nx.slice [Nx.I i; Nx.A; Nx.A] x in
        let cens = chroma_cens_of_chroma x and ton = tonnetz_of_chroma x in
        for i = 0 to 2 do
          same ~msg:"cens row"
            (Nx.slice [Nx.I i; Nx.A; Nx.A] cens)
            (chroma_cens_of_chroma (row i)) ;
          same ~msg:"tonnetz row"
            (Nx.slice [Nx.I i; Nx.A; Nx.A] ton)
            (tonnetz_of_chroma (row i))
        done ;
        let scaled = Nx.mul_s x 4. in
        same ~msg:"cens scale" cens (chroma_cens_of_chroma scaled) ;
        same ~msg:"tonnetz scale" ton (tonnetz_of_chroma scaled) )
  ; test "the centroid of a single class is its projection column" (fun () ->
        let phi = Nx.to_array (tonnetz_of_chroma (Nx.eye Nx.float64 12)) in
        let a = [|7. /. 6.; 3. /. 2.; 2. /. 3.|] and radius = [|1.; 1.; 0.5|] in
        (* the angles reach 13 pi, whose rounding moves a sine or cosine near
           zero by up to 2e-15 between the two spellings *)
        for c = 0 to 11 do
          for i = 0 to 2 do
            let angle = Float.pi *. a.(i) *. Float.of_int c in
            equal ~msg:"sine" (float 1e-14)
              (radius.(i) *. Float.sin angle)
              phi.((2 * i * 12) + c) ;
            equal ~msg:"cosine" (float 1e-14)
              (radius.(i) *. Float.cos angle)
              phi.((((2 * i) + 1) * 12) + c)
          done
        done ;
        (* an equal-weight C major triad is the mean of its three columns *)
        let triad =
          Nx.to_array (tonnetz_of_chroma (column [(0, 1.); (4, 1.); (7, 1.)]))
        in
        for p = 0 to 5 do
          equal ~msg:"triad" (float 1e-15)
            ( (phi.((p * 12) + 0) +. phi.((p * 12) + 4) +. phi.((p * 12) + 7))
            /. 3. )
            triad.(p)
        done )
  ; test "empty inputs and empty chunks keep their shapes" (fun () ->
        let shape t = Nx.shape t in
        equal ~msg:"cens" (array int) [|2; 12; 0|]
          (shape (chroma_cens_of_chroma (Nx.zeros Nx.float64 [|2; 12; 0|]))) ;
        equal ~msg:"tonnetz" (array int) [|2; 6; 0|]
          (shape (tonnetz_of_chroma (Nx.zeros Nx.float64 [|2; 12; 0|]))) ;
        equal ~msg:"tonnetz of no bands" (array float_exact) (Array.make 30 0.)
          (Nx.to_array (tonnetz_of_chroma (Nx.zeros Nx.float64 [|0; 5|]))) ;
        let s =
          Pipeline.Stream.prepare (chroma_cens_stage ())
            ~source:(source Nx.float64) ~max_chunk:4
        in
        ( match Pipeline.Stream.push s (Nx.zeros Nx.float64 [|12; 0|]) with
        | Some y ->
            equal ~msg:"cens chunk" (array int) [|12; 0|] (shape y)
        | None ->
            () ) ;
        let s =
          Pipeline.Stream.prepare (tonnetz_stage ()) ~source:(source Nx.float64)
            ~max_chunk:4
        in
        match Pipeline.Stream.push s (Nx.zeros Nx.float64 [|12; 0|]) with
        | Some y ->
            equal ~msg:"tonnetz chunk" (array int) [|6; 0|] (shape y)
        | None ->
            fail "tonnetz_stage withheld an empty chunk" )
  ; test "the dtype follows the input" (fun () ->
        let x = chromagram Nx.float32 [||] 50 in
        is_true ~msg:"cens float32"
          (Nx.dtype (chroma_cens_of_chroma x) = Nx.float32) ;
        is_true ~msg:"tonnetz float32"
          (Nx.dtype (tonnetz_of_chroma x) = Nx.float32) ;
        let h = Nx.cast Nx.float16 x in
        let c = chroma_cens_of_chroma ~smoothing:None ~norm:`None h in
        is_true ~msg:"cens float16" (Nx.dtype c = Nx.float16) ;
        (* the levels are exact in half precision *)
        same ~msg:"float16 levels" (Nx.cast Nx.float64 c)
          (chroma_cens_of_chroma ~smoothing:None ~norm:`None
             (Nx.cast Nx.float64 h) ) ) ]

let query_tests =
  [ test "latency is the window's reach ahead" (fun () ->
        let lat ?smoothing ?smoothing_window () =
          Pipeline.latency
            ( chroma_cens_stage ?smoothing ?smoothing_window ()
              : ((float, Nx.float64_elt) Nx.t, _, _) Pipeline.t )
        in
        equal ~msg:"hann 41" rate_t (r 20 1) (lat ()) ;
        equal ~msg:"hann 40" rate_t (r 20 1) (lat ~smoothing:(Some 40) ()) ;
        equal ~msg:"hann 2" rate_t (r 1 1) (lat ~smoothing:(Some 2) ()) ;
        equal ~msg:"hann 1" rate_t (r 0 1) (lat ~smoothing:(Some 1) ()) ;
        equal ~msg:"none" rate_t (r 0 1) (lat ~smoothing:None ()) ;
        equal ~msg:"hamming 41" rate_t (r 21 1)
          (lat ~smoothing_window:Window.Hamming ()) ;
        equal ~msg:"hamming 40" rate_t (r 21 1)
          (lat ~smoothing:(Some 40) ~smoothing_window:Window.Hamming ()) ;
        equal ~msg:"blackman 8" rate_t (r 4 1)
          (lat ~smoothing:(Some 8) ~smoothing_window:Window.Blackman ()) ;
        equal ~msg:"rectangular 4" rate_t (r 3 1)
          (lat ~smoothing:(Some 4) ~smoothing_window:Window.Rectangular ()) ;
        equal ~msg:"tonnetz" rate_t (r 0 1)
          (Pipeline.latency
             (tonnetz_stage () : ((float, Nx.float64_elt) Nx.t, _, _) Pipeline.t) ) ;
        equal ~msg:"rate" rate_t (r 1 1)
          (Pipeline.rate
             ( chroma_cens_stage ()
               : ((float, Nx.float64_elt) Nx.t, _, _) Pipeline.t ) ) )
  ; test "the even alignment reads one frame more ahead than behind" (fun () ->
        (* class 0 of a uniform chromagram, raised at frame 30: with smoothing
           40, output frame t reads frames t - 19 .. t + 20, so frames 10 to 49
           rise and no other *)
        let x raised =
          Nx.create Nx.float64 [|12; 61|]
            (Array.init (12 * 61) (fun i ->
                 if raised && i = 30 then 10. else 1. ) )
        in
        let cens raised =
          Nx.to_array
            (chroma_cens_of_chroma ~smoothing:(Some 40) ~norm:`None (x raised))
        in
        let base = cens false and raised = cens true in
        for t = 0 to 60 do
          equal
            ~msg:(Printf.sprintf "frame %d" t)
            bool
            (t >= 10 && t <= 49)
            (raised.(t) > base.(t))
        done ) ]

let error_tests =
  let x = chromagram Nx.float64 [||] 10 in
  [ test "smoothing must be at least 1" (fun () ->
        raises ~msg:"offline"
          (Invalid_argument
             "chroma_cens_of_chroma: cannot smooth over 0 frames (smoothing \
              must be at least 1)" ) (fun () ->
            ignore (chroma_cens_of_chroma ~smoothing:(Some 0) x) ) ;
        raises ~msg:"stage"
          (Invalid_argument
             "chroma_cens_stage: cannot smooth over -3 frames (smoothing must \
              be at least 1)" ) (fun () ->
            ignore (chroma_cens_stage ~smoothing:(Some (-3)) ()) ) ;
        raises ~msg:"audio"
          (Invalid_argument
             "chroma_cens: cannot smooth over 0 frames (smoothing must be at \
              least 1)" ) (fun () ->
            ignore
              (chroma_cens cqt ~smoothing:(Some 0) (audio Nx.float64 1 1024)) ) )
  ; test "the norm exponent must be finite and positive" (fun () ->
        raises ~msg:"norm"
          (Invalid_argument
             "chroma_cens_of_chroma: cannot normalise in the 0-norm (the \
              exponent must be finite and positive)" ) (fun () ->
            ignore (chroma_cens_of_chroma ~norm:(`P 0.) x) ) )
  ; test "the smoothing window must be valid and sum to a positive value"
      (fun () ->
        raises ~msg:"shape"
          (Invalid_argument
             "chroma_cens_of_chroma: cannot use a kaiser window with beta -1 \
              (beta must be finite and non-negative)" ) (fun () ->
            ignore
              (chroma_cens_of_chroma ~smoothing_window:(Window.Kaiser (-1.)) x) ) ;
        raises ~msg:"sum"
          (Invalid_argument
             "chroma_cens_stage: cannot smooth with a 4-point gaussian(0.01) \
              window summing to 0 (its weights must have a positive sum)" )
          (fun () ->
            ignore
              (chroma_cens_stage ~smoothing:(Some 2)
                 ~smoothing_window:(Window.Gaussian 0.01) () ) ) )
  ; test "the chromagram needs rank two" (fun () ->
        let v = Nx.ones Nx.float64 [|12|] in
        raises ~msg:"cens"
          (Invalid_argument
             "chroma_cens_of_chroma: cannot quantise a rank-1 tensor (the \
              chromagram needs [...; n_chroma; frames])" ) (fun () ->
            ignore (chroma_cens_of_chroma v) ) ;
        raises ~msg:"tonnetz"
          (Invalid_argument
             "tonnetz_of_chroma: cannot project a rank-1 tensor (the \
              chromagram needs [...; n_chroma; frames])" ) (fun () ->
            ignore (tonnetz_of_chroma v) ) ;
        raises ~msg:"tonnetz stage"
          (Invalid_argument
             "tonnetz_stage: cannot project a rank-1 chunk (the chromagram \
              needs [...; n_chroma; frames])" ) (fun () ->
            ignore
              (stream_outputs (tonnetz_stage ()) ~source:(source Nx.float64)
                 ~max_chunk:12 [v] ) ) )
  ; test "CENS refuses negative chroma; both refuse non-finite values"
      (fun () ->
        let with_value v =
          Nx.create Nx.float64 [|12; 2|]
            (Array.init 24 (fun i -> if i = 5 then v else 1.))
        in
        raises ~msg:"negative"
          (Invalid_argument
             "chroma_cens_of_chroma: cannot quantise a chromagram with \
              negative values (chroma energy must be non-negative)" ) (fun () ->
            ignore (chroma_cens_of_chroma (with_value (-1.))) ) ;
        raises ~msg:"nan"
          (Invalid_argument
             "chroma_cens_of_chroma: cannot normalise a chromagram with \
              infinite or NaN values (every entry must be finite)" ) (fun () ->
            ignore (chroma_cens_of_chroma (with_value Float.nan)) ) ;
        raises ~msg:"tonnetz infinity"
          (Invalid_argument
             "tonnetz_of_chroma: cannot normalise a chromagram with infinite \
              or NaN values (every entry must be finite)" ) (fun () ->
            ignore (tonnetz_of_chroma (with_value Float.infinity)) ) ;
        (* the centroid accepts signed chroma *)
        ignore (tonnetz_of_chroma (with_value (-1.))) ;
        raises ~msg:"stage, at the chunk"
          (Invalid_argument
             "chroma_cens_stage: cannot quantise a chromagram with negative \
              values (chroma energy must be non-negative)" ) (fun () ->
            ignore
              (stream_outputs (chroma_cens_stage ()) ~source:(source Nx.float64)
                 ~max_chunk:10
                 [x; with_value (-1.)] ) ) ;
        raises ~msg:"audio"
          (Invalid_argument
             "chroma_cens: cannot normalise a chromagram with infinite or NaN \
              values (every entry must be finite)" ) (fun () ->
            let y = Nx.to_array (audio Nx.float64 1 2048) in
            y.(1000) <- Float.nan ;
            ignore (chroma_cens cqt (Nx.create Nx.float64 [|1; 2048|] y)) ) )
  ; test "a chunk must keep the stream's leading shape" (fun () ->
        raises ~msg:"leading shape"
          (Invalid_argument
             "chroma_cens_stage: cannot continue a stream of leading shape \
              [12] with a chunk of leading shape [2; 12]" ) (fun () ->
            ignore
              (stream_outputs (chroma_cens_stage ()) ~source:(source Nx.float64)
                 ~max_chunk:10
                 [x; chromagram Nx.float64 [|2|] 10] ) ) ) ]

let suite =
  [ group "law" law_tests
  ; group "compositions" composition_tests
  ; group "properties" property_tests
  ; group "queries" query_tests
  ; group "errors" error_tests ]
