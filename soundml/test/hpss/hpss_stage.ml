(* The harmonic/percussive separation stages.

   - the pipeline law, bit for bit, for each of the three stages at both widths:

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s) = face x

   over one-frame pushes, the whole spectrogram, chunks of exactly k_h frames,
   chunks shorter than the kernel (k_h - 1 and lookahead + 1 frames) and seeded
   random partitions with empty chunks sprinkled in, across odd and even
   kernels, kernels longer than the spectrogram, frame counts around the
   lookahead and the kernel, several leading shapes including an empty bin axis,
   and data with silent bands, ties, signed zeros and NaN cells — the cases
   where a different selection algorithm would pick a different element.

   - emission: a push completes exactly the frames whose windows it closes, the
   flush the rest in one chunk, a second flush nothing, and a reset stream
   repeats itself.

   - the split across domains: a whole-chunk push of a full-size spectrogram,
   which the median splits across domains, and one-frame pushes, which it cannot
   split, give the same bits as the offline face.

   - the signal domain: Stft.stage >> map fst (or snd) of the complex stage >>
   Stft.synthesis_stage totals to Stft.invert at its default length of the
   offline components, and to harmonic and percussive on the samples both carry;
   the fanout over both components agrees with the two chains.

   - the latency queries and every precondition's message. *)

open Windtrap
open Soundml

let seed = 0x4b55

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

let lookahead k = k - 1 - (k / 2)

let step state =
  state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
  !state

(* {1 Fixtures} *)

(* A non-negative spectrogram of [shape]: LCG magnitudes with runs of tied small
   integers and negative zeros, the top two bins silent (zeros of both signs,
   the denormal branch of the masks), and with [nan] a NaN in about one cell in
   thirty. *)
let spectrogram_values ?(nan = false) ~salt shape =
  let st = ref (seed + salt) in
  let nd = Array.length shape in
  let n = shape.(nd - 1) and bins = shape.(nd - 2) in
  Array.init (Array.fold_left ( * ) 1 shape) (fun idx ->
      let b = idx / n mod bins in
      let x = step st in
      if nan && x mod 30 = 7 then Float.nan
      else if bins > 2 && b >= bins - 2 then if x mod 3 = 0 then -0. else 0.
      else
        match x mod 7 with
        | 0 ->
            -0.
        | 1 | 2 ->
            Float.of_int (x / 7 mod 3)
        | _ ->
            4. *. Float.of_int x /. Float.of_int (1 lsl 31) )

let spectrogram dtype ?nan ~salt shape =
  Nx.create dtype shape (spectrogram_values ?nan ~salt shape)

(* A complex spectrum of [shape]: parts from the LCG, a few bins exactly zero
   (the unit phase's offset), signed-zero parts, and the silent top bins. *)
let spectrum cdtype ?(nan = false) ~salt shape =
  let st = ref (seed + (7 * salt) + 1) in
  let nd = Array.length shape in
  let n = shape.(nd - 1) and bins = shape.(nd - 2) in
  let count = Array.fold_left ( * ) 1 shape in
  let part () =
    let x = step st in
    match x mod 9 with
    | 0 ->
        -0.
    | 1 ->
        0.
    | _ ->
        (2. *. Float.of_int x /. Float.of_int (1 lsl 31)) -. 1.
  in
  let re = Array.make count 0. and im = Array.make count 0. in
  for idx = 0 to count - 1 do
    let b = idx / n mod bins in
    let x = step st in
    if nan && x mod 30 = 7 then re.(idx) <- Float.nan
    else if bins > 2 && b >= bins - 2 then begin
      re.(idx) <- (if x mod 2 = 0 then -0. else 0.) ;
      im.(idx) <- (if x mod 3 = 0 then -0. else 0.)
    end
    else if x mod 11 = 0 then ()
    else begin
      re.(idx) <- part () ;
      im.(idx) <- part ()
    end
  done ;
  Nx.complex cdtype
    ~re:(Nx.create Nx.float64 shape re)
    ~im:(Nx.create Nx.float64 shape im)

let bits t = Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 t))

let complex_bits z =
  Array.append (bits (Nx.real Nx.float64 z)) (bits (Nx.imag Nx.float64 z))

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let frames_of t start stop =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
    t

let chunks_of x sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        frames_of x off (off + s) :: go (off + s) rest
  in
  go 0 sizes

let concat_pairs = function
  | [] ->
      None
  | parts ->
      Some
        ( Nx.concatenate ~axis:(-1) (List.map fst parts)
        , Nx.concatenate ~axis:(-1) (List.map snd parts) )

(* {1 Partitions} *)

let rec cycle size n = if n <= size then [n] else size :: cycle size (n - size)

let rec random_sizes rng ~most n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (Int.min n most) in
    s :: random_sizes rng ~most (n - s)

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 4 = 0 then [0; s] else [s])
    sizes

let partitions rng ~k n =
  let r = lookahead k in
  let fixed name size =
    if size >= 1 && n > 0 then [(name, cycle size n)] else []
  in
  [ ("ones", List.init n (fun _ -> 1))
  ; ("whole", if n = 0 then [] else [n])
  ; ("whole-then-empty", [n; 0]) ]
  @ fixed "kernel" k
  @ fixed "kernel-1" (k - 1)
  @ fixed "lookahead+1" (r + 1)
  @ [ ("random-1..7-with-empty", sprinkle_empty rng (random_sizes rng ~most:7 n))
    ; ("random-1..3k", random_sizes rng ~most:(3 * k) n) ]

(* {1 The law} *)

(* [law ~msg ~stage ~face ~bits x parts] checks [Pipeline.run] and every
   streamed partition against the offline [face], both components bit for
   bit. *)
let law ~msg ~stage ~face ~bits x parts =
  let h, p = face x in
  let expected = (bits h, bits p) and shape = Nx.shape h in
  let check name (gh, gp) =
    equal
      ~msg:(msg ^ " " ^ name ^ ": harmonic shape")
      (array int) shape (Nx.shape gh) ;
    equal
      ~msg:(msg ^ " " ^ name ^ ": percussive shape")
      (array int) shape (Nx.shape gp) ;
    equal
      ~msg:(msg ^ " " ^ name ^ ": harmonic")
      (array int64) (fst expected) (bits gh) ;
    equal
      ~msg:(msg ^ " " ^ name ^ ": percussive")
      (array int64) (snd expected) (bits gp)
  in
  check "run" (Pipeline.run ~source:(source ()) (stage ()) x) ;
  List.iter
    (fun (pname, sizes) ->
      let s =
        Pipeline.Stream.prepare (stage ()) ~source:(source ())
          ~max_chunk:(List.fold_left Int.max 1 sizes)
      in
      let pushed =
        List.filter_map (Pipeline.Stream.push s) (chunks_of x sizes)
      in
      match concat_pairs (pushed @ Pipeline.Stream.flush s) with
      | Some pair ->
          check pname pair
      | None ->
          equal
            ~msg:(msg ^ " " ^ pname ^ ": nothing emitted")
            (array int64) [||] (fst expected) )
    parts

let kernels =
  [ (1, 1)
  ; (2, 3)
  ; (3, 5)
  ; (4, 4)
  ; (7, 3)
  ; (8, 9)
  ; (17, 31)
  ; (31, 31)
  ; (32, 32)
  ; (33, 7)
  ; (64, 5)
  ; (129, 3) ]

let pick l i = List.nth l (i mod List.length l)

let frame_counts k =
  let r = lookahead k in
  List.sort_uniq Int.compare
    (List.filter
       (fun n -> n >= 0)
       [0; 1; 2; r; r + 1; r + 2; k - 1; k; k + 1; (2 * k) + 3; 40] )

let leads = [[|1|]; [|5|]; [|17|]; [|2; 9|]; [|2; 3; 5|]]

(* Every frame count over a leading shape with rows, then an empty bin axis. *)
let cases k =
  List.mapi (fun i n -> (n, pick leads (i + k))) (frame_counts k)
  @ [(0, [|0|]); (k + 1, [|0|]); (k + 1, [|3; 0|])]

let settings =
  [ (2., (1., 1.))
  ; (1., (1., 3.))
  ; (0.5, (2., 2.))
  ; (3.5, (1., 1.))
  ; (Float.infinity, (1., 1.))
  ; (Float.infinity, (1., 3.)) ]

type face =
  | Face :
      string
      * (   kernel_size:int * int
         -> power:float
         -> margin:float * float
         -> unit
         -> ( ('a, 'b) Nx.t
            , ('a, 'b) Nx.t * ('a, 'b) Nx.t
            , Pipeline.causal )
            Pipeline.t )
      * (   kernel_size:int * int
         -> power:float
         -> margin:float * float
         -> ('a, 'b) Nx.t
         -> ('a, 'b) Nx.t * ('a, 'b) Nx.t )
      * (?nan:bool -> salt:int -> int array -> ('a, 'b) Nx.t)
      * (('a, 'b) Nx.t -> int64 array)
      -> face

let spectrogram_face name dtype =
  Face
    ( "spectrogram " ^ name
    , (fun ~kernel_size ~power ~margin () ->
        hpss_of_spectrogram_stage ~kernel_size ~power ~margin () )
    , (fun ~kernel_size ~power ~margin s ->
        hpss_of_spectrogram ~kernel_size ~power ~margin s )
    , spectrogram dtype
    , bits )

let masks_face name dtype =
  Face
    ( "masks " ^ name
    , (fun ~kernel_size ~power ~margin () ->
        hpss_masks_stage ~kernel_size ~power ~margin () )
    , (fun ~kernel_size ~power ~margin s ->
        hpss_masks ~kernel_size ~power ~margin s )
    , spectrogram dtype
    , bits )

let stft_face name cdtype =
  Face
    ( "stft " ^ name
    , (fun ~kernel_size ~power ~margin () ->
        hpss_of_stft_stage ~kernel_size ~power ~margin () )
    , (fun ~kernel_size ~power ~margin z ->
        hpss_of_stft ~kernel_size ~power ~margin z )
    , spectrum cdtype
    , complex_bits )

let faces =
  [ spectrogram_face "float32" Nx.float32
  ; spectrogram_face "float64" Nx.float64
  ; masks_face "float32" Nx.float32
  ; masks_face "float64" Nx.float64
  ; stft_face "complex64" Nx.complex64
  ; stft_face "complex128" Nx.complex128 ]

let law_tests =
  List.concat_map
    (fun (Face (fname, stage, face, make, bits)) ->
      List.map
        (fun ((k_h, k_p) as kernel_size) ->
          test (Printf.sprintf "%s, kernel (%d, %d)" fname k_h k_p) (fun () ->
              let rng = Random.State.make [|seed; k_h; k_p|] in
              List.iteri
                (fun i (n, lead) ->
                  let power, margin = pick settings (i + k_p) in
                  let x =
                    make ~salt:((100 * i) + k_h) (Array.append lead [|n|])
                  in
                  law
                    ~msg:
                      (Printf.sprintf "n=%d lead=%d p=%g" n (Array.length lead)
                         power )
                    ~stage:(stage ~kernel_size ~power ~margin)
                    ~face:(face ~kernel_size ~power ~margin)
                    ~bits x (partitions rng ~k:k_h n) )
                (cases k_h) ) )
        kernels )
    faces

(* NaN cells: the same algorithm on the same values selects the same element,
   whatever the window holds. *)
let nan_tests =
  List.map
    (fun (Face (fname, stage, face, make, bits)) ->
      test (fname ^ " with NaN cells") (fun () ->
          List.iter
            (fun ((k_h, _) as kernel_size) ->
              let rng = Random.State.make [|seed; k_h; 1|] in
              let x = make ~nan:true ~salt:k_h [|9; 45|] in
              law
                ~msg:(Printf.sprintf "k_h=%d" k_h)
                ~stage:(stage ~kernel_size ~power:2. ~margin:(1., 1.))
                ~face:(face ~kernel_size ~power:2. ~margin:(1., 1.))
                ~bits x (partitions rng ~k:k_h 45) )
            [(3, 3); (8, 5); (31, 7)] ) )
    faces

(* {1 Emission} *)

let emission_tests =
  [ test "a push completes the frames whose windows it closes" (fun () ->
        List.iter
          (fun k_h ->
            let r = lookahead k_h in
            List.iter
              (fun n ->
                let x = spectrogram Nx.float64 ~salt:n [|6; n|] in
                let msg = Printf.sprintf "k_h=%d n=%d" k_h n in
                let s =
                  Pipeline.Stream.prepare
                    (hpss_of_spectrogram_stage ~kernel_size:(k_h, 3) ())
                    ~source:(source ()) ~max_chunk:1
                in
                let emitted = ref 0 and outs = ref [] in
                for i = 0 to n - 1 do
                  ( match Pipeline.Stream.push s (frames_of x i (i + 1)) with
                  | None ->
                      ()
                  | Some ((h, p) as pair) ->
                      equal
                        ~msg:(msg ^ ": components align")
                        int (last_dim h) (last_dim p) ;
                      is_true
                        ~msg:(msg ^ ": a chunk is never empty")
                        (last_dim h > 0) ;
                      emitted := !emitted + last_dim h ;
                      outs := pair :: !outs ) ;
                  equal
                    ~msg:(Printf.sprintf "%s: after %d frames" msg (i + 1))
                    int
                    (Int.max 0 (i + 1 - r))
                    !emitted
                done ;
                equal
                  ~msg:(msg ^ ": an empty push completes nothing")
                  bool true
                  (Pipeline.Stream.push s (frames_of x 0 0) = None) ;
                let tail = Pipeline.Stream.flush s in
                equal
                  ~msg:(msg ^ ": the flush is one chunk at most")
                  bool true
                  (List.length tail <= 1) ;
                equal
                  ~msg:(msg ^ ": the flush completes the rest")
                  int (Int.min n r)
                  (List.fold_left (fun acc (h, _) -> acc + last_dim h) 0 tail) ;
                equal
                  ~msg:(msg ^ ": a second flush is empty")
                  int 0
                  (List.length (Pipeline.Stream.flush s)) ;
                let first =
                  Option.map
                    (fun (h, p) -> (bits h, bits p))
                    (concat_pairs (List.rev_append !outs tail))
                in
                Pipeline.Stream.reset s ;
                let again =
                  let pushed =
                    List.filter_map (Pipeline.Stream.push s)
                      (chunks_of x (cycle 1 n))
                  in
                  Option.map
                    (fun (h, p) -> (bits h, bits p))
                    (concat_pairs (pushed @ Pipeline.Stream.flush s))
                in
                equal
                  ~msg:(msg ^ ": a reset stream repeats itself")
                  bool true (first = again) )
              [0; 1; r; r + 1; (2 * k_h) + 3; 50] )
          [1; 2; 3; 4; 7; 8; 31; 32] ) ]

(* {1 The split across domains} *)

let split_tests =
  let check name ~stage ~face ~bits x =
    let h, p = face x in
    let expected = (bits h, bits p) in
    let whole =
      match Pipeline.run ~source:(source ()) (stage ()) x with
      | h, p ->
          (bits h, bits p)
    in
    let ones =
      let s =
        Pipeline.Stream.prepare (stage ()) ~source:(source ()) ~max_chunk:1
      in
      let pushed =
        List.filter_map (Pipeline.Stream.push s)
          (chunks_of x (cycle 1 (last_dim x)))
      in
      match concat_pairs (pushed @ Pipeline.Stream.flush s) with
      | Some (h, p) ->
          (bits h, bits p)
      | None ->
          ([||], [||])
    in
    equal
      ~msg:(name ^ ": whole push, harmonic")
      (array int64) (fst expected) (fst whole) ;
    equal
      ~msg:(name ^ ": whole push, percussive")
      (array int64) (snd expected) (snd whole) ;
    equal
      ~msg:(name ^ ": one-frame pushes, harmonic")
      (array int64) (fst expected) (fst ones) ;
    equal
      ~msg:(name ^ ": one-frame pushes, percussive")
      (array int64) (snd expected) (snd ones)
  in
  [ test "a full-size spectrogram, split or not" (fun () ->
        let shape = [|1025; 300|] in
        check "float32" ~stage:hpss_of_spectrogram_stage
          ~face:(fun s -> hpss_of_spectrogram s)
          ~bits
          (spectrogram Nx.float32 ~salt:1 shape) ;
        check "float64" ~stage:hpss_of_spectrogram_stage
          ~face:(fun s -> hpss_of_spectrogram s)
          ~bits
          (spectrogram Nx.float64 ~salt:2 shape) ;
        check "complex128" ~stage:hpss_of_stft_stage
          ~face:(fun z -> hpss_of_stft z)
          ~bits:complex_bits
          (spectrum Nx.complex128 ~salt:3 shape) ) ]

(* {1 The signal domain} *)

let lcg_signal n salt =
  let st = ref (seed + salt) in
  Array.init n (fun _ ->
      (Float.of_int (step st) /. Float.of_int (1 lsl 30)) -. 1. )

let stream_signal p x sizes =
  let s =
    Pipeline.Stream.prepare p ~source:(source ())
      ~max_chunk:(List.fold_left Int.max 1 sizes)
  in
  let pushed = List.filter_map (Pipeline.Stream.push s) (chunks_of x sizes) in
  match pushed @ Pipeline.Stream.flush s with
  | [] ->
      None
  | parts ->
      Some (Nx.concatenate ~axis:(-1) parts)

let signal_case : type a c.
       string
    -> (float, a) Nx.dtype
    -> (Complex.t, c) Nx.dtype
    -> Stft.Config.t
    -> int array
    -> kernel_size:int * int
    -> power:float
    -> margin:float * float
    -> unit =
 fun name dtype cdtype c shape ~kernel_size ~power ~margin ->
  let n = shape.(Array.length shape - 1) in
  let x =
    Nx.create dtype shape
      (lcg_signal (Array.fold_left ( * ) 1 shape) (n + Array.length shape))
  in
  let separation () = hpss_of_stft_stage ~kernel_size ~power ~margin () in
  let chain pick =
    Pipeline.(
      Stft.stage cdtype c
      >> map pick (separation ())
      >> Stft.synthesis_stage dtype c )
  in
  let both =
    Pipeline.(
      Stft.stage cdtype c >> separation ()
      >> fanout
           (stateless fst >> Stft.synthesis_stage dtype c)
           (stateless snd >> Stft.synthesis_stage dtype c) )
  in
  let z_h, z_p =
    hpss_of_stft ~kernel_size ~power ~margin (Stft.transform cdtype c x)
  in
  let inverse_h = Stft.invert dtype c z_h
  and inverse_p = Stft.invert dtype c z_p in
  let total = last_dim inverse_h in
  let offline_h = harmonic c ~kernel_size ~power ~margin x
  and offline_p = percussive c ~kernel_size ~power ~margin x in
  let carried = Int.min total n in
  if Stft.Config.hop c * (n / Stft.Config.hop c) = n then
    equal ~msg:(name ^ ": centred frames tile the signal") int n total ;
  let hop = Stft.Config.hop c in
  let rng = Random.State.make [|seed; n; hop|] in
  List.iter
    (fun (pname, sizes) ->
      let msg what = Printf.sprintf "%s %s: %s" name pname what in
      let got_h = Option.get (stream_signal (chain fst) x sizes)
      and got_p = Option.get (stream_signal (chain snd) x sizes) in
      equal
        ~msg:(msg "harmonic is the inverse of the components")
        (array int64) (bits inverse_h) (bits got_h) ;
      equal
        ~msg:(msg "percussive is the inverse of the components")
        (array int64) (bits inverse_p) (bits got_p) ;
      equal
        ~msg:(msg "harmonic is the signal-domain face where both carry")
        (array int64)
        (bits (frames_of offline_h 0 carried))
        (bits (frames_of got_h 0 carried)) ;
      equal
        ~msg:(msg "percussive is the signal-domain face where both carry")
        (array int64)
        (bits (frames_of offline_p 0 carried))
        (bits (frames_of got_p 0 carried)) ;
      let s =
        Pipeline.Stream.prepare both ~source:(source ())
          ~max_chunk:(List.fold_left Int.max 1 sizes)
      in
      let pushed =
        List.filter_map (Pipeline.Stream.push s) (chunks_of x sizes)
      in
      let pairs = pushed @ Pipeline.Stream.flush s in
      let branch f = Nx.concatenate ~axis:(-1) (List.filter_map f pairs) in
      equal
        ~msg:(msg "the fanout's harmonic branch")
        (array int64) (bits got_h)
        (bits (branch fst)) ;
      equal
        ~msg:(msg "the fanout's percussive branch")
        (array int64) (bits got_p)
        (bits (branch snd)) )
    [ ("ones", cycle 1 n)
    ; ("hop", cycle hop n)
    ; ("random-1..300", random_sizes rng ~most:300 n) ]

let signal_tests =
  let configs =
    [ ("reflect", Stft.Config.create ~fft_size:256 ~hop:64 ())
    ; ( "constant"
      , Stft.Config.create ~fft_size:256 ~hop:64 ~pad:(`Constant 0.) () ) ]
  in
  let params =
    [ ((31, 31), 2., (1., 1.))
    ; ((17, 31), Float.infinity, (1., 3.))
    ; ((32, 32), 1., (1., 1.)) ]
  in
  let shapes = [[|2048|]; [|2000|]; [|2; 1536|]] in
  List.concat_map
    (fun (cname, c) ->
      List.mapi
        (fun i shape ->
          let label =
            Printf.sprintf "%s, [%s]" cname
              (String.concat "; "
                 (Array.to_list (Array.map string_of_int shape)) )
          in
          test label (fun () ->
              List.iteri
                (fun j (kernel_size, power, margin) ->
                  if (i + j) mod 2 = 0 then
                    signal_case (label ^ " float64") Nx.float64 Nx.complex128 c
                      shape ~kernel_size ~power ~margin
                  else
                    signal_case (label ^ " float32") Nx.float32 Nx.complex64 c
                      shape ~kernel_size ~power ~margin )
                params ) )
        shapes )
    configs

(* {1 Queries} *)

let query_tests =
  [ test "the lookahead is k_h - 1 - k_h / 2 frames" (fun () ->
        List.iter
          (fun (k_h, expected) ->
            let kernel_size = (k_h, 5) in
            let msg = Printf.sprintf "k_h=%d" k_h in
            equal ~msg:(msg ^ " spectrogram") rate_t (r expected 1)
              (Pipeline.latency (hpss_of_spectrogram_stage ~kernel_size ())) ;
            equal ~msg:(msg ^ " masks") rate_t (r expected 1)
              (Pipeline.latency (hpss_masks_stage ~kernel_size ())) ;
            equal ~msg:(msg ^ " stft") rate_t (r expected 1)
              (Pipeline.latency (hpss_of_stft_stage ~kernel_size ())) ;
            equal ~msg:(msg ^ " rate") rate_t (r 1 1)
              (Pipeline.rate (hpss_of_stft_stage ~kernel_size ())) )
          [(1, 0); (2, 0); (3, 1); (4, 1); (31, 15); (32, 15)] )
  ; test "behind an analysis, the lookahead counts hops" (fun () ->
        List.iter
          (fun (fft_size, hop, k_h) ->
            let c = Stft.Config.create ~fft_size ~hop () in
            let msg = Printf.sprintf "fft %d hop %d k_h %d" fft_size hop k_h in
            let chain =
              Pipeline.(
                Stft.stage Nx.complex128 c
                >> map fst (hpss_of_stft_stage ~kernel_size:(k_h, 31) ())
                >> Stft.synthesis_stage Nx.float64 c )
            in
            let round_trip =
              Pipeline.(
                Stft.stage Nx.complex128 c >> Stft.synthesis_stage Nx.float64 c )
            in
            let base = Pipeline.latency round_trip in
            equal ~msg rate_t
              (r (base.num + (lookahead k_h * hop * base.den)) base.den)
              (Pipeline.latency chain) )
          [(256, 64, 31); (2048, 512, 31); (512, 128, 4); (255, 64, 2)] ;
        (* the figure the documentation quotes *)
        equal ~msg:"fft 2048, hop 512, the default kernel" rate_t (r 9728 1)
          (Pipeline.latency
             Pipeline.(
               Stft.stage Nx.complex128
                 (Stft.Config.create ~fft_size:2048 ~hop:512 ())
               >> map fst (hpss_of_stft_stage ())
               >> Stft.synthesis_stage Nx.float64
                    (Stft.Config.create ~fft_size:2048 ~hop:512 ()) ) ) ) ]

(* {1 Errors} *)

let push_all p chunks =
  let s =
    Pipeline.Stream.prepare p ~source:(source ())
      ~max_chunk:
        (List.fold_left (fun acc c -> Int.max acc (last_dim c)) 1 chunks)
  in
  List.iter (fun c -> ignore (Pipeline.Stream.push s c)) chunks ;
  s

let error_tests =
  [ test "the parameters are checked when the stage is built" (fun () ->
        raises ~msg:"kernel"
          (Invalid_argument
             "hpss_of_spectrogram_stage: cannot median-filter with a kernel of \
              (0, 3) (both kernel sizes must be at least 1)" ) (fun () ->
            ignore (hpss_of_spectrogram_stage ~kernel_size:(0, 3) ()) ) ;
        raises ~msg:"kernel, stft"
          (Invalid_argument
             "hpss_of_stft_stage: cannot median-filter with a kernel of (3, \
              -1) (both kernel sizes must be at least 1)" ) (fun () ->
            ignore (hpss_of_stft_stage ~kernel_size:(3, -1) ()) ) ;
        List.iter
          (fun power ->
            raises
              ~msg:(Printf.sprintf "power %g" power)
              (Invalid_argument
                 (Printf.sprintf
                    "hpss_masks_stage: cannot raise the mask to the power %g \
                     (power must be strictly positive, or infinite for a hard \
                     mask)"
                    power ) )
              (fun () -> ignore (hpss_masks_stage ~power ())) )
          [0.; Float.nan; -1.] ;
        List.iter
          (fun ((m_h, m_p) as margin) ->
            raises
              ~msg:(Printf.sprintf "margin (%g, %g)" m_h m_p)
              (Invalid_argument
                 (Printf.sprintf
                    "hpss_of_stft_stage: cannot bias the decision by a margin \
                     of (%g, %g) (both margins must be finite and at least 1)"
                    m_h m_p ) )
              (fun () -> ignore (hpss_of_stft_stage ~margin ())) )
          [(0.5, 1.); (1., Float.nan); (Float.infinity, 1.)] )
  ; test "a chunk carries a bin axis and a frame axis" (fun () ->
        raises ~msg:"spectrogram"
          (Invalid_argument
             "hpss_of_spectrogram_stage: cannot separate a rank-1 tensor (a \
              spectrogram carries a bin axis and a frame axis)" ) (fun () ->
            ignore
              (push_all
                 (hpss_of_spectrogram_stage ())
                 [Nx.zeros Nx.float64 [|8|]] ) ) ;
        raises ~msg:"stft, offline driver"
          (Invalid_argument
             "hpss_of_stft_stage: cannot separate a rank-1 tensor (a \
              spectrogram carries a bin axis and a frame axis)" ) (fun () ->
            ignore
              (Pipeline.run ~source:(source ()) (hpss_of_stft_stage ())
                 (Nx.zeros Nx.complex128 [|8|]) ) ) )
  ; test "the median kernel carries float32 and float64" (fun () ->
        raises ~msg:"float16"
          (Invalid_argument
             "hpss_masks_stage: cannot separate float16 spectra (the median \
              kernel carries float32 and float64)" ) (fun () ->
            ignore
              (push_all (hpss_masks_stage ()) [Nx.zeros Nx.float16 [|4; 4|]]) ) )
  ; test "a chunk keeps the stream's leading shape" (fun () ->
        raises ~msg:"bins"
          (Invalid_argument
             "hpss_masks_stage: cannot continue a stream of leading shape [4] \
              with a chunk of leading shape [5]" ) (fun () ->
            ignore
              (push_all (hpss_masks_stage ())
                 [Nx.zeros Nx.float64 [|4; 3|]; Nx.zeros Nx.float64 [|5; 3|]] ) ) ;
        raises ~msg:"channels"
          (Invalid_argument
             "hpss_of_stft_stage: cannot continue a stream of leading shape \
              [2; 9] with a chunk of leading shape [1; 9]" ) (fun () ->
            ignore
              (push_all (hpss_of_stft_stage ())
                 [ Nx.zeros Nx.complex64 [|2; 9; 3|]
                 ; Nx.zeros Nx.complex64 [|1; 9; 3|] ] ) ) )
  ; test "a drained stream takes no chunk until it is reset" (fun () ->
        let x = spectrogram Nx.float64 ~salt:5 [|4; 20|] in
        let s = push_all (hpss_of_spectrogram_stage ()) [frames_of x 0 10] in
        ignore (Pipeline.Stream.flush s) ;
        raises ~msg:"push after flush"
          (Invalid_argument
             "hpss_of_spectrogram_stage: cannot feed a drained stream (flush \
              consumed the tail; reset before reusing)" ) (fun () ->
            ignore (Pipeline.Stream.push s (frames_of x 10 20)) ) ;
        Pipeline.Stream.reset s ;
        (* a reset stream starts afresh, the leading shape included *)
        let y = spectrogram Nx.float64 ~salt:6 [|7; 20|] in
        let pushed = List.filter_map (Pipeline.Stream.push s) [y] in
        let h, p =
          Option.get (concat_pairs (pushed @ Pipeline.Stream.flush s))
        in
        let eh, ep = hpss_of_spectrogram y in
        equal ~msg:"harmonic" (array int64) (bits eh) (bits h) ;
        equal ~msg:"percussive" (array int64) (bits ep) (bits p) ) ]

let suite =
  [ group "stage-law" law_tests
  ; group "stage-nan" nan_tests
  ; group "stage-emission" emission_tests
  ; group "stage-split" split_tests
  ; group "stage-signal" signal_tests
  ; group "stage-queries" query_tests
  ; group "stage-errors" error_tests ]
