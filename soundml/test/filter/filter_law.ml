(* The laws of Soundml.Filter, without an oracle.

   The pipeline law for every stage,

   run ~source p x = concat (List.filter_map (Stream.push s) (chunks x) @
   Stream.flush s) = the offline function

   exactly, over seeded partitionings — one-sample chunks, sizes cycling through
   1 .. 7, random sizes, empty chunks in mid-stream, a first chunk of a single
   sample (the emphasis stages' startup hold), the whole signal — at both
   dtypes, over a single row and a batch. Also here: chaining blocks through the
   final state, the independence of the rows of a batch, the identities between
   the functions, empty signals, the latency queries and every error. *)

open Windtrap
open Soundml

let seed = 0xf117e

let source () = Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1

let rate_t = Testable.make ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal

let r num den = Pipeline.Rate.{num; den}

let numel = Array.fold_left ( * ) 1

let signal dtype shape =
  let n = numel shape in
  Nx.create dtype shape
    (Array.init n (fun i ->
         Float.sin (0.37 *. Float.of_int (i * i mod 211))
         +. (0.25 *. Float.cos (0.011 *. Float.of_int i)) ) )

let vector v = Nx.create Nx.float64 [|Array.length v|] v

let bits t = Nx.to_array (Nx.cast Nx.float64 t)

let same ~msg a b = equal ~msg (array float_exact) (bits a) (bits b)

let last_slice x lo hi =
  let nd = Nx.ndim x in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (lo, hi) else Nx.A))
    x

let concat = function
  | [] ->
      invalid_arg "concat: no chunks"
  | parts ->
      Nx.concatenate ~axis:(-1) parts

(* {1 Partitions} *)

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (min n 7) in
    s :: random_sizes rng (n - s)

let cycling n =
  let rec go k left =
    if left = 0 then []
    else
      let s = min left (1 + (k mod 7)) in
      s :: go (k + 1) (left - s)
  in
  go 0 n

let sprinkle_empty rng sizes =
  List.concat_map
    (fun s -> if Random.State.int rng 3 = 0 then [0; s] else [s])
    sizes

let partitions rng n =
  [ ("ones", List.init n (fun _ -> 1))
  ; ("cycling 1..7", cycling n)
  ; ("whole", [n])
  ; ("random", random_sizes rng n)
  ; ("random with empty chunks", sprinkle_empty rng (random_sizes rng n))
  ; ("one sample first", 1 :: random_sizes rng (n - 1))
  ; ("one sample, empty, rest", [1; 0] @ random_sizes rng (n - 1)) ]

let chunks_of x sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        last_slice x off (off + s) :: go (off + s) rest
  in
  go 0 sizes

let stream_outputs p ~max_chunk chunks =
  let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk in
  (* [@] evaluates its right operand first: sequence the pushes explicitly *)
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  pushed @ Pipeline.Stream.flush s

(* [law ~name dtype shape stage offline] checks the stage against the offline
   function over every partitioning of a signal of [shape]. *)
let law (type a) ~name (dtype : (float, a) Nx.dtype) shape
    (stage :
      unit -> ((float, a) Nx.t, (float, a) Nx.t, Pipeline.causal) Pipeline.t )
    (offline : (float, a) Nx.t -> (float, a) Nx.t) =
  test name (fun () ->
      let rng = Random.State.make [|seed; Array.length shape|] in
      let x = signal dtype shape in
      let expected = offline x in
      same ~msg:"run" expected (Pipeline.run ~source:(source ()) (stage ()) x) ;
      List.iter
        (fun (pname, sizes) ->
          let max_chunk = List.fold_left max 1 sizes in
          let outs = stream_outputs (stage ()) ~max_chunk (chunks_of x sizes) in
          same ~msg:pname expected (concat outs) )
        (partitions rng shape.(Array.length shape - 1)) )

let b4, a4 = Filter.butter ~order:4 ~sample_rate:16000 (`Lowpass 1000.)

let sos6 =
  Filter.butter_sos ~order:6 ~sample_rate:16000 (`Bandpass (300., 3400.))

let zi_tf lead =
  Nx.create Nx.float64 (Array.append lead [|4|])
    (Array.init (numel lead * 4) (fun i -> 0.1 *. Float.of_int (i + 1)))

let zi_sos lead =
  Nx.create Nx.float64
    (Array.append lead [|6; 2|])
    (Array.init (numel lead * 12) (fun i -> 0.05 *. Float.of_int (i - 5)))

let zi_one lead =
  Nx.create Nx.float64 (Array.append lead [|1|])
    (Array.init (numel lead) (fun i -> 0.3 -. (0.2 *. Float.of_int i)))

(* a stage and its offline function, at every dtype *)
type pair =
  { stage:
      'a.
      unit -> ((float, 'a) Nx.t, (float, 'a) Nx.t, Pipeline.causal) Pipeline.t
  ; offline: 'a. (float, 'a) Nx.t -> (float, 'a) Nx.t }

let pairs lead =
  let zt = zi_tf lead and zs = zi_sos lead and z1 = zi_one lead in
  [ ( "lfilter_stage"
    , { stage= (fun () -> Filter.lfilter_stage b4 a4)
      ; offline= (fun x -> Filter.lfilter b4 a4 x) } )
  ; ( "lfilter_stage ~zi"
    , { stage= (fun () -> Filter.lfilter_stage ~zi:zt b4 a4)
      ; offline= (fun x -> Filter.lfilter ~zi:zt b4 a4 x) } )
  ; ( "sosfilt_stage"
    , { stage= (fun () -> Filter.sosfilt_stage sos6)
      ; offline= (fun x -> Filter.sosfilt sos6 x) } )
  ; ( "sosfilt_stage ~zi"
    , { stage= (fun () -> Filter.sosfilt_stage ~zi:zs sos6)
      ; offline= (fun x -> Filter.sosfilt ~zi:zs sos6 x) } )
  ; ( "preemphasis_stage"
    , { stage= (fun () -> Filter.preemphasis_stage ())
      ; offline= (fun x -> Filter.preemphasis x) } )
  ; ( "preemphasis_stage ~zi"
    , { stage= (fun () -> Filter.preemphasis_stage ~coef:0.9 ~zi:z1 ())
      ; offline= (fun x -> Filter.preemphasis ~coef:0.9 ~zi:z1 x) } )
  ; ( "deemphasis_stage"
    , { stage= (fun () -> Filter.deemphasis_stage ())
      ; offline= (fun x -> Filter.deemphasis x) } )
  ; ( "deemphasis_stage ~zi"
    , { stage= (fun () -> Filter.deemphasis_stage ~coef:0.5 ~zi:z1 ())
      ; offline= (fun x -> Filter.deemphasis ~coef:0.5 ~zi:z1 x) } ) ]

let law_tests =
  List.concat_map
    (fun shape ->
      let lead = Array.sub shape 0 (Array.length shape - 1) in
      let tag =
        String.concat "; " (Array.to_list (Array.map string_of_int shape))
      in
      List.concat_map
        (fun (name, p) ->
          [ law
              ~name:(Printf.sprintf "%s, [%s], float64" name tag)
              Nx.float64 shape p.stage p.offline
          ; law
              ~name:(Printf.sprintf "%s, [%s], float32" name tag)
              Nx.float32 shape p.stage p.offline ] )
        (pairs lead) )
    [[|53|]; [|2; 3; 41|]]

(* {1 Stages} *)

let raises_prefix ~msg prefix f =
  raises_match ~msg (Exn.invalid_arg ~substring:prefix) f

let stage_tests =
  [ test "latency: none, but one sample for librosa's emphasis start" (fun () ->
        let lat p = Pipeline.latency p in
        equal ~msg:"lfilter" rate_t (r 0 1) (lat (Filter.lfilter_stage b4 a4)) ;
        equal ~msg:"sosfilt" rate_t (r 0 1) (lat (Filter.sosfilt_stage sos6)) ;
        equal ~msg:"preemphasis" rate_t (r 1 1)
          (lat (Filter.preemphasis_stage ())) ;
        equal ~msg:"deemphasis" rate_t (r 1 1)
          (lat (Filter.deemphasis_stage ())) ;
        let zi = zi_one [||] in
        equal ~msg:"preemphasis ~zi" rate_t (r 0 1)
          (lat (Filter.preemphasis_stage ~zi ())) ;
        equal ~msg:"deemphasis ~zi" rate_t (r 0 1)
          (lat (Filter.deemphasis_stage ~zi ())) ;
        equal ~msg:"rate" rate_t (r 1 1)
          (Pipeline.rate (Filter.lfilter_stage b4 a4)) )
  ; test "the emphasis stages hold the first sample alone" (fun () ->
        let s =
          Pipeline.Stream.prepare
            (Filter.preemphasis_stage ())
            ~source:(source ()) ~max_chunk:4
        in
        let x = signal Nx.float64 [|5|] in
        let first = Option.get (Pipeline.Stream.push s (last_slice x 0 1)) in
        equal ~msg:"nothing yet" (array int) [|0|] (Nx.shape first) ;
        let next = Option.get (Pipeline.Stream.push s (last_slice x 1 3)) in
        equal ~msg:"both released" (array int) [|3|] (Nx.shape next) ;
        same ~msg:"values" (last_slice (Filter.preemphasis x) 0 3) next )
  ; test "a stream of one sample flushes like the offline call raises"
      (fun () ->
        let x = signal Nx.float64 [|2; 1|] in
        List.iter
          (fun (name, p, offline) ->
            raises_prefix ~msg:(name ^ " offline")
              ( name
              ^ ": cannot extrapolate a starting state from a single sample" )
              (fun () -> offline x ) ;
            raises_prefix ~msg:(name ^ "_stage flush")
              ( name
              ^ "_stage: cannot extrapolate a starting state from a single \
                 sample" ) (fun () -> stream_outputs p ~max_chunk:1 [x] ) )
          [ ( "preemphasis"
            , Filter.preemphasis_stage ()
            , fun x -> Filter.preemphasis x )
          ; ( "deemphasis"
            , Filter.deemphasis_stage ()
            , fun x -> Filter.deemphasis x ) ] ;
        (* an empty stream is not a short one *)
        let outs =
          stream_outputs
            (Filter.preemphasis_stage ())
            ~max_chunk:1
            [last_slice x 0 0]
        in
        equal ~msg:"empty stream" (array int) [|2; 0|] (Nx.shape (concat outs)) )
  ; test "reset restarts every stage" (fun () ->
        let x = signal Nx.float64 [|2; 17|] in
        let check name p =
          let s = Pipeline.Stream.prepare p ~source:(source ()) ~max_chunk:17 in
          let first = Option.get (Pipeline.Stream.push s x) in
          ignore (Pipeline.Stream.push s x) ;
          Pipeline.Stream.reset s ;
          let again = Option.get (Pipeline.Stream.push s x) in
          same ~msg:name first again
        in
        check "lfilter" (Filter.lfilter_stage b4 a4) ;
        check "sosfilt" (Filter.sosfilt_stage sos6) ;
        check "preemphasis" (Filter.preemphasis_stage ()) ;
        check "deemphasis" (Filter.deemphasis_stage ()) )
  ; test "a chunk must keep the stream's leading shape" (fun () ->
        List.iter
          (fun (name, p) ->
            raises ~msg:name
              (Invalid_argument
                 ( name
                 ^ ": cannot continue a stream of leading shape [3] with a \
                    chunk of leading shape [4]" ) )
              (fun () ->
                stream_outputs p ~max_chunk:5
                  [signal Nx.float64 [|3; 5|]; signal Nx.float64 [|4; 5|]] ) )
          [ ("lfilter_stage", Filter.lfilter_stage b4 a4)
          ; ("sosfilt_stage", Filter.sosfilt_stage sos6)
          ; ("preemphasis_stage", Filter.preemphasis_stage ())
          ; ("deemphasis_stage", Filter.deemphasis_stage ()) ] ) ]

(* {1 States} *)

(* a filter run from an optional state, returning its final state *)
type runner =
  { run:
      'a.
      Filter.state option -> (float, 'a) Nx.t -> (float, 'a) Nx.t * Filter.state
  }

let chain_case (type a) ~name (dtype : (float, a) Nx.dtype) r =
  test name (fun () ->
      let x = signal dtype [|2; 3; 50|] in
      let whole, zf = r.run None x in
      let y1, z1 = r.run None (last_slice x 0 21) in
      let y2, z2 = r.run (Some z1) (last_slice x 21 50) in
      same ~msg:"output" whole (concat [y1; y2]) ;
      same ~msg:"state" zf z2 )

let chain_tests =
  let zt = zi_tf [|3|] and zs = zi_sos [|2; 1|] and z1 = zi_one [|2; 3|] in
  let start zi z0 = Option.value zi ~default:z0 in
  List.concat_map
    (fun (name, r) ->
      [ chain_case ~name:(name ^ ", float64") Nx.float64 r
      ; chain_case ~name:(name ^ ", float32") Nx.float32 r ] )
    [ ( "lfilter_zf chains"
      , {run= (fun zi x -> Filter.lfilter_zf ~zi:(start zi zt) b4 a4 x)} )
    ; ( "sosfilt_zf chains"
      , {run= (fun zi x -> Filter.sosfilt_zf ~zi:(start zi zs) sos6 x)} )
    ; ( "preemphasis_zf chains from librosa's start"
      , {run= (fun zi x -> Filter.preemphasis_zf ?zi x)} )
    ; ( "deemphasis_zf chains from a given state"
      , {run= (fun zi x -> Filter.deemphasis_zf ~zi:(start zi z1) x)} ) ]

let state_tests =
  [ test "deemphasis_zf without ~zi continues the output to the arithmetic"
      (fun () ->
        let x = signal Nx.float64 [|2; 80|] in
        let whole = Filter.deemphasis x in
        let y1, z1 = Filter.deemphasis_zf (last_slice x 0 31) in
        let y2 = Filter.deemphasis ~zi:z1 (last_slice x 31 80) in
        let got = bits (concat [y1; y2]) and want = bits whole in
        Array.iteri
          (fun i w -> equal ~msg:(string_of_int i) (float 1e-13) w got.(i))
          want ;
        (* the state is coef times the last output *)
        let y, zf = Filter.deemphasis_zf ~coef:0.8 x in
        let y = bits y in
        equal ~msg:"zf" (array float_exact)
          [|0.8 *. y.(79); 0.8 *. y.(159)|]
          (bits zf) )
  ; test "a state broadcasts over the leading axes" (fun () ->
        let x = signal Nx.float64 [|2; 3; 30|] in
        let zi = zi_tf [||] in
        let full = Nx.broadcast_to [|2; 3; 4|] zi in
        same ~msg:"rank one"
          (Filter.lfilter ~zi:full b4 a4 x)
          (Filter.lfilter ~zi b4 a4 x) ;
        let mid = Nx.reshape [|1; 3; 4|] (Nx.broadcast_to [|3; 4|] zi) in
        same ~msg:"size one"
          (Filter.lfilter ~zi:full b4 a4 x)
          (Filter.lfilter ~zi:mid b4 a4 x) ;
        let zs = zi_sos [||] in
        same ~msg:"sections"
          (Filter.sosfilt ~zi:(Nx.broadcast_to [|2; 3; 6; 2|] zs) sos6 x)
          (Filter.sosfilt ~zi:zs sos6 x) )
  ; test "an empty signal returns its state" (fun () ->
        let x = Nx.zeros Nx.float64 [|2; 0|] in
        let zi = zi_tf [|2|] in
        let y, zf = Filter.lfilter_zf ~zi b4 a4 x in
        equal ~msg:"lfilter output" (array int) [|2; 0|] (Nx.shape y) ;
        same ~msg:"lfilter state" zi zf ;
        let zs = zi_sos [|2|] in
        let y, zf = Filter.sosfilt_zf ~zi:zs sos6 x in
        equal ~msg:"sosfilt output" (array int) [|2; 0|] (Nx.shape y) ;
        same ~msg:"sosfilt state" zs zf ;
        let y, zf = Filter.preemphasis_zf x in
        equal ~msg:"preemphasis output" (array int) [|2; 0|] (Nx.shape y) ;
        same ~msg:"preemphasis state" (Nx.zeros Nx.float64 [|2; 1|]) zf ;
        let y, zf = Filter.deemphasis_zf x in
        equal ~msg:"deemphasis output" (array int) [|2; 0|] (Nx.shape y) ;
        same ~msg:"deemphasis state" (Nx.zeros Nx.float64 [|2; 1|]) zf )
  ; test "the steady state is the response to a long step" (fun () ->
        (* a filter started from lfilter_zi under a constant input stays at its
           DC gain from the first sample *)
        let x = Nx.full Nx.float64 [|64|] 0.7 in
        let zi = Nx.mul_s (Filter.lfilter_zi b4 a4) 0.7 in
        Array.iter
          (fun v -> equal ~msg:"lfilter" (float 1e-12) 0.7 v)
          (bits (Filter.lfilter ~zi b4 a4 x)) ;
        let zs = Nx.mul_s (Filter.sosfilt_zi sos6) 0.7 in
        Array.iter
          (fun v -> equal ~msg:"sosfilt, bandpass: DC gain 0" (float 1e-12) 0. v)
          (bits (Filter.sosfilt ~zi:zs sos6 x)) )
  ; test "the steady state keeps scipy's signs of zero" (fun () ->
        (* numpy sums negative zeros to +0, so a numerator of negative zeros
           settles to negative zeros, as in scipy *)
        same ~msg:"lfilter_zi" (vector [|-0.|])
          (Filter.lfilter_zi (vector [|-0.; -0.|]) (vector [|1.; 0.5|])) ;
        same ~msg:"sosfilt_zi"
          (Nx.create Nx.float64 [|1; 2|] [|-0.; -0.|])
          (Filter.sosfilt_zi
             (Nx.create Nx.float64 [|1; 6|] [|-0.; -0.; -0.; 1.; 0.5; 0.25|]) ) )
  ]

(* {1 Rows and identities} *)

let row_tests =
  [ test "a row of a batch is filtered as if alone" (fun () ->
        let x = signal Nx.float64 [|5; 3; 70|] in
        let check name f =
          let all = f x in
          for i = 0 to 4 do
            for j = 0 to 2 do
              let row = Nx.slice [Nx.I i; Nx.I j; Nx.A] x in
              same
                ~msg:(Printf.sprintf "%s row %d %d" name i j)
                (Nx.slice [Nx.I i; Nx.I j; Nx.A] all)
                (f row)
            done
          done
        in
        check "lfilter" (Filter.lfilter b4 a4) ;
        check "sosfilt" (Filter.sosfilt sos6) ;
        check "filtfilt" (Filter.filtfilt b4 a4) ;
        check "sosfiltfilt" (Filter.sosfiltfilt sos6) ;
        check "preemphasis" (fun x -> Filter.preemphasis x) ;
        check "deemphasis" (fun x -> Filter.deemphasis x) ) ]

let identity_tests =
  [ test "the emphasis filters are their transfer functions" (fun () ->
        let x = signal Nx.float64 [|2; 40|] in
        let zi = zi_one [|2|] in
        same ~msg:"preemphasis"
          (Filter.lfilter ~zi (vector [|1.; -0.9|]) (vector [|1.|]) x)
          (Filter.preemphasis ~coef:0.9 ~zi x) ;
        same ~msg:"deemphasis"
          (Filter.lfilter ~zi (vector [|1.|]) (vector [|1.; -0.9|]) x)
          (Filter.deemphasis ~coef:0.9 ~zi x) ;
        (* librosa's start is the state 2 x0 - x1 *)
        let x0 = Nx.slice [Nx.A; Nx.I 0] x and x1 = Nx.slice [Nx.A; Nx.I 1] x in
        let start = Nx.reshape [|2; 1|] (Nx.sub (Nx.mul_s x0 2.) x1) in
        same ~msg:"librosa's start"
          (Filter.preemphasis ~zi:start x)
          (Filter.preemphasis x) )
  ; test "de-emphasis by the opposite state inverts pre-emphasis" (fun () ->
        let x = signal Nx.float64 [|2; 60|] in
        let zi = zi_one [|2|] in
        let back =
          Filter.deemphasis ~zi:(Nx.neg zi) (Filter.preemphasis ~zi x)
        in
        Array.iter2
          (fun w g -> equal ~msg:"round trip" (float 1e-13) w g)
          (bits x) (bits back) ;
        let back = Filter.deemphasis (Filter.preemphasis x) in
        Array.iter2
          (fun w g -> equal ~msg:"librosa's defaults" (float 1e-13) w g)
          (bits x) (bits back) )
  ; test "sections and transfer function are one filter" (fun () ->
        let x = signal Nx.float64 [|120|] in
        List.iter
          (fun band ->
            let b, a = Filter.butter ~order:4 ~sample_rate:16000 band
            and sos = Filter.butter_sos ~order:4 ~sample_rate:16000 band in
            Array.iter2
              (fun w g -> equal ~msg:"sosfilt = lfilter" (float 1e-11) w g)
              (bits (Filter.lfilter b a x))
              (bits (Filter.sosfilt sos x)) )
          [ `Lowpass 1000.
          ; `Highpass 300.
          ; `Bandpass (300., 3400.)
          ; `Bandstop (2000., 2500.) ] )
  ; test "Butterworth gains at DC and Nyquist" (fun () ->
        let gain_at sign (b, a) =
          let s v =
            snd
              (Array.fold_left
                 (fun (k, acc) c ->
                   (k + 1, acc +. if k mod 2 = 0 then c else sign *. c) )
                 (0, 0.) (bits v) )
          in
          s b /. s a
        in
        let f = Filter.butter ~sample_rate:16000 in
        equal ~msg:"lowpass at DC" (float 1e-9) 1.
          (gain_at 1. (f ~order:5 (`Lowpass 1000.))) ;
        equal ~msg:"lowpass at Nyquist" (float 1e-9) 0.
          (gain_at (-1.) (f ~order:5 (`Lowpass 1000.))) ;
        equal ~msg:"highpass at Nyquist" (float 1e-9) 1.
          (Float.abs (gain_at (-1.) (f ~order:5 (`Highpass 3000.)))) ;
        equal ~msg:"bandstop at DC" (float 1e-9) 1.
          (gain_at 1. (f ~order:3 (`Bandstop (1000., 2000.)))) )
  ; test "order 0 is the identity" (fun () ->
        let b, a = Filter.butter ~order:0 ~sample_rate:16000 (`Lowpass 1000.) in
        equal ~msg:"b" (array float_exact) [|1.|] (bits b) ;
        equal ~msg:"a" (array float_exact) [|1.|] (bits a) ;
        equal ~msg:"sos" (array float_exact) [|1.; 0.; 0.; 1.; 0.; 0.|]
          (bits
             (Filter.butter_sos ~order:0 ~sample_rate:16000
                (`Bandpass (1., 2.)) ) ) ;
        let x = signal Nx.float32 [|2; 40|] in
        same ~msg:"filtfilt" x (Filter.filtfilt b a x) )
  ; test "zero-phase filtering keeps a constant" (fun () ->
        let x = Nx.full Nx.float64 [|100|] (-0.4) in
        Array.iter
          (fun v -> equal ~msg:"filtfilt" (float 1e-12) (-0.4) v)
          (bits (Filter.filtfilt b4 a4 x)) ;
        let sos =
          Filter.butter_sos ~order:5 ~sample_rate:16000 (`Lowpass 1000.)
        in
        Array.iter
          (fun v -> equal ~msg:"sosfiltfilt" (float 1e-12) (-0.4) v)
          (bits (Filter.sosfiltfilt sos x)) ) ]

(* {1 Errors} *)

let x64 = signal Nx.float64 [|2; 30|]

let error_tests =
  let inv msg f = raises ~msg (Invalid_argument msg) f in
  [ test "coefficients" (fun () ->
        inv "lfilter: cannot use b of shape [1; 2] (b must have rank 1)"
          (fun () -> Filter.lfilter (Nx.zeros Nx.float64 [|1; 2|]) a4 x64 ) ;
        inv "lfilter: cannot use an empty a (a needs at least one coefficient)"
          (fun () -> Filter.lfilter b4 (Nx.zeros Nx.float64 [|0|]) x64 ) ;
        inv
          "lfilter_stage: cannot use the non-finite coefficients of b (every \
           coefficient must be finite)" (fun () ->
            Filter.lfilter_stage (vector [|1.; Float.nan|]) a4 ) ;
        inv
          "lfilter: cannot normalise by a.(0) = 0 (the leading denominator \
           coefficient must be nonzero)" (fun () ->
            Filter.lfilter b4 (vector [|0.; 1.|]) x64 ) ;
        inv
          "sosfilt: cannot use sections of shape [2; 5] (sos must be \
           [n_sections; 6] with at least one section)" (fun () ->
            Filter.sosfilt (Nx.zeros Nx.float64 [|2; 5|]) x64 ) ;
        inv
          "sosfilt_stage: cannot use section 1 with a0 = 2 (every section must \
           have a0 = 1)" (fun () ->
            Filter.sosfilt_stage
              (Nx.create Nx.float64 [|2; 6|]
                 [|1.; 0.; 0.; 1.; 0.; 0.; 1.; 0.; 0.; 2.; 0.; 0.|] ) ) ;
        inv
          "sosfiltfilt: cannot use non-finite section coefficients (every \
           coefficient must be finite)" (fun () ->
            Filter.sosfiltfilt
              (Nx.create Nx.float64 [|1; 6|]
                 [|1.; 0.; 0.; 1.; Float.infinity; 0.|] )
              x64 ) )
  ; test "signals" (fun () ->
        inv
          "lfilter: cannot filter a rank-zero tensor (the time axis must exist)"
          (fun () -> Filter.lfilter b4 a4 (Nx.scalar Nx.float64 1.) ) ;
        inv "sosfilt: cannot filter this dtype (float32 and float64 only)"
          (fun () -> Filter.sosfilt sos6 (Nx.zeros Nx.float16 [|4|]) ) ;
        inv
          "preemphasis: cannot filter a rank-zero tensor (the time axis must \
           exist)" (fun () -> Filter.preemphasis (Nx.scalar Nx.float64 1.) ) )
  ; test "states" (fun () ->
        inv
          "lfilter: cannot start from a state of shape [2; 3] (zi must be \
           [...; 4])" (fun () ->
            Filter.lfilter ~zi:(Nx.zeros Nx.float64 [|2; 3|]) b4 a4 x64 ) ;
        inv
          "lfilter_zf: cannot start from a state of shape [3; 4] (zi must \
           broadcast to [2; 4])" (fun () ->
            Filter.lfilter_zf ~zi:(Nx.zeros Nx.float64 [|3; 4|]) b4 a4 x64 ) ;
        inv
          "lfilter: cannot start from a state of shape [1; 2; 4] (zi must \
           broadcast to [2; 4])" (fun () ->
            Filter.lfilter ~zi:(Nx.zeros Nx.float64 [|1; 2; 4|]) b4 a4 x64 ) ;
        inv
          "sosfilt_stage: cannot start from a state of shape [6] (zi must be \
           [...; 6; 2])" (fun () ->
            Filter.sosfilt_stage ~zi:(Nx.zeros Nx.float64 [|6|]) sos6 ) ;
        inv
          "preemphasis_stage: cannot start from a state of shape [2] (zi must \
           be [...; 1])" (fun () ->
            Filter.preemphasis_stage ~zi:(Nx.zeros Nx.float64 [|2|]) () ) ;
        inv
          "deemphasis: cannot start from a state of shape [3; 1] (zi must \
           broadcast to [2; 1])" (fun () ->
            Filter.deemphasis ~zi:(Nx.zeros Nx.float64 [|3; 1|]) x64 ) )
  ; test "steady states" (fun () ->
        inv
          "lfilter_zi: cannot settle a filter whose denominator sums to 0 (a \
           pole at z = 1 has no steady state)" (fun () ->
            Filter.lfilter_zi (vector [|1.|]) (vector [|1.; -1.|]) ) ;
        inv
          "sosfilt_zi: cannot settle a filter whose denominator sums to 0 (a \
           pole at z = 1 has no steady state)" (fun () ->
            Filter.sosfilt_zi
              (Nx.create Nx.float64 [|1; 6|] [|1.; 0.; 0.; 1.; -2.; 1.|]) ) ;
        inv
          "filtfilt: cannot settle a filter whose denominator sums to 0 (a \
           pole at z = 1 has no steady state)" (fun () ->
            Filter.filtfilt (vector [|1.|]) (vector [|1.; -1.|]) x64 ) )
  ; test "zero-phase padding" (fun () ->
        inv
          "filtfilt: cannot extend a signal of 15 samples by 15 at each end \
           (the signal must be longer than padlen)" (fun () ->
            Filter.filtfilt b4 a4 (signal Nx.float64 [|15|]) ) ;
        inv
          "sosfiltfilt: cannot extend a signal of 18 samples by 18 at each end \
           (the signal must be longer than padlen)" (fun () ->
            Filter.sosfiltfilt
              (Filter.butter_sos ~order:5 ~sample_rate:16000 (`Lowpass 1000.))
              (signal Nx.float64 [|18|]) ) ;
        inv "filtfilt: cannot pad by -1 samples (padlen must be non-negative)"
          (fun () -> Filter.filtfilt ~padlen:(-1) b4 a4 x64 ) ;
        inv
          "sosfiltfilt: cannot extend a signal of 0 samples by 0 at each end \
           (the signal must be longer than padlen)" (fun () ->
            Filter.sosfiltfilt ~padlen:0 sos6 (Nx.zeros Nx.float64 [|0|]) ) )
  ; test "design" (fun () ->
        inv
          "butter: cannot design a filter of order -1 (order must be \
           non-negative)" (fun () ->
            Filter.butter ~order:(-1) ~sample_rate:16000 (`Lowpass 1000.) ) ;
        inv "butter_sos: cannot design at 0 Hz (sample_rate must be at least 1)"
          (fun () ->
            Filter.butter_sos ~order:2 ~sample_rate:0 (`Lowpass 1000.) ) ;
        inv
          "butter: cannot place a cutoff at 8000 Hz (cutoffs must lie strictly \
           between 0 and sample_rate / 2 = 8000 Hz)" (fun () ->
            Filter.butter ~order:2 ~sample_rate:16000 (`Highpass 8000.) ) ;
        inv
          "butter: cannot place a cutoff at 0 Hz (cutoffs must lie strictly \
           between 0 and sample_rate / 2 = 8000 Hz)" (fun () ->
            Filter.butter ~order:2 ~sample_rate:16000 (`Bandpass (0., 100.)) ) ;
        inv
          "butter_sos: cannot place a cutoff at nan Hz (cutoffs must lie \
           strictly between 0 and sample_rate / 2 = 8000 Hz)" (fun () ->
            Filter.butter_sos ~order:2 ~sample_rate:16000 (`Lowpass Float.nan) ) ;
        inv
          "butter: cannot use the band (3000, 2000) Hz (its lower edge must \
           lie below its upper edge)" (fun () ->
            Filter.butter ~order:2 ~sample_rate:16000 (`Bandstop (3000., 2000.)) ) )
  ; test "emphasis" (fun () ->
        inv "preemphasis: cannot use coef = nan (coef must be finite)"
          (fun () -> Filter.preemphasis ~coef:Float.nan x64 ) ;
        inv "deemphasis_stage: cannot use coef = inf (coef must be finite)"
          (fun () -> Filter.deemphasis_stage ~coef:Float.infinity () ) ;
        inv
          "preemphasis_zf: cannot extrapolate a starting state from a single \
           sample (without ~zi the signal needs at least two)" (fun () ->
            Filter.preemphasis_zf (signal Nx.float64 [|3; 1|]) ) ;
        inv
          "deemphasis_zf: cannot extrapolate a starting state from a single \
           sample (without ~zi the signal needs at least two)" (fun () ->
            Filter.deemphasis_zf (signal Nx.float64 [|1|]) ) ) ]

let suite =
  [ group "law" law_tests
  ; group "stages" stage_tests
  ; group "chaining" chain_tests
  ; group "states" state_tests
  ; group "rows" row_tests
  ; group "identities" identity_tests
  ; group "errors" error_tests ]
