(* What Griffin-Lim guarantees, checked without an oracle.

   The measure is the spectral convergence of the returned signal,

   SC(y) = ‖ |transform y| - s ‖ / ‖ s ‖,

   the normalised distance between the magnitudes asked for and the magnitudes
   obtained. Reconstructing from magnitudes alone has no exact answer in
   general, so the gates are the ones the algorithm actually promises: the
   momentum-free iteration never increases SC, the accelerated one is allowed to
   overshoot slightly but must still descend over the run, and running the
   default schedule out to 32 iterations must land well below its first step.

   Also here: the consistent-spectrogram fixed point (starting from the true
   phase of a signal returns that signal's synthesis), the padding grid — the
   iteration re-analyses what it synthesises, so unlike [Stft.invert] it reads
   [pad] — shape stability across alignments, the one geometry that leaves the
   iteration no length to run at, and the argument rejections. Nothing in the
   module draws a random number, so every case is a plain equality on rerun. *)

open Windtrap
open Soundml

let lcg_signal n =
  let state = ref 20250803 in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let config ?(alignment = `Centered) ?(pad = `Reflect) () =
  Stft.Config.create ~fft_size:512 ~hop:128 ~alignment ~pad ()

let signals =
  let half = lcg_signal 1024 in
  [ ("lcg", Nx.create Nx.float64 [|2048|] (lcg_signal 2048))
  ; ( "impulse-train"
    , Nx.create Nx.float64 [|2048|]
        (Array.init 2048 (fun i -> if i mod 97 = 0 then 1. else 0.)) )
  ; ( "half-silent"
    , Nx.create Nx.float64 [|2048|]
        (Array.init 2048 (fun i -> if i < 1024 then half.(i) else 0.)) ) ]

let magnitudes c x = Stft.power_spectrum ~power:1. c x

let frobenius t = Float.sqrt (Nx.item [] (Nx.sum (Nx.square t)))

(* [convergence c s y] is SC(y): the returned signal is re-analysed on the same
   geometry, which its length guarantees produces the frame count of [s]. *)
let convergence c s y = frobenius (Nx.sub (magnitudes c y) s) /. frobenius s

let iterations = List.init 10 (fun i -> i + 1) @ [32]

(* {2 Descent} *)

let descent_case (name, x) momentum =
  test (Printf.sprintf "%s momentum %g" name momentum) (fun () ->
      let c = config () in
      let s = magnitudes c x in
      let trajectory =
        List.map
          (fun n_iter ->
            (n_iter, convergence c s (Stft.griffin_lim ~n_iter ~momentum c s)) )
          iterations
      in
      (* the classic iteration is a descent method; the accelerated one trades
         monotonicity for speed, so it is only held to a small overshoot *)
      let slack = if Float.equal momentum 0. then 1. else 1.01 in
      let rec check = function
        | (k0, sc0) :: ((k1, sc1) :: _ as rest) ->
            if sc1 > (sc0 *. slack) +. 1e-12 then
              failf "%s: SC rose from %.6g at %d iterations to %.6g at %d" name
                sc0 k0 sc1 k1 ;
            check rest
        | _ ->
            ()
      in
      check trajectory ;
      let sc_1 = List.assoc 1 trajectory and sc_32 = List.assoc 32 trajectory in
      is_true
        ~msg:
          (Printf.sprintf "%s: 32 iterations descend below the first (%.6g)"
             name sc_1 )
        (sc_32 < sc_1) ;
      is_true
        ~msg:(Printf.sprintf "%s: SC after 32 iterations is %.6g" name sc_32)
        (sc_32 <= 0.25) )

let descent_tests =
  List.concat_map
    (fun signal -> List.map (descent_case signal) [0.; 0.5; 0.99])
    signals

(* {2 The consistent fixed point} *)

let fixed_point_tests =
  [ test "the true phase of a signal is a fixed point" (fun () ->
        let c = config () in
        List.iter
          (fun (name, x) ->
            let z = Stft.transform Nx.complex128 c x in
            let s = Nx.magnitude Nx.float64 z in
            let init = `Phase (Nx.angle Nx.float64 z) in
            let expected = Nx.to_array (Stft.invert Nx.float64 c z) in
            List.iter
              (fun (n_iter, momentum) ->
                Tutils.check_close ~rtol:0. ~atol:1e-12
                  ~msg:
                    (Printf.sprintf "%s/n_iter=%d/momentum=%g" name n_iter
                       momentum )
                  ~expected
                  (Stft.griffin_lim ~n_iter ~momentum ~init c s) )
              [(1, 0.); (8, 0.99)] )
          signals )
  ; test "the loop is shape stable across alignments" (fun () ->
        List.iter
          (fun (aname, alignment) ->
            let c = config ~alignment () in
            let _, x = List.hd signals in
            let s = magnitudes c x in
            let y = Stft.griffin_lim ~n_iter:3 c s in
            equal
              ~msg:(Printf.sprintf "%s frames" aname)
              int (Nx.dim (-1) s)
              (Stft.frames c ~n:(Nx.dim (-1) y)) ;
            let cut = Stft.griffin_lim ~n_iter:3 ~length:1500 c s in
            equal
              ~msg:(Printf.sprintf "%s length" aname)
              (array int) [|1500|] (Nx.shape cut) )
          [("centered", `Centered); ("left", `Left); ("right", `Right)] )
  ; test "float32 magnitudes stay float32" (fun () ->
        let c = config () in
        let x = Nx.create Nx.float32 [|2048|] (lcg_signal 2048) in
        let s = Stft.power_spectrum ~power:1. c x in
        let y = Stft.griffin_lim ~n_iter:2 c s in
        equal ~msg:"dtype preserved" bool true
          (match Nx.dtype y with Nx.Float32 -> true | _ -> false) ) ]

(* {2 The padding mode is part of the geometry} *)

(* Each iteration re-analyses the signal it has just synthesised, and that
   analysis pads: the loop reads [pad] where [Stft.invert] does not. [`Left]
   extends nothing, so under it the three modes are the same computation and
   agree bit for bit. [`Centered] and [`Right] do extend, the extension feeds
   the next phase estimate, and the reconstructions separate — by a fair
   fraction of the signal's own range, far above anything rounding explains.
   Each mode is still a Griffin-Lim iteration on its own geometry, and the
   descent gate above holds for all of them. *)

let pads = [("reflect", `Reflect); ("constant", `Constant 0.); ("edge", `Edge)]

(* the reconstructions are O(1) signals, and the measured separation is above
   1.6; a tenth is a wide margin either way *)
let separated = 0.1

let max_gap a b =
  let a = Nx.to_array a and b = Nx.to_array b in
  Array.fold_left Float.max 0.
    (Array.mapi (fun i v -> Float.abs (v -. b.(i))) a)

let padding_tests =
  let name, x = List.hd signals in
  let reconstruct ~alignment pad =
    let c = config ~alignment ~pad () in
    Stft.griffin_lim ~n_iter:4 c (magnitudes c x)
  in
  [ test "`Left` extends nothing to disagree about" (fun () ->
        let reference = reconstruct ~alignment:`Left `Reflect in
        List.iter
          (fun (pname, pad) ->
            Tutils.check_close ~rtol:0. ~atol:0.
              ~msg:(Printf.sprintf "%s/%s" name pname)
              ~expected:(Nx.to_array reference)
              (reconstruct ~alignment:`Left pad) )
          pads )
  ; test "the extension reaches the reconstruction" (fun () ->
        List.iter
          (fun (aname, alignment) ->
            let reference = reconstruct ~alignment `Reflect in
            List.iter
              (fun (pname, pad) ->
                let gap = max_gap reference (reconstruct ~alignment pad) in
                is_true
                  ~msg:
                    (Printf.sprintf "%s: reflect and %s differ by %.6g" aname
                       pname gap )
                  (gap > separated) )
              [("constant", `Constant 0.); ("edge", `Edge)] )
          [("centered", `Centered); ("right", `Right)] )
  ; test "every mode converges on its own geometry" (fun () ->
        List.iter
          (fun (aname, alignment) ->
            List.iter
              (fun (pname, pad) ->
                let c = config ~alignment ~pad () in
                let s = magnitudes c x in
                let sc n_iter =
                  convergence c s (Stft.griffin_lim ~n_iter c s)
                in
                let sc_1 = sc 1 and sc_16 = sc 16 in
                is_true
                  ~msg:
                    (Printf.sprintf
                       "%s/%s: SC went from %.6g at 1 iteration to %.6g at 16"
                       aname pname sc_1 sc_16 )
                  (sc_16 < sc_1 && sc_16 <= 0.25) )
              pads )
          [("centered", `Centered); ("left", `Left); ("right", `Right)] ) ]

(* {2 The geometry with no natural length} *)

(* The iteration runs at the default length of [Stft.invert], and under
   [`Centered] with an even [fft_size] a single frame has none: the boundary
   extension is the whole frame, so the frame spans no signal. There is then
   nothing to synthesise and re-analyse, no iteration runs however many are
   asked for, and what comes back is [init] put through one synthesis — which at
   [`Zero_phase] is the magnitudes taken as a real spectrum. The last case shows
   the equality is not vacuous: two frames leave a signal to iterate on, and the
   iteration count changes the answer. *)

let no_length_tests =
  [ test "a single centered frame leaves nothing to iterate on" (fun () ->
        let c = config () in
        (* 100 samples is one frame on this geometry, 200 is two *)
        let one = Nx.create Nx.float64 [|100|] (lcg_signal 100) in
        let s = magnitudes c one in
        equal ~msg:"one frame" int 1 (Nx.dim (-1) s) ;
        equal ~msg:"empty by default" (array int) [|0|]
          (Nx.shape (Stft.griffin_lim c s)) ;
        let zero_phase = Nx.complex Nx.complex128 ~re:s ~im:(Nx.zeros_like s) in
        List.iter
          (fun length ->
            let once =
              Nx.to_array (Stft.invert Nx.float64 c ~length zero_phase)
            in
            List.iter
              (fun n_iter ->
                Tutils.check_close ~rtol:0. ~atol:0.
                  ~msg:
                    (Printf.sprintf "length %d after %d iterations" length
                       n_iter )
                  ~expected:once
                  (Stft.griffin_lim ~n_iter c ~length s) )
              [1; 32] )
          [1; 300; 4096] ;
        let two = Nx.create Nx.float64 [|200|] (lcg_signal 200) in
        let s = magnitudes c two in
        equal ~msg:"two frames" int 2 (Nx.dim (-1) s) ;
        let gap =
          max_gap
            (Stft.griffin_lim ~n_iter:1 c ~length:200 s)
            (Stft.griffin_lim ~n_iter:32 c ~length:200 s)
        in
        is_true
          ~msg:(Printf.sprintf "two frames iterate, and move by %.6g" gap)
          (gap > separated) ) ]

(* {2 Rejections} *)

let error_tests =
  [ test "the iteration parameters are checked" (fun () ->
        let c = config () in
        let s = Nx.zeros Nx.float64 [|257; 4|] in
        raises_invalid_arg ~msg:"no iterations"
          "griffin_lim: cannot run 0 iterations (n_iter must be at least 1)"
          (fun () -> ignore (Stft.griffin_lim ~n_iter:0 c s) ) ;
        raises_invalid_arg ~msg:"negative momentum"
          "griffin_lim: cannot use a momentum of -0.1 (momentum must be \
           non-negative)" (fun () ->
            ignore (Stft.griffin_lim ~momentum:(-0.1) c s) ) ;
        raises_invalid_arg ~msg:"mismatched initial phase"
          "griffin_lim: cannot start from a [257; 3] phase for a [257; 4] \
           spectrogram (the initial phase must have the shape of the \
           magnitudes)" (fun () ->
            ignore
              (Stft.griffin_lim
                 ~init:(`Phase (Nx.zeros Nx.float64 [|257; 3|]))
                 c s ) ) ;
        raises_invalid_arg ~msg:"wrong bin count"
          "griffin_lim: cannot invert 128 frequency bins of a 512-point \
           transform (the bin axis must hold fft_size / 2 + 1 = 257 values)"
          (fun () ->
            ignore (Stft.griffin_lim c (Nx.zeros Nx.float64 [|128; 4|])) ) ) ]

let suite =
  [ group "descent" descent_tests
  ; group "fixed-point" fixed_point_tests
  ; group "padding" padding_tests
  ; group "no-natural-length" no_length_tests
  ; group "errors" error_tests ]
