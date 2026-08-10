(* The synthesis laws of Soundml.Stft, checked without an oracle:

   - the round trip [invert ~length:n (transform x) = x] on the positions the
   overlap-added squared window fully covers, over the golden parameter grid
   crossed with every alignment, every padding mode and both dtypes. The
   comparison is restricted to that interior on purpose: outside it the
   least-squares estimate is built from fewer window taps than the analysis
   used, so it is not the original signal and no tolerance would make it one.

   - the boundary extension is invisible to the round trip: the three padding
   modes reconstruct the same interior, since the padded stream agrees with the
   signal there whatever the borders hold.

   - [`Right] analysis is [`Left] analysis of the left-extended signal, so its
   synthesis is the [`Left] synthesis with that extension dropped.

   - the default output length is the fixed point of the frame geometry, and
   leading axes broadcast.

   - the invertibility criterion rejects both shapes of gap, and the shape and
   length checks fire before any transform runs. *)

open Windtrap
open Soundml

(* The generator's LCG, reproduced with the same integer arithmetic. *)
let lcg_signal n =
  let state = ref 20250803 in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let left_width fft = function
  | `Centered ->
      fft / 2
  | `Left ->
      0
  | `Right ->
      fft - 1

(* float64 round trips lose only the two FFTs and the envelope division; the
   bound is absolute because the interior samples are O(1) and their relative
   error explodes at the zero crossings the LCG signal has. *)
let float64_atol = 1e-12

let cells =
  [ (16, 4, 16)
  ; (32, 7, 32)
  ; (32, 8, 20)
  ; (64, 16, 64)
  ; (64, 17, 40)
  ; (31, 5, 31) ]

let alignments = [("centered", `Centered); ("left", `Left); ("right", `Right)]

let pads = [("reflect", `Reflect); ("constant", `Constant 0.); ("edge", `Edge)]

let lengths = [96; 127; 128; 1000]

let config (fft, hop, wl) alignment pad =
  Stft.Config.create ~fft_size:fft ~hop ~win_length:wl ~alignment ~pad ()

(* [interior c ~n] is the half-open range of signal positions every window tap
   of the analysis reaches: padded position [q] is covered by the full frame
   pattern once [q >= fft_size - hop] and while [q < frames * hop]. *)
let interior (fft, hop, _) alignment ~frames ~n =
  let left = left_width fft alignment in
  (Stdlib.max 0 (fft - hop - left), Stdlib.min n ((frames * hop) - left))

let sub_array a lo hi = Array.sub a lo (hi - lo)

let round_trip_case (fft, hop, wl) (aname, alignment) (pname, pad) =
  let cell = (fft, hop, wl) in
  test (Printf.sprintf "fft%d_hop%d_win%d %s %s" fft hop wl aname pname)
    (fun () ->
      List.iter
        (fun n ->
          let c = config cell alignment pad in
          let frames = Stft.frames c ~n in
          let lo, hi = interior cell alignment ~frames ~n in
          if hi > lo then begin
            let signal = lcg_signal n in
            let x64 = Nx.create Nx.float64 [|n|] signal in
            let y64 =
              Stft.invert Nx.float64 c ~length:n
                (Stft.transform Nx.complex128 c x64)
            in
            Tutils.check_close ~rtol:0. ~atol:float64_atol
              ~msg:(Printf.sprintf "float64/n=%d" n)
              ~expected:(sub_array signal lo hi)
              (Nx.shrink [|(lo, hi)|] y64) ;
            let x32 = Nx.create Nx.float32 [|n|] signal in
            let y32 =
              Stft.invert Nx.float32 c ~length:n
                (Stft.transform Nx.complex64 c x32)
            in
            Tutils.check_close ~rtol:Tutils.float32_rtol
              ~atol:Tutils.float32_atol
              ~msg:(Printf.sprintf "float32/n=%d" n)
              ~expected:(sub_array (Nx.to_array x32) lo hi)
              (Nx.shrink [|(lo, hi)|] y32)
          end )
        lengths )

let round_trip_tests =
  List.concat_map
    (fun cell ->
      List.concat_map
        (fun alignment -> List.map (round_trip_case cell alignment) pads)
        alignments )
    cells

(* The 2048-point cell, at one length and one padding mode per alignment: the
   grid above already crosses the knobs, this pins the geometry the benches and
   the big goldens use. *)
let big_round_trip_tests =
  List.map
    (fun ((fft, hop, wl), (aname, alignment)) ->
      let cell = (fft, hop, wl) in
      test (Printf.sprintf "fft%d_hop%d_win%d %s" fft hop wl aname) (fun () ->
          let n = 8000 in
          let c = config cell alignment (`Constant 0.) in
          let frames = Stft.frames c ~n in
          let lo, hi = interior cell alignment ~frames ~n in
          let signal = lcg_signal n in
          let x = Nx.create Nx.float64 [|n|] signal in
          let y =
            Stft.invert Nx.float64 c ~length:n
              (Stft.transform Nx.complex128 c x)
          in
          Tutils.check_close ~rtol:0. ~atol:float64_atol ~msg:"float64"
            ~expected:(sub_array signal lo hi)
            (Nx.shrink [|(lo, hi)|] y) ) )
    [ ((2048, 512, 2048), ("centered", `Centered))
    ; ((2048, 512, 2048), ("left", `Left))
    ; ((2048, 500, 1200), ("centered", `Centered))
    ; ((2048, 500, 1200), ("right", `Right)) ]

(* {2 The borders are invisible to the interior} *)

let pad_independence_tests =
  List.map
    (fun ((fft, hop, wl) as cell) ->
      test (Printf.sprintf "fft%d_hop%d_win%d" fft hop wl) (fun () ->
          List.iter
            (fun (aname, alignment) ->
              let n = 300 in
              let signal = lcg_signal n in
              let x = Nx.create Nx.float64 [|n|] signal in
              let reconstruct pad =
                let c = config cell alignment pad in
                Nx.to_array
                  (Stft.invert Nx.float64 c ~length:n
                     (Stft.transform Nx.complex128 c x) )
              in
              let c = config cell alignment `Reflect in
              let frames = Stft.frames c ~n in
              let lo, hi = interior cell alignment ~frames ~n in
              let reference = reconstruct `Reflect in
              List.iter
                (fun (pname, pad) ->
                  let got = reconstruct pad in
                  Tutils.check_close ~rtol:0. ~atol:float64_atol
                    ~msg:(Printf.sprintf "%s/%s" aname pname)
                    ~expected:(sub_array reference lo hi)
                    (Nx.create Nx.float64 [|hi - lo|] (sub_array got lo hi)) )
                [("constant", `Constant 0.); ("edge", `Edge)] )
            alignments ) )
    cells

(* {2 [`Right] is [`Left] on the left-extended signal} *)

let right_shift_tests =
  List.map
    (fun ((fft, hop, wl) as cell) ->
      test (Printf.sprintf "fft%d_hop%d_win%d" fft hop wl) (fun () ->
          let n = 300 in
          let signal = lcg_signal n in
          let x = Nx.create Nx.float64 [|n|] signal in
          let extended =
            Nx.create Nx.float64
              [|n + fft - 1|]
              (Array.append (Array.make (fft - 1) 0.) signal)
          in
          let right = config cell `Right (`Constant 0.) in
          let left = config cell `Left (`Constant 0.) in
          let zr = Stft.transform Nx.complex128 right x in
          let zl = Stft.transform Nx.complex128 left extended in
          equal ~msg:"same frames" int (Nx.dim (-1) zl) (Nx.dim (-1) zr) ;
          let yr = Stft.invert Nx.float64 right zr in
          let yl = Stft.invert Nx.float64 left zl in
          let expected = Nx.to_array yl in
          Tutils.check_close ~rtol:0. ~atol:0. ~msg:"right is left shifted"
            ~expected:(sub_array expected (fft - 1) (Array.length expected))
            yr ) )
    cells

(* {2 Lengths, shapes and batching} *)

let shape_tests =
  [ test "the default length is the frame-count fixed point" (fun () ->
        List.iter
          (fun ((fft, hop, wl) as cell) ->
            List.iter
              (fun (aname, alignment) ->
                List.iter
                  (fun n ->
                    let c = config cell alignment (`Constant 0.) in
                    let frames = Stft.frames c ~n in
                    if frames > 0 then begin
                      let x = Nx.zeros Nx.float64 [|n|] in
                      let y =
                        Stft.invert Nx.float64 c
                          (Stft.transform Nx.complex128 c x)
                      in
                      let out = Nx.dim (-1) y in
                      if out > 0 then
                        equal
                          ~msg:
                            (Printf.sprintf "fft%d_hop%d_win%d %s n=%d" fft hop
                               wl aname n )
                          int frames (Stft.frames c ~n:out)
                    end )
                  lengths )
              alignments )
          cells )
  ; test "leading axes broadcast" (fun () ->
        let cell = (32, 8, 20) in
        let n = 200 in
        let c = config cell `Centered `Reflect in
        let a = lcg_signal n and b = Array.map (fun v -> -.v) (lcg_signal n) in
        let stacked = Nx.create Nx.float64 [|2; n|] (Array.append a b) in
        let batched =
          Stft.invert Nx.float64 c ~length:n
            (Stft.transform Nx.complex128 c stacked)
        in
        equal ~msg:"batch shape" (array int) [|2; n|] (Nx.shape batched) ;
        List.iteri
          (fun i signal ->
            let single =
              Stft.invert Nx.float64 c ~length:n
                (Stft.transform Nx.complex128 c
                   (Nx.create Nx.float64 [|n|] signal) )
            in
            Tutils.check_close ~rtol:0. ~atol:0.
              ~msg:(Printf.sprintf "row %d" i)
              ~expected:(Nx.to_array single)
              (Nx.shrink [|(i, i + 1); (0, n)|] batched) )
          [a; b] )
  ; test "an empty request synthesises nothing" (fun () ->
        let c = config (32, 8, 20) `Centered `Reflect in
        let z = Nx.zeros Nx.complex128 [|17; 4|] in
        equal ~msg:"length 0" (array int) [|0|]
          (Nx.shape (Stft.invert Nx.float64 c ~length:0 z)) ;
        equal ~msg:"no frames" (array int) [|0|]
          (Nx.shape
             (Stft.invert Nx.float64 c (Nx.zeros Nx.complex128 [|17; 0|])) ) ;
        equal ~msg:"no signals at all" (array int) [|0; 24|]
          (Nx.shape
             (Stft.invert Nx.float64 c ~length:24
                (Nx.zeros Nx.complex128 [|0; 17; 4|]) ) ) ) ]

(* {2 Rejections} *)

let error_tests =
  [ test "a hop the squared window cannot cover is rejected" (fun () ->
        (* the window's own zero at the frame edge: a periodic Hann advanced by
           its whole length leaves that position with no tap at all *)
        let c = Stft.Config.create ~fft_size:2048 ~hop:2048 () in
        raises_invalid_arg ~msg:"hop equal to the frame"
          "invert: cannot invert a 2048-point window advanced by 2048 samples \
           inside a 2048-point frame (the squared window must overlap-add to a \
           nonzero value at every position)" (fun () ->
            ignore
              (Stft.invert Nx.float64 c (Nx.zeros Nx.complex128 [|1025; 3|])) ) ;
        (* a gap between the window supports: the taper is shorter than the
           advance, so whole stretches of the signal are never analysed *)
        let gapped =
          Stft.Config.create ~fft_size:2048 ~win_length:1200 ~hop:1500 ()
        in
        raises_invalid_arg ~msg:"window shorter than the hop"
          "invert: cannot invert a 1200-point window advanced by 1500 samples \
           inside a 2048-point frame (the squared window must overlap-add to a \
           nonzero value at every position)" (fun () ->
            ignore
              (Stft.invert Nx.float64 gapped
                 (Nx.zeros Nx.complex128 [|1025; 3|]) ) ) ;
        raises_invalid_arg ~msg:"griffin_lim checks the same criterion"
          "griffin_lim: cannot invert a 2048-point window advanced by 2048 \
           samples inside a 2048-point frame (the squared window must \
           overlap-add to a nonzero value at every position)" (fun () ->
            ignore (Stft.griffin_lim c (Nx.zeros Nx.float64 [|1025; 3|])) ) )
  ; test "the spectral shape is checked before any transform" (fun () ->
        let c = config (32, 8, 20) `Centered `Reflect in
        raises_invalid_arg ~msg:"rank one"
          "invert: cannot invert a rank-1 tensor (the bin and frame axes must \
           exist)" (fun () ->
            ignore (Stft.invert Nx.float64 c (Nx.zeros Nx.complex128 [|17|])) ) ;
        raises_invalid_arg ~msg:"wrong bin count"
          "invert: cannot invert 16 frequency bins of a 32-point transform \
           (the bin axis must hold fft_size / 2 + 1 = 17 values)" (fun () ->
            ignore (Stft.invert Nx.float64 c (Nx.zeros Nx.complex128 [|16; 4|])) ) ;
        raises_invalid_arg ~msg:"negative length"
          "invert: cannot synthesise a signal of length -1 (length must be \
           non-negative)" (fun () ->
            ignore
              (Stft.invert Nx.float64 c ~length:(-1)
                 (Nx.zeros Nx.complex128 [|17; 4|]) ) ) ) ]

let suite =
  [ group "round-trip" round_trip_tests
  ; group "round-trip-2048" big_round_trip_tests
  ; group "pad-independence" pad_independence_tests
  ; group "right-is-shifted-left" right_shift_tests
  ; group "shape" shape_tests
  ; group "errors" error_tests ]
