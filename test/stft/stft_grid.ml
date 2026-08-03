(* The frame grid: transform_range tiling (adjacent ranges reassemble the
   transform exactly, including across the internal framing-block boundary),
   first_complete/last_complete identifying exactly the frames that touch the
   boundary extension, and strict causality of `Right alignment under causal
   padding, tested by truncation invariance. *)

open Windtrap
open Soundml

let signal n = Array.init n (fun i -> Float.sin (0.37 *. Float.of_int i))

let complex_array z = Nx.to_array (z : (Complex.t, _) Nx.t)

(* [check_exact msg expected actual] asserts elementwise bit-equality of two
   complex tensors: every value flows through the identical per-frame
   computation, so tiling and truncation reassemble transforms exactly. *)
let check_exact msg expected actual =
  equal ~msg:(msg ^ "/shape") (array int) (Nx.shape expected) (Nx.shape actual) ;
  let e = complex_array expected and a = complex_array actual in
  Array.iteri
    (fun i (x : Complex.t) ->
      let y = a.(i) in
      if not (Float.equal x.re y.re && Float.equal x.im y.im) then
        failf "%s: value %d differs: (%.17g, %.17g) vs (%.17g, %.17g)" msg i
          x.re x.im y.re y.im )
    e

let shrink_frames z p0 p1 = Nx.shrink [|(0, Nx.dim 0 z); (p0, p1)|] z

(* {2 transform_range tiling} *)

let tiling_case name config n cuts =
  test name (fun () ->
      let x = Nx.create Nx.float64 [|n|] (signal n) in
      let total = Stft.frames config ~n in
      let whole = Stft.transform Nx.complex128 config x in
      equal ~msg:(name ^ "/frames") int total (Nx.dim 1 whole) ;
      let cuts = List.sort_uniq Int.compare ((0 :: cuts) @ [total]) in
      let rec ranges = function
        | p0 :: (p1 :: _ as rest) ->
            (p0, p1) :: ranges rest
        | _ ->
            []
      in
      let parts =
        List.map
          (fun (p0, p1) -> Stft.transform_range Nx.complex128 config ~p0 ~p1 x)
          (ranges cuts)
      in
      let tiled = Nx.concatenate ~axis:(-1) parts in
      check_exact (name ^ "/tiling") whole tiled ;
      (* an empty range is a zero-frame spectrum on the same grid *)
      let empty = Stft.transform_range Nx.complex128 config ~p0:0 ~p1:0 x in
      equal ~msg:(name ^ "/empty-range") (array int)
        [|Stft.Config.bins config; 0|]
        (Nx.shape empty) )

let tiling_tests =
  [ tiling_case "centered tiling"
      (Stft.Config.create ~fft_size:16 ~hop:4 ())
      127 [1; 5; 6; 19]
  ; tiling_case "left tiling, non-divisor hop"
      (Stft.Config.create ~alignment:`Left ~fft_size:32 ~hop:7 ())
      257 [3; 11; 30]
  ; tiling_case "right tiling"
      (Stft.Config.create ~alignment:`Right ~pad:(`Constant 0.) ~fft_size:16
         ~hop:5 () )
      101 [2; 9]
  ; (* frames * fft_size exceeds the internal framing-block budget: the blocked
       path must tile exactly too *)
    tiling_case "tiling across framing blocks"
      (Stft.Config.create ~alignment:`Left ~fft_size:64 ~hop:16 ())
      20000 [400; 1100] ]

(* {2 first_complete / last_complete} *)

(* Complete frames read no boundary extension, so they cannot depend on the
   padding mode; frames outside [first_complete, last_complete) read the
   extension and (for this signal) must change when the mode changes. *)
let complete_case name ~alignment ~fft_size ~hop n =
  test name (fun () ->
      let x = Nx.create Nx.float64 [|n|] (signal n) in
      let reflect =
        Stft.Config.create ~alignment ~pad:`Reflect ~fft_size ~hop ()
      in
      let constant =
        Stft.Config.create ~alignment ~pad:(`Constant 0.5) ~fft_size ~hop ()
      in
      let total = Stft.frames reflect ~n in
      let first = Stft.first_complete reflect in
      let last = Stft.last_complete reflect ~n in
      is_true ~msg:(name ^ "/ordered")
        (0 <= first && first <= last && last <= total) ;
      let zr = Stft.transform Nx.complex128 reflect x in
      let zc = Stft.transform Nx.complex128 constant x in
      if first < last then
        check_exact
          (name ^ "/complete frames are pad-independent")
          (shrink_frames zr first last)
          (shrink_frames zc first last) ;
      let differ p =
        let a = complex_array (shrink_frames zr p (p + 1)) in
        let b = complex_array (shrink_frames zc p (p + 1)) in
        Array.exists2
          (fun (x : Complex.t) (y : Complex.t) ->
            not (Float.equal x.re y.re && Float.equal x.im y.im) )
          a b
      in
      if first > 0 then
        is_true
          ~msg:(name ^ "/frame before first_complete touches the left border")
          (differ (first - 1)) ;
      if last < total then
        is_true
          ~msg:(name ^ "/frame at last_complete touches the right border")
          (differ last) )

let complete_tests =
  [ complete_case "centered borders" ~alignment:`Centered ~fft_size:16 ~hop:4 127
  ; complete_case "centered borders, non-divisor hop" ~alignment:`Centered
      ~fft_size:16 ~hop:5 61
  ; complete_case "left has no borders" ~alignment:`Left ~fft_size:16 ~hop:4 127
  ; complete_case "right left-border only" ~alignment:`Right ~fft_size:16 ~hop:5
      101
  ; test "grid counts match transform shapes" (fun () ->
        List.iter
          (fun config ->
            List.iter
              (fun n ->
                let x = Nx.create Nx.float64 [|n|] (signal n) in
                let z = Stft.transform Nx.complex128 config x in
                equal
                  ~msg:(Printf.sprintf "frames at n=%d" n)
                  int (Stft.frames config ~n) (Nx.dim 1 z) ;
                equal
                  ~msg:(Printf.sprintf "times at n=%d" n)
                  int (Stft.frames config ~n)
                  (Nx.dim 0 (Stft.times Nx.float64 config ~sample_rate:100 ~n)) )
              [0; 1; 2; 7; 16; 17; 61] )
          [ Stft.Config.create ~fft_size:16 ~hop:4 ()
          ; Stft.Config.create ~alignment:`Left ~fft_size:16 ~hop:4 ()
          ; Stft.Config.create ~alignment:`Right ~fft_size:16 ~hop:3 ()
          ; Stft.Config.create ~alignment:`Left ~fft_size:4 ~hop:6 () ] ) ]

(* {2 `Right strict causality} *)

(* No frame of a `Right transform depends on samples after its grid position:
   truncating the signal to any prefix leaves every frame the prefix still
   produces bit-identical. Holds for the causal padding modes; `Reflect mirrors
   early samples into the left border and is exempt by documentation. *)
let causality_case name pad =
  test name (fun () ->
      let n = 61 in
      let config =
        Stft.Config.create ~alignment:`Right ~pad ~fft_size:16 ~hop:5 ()
      in
      let samples = signal n in
      let full =
        Stft.transform Nx.complex128 config (Nx.create Nx.float64 [|n|] samples)
      in
      List.iter
        (fun m ->
          let prefix =
            Stft.transform Nx.complex128 config
              (Nx.create Nx.float64 [|m|] (Array.sub samples 0 m))
          in
          let count = Stft.frames config ~n:m in
          check_exact
            (Printf.sprintf "%s/prefix %d" name m)
            (shrink_frames full 0 count)
            prefix )
        [1; 2; 16; 17; 40; 60] )

let causality_tests =
  [ causality_case "right is causal under constant padding" (`Constant 0.)
  ; causality_case "right is causal under edge padding" `Edge ]

(* {2 Leading axes broadcast} *)

let broadcast_tests =
  [ test "leading axes map to per-signal transforms" (fun () ->
        let n = 61 in
        let config = Stft.Config.create ~fft_size:16 ~hop:4 () in
        let batch =
          Nx.init Nx.float64 [|2; 3; n|] (fun index ->
              Float.sin
                ( 0.1
                *. Float.of_int
                     ((index.(0) * 1000) + (index.(1) * 100) + index.(2)) ) )
        in
        let z = Stft.transform Nx.complex128 config batch in
        equal ~msg:"batched shape" (array int)
          [|2; 3; Stft.Config.bins config; Stft.frames config ~n|]
          (Nx.shape z) ;
        for i = 0 to 1 do
          for j = 0 to 2 do
            let one =
              Stft.transform Nx.complex128 config (Nx.slice [I i; I j; A] batch)
            in
            check_exact
              (Printf.sprintf "slice %d,%d" i j)
              one
              (Nx.slice [I i; I j; A; A] z)
          done
        done )
  ; test "rank-two power spectrum preserves channels" (fun () ->
        let config = Stft.Config.create ~fft_size:16 ~hop:4 () in
        let x =
          Nx.init Nx.float32 [|2; 40|] (fun i -> Float.of_int (i.(0) + i.(1)))
        in
        equal ~msg:"shape" (array int)
          [|2; Stft.Config.bins config; Stft.frames config ~n:40|]
          (Nx.shape (Stft.power_spectrum config x)) ) ]

let suite =
  [ group "grid-tiling" tiling_tests
  ; group "grid-complete" complete_tests
  ; group "grid-causality" causality_tests
  ; group "grid-broadcast" broadcast_tests ]
