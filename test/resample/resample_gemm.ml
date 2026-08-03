(* The GEMM surface, cross-validated against the normative executor.

   [apply_gemm] promises the same geometry as [apply] — exact output length,
   group-delay compensation, zeros outside the extent — with different bits: the
   matrix product reassociates each output's dot product. The binding contract
   is a tolerance in ULP of the signal peak — one unit is [peak * 2^-52]
   (float64) or [peak * 2^-23] (float32) — measured here across a seeded corpus
   of rates, presets and dtypes; the mli documents exactly the bound this suite
   gates (16 units at the common conversions, 32 across the standard-rate matrix
   in the quality suite), so the documentation and the gate cannot drift apart.
   Geometry, by contrast, is asserted exactly: lengths, batch broadcasting, the
   identity passthrough, and the error matrix are the same statements the C path
   already passes. *)

open Windtrap
open Soundml

let seed = 0x67e4

(* {2 ULP-of-peak distance} *)

(* [ulp_scale dtype peak] is one unit in the last place of [peak] at the
   arithmetic's precision: [2^-52] (float64) or [2^-23] (float32) times the
   peak's magnitude, the natural yardstick for "same computation, different
   summation order". *)
let ulp_scale : type a. (float, a) Nx.dtype -> float -> float =
 fun dtype peak ->
  let eps =
    match dtype with
    | Nx.Float32 ->
        Float.of_int (1 lsl 23)
    | Nx.Float64 ->
        Float.ldexp 1. 52
    | _ ->
        assert false
  in
  Float.max (Float.abs peak) Float.min_float /. eps

let max_abs a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a

let ulp_distance dtype expected actual =
  let e = Nx.to_array (Nx.cast Nx.float64 expected) in
  let a = Nx.to_array (Nx.cast Nx.float64 actual) in
  if Array.length e <> Array.length a then
    failf "ulp_distance: length mismatch (%d vs %d)" (Array.length e)
      (Array.length a) ;
  let peak = max_abs e in
  let scale = ulp_scale dtype peak in
  let worst = ref 0. in
  Array.iteri
    (fun i v ->
      let d = Float.abs (v -. a.(i)) /. scale in
      if d > !worst then worst := d )
    e ;
  !worst

(* {2 The seeded corpus} *)

let signal rng dtype n =
  Nx.create dtype [|n|]
    (Array.init n (fun _ -> (2. *. Random.State.float rng 1.) -. 1.))

let batch_signal rng dtype lead n =
  let count = Array.fold_left ( * ) 1 lead * n in
  Nx.reshape (Array.append lead [|n|])
    (Nx.create dtype [|count|]
       (Array.init count (fun _ -> (2. *. Random.State.float rng 1.) -. 1.)) )

let rates =
  [ (44100, 48000)
  ; (48000, 44100)
  ; (44100, 16000)
  ; (22050, 44100)
  ; (44100, 22050)
  ; (8000, 44100) ]

let qualities = [("fast", `Fast); ("high", `High); ("best", `Best)]

(* The binding tolerance, which the mli documents verbatim. The measured worst
   case across this corpus is 9.3 ULP of peak (float32, [`Best] — the longest
   dot products diverge most under the matmul's blocking); the gate is the next
   power of two, leaving margin for backend blocking changes while still failing
   well before the divergence could matter (~ -290 dBFS at float64). *)
let tolerance_ulp = 16.

let cross_case name dtype ~ns () =
  test name (fun () ->
      let rng = Random.State.make [|seed|] in
      List.iter
        (fun (sample_rate, target) ->
          List.iter
            (fun (qname, quality) ->
              let cfg =
                Resample.Config.create ~quality ~sample_rate ~target ()
              in
              List.iter
                (fun n ->
                  let x = signal rng dtype n in
                  let expected = Resample.apply cfg x in
                  let got = Resample.apply_gemm cfg x in
                  equal
                    ~msg:
                      (Printf.sprintf "%s/%d->%d/%s/n=%d/shape" name sample_rate
                         target qname n )
                    (array int) (Nx.shape expected) (Nx.shape got) ;
                  let d = ulp_distance dtype expected got in
                  if d > tolerance_ulp then
                    failf
                      "%s: %d -> %d Hz (%s, n=%d): %.2f ULP of peak (gate %g)"
                      name sample_rate target qname n d tolerance_ulp )
                ns )
            qualities )
        rates )

(* {2 Geometry: lengths, batches, identity, errors} *)

let length_case () =
  test "output length equals output_frames on both surfaces" (fun () ->
      let rng = Random.State.make [|seed + 1|] in
      List.iter
        (fun (sample_rate, target) ->
          let cfg = Resample.Config.create ~sample_rate ~target () in
          List.iter
            (fun n ->
              let x = signal rng Nx.float64 n in
              let y = Resample.apply_gemm cfg x in
              equal
                ~msg:(Printf.sprintf "%d->%d/n=%d" sample_rate target n)
                int
                (Resample.Config.output_frames cfg ~n)
                (Nx.dim (Nx.ndim y - 1) y) )
            [1; 2; 7; 146; 147; 148; 400; 1000; 4097] )
        rates )

let batch_case () =
  test "batched channels equal standalone conversions" (fun () ->
      let rng = Random.State.make [|seed + 2|] in
      let cfg = Resample.Config.create ~sample_rate:44100 ~target:16000 () in
      let x = batch_signal rng Nx.float64 [|2; 3|] 512 in
      let y = Resample.apply_gemm cfg x in
      equal ~msg:"leading shape" (array int)
        [|2; 3; Resample.Config.output_frames cfg ~n:512|]
        (Nx.shape y) ;
      for i = 0 to 1 do
        for j = 0 to 2 do
          let one =
            Resample.apply_gemm cfg
              (Nx.shrink [|(i, i + 1); (j, j + 1); (0, 512)|] x)
          in
          let bits t = Array.map Int64.bits_of_float (Nx.to_array t) in
          let row = Nx.shrink [|(i, i + 1); (j, j + 1); (0, Nx.dim 2 y)|] y in
          if bits (Nx.cast Nx.float64 row) <> bits (Nx.cast Nx.float64 one) then
            failf "channel (%d, %d) diverges from its standalone run" i j
        done
      done )

let identity_case () =
  test "identity configuration returns the input physically" (fun () ->
      let cfg = Resample.Config.create ~sample_rate:48000 ~target:48000 () in
      let x = signal (Random.State.make [|seed + 3|]) Nx.float32 64 in
      is_true ~msg:"physical equality" (Resample.apply_gemm cfg x == x) )

let empty_case () =
  test "empty inputs produce exact-length zeros" (fun () ->
      let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
      let y = Resample.apply_gemm cfg (Nx.zeros Nx.float64 [|2; 0|]) in
      equal ~msg:"empty time axis" (array int) [|2; 0|] (Nx.shape y) ;
      let y = Resample.apply_gemm cfg (Nx.zeros Nx.float64 [|0; 100|]) in
      equal ~msg:"zero channels" (array int)
        [|0; Resample.Config.output_frames cfg ~n:100|]
        (Nx.shape y) )

let error_case () =
  test "rank-zero and non-executor dtypes are refused" (fun () ->
      let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
      raises_invalid_arg ~msg:"rank zero"
        "apply_gemm: cannot resample a rank-zero tensor (the time axis must \
         exist)" (fun () ->
          ignore (Resample.apply_gemm cfg (Nx.scalar Nx.float64 1.)) ) ;
      raises_invalid_arg ~msg:"float16"
        "apply_gemm: cannot resample float16 audio (the executor carries \
         float32 and float64)" (fun () ->
          ignore (Resample.apply_gemm cfg (Nx.zeros Nx.float16 [|8|])) ) )

(* {2 An impulse probe: the two surfaces place energy identically} *)

let impulse_case () =
  test "impulse peaks land on the same grid on both surfaces" (fun () ->
      let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
      let n = 512 in
      List.iter
        (fun j ->
          let a = Array.make n 0. in
          a.(j) <- 1. ;
          let x = Nx.create Nx.float64 [|n|] a in
          let peak t =
            let v = Nx.to_array t in
            let best = ref 0 in
            Array.iteri
              (fun i x -> if Float.abs x > Float.abs v.(!best) then best := i)
              v ;
            !best
          in
          equal
            ~msg:(Printf.sprintf "impulse at %d" j)
            int
            (peak (Resample.apply cfg x))
            (peak (Resample.apply_gemm cfg x)) )
        [0; 147; 200; 441] )

let suite =
  [ group "gemm"
      [ cross_case "cross-validation f64" Nx.float64 ~ns:[1; 96; 147; 1024] ()
      ; cross_case "cross-validation f32" Nx.float32 ~ns:[1; 96; 147; 1024] ()
      ; length_case ()
      ; batch_case ()
      ; identity_case ()
      ; empty_case ()
      ; error_case ()
      ; impulse_case () ] ]
