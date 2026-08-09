(* Config: the design geometry the docstrings promise, the accessors, and the
   creation-time error matrix.

   The tier numbers pinned here are the documented ones: at 44.1 -> 48 kHz the
   presets land at K = 74 / 95 / 134 input samples of group delay, and the
   prototype length is exactly 2*K*L + 1 — the rounding that makes the
   linear-phase delay integral on the input grid. *)

open Windtrap
open Soundml

let rate_t = testable ~pp:Pipeline.Rate.pp ~equal:Pipeline.Rate.equal ()

let r num den = {Pipeline.Rate.num; den}

(* {2 Geometry} *)

let geometry_tests =
  [ test "the ratio reduces exactly" (fun () ->
        List.iter
          (fun (sample_rate, target, num, den) ->
            let cfg = Resample.Config.create ~sample_rate ~target () in
            equal
              ~msg:(Printf.sprintf "%d->%d" sample_rate target)
              rate_t (r num den) (Resample.Config.rate cfg) )
          [ (44100, 48000, 160, 147)
          ; (48000, 44100, 147, 160)
          ; (44100, 16000, 160, 441)
          ; (44100, 22050, 1, 2)
          ; (8000, 192000, 24, 1)
          ; (44100, 44100, 1, 1) ] )
  ; test "the tier ladder lands on the documented delays at 44.1 -> 48"
      (fun () ->
        (* every tier's plan is pinned by its composite delay; the plan shapes
           themselves are pinned by the pp test below *)
        List.iter
          (fun (quality, k) ->
            let cfg =
              Resample.Config.create ~quality ~sample_rate:44100 ~target:48000
                ()
            in
            equal
              ~msg:(Printf.sprintf "K at tier")
              int k
              (Resample.Config.latency cfg) )
          [(`Fast, 74); (`High, 105); (`Best, 145)] )
  ; test "a single-stage prototype has length 2*K*L + 1" (fun () ->
        (* 11025 -> 8000 stays a single direct stage (near-unity class, and its
           block rule sits past the emission ceiling) *)
        let cfg = Resample.Config.create ~sample_rate:11025 ~target:8000 () in
        let k = Resample.Config.latency cfg in
        let proto = Resample.Config.prototype Nx.float64 cfg in
        equal ~msg:"prototype length" int ((2 * k * 320) + 1) (Nx.dim 0 proto) )
  ; test "the prototype is linear-phase and sums to L" (fun () ->
        (* a single-stage plan (x2, FFT-executed — the executor cannot touch the
           designed filter): symmetry is exact by construction *)
        let cfg = Resample.Config.create ~sample_rate:22050 ~target:44100 () in
        let h = Nx.to_array (Resample.Config.prototype Nx.float64 cfg) in
        let n = Array.length h in
        for i = 0 to (n / 2) - 1 do
          if not (Float.equal h.(i) h.(n - 1 - i)) then
            failf "prototype asymmetric at %d" i
        done ;
        let sum = Array.fold_left ( +. ) 0. h in
        if Float.abs (sum -. 2.) > 1e-9 then
          failf "prototype sums to %.17g, expected 2" sum ;
        (* the center tap is the peak *)
        let peak = ref 0 in
        Array.iteri (fun i v -> if v > h.(!peak) then peak := i) h ;
        equal ~msg:"peak at the center" int (n / 2) !peak )
  ; test "prototype returns a fresh copy" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let p1 = Resample.Config.prototype Nx.float64 cfg in
        Nx.fill 42. p1 |> ignore ;
        let p2 = Resample.Config.prototype Nx.float64 cfg in
        if Float.equal (Nx.item [0] p2) 42. then
          fail "mutating the returned prototype leaked into the config" )
  ; test "wide-ratio plans cascade with exact composed latency" (fun () ->
        (* the planner's decompositions at `High, pinned like the tier ladder:
           the composite delay is K1 + K2*M1/L1, integral by the plan-time
           rounding of K2 onto stage 1's grid (44.1 -> 16 k rounds K2 from 199
           to 320 for exactly this), and output_latency stays the exact rational
           latency*L/M. Near-unity pairs stay single-stage — the shipped design,
           bit for bit; the plain octave keeps its single stage and its latency
           too (the FFT executor runs the same filter, so the accessors cannot
           move). *)
        List.iter
          (fun (sample_rate, target, latency) ->
            let cfg = Resample.Config.create ~sample_rate ~target () in
            equal
              ~msg:(Printf.sprintf "%d->%d latency" sample_rate target)
              int latency
              (Resample.Config.latency cfg) )
          [ (44100, 16000, 453)
          ; (16000, 44100, 105)
          ; (48000, 8000, 622)
          ; (8000, 48000, 105)
          ; (44100, 22050, 190) (* single stage: same filter, FFT-executed *)
          ; (44100, 48000, 105)
            (* near-unity: the measured decision rule takes the x2-first
               FFT-executed shape (the recorded gate in resample.ml) *)
          ; (48000, 44100, 113) ] ;
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:16000 () in
        equal ~msg:"cascade output latency, exact rational" rate_t (r 24160 147)
          (Resample.Config.output_latency cfg) )
  ; test "pp pins the plans of record" (fun () ->
        (* the executor tag and block length are observable only here, so the
           shipped plans — stage shapes, filter lengths, transform lengths and
           latencies — are pinned as exact strings. A cost-model or block-rule
           change that moves any plan fails this test and must be re-blessed
           deliberately. *)
        List.iter
          (fun (sample_rate, target, expected) ->
            let cfg = Resample.Config.create ~sample_rate ~target () in
            equal
              ~msg:(Printf.sprintf "%d->%d" sample_rate target)
              string expected
              (Format.asprintf "%a" Resample.Config.pp cfg) )
          [ ( 44100
            , 48000
            , "resample(44100 -> 48000 Hz, quality=high, L/M=160/147, \
               stages=2/1:401(ols,N=2048) >> 80/147:21, latency=105)" )
          ; ( 48000
            , 44100
            , "resample(48000 -> 44100 Hz, quality=high, L/M=147/160, \
               stages=2/1:437(ols,N=4096) >> 147/320:17, latency=113)" )
          ; ( 44100
            , 16000
            , "resample(44100 -> 16000 Hz, quality=high, L/M=160/441, \
               stages=320/441:25 >> 1/2:641(ols,N=4096), latency=453)" )
          ; ( 16000
            , 44100
            , "resample(16000 -> 44100 Hz, quality=high, L/M=441/160, \
               stages=2/1:401(ols,N=2048) >> 441/320:21, latency=105)" )
          ; ( 8000
            , 48000
            , "resample(8000 -> 48000 Hz, quality=high, L/M=6/1, \
               stages=2/1:401(ols,N=2048) >> 3/1:21, latency=105)" )
          ; ( 48000
            , 8000
            , "resample(48000 -> 8000 Hz, quality=high, L/M=1/6, stages=1/3:51 \
               >> 1/2:399(ols,N=2048), latency=622)" )
          ; ( 44100
            , 22050
            , "resample(44100 -> 22050 Hz, quality=high, L/M=1/2, \
               taps=381(ols,N=2048), latency=190)" )
          ; ( 22050
            , 44100
            , "resample(22050 -> 44100 Hz, quality=high, L/M=2/1, \
               taps=381(ols,N=2048), latency=95)" ) ] )
  ; test "identity configuration designs nothing" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:48000 ~target:48000 () in
        equal ~msg:"latency" int 0 (Resample.Config.latency cfg) ;
        equal ~msg:"rate" rate_t (r 1 1) (Resample.Config.rate cfg) ;
        equal ~msg:"output latency" rate_t (r 0 1)
          (Resample.Config.output_latency cfg) ;
        equal ~msg:"output_frames n" int 123
          (Resample.Config.output_frames cfg ~n:123) ) ]

(* {2 Accessors, equality, printing} *)

let accessor_tests =
  [ test "accessors echo the creation parameters" (fun () ->
        let cfg =
          Resample.Config.create ~quality:`Fast ~sample_rate:22050 ~target:8000
            ()
        in
        equal ~msg:"sample_rate" int 22050 (Resample.Config.sample_rate cfg) ;
        equal ~msg:"target" int 8000 (Resample.Config.target cfg) ;
        is_true ~msg:"quality"
          (match Resample.Config.quality cfg with `Fast -> true | _ -> false) )
  ; test "equal compares rates and quality" (fun () ->
        let c1 = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let c2 = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let c3 =
          Resample.Config.create ~quality:`Best ~sample_rate:44100 ~target:48000
            ()
        in
        let c4 = Resample.Config.create ~sample_rate:44100 ~target:32000 () in
        is_true ~msg:"same" (Resample.Config.equal c1 c2) ;
        is_true ~msg:"different quality" (not (Resample.Config.equal c1 c3)) ;
        is_true ~msg:"different target" (not (Resample.Config.equal c1 c4)) ;
        let s = {Resample.attenuation= 90.; passband= 0.9} in
        is_true ~msg:"custom equal"
          (Resample.Config.equal
             (Resample.Config.create ~quality:(`Custom s) ~sample_rate:2
                ~target:3 () )
             (Resample.Config.create ~quality:(`Custom s) ~sample_rate:2
                ~target:3 () ) ) )
  ; test "pp prints a compact line" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let s = Format.asprintf "%a" Resample.Config.pp cfg in
        is_true ~msg:"names both rates"
          ( String.length s > 0
          && String.length (String.concat "" (String.split_on_char '4' s))
             < String.length s ) ;
        let idc = Resample.Config.create ~sample_rate:8000 ~target:8000 () in
        let si = Format.asprintf "%a" Resample.Config.pp idc in
        is_true ~msg:"identity says so"
          ( String.length si >= 8
          &&
          let rec contains i =
            i + 8 <= String.length si
            && (String.sub si i 8 = "identity" || contains (i + 1))
          in
          contains 0 ) ) ]

(* {2 The creation error matrix} *)

let error_tests =
  [ test "rates must be positive" (fun () ->
        raises_invalid_arg ~msg:"sample_rate"
          "create: cannot resample from 0 Hz (sample_rate must be at least 1)"
          (fun () ->
            ignore (Resample.Config.create ~sample_rate:0 ~target:48000 ()) ) ;
        raises_invalid_arg ~msg:"target"
          "create: cannot resample to -1 Hz (target must be at least 1)"
          (fun () ->
            ignore (Resample.Config.create ~sample_rate:44100 ~target:(-1) ()) ) )
  ; test "custom specs are validated" (fun () ->
        let mk attenuation passband () =
          ignore
            (Resample.Config.create
               ~quality:(`Custom {Resample.attenuation; passband})
               ~sample_rate:44100 ~target:48000 () )
        in
        raises_invalid_arg ~msg:"attenuation low"
          "create: cannot design a filter with 39.5 dB of stop-band rejection \
           (attenuation must be finite, in [40, 200])"
          (mk 39.5 0.9) ;
        raises_invalid_arg ~msg:"attenuation high"
          "create: cannot design a filter with 220 dB of stop-band rejection \
           (attenuation must be finite, in [40, 200])"
          (mk 220. 0.9) ;
        raises_invalid_arg ~msg:"attenuation nan"
          "create: cannot design a filter with nan dB of stop-band rejection \
           (attenuation must be finite, in [40, 200])"
          (mk Float.nan 0.9) ;
        raises_invalid_arg ~msg:"passband low"
          "create: cannot preserve 0.4 of the band (passband must be finite, \
           in [0.5, 0.99])"
          (mk 100. 0.4) ;
        raises_invalid_arg ~msg:"passband high"
          "create: cannot preserve 0.995 of the band (passband must be finite, \
           in [0.5, 0.99])"
          (mk 100. 0.995) ;
        raises_invalid_arg ~msg:"passband inf"
          "create: cannot preserve inf of the band (passband must be finite, \
           in [0.5, 0.99])"
          (mk 100. Float.infinity) )
  ; test "near-unity ratios are rejected with the drift hint" (fun () ->
        match Resample.Config.create ~sample_rate:44100 ~target:44099 () with
        | _ ->
            fail "44100 -> 44099 was accepted"
        | exception Invalid_argument message ->
            List.iter
              (fun needle ->
                let rec contains i =
                  i + String.length needle <= String.length message
                  && ( String.sub message i (String.length needle) = needle
                     || contains (i + 1) )
                in
                if not (contains 0) then
                  failf "budget message misses %S: %s" needle message )
              [ "44100 Hz to 44099 Hz"
              ; "44099 phases"
              ; "budget is 8 MB"
              ; "clock-drift correction" ] )
  ; test "the worst standard pair fits with margin" (fun () ->
        (* 11025 -> 192000 reduces to L = 2560 and needs the largest bank of any
           standard pair — under 4 MB at `High, under the 8 MB budget even at
           `Best *)
        List.iter
          (fun quality ->
            ignore
              (Resample.Config.create ~quality ~sample_rate:11025 ~target:192000
                 () ) )
          [`High; `Best] ) ]

let suite =
  [ group "geometry" geometry_tests
  ; group "accessors" accessor_tests
  ; group "create-errors" error_tests ]
