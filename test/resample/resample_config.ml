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
        List.iter
          (fun (quality, k) ->
            let cfg =
              Resample.Config.create ~quality ~sample_rate:44100 ~target:48000
                ()
            in
            equal
              ~msg:(Printf.sprintf "K at tier")
              int k
              (Resample.Config.latency cfg) ;
            let proto = Resample.Config.prototype Nx.float64 cfg in
            equal
              ~msg:(Printf.sprintf "prototype length 2*K*L + 1")
              int
              ((2 * k * 160) + 1)
              (Nx.dim 0 proto) )
          [(`Fast, 74); (`High, 95); (`Best, 134)] )
  ; test "the prototype is linear-phase and sums to L" (fun () ->
        let cfg = Resample.Config.create ~sample_rate:44100 ~target:48000 () in
        let h = Nx.to_array (Resample.Config.prototype Nx.float64 cfg) in
        let n = Array.length h in
        for i = 0 to (n / 2) - 1 do
          if not (Float.equal h.(i) h.(n - 1 - i)) then
            failf "prototype asymmetric at %d" i
        done ;
        let sum = Array.fold_left ( +. ) 0. h in
        if Float.abs (sum -. 160.) > 1e-9 then
          failf "prototype sums to %.17g, expected 160" sum ;
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
