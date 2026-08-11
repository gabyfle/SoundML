(* Contracts of the harmonic/percussive separation faces: parameter validation
   (every precondition raises its documented message), the algebra of the masks
   (partition at the default margin, exclusion beyond it, the [0, 1] range, the
   0/1 values and disjointness of the hard mask), the degenerate kernel, the
   Fitzgerald orientation test — a horizontal ridge is harmonic, a vertical one
   percussive — the median kernels against a naive reference over random line
   lengths and kernel sizes including even kernels and kernels beyond twice the
   line, and the agreement of the complex face with the signal-domain face. *)

open Windtrap
open Soundml

(* {1 Fixtures} *)

let seed = 20260813

let lcg n state0 =
  let state = ref state0 in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      Float.of_int !state /. Float.of_int (1 lsl 30) )

(* A spectrogram with sustained partials, transients and a noise floor. *)
let mixture ~bins ~frames =
  let noise = lcg (bins * frames) seed in
  Nx.create Nx.float64 [|bins; frames|]
    (Array.init (bins * frames) (fun i ->
         let b = i / frames and t = i mod frames in
         let v = noise.(i) in
         let v = v +. if b mod 5 = 2 then 4. else 0. in
         v +. if t mod 6 = 1 then 3. else 0. ) )

let to_array t = Nx.to_array (Nx.cast Nx.float64 t)

let max_abs_diff a b =
  let worst = ref 0. in
  Array.iteri (fun i v -> worst := Float.max !worst (Float.abs (v -. b.(i)))) a ;
  !worst

let max_rel_diff a b =
  let worst = ref 0. in
  Array.iteri
    (fun i v ->
      let scale = Float.max (Float.abs v) (Float.abs b.(i)) in
      if scale > 0. then
        worst := Float.max !worst (Float.abs (v -. b.(i)) /. scale) )
    a ;
  !worst

(* {1 Validation} *)

let stft8 () = Stft.Config.create ~fft_size:8 ~hop:2 ()

let square () = Nx.zeros Nx.float64 [|4; 4|]

let validation_tests =
  [ test "kernel sizes must be at least 1" (fun () ->
        List.iter
          (fun (name, kernel_size) ->
            let k_h, k_p = kernel_size in
            raises_invalid_arg ~msg:name
              (Printf.sprintf
                 "hpss_of_spectrogram: cannot median-filter with a kernel of \
                  (%d, %d) (both kernel sizes must be at least 1)"
                 k_h k_p ) (fun () ->
                ignore (hpss_of_spectrogram ~kernel_size (square ())) ) )
          [("zero time", (0, 3)); ("negative frequency", (3, -1))] )
  ; test "power must be strictly positive or infinite" (fun () ->
        List.iter
          (fun (name, power) ->
            raises_invalid_arg ~msg:name
              (Printf.sprintf
                 "hpss_masks: cannot raise the mask to the power %g (power \
                  must be strictly positive, or infinite for a hard mask)"
                 power ) (fun () -> ignore (hpss_masks ~power (square ())) ) )
          [("zero", 0.); ("negative", -1.); ("nan", Float.nan)] ;
        no_raise ~msg:"infinite" (fun () ->
            hpss_masks ~power:Float.infinity (square ()) ) )
  ; test "margins must be finite and at least 1" (fun () ->
        List.iter
          (fun (name, margin) ->
            let m_h, m_p = margin in
            raises_invalid_arg ~msg:name
              (Printf.sprintf
                 "hpss_of_spectrogram: cannot bias the decision by a margin of \
                  (%g, %g) (both margins must be finite and at least 1)"
                 m_h m_p ) (fun () ->
                ignore (hpss_of_spectrogram ~margin (square ())) ) )
          [ ("below one", (0.5, 1.))
          ; ("nan", (1., Float.nan))
          ; ("infinite", (Float.infinity, 1.)) ] )
  ; test "a spectrogram carries two axes" (fun () ->
        raises_invalid_arg ~msg:"rank one"
          "hpss_of_spectrogram: cannot separate a rank-1 tensor (a spectrogram \
           carries a bin axis and a frame axis)" (fun () ->
            ignore (hpss_of_spectrogram (Nx.zeros Nx.float64 [|8|])) ) ;
        raises_invalid_arg ~msg:"rank zero audio"
          "hpss: cannot separate a rank-zero tensor (the time axis must exist)"
          (fun () -> ignore (hpss (stft8 ()) (Nx.zeros Nx.float64 [||])) ) )
  ; test "the median kernel carries float32 and float64" (fun () ->
        raises_invalid_arg ~msg:"float16"
          "hpss_of_spectrogram: cannot separate float16 spectra (the median \
           kernel carries float32 and float64)" (fun () ->
            ignore (hpss_of_spectrogram (Nx.zeros Nx.float16 [|4; 4|])) ) ;
        raises_invalid_arg ~msg:"float16 audio"
          "hpss: cannot separate float16 spectra (the median kernel carries \
           float32 and float64)" (fun () ->
            ignore (hpss (stft8 ()) (Nx.zeros Nx.float16 [|32|])) ) )
  ; test "an empty axis maps to an empty result" (fun () ->
        List.iter
          (fun shape ->
            let h, p = hpss_of_spectrogram (Nx.zeros Nx.float64 shape) in
            equal (array int) ~msg:"harmonic shape" shape (Nx.shape h) ;
            equal (array int) ~msg:"percussive shape" shape (Nx.shape p) )
          [[|6; 0|]; [|0; 6|]; [|0; 0|]] )
  ; test "shapes and dtypes are preserved" (fun () ->
        let s = mixture ~bins:12 ~frames:9 in
        let h, p = hpss_of_spectrogram s in
        equal (array int) ~msg:"harmonic shape" [|12; 9|] (Nx.shape h) ;
        equal (array int) ~msg:"percussive shape" [|12; 9|] (Nx.shape p) ;
        let h32, _ = hpss_of_spectrogram (Nx.cast Nx.float32 s) in
        is_true ~msg:"float32 in, float32 out" (Nx.dtype h32 = Nx.Float32) ;
        let batched =
          Nx.concatenate ~axis:0
            [Nx.reshape [|1; 12; 9|] s; Nx.reshape [|1; 12; 9|] s]
        in
        let hb, _ = hpss_of_spectrogram batched in
        equal (array int) ~msg:"batched shape" [|2; 12; 9|] (Nx.shape hb) ;
        equal
          (array (float 0.))
          ~msg:"leading axes map independently" (to_array h)
          (to_array
             (Nx.reshape [|12; 9|] (Nx.shrink [|(0, 1); (0, 12); (0, 9)|] hb)) ) )
  ]

(* {1 Mask algebra} *)

let mask_tests =
  [ test "finite power at the default margin partitions the spectrogram"
      (fun () ->
        let s = mixture ~bins:40 ~frames:34 in
        List.iter
          (fun power ->
            let mask_h, mask_p = hpss_masks ~power s in
            let sum = to_array (Nx.add mask_h mask_p) in
            is_true
              ~msg:(Printf.sprintf "masks sum to one at p = %g" power)
              (max_abs_diff sum (Array.make (Array.length sum) 1.) <= 1e-12) ;
            let h, p = hpss_of_spectrogram ~power s in
            is_true
              ~msg:
                (Printf.sprintf "components sum to the input at p = %g" power)
              (max_rel_diff (to_array (Nx.add h p)) (to_array s) <= 1e-12) )
          [0.5; 1.; 2.; 3.5] )
  ; test "masks lie in the unit interval and never overlap" (fun () ->
        let s = mixture ~bins:40 ~frames:34 in
        List.iter
          (fun (power, margin) ->
            let mask_h, mask_p = hpss_masks ~power ~margin s in
            List.iter
              (fun (name, m) ->
                let v = to_array m in
                is_true
                  ~msg:(Printf.sprintf "%s in [0, 1] at p = %g" name power)
                  (Array.for_all (fun x -> x >= 0. && x <= 1.) v) )
              [("harmonic", mask_h); ("percussive", mask_p)] ;
            let sum = to_array (Nx.add mask_h mask_p) in
            is_true
              ~msg:(Printf.sprintf "masks sum to at most one at p = %g" power)
              (Array.for_all (fun x -> x <= 1. +. 1e-12) sum) )
          [ (1., (1., 3.))
          ; (2., (1., 3.))
          ; (2., (2., 2.))
          ; (Float.infinity, (1., 3.)) ] )
  ; test "the hard mask is a disjoint 0/1 decision" (fun () ->
        let s = mixture ~bins:40 ~frames:34 in
        List.iter
          (fun margin ->
            let mask_h, mask_p = hpss_masks ~power:Float.infinity ~margin s in
            let h = to_array mask_h and p = to_array mask_p in
            is_true ~msg:"values are exactly 0 or 1"
              ( Array.for_all (fun x -> x = 0. || x = 1.) h
              && Array.for_all (fun x -> x = 0. || x = 1.) p ) ;
            is_true ~msg:"the two masks never select the same bin"
              (Array.for_all
                 (fun i -> h.(i) *. p.(i) = 0.)
                 (Array.init (Array.length h) Fun.id) ) )
          [(1., 1.); (1., 3.)] ;
        let h, p = hpss_of_spectrogram ~power:Float.infinity s in
        let energy t =
          Array.fold_left (fun acc v -> acc +. (v *. v)) 0. (to_array t)
        in
        is_true ~msg:"the hard split loses energy, never gains it"
          (energy h +. energy p <= energy s +. 1e-9) )
  ; test "the unit kernel splits the spectrogram evenly" (fun () ->
        let s = mixture ~bins:12 ~frames:11 in
        let h, p = hpss_of_spectrogram ~kernel_size:(1, 1) ~power:1.5 s in
        let half = Array.map (fun v -> v /. 2.) (to_array s) in
        equal (array (float 0.)) ~msg:"harmonic is half" half (to_array h) ;
        equal (array (float 0.)) ~msg:"percussive is half" half (to_array p) )
  ; test "a horizontal ridge is harmonic and a vertical one percussive"
      (fun () ->
        let bins = 24 and frames = 24 in
        let values =
          Array.init (bins * frames) (fun i ->
              let b = i / frames and t = i mod frames in
              if b = 9 then 1. else if t = 15 then 1. else 0.01 )
        in
        let s = Nx.create Nx.float64 [|bins; frames|] values in
        let mask_h, mask_p =
          hpss_masks ~kernel_size:(9, 9) ~power:Float.infinity s
        in
        let h = to_array mask_h and p = to_array mask_p in
        let at b t = (b * frames) + t in
        is_true ~msg:"the ridge lands in the harmonic mask"
          (h.(at 9 3) = 1. && p.(at 9 3) = 0.) ;
        is_true ~msg:"the column lands in the percussive mask"
          (p.(at 3 15) = 1. && h.(at 3 15) = 0.) ) ]

(* {1 The median kernels against a naive reference}

   The reference gathers the window the library documents — indices reflected
   half-sample-symmetrically with period [2n] — sorts it and takes rank [k / 2],
   with no sliding state at all. It is compared over random line lengths and
   kernel sizes, including even kernels and kernels beyond twice the line, where
   the reflection wraps repeatedly. *)

let refl i n =
  let p = 2 * n in
  let j = ((i mod p) + p) mod p in
  if j < n then j else p - 1 - j

let reference_time values ~bins ~frames ~k =
  Array.init (bins * frames) (fun idx ->
      let b = idx / frames and t = idx mod frames in
      let w =
        Array.init k (fun j ->
            values.((b * frames) + refl (t - (k / 2) + j) frames) )
      in
      Array.sort compare w ;
      w.(k / 2) )

let reference_freq values ~bins ~frames ~k =
  Array.init (bins * frames) (fun idx ->
      let b = idx / frames and t = idx mod frames in
      let w =
        Array.init k (fun j ->
            values.((refl (b - (k / 2) + j) bins * frames) + t) )
      in
      Array.sort compare w ;
      w.(k / 2) )

(* The kernels are reached through the unit-power mask at a unit margin: there
   [mask_h = harm / (harm + perc)] and [mask_p = perc / (harm + perc)], so
   [mask_h / mask_p] recovers the ratio and, applied to a spectrogram whose
   percussive median is one, the harmonic median itself. A cleaner probe is the
   infinite power at kernels [(k, 1)] and [(1, k)]: the frequency median of a
   [(k, 1)] filter is the spectrogram itself, so the mask compares the time
   median against the input directly. The reference values below are compared
   through [hpss_masks] with one kernel degenerate, which pins each filter
   separately. *)
let kernel_tests =
  let one_shape (bins, frames) =
    List.concat_map
      (fun k ->
        let values = lcg (bins * frames) (seed + (7 * k) + bins + frames) in
        let s = Nx.create Nx.float64 [|bins; frames|] values in
        (* the time filter: kernel (k, 1) leaves the frequency median equal to
           the input, so the harmonic mask is decided by [harm > s] *)
        let time_ref = reference_time values ~bins ~frames ~k in
        let freq_ref = reference_freq values ~bins ~frames ~k in
        let decide reference other =
          Array.init (bins * frames) (fun i ->
              if reference.(i) > other.(i) then 1. else 0. )
        in
        [ test (Printf.sprintf "time median %dx%d k=%d" bins frames k) (fun () ->
              let mask_h, _ =
                hpss_masks ~kernel_size:(k, 1) ~power:Float.infinity s
              in
              equal
                (array (float 0.))
                ~msg:"time median" (decide time_ref values) (to_array mask_h) )
        ; test (Printf.sprintf "frequency median %dx%d k=%d" bins frames k)
            (fun () ->
              let _, mask_p =
                hpss_masks ~kernel_size:(1, k) ~power:Float.infinity s
              in
              equal
                (array (float 0.))
                ~msg:"frequency median" (decide freq_ref values)
                (to_array mask_p) ) ] )
      [1; 2; 3; 4; 7; 8; 15; 16; 31; 32; 33; 64; 129]
  in
  List.concat_map one_shape [(1, 1); (2, 3); (5, 4); (9, 17); (17, 9); (40, 33)]

(* {1 The complex face} *)

let complex_tests =
  [ test "the complex face agrees with the signal-domain face" (fun () ->
        let length = 2048 in
        let x =
          Nx.create Nx.float64 [|length|]
            (Array.map (fun v -> v -. 0.5) (lcg length seed))
        in
        let config =
          Stft.Config.create ~fft_size:256 ~hop:64 ~pad:(`Constant 0.) ()
        in
        List.iter
          (fun (kernel_size, power, margin) ->
            let z_h, z_p =
              hpss_of_stft ~kernel_size ~power ~margin
                (Stft.transform Nx.complex128 config x)
            in
            let h, p = hpss config ~kernel_size ~power ~margin x in
            equal
              (array (float 0.))
              ~msg:"harmonic"
              (to_array (Stft.invert Nx.float64 config ~length z_h))
              (to_array h) ;
            equal
              (array (float 0.))
              ~msg:"percussive"
              (to_array (Stft.invert Nx.float64 config ~length z_p))
              (to_array p) )
          [ ((31, 31), 2., (1., 1.))
          ; ((17, 31), Float.infinity, (1., 3.))
          ; ((32, 32), 1., (1., 1.)) ] )
  ; test "the complex components keep the phase and sum at the default margin"
      (fun () ->
        let x =
          Nx.create Nx.float64 [|1024|]
            (Array.map (fun v -> v -. 0.5) (lcg 1024 (seed + 1)))
        in
        let config = Stft.Config.create ~fft_size:128 ~hop:32 () in
        let z = Stft.transform Nx.complex128 config x in
        let z_h, z_p = hpss_of_stft z in
        let recombined = Nx.add z_h z_p in
        is_true ~msg:"the components sum to the spectrum"
          ( max_rel_diff
              (to_array (Nx.real Nx.float64 recombined))
              (to_array (Nx.real Nx.float64 z))
          <= 1e-12 ) ;
        is_true ~msg:"complex64 in, complex64 out"
          (Nx.dtype (fst (hpss_of_stft (Nx.cast Nx.complex64 z))) = Nx.Complex64) )
  ; test "harmonic and percussive are the two sides of hpss" (fun () ->
        let x =
          Nx.create Nx.float32 [|1500|]
            (Array.map (fun v -> v -. 0.5) (lcg 1500 (seed + 2)))
        in
        let config = Stft.Config.create ~fft_size:256 ~hop:64 () in
        let h, p = hpss config ~kernel_size:(9, 11) x in
        equal
          (array (float 0.))
          ~msg:"harmonic" (to_array h)
          (to_array (harmonic config ~kernel_size:(9, 11) x)) ;
        equal
          (array (float 0.))
          ~msg:"percussive" (to_array p)
          (to_array (percussive config ~kernel_size:(9, 11) x)) ;
        equal (array int) ~msg:"the length of the input" [|1500|] (Nx.shape h) )
  ]

let suite =
  [ group "validation" validation_tests
  ; group "masks" mask_tests
  ; group "kernels" kernel_tests
  ; group "complex" complex_tests ]
