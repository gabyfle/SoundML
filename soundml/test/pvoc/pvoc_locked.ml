(* The identity phase-locked variant: its regression pins, and the properties
   that say what locking buys.

   The locked path has no reference implementation to be parity-checked against
   — librosa ships the classical algorithm only — so it is pinned two ways.
   Three golden cells come from a replica of the rule inside
   [dev/generate_vectors.py], which the vector file records as its own oracle;
   they freeze peak picking, region splitting and the write-back against any
   later drift. The properties below are what makes the rule worth having, and
   are stated as inequalities with the margins measured on this implementation.

   The metric the properties use is the consistency of the produced spectrum,

   20 log10 (‖transform (invert Y) - Y‖ / ‖Y‖)

   — how far the frames are from being the transform of an actual signal.
   Independent propagation lets the bins of one partial drift apart, the
   synthesis overlap-adds frames that no longer agree, and what the ear hears
   from that disagreement is the loss of waveform shape the classical algorithm
   is known for. The energy the synthesis retains measures the same disagreement
   from the other side: frames that cancel each other return less signal than
   they were given. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* The golden tolerance of the parity suite, for the same reason: these cells
   differ from their oracle only by the FFT implementation beneath, amplified
   through the recurrence. They clear it by two orders — the worst observed is
   below 1e-13 — and the shared number keeps the two files' pins comparable. *)
let float64_atol = 1e-11

let lcg_stream seed n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let config ~fft_size ~hop =
  Stft.Config.create ~fft_size ~hop ~pad:(`Constant 0.) ()

let golden_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let c =
        config
          ~fft_size:(Tutils.Golden.int_param case "fft_size")
          ~hop:(Tutils.Golden.int_param case "hop")
      in
      let n = Tutils.Golden.int_param case "n" in
      let rate = Tutils.Golden.float_param case "rate" in
      let x = Nx.create Nx.float64 [|n|] (lcg_stream 20250803 n) in
      Tutils.check_close ~rtol:0. ~atol:float64_atol ~shape:case.shape
        ~msg:case.name ~expected:case.values
        (Effects.time_stretch ~phase:`Locked c ~rate x) )

let golden_tests =
  List.concat_map
    (fun (file : Tutils.Golden.file) -> List.map golden_case file.cases)
    (Tutils.Golden.load_dir
       ~filter:(fun name -> name = "locked.json")
       vectors_dir )

(* {2 Determinism} *)

let determinism_tests =
  [ test "both paths are functions of their arguments alone" (fun () ->
        let x = Pvoc_signals.sweep (Pvoc_signals.sample_rate / 2) in
        let c = config ~fft_size:1024 ~hop:256 in
        List.iter
          (fun phase ->
            let a = Effects.time_stretch ~phase c ~rate:1.37 x in
            let b = Effects.time_stretch ~phase c ~rate:1.37 x in
            let n = Nx.dim 0 a in
            is_true ~msg:"two runs agree bit for bit"
              ( Nx.dim 0 b = n
              && Float.equal (Nx.item [] (Nx.max (Nx.abs (Nx.sub a b)))) 0. ) )
          [`Independent; `Locked] ) ]

(* {2 Consistency and retained energy} *)

let frobenius z =
  Float.sqrt (Nx.item [] (Nx.sum (Nx.square (Nx.magnitude Nx.float64 z))))

let leading_frames t frames =
  let rank = Nx.ndim t in
  Nx.shrink
    (Array.init rank (fun i ->
         if i = rank - 1 then (0, frames) else (0, Nx.dim i t) ) )
    t

(* [measure c ~rate ~phase x] is the consistency of the stretched spectrum in
   decibels and the energy its synthesis retains, as a fraction of the input's.
   The default synthesis length re-analyses to the frame count of [Y], so the
   two spectra line up frame for frame; the shrink guards the geometries where
   the last frame is a boundary frame. *)
let measure c ~rate ~phase x =
  let spectrum = Stft.transform Nx.complex128 c x in
  let stretched = Effects.phase_vocoder ~phase c ~rate spectrum in
  let signal = Stft.invert Nx.float64 c stretched in
  let again = Stft.transform Nx.complex128 c signal in
  let frames =
    Stdlib.min
      (Nx.dim (Nx.ndim stretched - 1) stretched)
      (Nx.dim (Nx.ndim again - 1) again)
  in
  let a = leading_frames stretched frames and b = leading_frames again frames in
  ( 20. *. Float.log10 (frobenius (Nx.sub b a) /. frobenius a)
  , Nx.item [] (Nx.mean (Nx.square signal))
    /. Nx.item [] (Nx.mean (Nx.square x)) )

let duration = 3 * Pvoc_signals.sample_rate / 2

(* Six cells of the grid the module's documentation summarises, one per signal
   kind at two geometries each, chosen where locking wins. The margin asked for
   is 2 dB; the smallest of the six measures 6.4 dB. *)
let consistency_cells =
  [ ("sweep 1024/256 r0.75", Pvoc_signals.sweep, 1024, 256, 0.75)
  ; ("sweep 1024/128 r1.5", Pvoc_signals.sweep, 1024, 128, 1.5)
  ; ("harmonics 2048/512 r0.75", Pvoc_signals.harmonics, 2048, 512, 0.75)
  ; ("harmonics 2048/256 r1.5", Pvoc_signals.harmonics, 2048, 256, 1.5)
  ; ("buzz 2048/512 r0.75", Pvoc_signals.buzz, 2048, 512, 0.75)
  ; ("buzz 1024/256 r0.75", Pvoc_signals.buzz, 1024, 256, 0.75) ]

let consistency_tests =
  List.map
    (fun (name, signal, fft_size, hop, rate) ->
      test ("locking improves consistency: " ^ name) (fun () ->
          let x = signal duration in
          let c = config ~fft_size ~hop in
          let naive, naive_energy = measure c ~rate ~phase:`Independent x in
          let locked, locked_energy = measure c ~rate ~phase:`Locked x in
          is_true
            ~msg:
              (Printf.sprintf "%s: locked %.2f dB against naive %.2f dB" name
                 locked naive )
            (locked <= naive -. 2.) ;
          is_true
            ~msg:
              (Printf.sprintf "%s: locked keeps %.3f against naive %.3f" name
                 locked_energy naive_energy )
            (locked_energy >= naive_energy) ) )
    consistency_cells

(* {2 The pure-tone gate} *)

(* [partial z ~f0] is the frequency of the strongest component of [z] in cents
   from [f0], and the energy outside its main lobe in decibels below the total.
   The lobe is the thirteen bins around the peak, which covers the main lobe of
   the Hann window this measurement applies and its first sidelobes. The peak
   position is refined by parabolic interpolation of the log magnitudes, which
   is exact for a Gaussian and accurate to a thousandth of a bin here. *)
let partial z ~f0 =
  let n = Nx.dim 0 z in
  let window =
    Nx.create Nx.float64 [|n|]
      (Array.init n (fun i ->
           0.5
           -. 0.5
              *. Float.cos
                   (2. *. Float.pi *. Float.of_int i /. Float.of_int (n - 1)) )
      )
  in
  let magnitudes =
    Nx.magnitude Nx.float64 (Nx.rfft Nx.complex128 (Nx.mul z window))
  in
  let bins = Nx.dim 0 magnitudes in
  let at i = Nx.item [i] magnitudes in
  let peak = ref 1 in
  for i = 1 to bins - 2 do
    if at i > at !peak then peak := i
  done ;
  let logarithm i = Float.log (at i +. 1e-300) in
  let left = logarithm (!peak - 1)
  and centre = logarithm !peak
  and right = logarithm (!peak + 1) in
  let offset = 0.5 *. (left -. right) /. (left -. (2. *. centre) +. right) in
  let frequency =
    (Float.of_int !peak +. offset)
    *. Float.of_int Pvoc_signals.sample_rate
    /. Float.of_int n
  in
  let total = ref 0. and lobe = ref 0. in
  for i = 0 to bins - 1 do
    let energy = at i *. at i in
    total := !total +. energy ;
    if i >= !peak - 6 && i <= !peak + 6 then lobe := !lobe +. energy
  done ;
  ( 1200. *. Float.log2 (frequency /. f0)
  , 10. *. Float.log10 (Float.max (!total -. !lobe) 1e-300 /. !total) )

let tone_rates = [0.5; 0.75; 1.25; 1.5; 2.0]

let tone_frequencies = [440.; 1000.; 987.7666]

(* The two cells where locking clears the naive path by more than 15 dB: an
   inharmonically placed tone at the two rates that stress the naive path most.
   They measure 18.0 and 24.3 dB. *)
let locked_wins = [(987.7666, 0.5); (987.7666, 0.75)]

let tone_tests =
  List.concat_map
    (fun f0 ->
      List.map
        (fun rate ->
          test
            (Printf.sprintf
               "a pure tone at %g Hz stretched by %g stays that tone" f0 rate )
            (fun () ->
              let x = Pvoc_signals.tone ~f0 (3 * Pvoc_signals.sample_rate) in
              let c = config ~fft_size:2048 ~hop:512 in
              let evaluate phase =
                let y = Effects.time_stretch ~phase c ~rate x in
                let n = Nx.dim 0 y in
                partial (Nx.shrink [|(4096, n - 4096)|] y) ~f0
              in
              let naive_cents, naive_floor = evaluate `Independent in
              let locked_cents, locked_floor = evaluate `Locked in
              List.iter
                (fun (tag, cents) ->
                  is_true
                    ~msg:
                      (Printf.sprintf "%s path holds the frequency: %+.4f cents"
                         tag cents )
                    (Float.abs cents <= 0.05) )
                [("independent", naive_cents); ("locked", locked_cents)] ;
              is_true
                ~msg:
                  (Printf.sprintf
                     "locked leaves no more out of the lobe: %.2f against %.2f \
                      dB"
                     locked_floor naive_floor )
                (locked_floor <= naive_floor +. 1e-6) ;
              if List.mem (f0, rate) locked_wins then
                is_true
                  ~msg:
                    (Printf.sprintf
                       "locked clears the naive path: %.2f against %.2f dB"
                       locked_floor naive_floor )
                  (locked_floor <= naive_floor -. 15.) ) )
        tone_rates )
    tone_frequencies

let suite =
  [ group "golden-locked" golden_tests
  ; group "locked-properties"
      (determinism_tests @ consistency_tests @ tone_tests) ]
