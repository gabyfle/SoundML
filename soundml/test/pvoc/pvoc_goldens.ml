(* Golden parity against librosa 0.11 for the phase vocoder, replayed from the
   committed vectors under the suite's vectors/ directory. Inputs are the
   generator's 31-bit LCG stream, reproduced here with the same integer
   arithmetic, so the float64 signals are bit-identical; float32 cases quantize
   the signal exactly as the generator does and compare against a reference
   computed in float64.

   The configurations pin librosa's own defaults where they differ from this
   library's: zero padding rather than reflection, and centered frames. The
   phase mode is [`Independent], the setting parity is defined at.

   Three files:

   - stretch: {!Effects.time_stretch} across the geometry, rate, length, dtype
   and batching axes.

   - pitchstretch: the stretch stage of the pitch cases at the exact float64
   quotient their rational ratio produces. It pins that the rate the composition
   derives from the ratio is the one librosa stretched by, which matters more
   than it looks: the recurrence is chaotic in its rate, and a rate differing in
   the last place moves the result far above these tolerances.

   - pitch: the whole composition. SoundML resamples through its own polyphase
   bank rather than through soxr, so these cases carry the resampler
   substitution and are compared at the fraction of peak the suite measures for
   it, below. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* The stretch tolerance. The vocoder's accumulator is a recurrence, not a
   pointwise map: a difference in the analysis spectrum — which is all an FFT
   implementation can differ by — re-enters the phase of every later output
   frame, and the amplification from the analysis gap to the output reaches
   2.9e4x through this grid. The worst deviation actually observed against
   librosa over the committed cases is 8e-14 absolute, 4.3e-14 of the case's
   peak, on the longest expanding cell; this tolerance clears it by more than a
   hundred times. It is looser than the 1e-12 the synthesis suite uses for
   exactly that reason — that suite's map is pointwise, this one's is a
   recurrence — and the two are consistent: the shortest cells here, where the
   recurrence has no room to run, land at 1e-16. *)
let float64_atol = 1e-11

(* The pitch tolerance, as a fraction of the case's peak. The stretch stages of
   these very cases are pinned at [float64_atol] above, so this number is the
   resampler substitution and nothing else. Both resamplers are windowed-sinc
   designs at the same published specification (126 dB stop band, passband flat
   to 0.913 of the lower Nyquist), and they may differ arbitrarily in the
   transition band above that passband — which is where a full-band LCG signal
   puts a fifth of its energy. Measured on these vectors, the substitution moves
   the result by at most 2.7e-2 of peak, against the 6.5e-3 to 9.0e-3 librosa
   moves them itself by switching its own soxr from its default HQ tier to VHQ:
   the substitution is of the order of the reference's own choice of tier, three
   times it at worst. It is not spread evenly over the output — the two
   resamplers start and flush their filters differently — and the first five
   samples carry that worst figure, against 6.8e-3 between the ends of the
   signal and 7.7e-3 in the final sample. On material with a spectrum that rolls
   off — two seconds of the swept sine the locking suite analyses — the same
   three figures are 4.2e-3, 8.1e-4 and 5.1e-2, against 3.3e-4 to 6.3e-4 for
   that tier switch. *)
let pitch_fraction = 4e-2

let lcg_stream seed n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let config (case : Tutils.Golden.case) =
  Stft.Config.create
    ~hop:(Tutils.Golden.int_param case "hop")
    ~pad:(`Constant 0.)
    ~fft_size:(Tutils.Golden.int_param case "fft_size")
    ()

let shape (case : Tutils.Golden.case) =
  let n = Tutils.Golden.int_param case "n" in
  match Tutils.Golden.int_param case "channels" with
  | 1 ->
      [|n|]
  | channels ->
      [|channels; n|]

let signal dtype (case : Tutils.Golden.case) =
  let shape = shape case in
  Nx.create dtype shape (lcg_stream 20250803 (Array.fold_left ( * ) 1 shape))

(* The dtype-polymorphic map a case replays; the record makes it usable at both
   storage widths from one call site. *)
type transform = {apply: 'a. (float, 'a) Nx.t -> (float, 'a) Nx.t}

(* [replay case t] evaluates [t] at the dtype of [case] and checks it against
   the recorded values: float64 at the suite tolerance, float32 at the house
   pair over a signal quantized the way the generator quantized it. *)
let replay ?fraction (case : Tutils.Golden.case) t =
  let atol =
    match fraction with
    | None ->
        float64_atol
    | Some f ->
        f
        *. Array.fold_left (fun a v -> Float.max a (Float.abs v)) 0. case.values
  in
  match Tutils.Golden.string_param case "dtype" with
  | "float64" ->
      Tutils.check_close ~rtol:0. ~atol ~shape:case.shape ~msg:case.name
        ~expected:case.values
        (t.apply (signal Nx.float64 case))
  | "float32" ->
      let rtol, atol =
        match fraction with
        | None ->
            (Tutils.float32_rtol, Tutils.float32_atol)
        | Some _ ->
            (0., atol)
      in
      Tutils.check_close ~rtol ~atol ~shape:case.shape ~msg:case.name
        ~expected:case.values
        (t.apply (signal Nx.float32 case))
  | other ->
      failf "golden case %s: unknown dtype %s" case.name other

let stretch_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let c = config case in
      let rate = Tutils.Golden.float_param case "rate" in
      replay case {apply= (fun x -> Effects.time_stretch c ~rate x)} )

let pitch_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let c = config case in
      let ratio =
        { Pipeline.Rate.num= Tutils.Golden.int_param case "num"
        ; den= Tutils.Golden.int_param case "den" }
      in
      replay ~fraction:pitch_fraction case
        {apply= (fun x -> Effects.pitch_shift c ~ratio x)} )

let suite =
  List.map
    (fun (file : Tutils.Golden.file) ->
      let stem = Filename.remove_extension (Filename.basename file.path) in
      let case = if stem = "pitch" then pitch_case else stretch_case in
      group ("golden-" ^ stem) (List.map case file.cases) )
    (Tutils.Golden.load_dir
       ~filter:(fun name ->
         List.mem name ["stretch.json"; "pitchstretch.json"; "pitch.json"] )
       vectors_dir )
