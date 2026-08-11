(* The deterministic signals the phase-vocoder suites analyse.

   [lcg] is the 31-bit stream the generator emits, reproduced here with the same
   integer arithmetic so the golden inputs are bit-identical. The three musical
   signals are built from it and from closed-form oscillators only, so they
   carry no dependence on any library beyond the arithmetic: a swept sine in
   one-pole noise, a vibrato-laden harmonic mixture, and a pulse train through
   three resonators. They stand for the material a vocoder is used on — a moving
   partial, a dense harmonic stack, and a periodic excitation with formant
   structure — where white noise, which the parity vectors use, has no partials
   at all and so nothing for phase locking to lock to. *)

let two_pi = 2. *. Float.pi

let lcg ?(seed = 20250803) n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let sample_rate = 22050

let normalise scale y =
  let peak = Array.fold_left (fun a v -> Float.max a (Float.abs v)) 0. y in
  if peak = 0. then y else Array.map (fun v -> v *. scale /. peak) y

let tensor y = Nx.create Nx.float64 [|Array.length y|] y

(* A linear chirp from 220 Hz to 2 kHz over the whole clip, plus one-pole
   lowpass noise at a fifth of its amplitude. *)
let sweep n =
  let duration = Float.of_int n /. Float.of_int sample_rate in
  let noise = lcg n in
  let filtered = Array.make n 0. in
  let previous = ref 0. in
  Array.iteri
    (fun i v ->
      previous := (0.15 *. v) +. (0.85 *. !previous) ;
      filtered.(i) <- !previous )
    noise ;
  let filtered = normalise 0.15 filtered in
  tensor
    (normalise 0.6
       (Array.init n (fun i ->
            let t = Float.of_int i /. Float.of_int sample_rate in
            0.45
            *. Float.sin
                 ( two_pi
                 *. ( (220. *. t)
                    +. ((2000. -. 220.) /. (2. *. duration) *. t *. t) ) )
            +. filtered.(i) ) ) )

(* Three notes an octave below middle C, eight harmonics each, with a 5.3 Hz
   vibrato of four tenths of a percent, over a 50 ms fade-in. *)
let harmonics n =
  let y = Array.make n 0. in
  List.iteri
    (fun k f0 ->
      let phase = ref 0. in
      for i = 0 to n - 1 do
        let t = Float.of_int i /. Float.of_int sample_rate in
        let vibrato = 1. +. (0.004 *. Float.sin (two_pi *. 5.3 *. t)) in
        phase := !phase +. (two_pi *. f0 *. vibrato /. Float.of_int sample_rate) ;
        for h = 1 to 8 do
          y.(i) <-
            y.(i)
            +. 1. /. Float.of_int h
               *. Float.sin
                    ( (Float.of_int h *. !phase)
                    +. (Float.of_int ((k * 8) + h) *. 0.7) )
        done
      done )
    [196.; 246.94; 293.66] ;
  let fade = 0.05 *. Float.of_int sample_rate in
  Array.iteri
    (fun i v ->
      let ramp = Float.min (Float.of_int i /. fade) 1. in
      y.(i) <- v *. 0.5 *. (1. -. Float.cos (Float.pi *. ramp)) )
    y ;
  tensor (normalise 0.6 y)

(* A pulse train whose fundamental wanders around 120 Hz, through three two-pole
   resonators at 700, 1220 and 2600 Hz. *)
let buzz n =
  let excitation = Array.make n 0. in
  let phase = ref 0. in
  for i = 0 to n - 1 do
    let t = Float.of_int i /. Float.of_int sample_rate in
    let f0 = 120. *. (1. +. (0.15 *. Float.sin (two_pi *. 1.7 *. t))) in
    let previous = !phase in
    phase := !phase +. (f0 /. Float.of_int sample_rate) ;
    if Float.to_int !phase > Float.to_int previous then excitation.(i) <- 1.
  done ;
  let noise = lcg ~seed:20250804 n in
  let y = Array.make n 0. in
  List.iter
    (fun (centre, bandwidth, gain) ->
      let r = Float.exp (-.Float.pi *. bandwidth /. Float.of_int sample_rate) in
      let theta = two_pi *. centre /. Float.of_int sample_rate in
      let a1 = 2. *. r *. Float.cos theta and a2 = -.(r *. r) in
      let z1 = ref 0. and z2 = ref 0. in
      for i = 0 to n - 1 do
        let v = excitation.(i) +. (a1 *. !z1) +. (a2 *. !z2) in
        z2 := !z1 ;
        z1 := v ;
        y.(i) <- y.(i) +. (gain *. (1. -. r) *. v)
      done )
    [(700., 90., 1.); (1220., 110., 0.6); (2600., 170., 0.3)] ;
  Array.iteri (fun i v -> y.(i) <- v +. (0.02 *. noise.(i))) y ;
  tensor (normalise 0.6 y)

(* A pure tone of amplitude 0.6. *)
let tone ~f0 n =
  tensor
    (Array.init n (fun i ->
         0.6
         *. Float.sin
              (two_pi *. f0 *. Float.of_int i /. Float.of_int sample_rate) ) )
