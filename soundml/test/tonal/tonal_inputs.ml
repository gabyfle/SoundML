(* The generator's inputs, reproduced: the LCG stream under the suite's seed,
   the audio recipes (chords, a scale, noise, silence) and the given chromagrams
   (templates, a scale, |LCG|, signed LCG, the identity, zeros). Keep in step
   with TonalVectorGenerator in dev/generate_vectors.py. *)

let seed = 20261113

let sample_rate = 22050

let lcg n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let midi_hz midi = 440.0 *. Float.pow 2.0 (Float.of_int (midi - 69) /. 12.0)

(* [tone y midi weight onset offset] adds the note [midi] to [y] over samples
   [onset, offset): six partials below 0.45 * sample_rate, partial [h] at
   [weight * 0.6 ** (h - 1)], summed per sample in partial order before the note
   is added, as numpy accumulates them. The products are let-bound so that no
   multiply-add fuses. *)
let tone y midi weight onset offset =
  let sr = Float.of_int sample_rate in
  let f0 = midi_hz midi in
  let v = Array.make (offset - onset) 0. in
  for h = 1 to 6 do
    let fh = f0 *. Float.of_int h in
    if fh < 0.45 *. sr then begin
      let amp = weight *. Float.pow 0.6 (Float.of_int (h - 1)) in
      let rate = 2.0 *. Float.pi *. f0 *. Float.of_int h in
      for i = onset to offset - 1 do
        let t = Float.of_int i /. sr in
        let p = amp *. Float.sin (rate *. t) in
        v.(i - onset) <- v.(i - onset) +. p
      done
    end
  done ;
  for i = onset to offset - 1 do
    y.(i) <- y.(i) +. v.(i - onset)
  done

let chords = [|[|60; 64; 67|]; [|57; 60; 64|]; [|53; 57; 60|]; [|55; 59; 62|]|]

let scale = [|60; 62; 64; 65; 67; 69; 71; 72|]

(* [floor y gain] is [gain * y + 0.002 * lcg], the recipes' noise floor. *)
let with_floor y gain =
  let base = lcg (Array.length y) in
  Array.mapi
    (fun i v ->
      let a = gain *. v and b = 0.002 *. base.(i) in
      a +. b )
    y

let signal name n =
  let y = Array.make n 0. in
  match name with
  | "chords" ->
      Array.iteri
        (fun k chord ->
          Array.iteri
            (fun j midi ->
              tone y midi
                (1.0 -. (0.2 *. Float.of_int j))
                (k * n / 4)
                ((k + 1) * n / 4) )
            chord )
        chords ;
      with_floor y 0.1
  | "scale" ->
      Array.iteri
        (fun k midi -> tone y midi 1.0 (k * n / 8) ((k + 1) * n / 8))
        scale ;
      with_floor y 0.2
  | "noise" ->
      Array.map (fun v -> 0.1 *. v) (lcg n)
  | "silence" ->
      y
  | other ->
      failwith ("unknown signal " ^ other)

(* The chromagram templates, (pitch class, weight) lists with C = 0: triads and
   a seventh chord, the diminished seventh and the augmented triad, a single
   note, the uniform frame, and ties — shares of exactly 0.4 and 0.2, 0.1 and
   0.05. Each spans [template_frames] frames, then [template_gap] silent
   ones. *)
let templates =
  [ [(0, 1.0); (4, 0.8); (7, 0.6)]
  ; [(9, 1.0); (0, 0.8); (4, 0.6)]
  ; [(5, 1.0); (9, 0.8); (0, 0.6)]
  ; [(7, 1.0); (11, 0.8); (2, 0.6); (5, 0.4)]
  ; [(0, 1.0); (3, 1.0); (6, 1.0); (9, 1.0)]
  ; [(0, 1.0); (4, 1.0); (8, 1.0)]
  ; [(2, 1.0)]
  ; List.init 12 (fun k -> (k, 1.0))
  ; [(0, 2.0); (4, 1.0); (7, 1.0); (11, 1.0)]
  ; List.init 10 (fun k -> (k, 1.0))
  ; (0, 9.0) :: List.init 11 (fun k -> (k + 1, 1.0)) ]

let template_frames = 8

let template_gap = 4

let scale_classes = [|0; 2; 4; 5; 7; 9; 11; 0|]

(* [chroma name shape] is the given chromagram [name] of [shape], C order. *)
let chroma name shape =
  let size = Array.fold_left ( * ) 1 shape in
  let nd = Array.length shape in
  match name with
  | "lcg" ->
      Array.map Float.abs (lcg size)
  | "signed" ->
      lcg size
  | "zeros" ->
      Array.make size 0.
  | "identity" ->
      let k = shape.(nd - 1) in
      Array.init size (fun i -> if i / k = i mod k then 1. else 0.)
  | "templates" ->
      let frames = shape.(1) in
      let block = template_frames + template_gap in
      let c = Array.make size 0. in
      List.iteri
        (fun i template ->
          List.iter
            (fun (k, w) ->
              for t = i * block to (i * block) + template_frames - 1 do
                c.((k * frames) + t) <- w
              done )
            template )
        templates ;
      c
  | "scale" ->
      let frames = shape.(1) in
      let c = Array.make size 0. in
      for t = 0 to frames - 1 do
        let step = t / 6 mod Array.length scale_classes in
        let k = scale_classes.(step) in
        c.((k * frames) + t) <- c.((k * frames) + t) +. 1.0 ;
        if step > 0 then begin
          let k = scale_classes.(step - 1) in
          c.((k * frames) + t) <- c.((k * frames) + t) +. 0.3
        end
      done ;
      c
  | other ->
      failwith ("unknown chromagram " ^ other)

(* [quantize v] is [v] rounded to float32 and widened back. *)
let quantize v = Int32.float_of_bits (Int32.bits_of_float v)
