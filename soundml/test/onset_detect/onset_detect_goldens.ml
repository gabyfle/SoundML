(* Golden parity against librosa 1.0: librosa.util.peak_pick (all three
   methods), librosa.onset.onset_backtrack and librosa.onset.onset_detect, from
   an envelope and from audio, replayed from the committed vectors. Every
   expected value is a frame index and every comparison is exact.

   Given an envelope, SoundML runs the reference's float64 arithmetic term for
   term, so those cases hold ties and need no margin; float32 cases quantize the
   input and take the reference on its float64 image. From audio, the two
   envelopes differ by rounding, which the generator absorbs by keeping every
   picking decision and every backtracking comparison a margin away from
   flipping; the agreement those margins must cover is checked below, on the
   stored envelopes and RMS curves of the same signals. *)

open Windtrap
open Soundml

let vectors_dir = "vectors"

let seed = 20261103

(* [lcg ?offset n] is the generator's StftVectorGenerator.lcg_signal under the
   seed [seed + offset]: integer arithmetic and one exact division. *)
let lcg ?(offset = 0) n =
  let state = ref (seed + offset) in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

let frames_t = list int

let ints values = Array.to_list (Array.map Float.to_int values)

(* {1 Case parameters} *)

let has (case : Tutils.Golden.case) key = List.mem_assoc key case.params

let floats (case : Tutils.Golden.case) key =
  match Tutils.Golden.param case key with
  | `List l ->
      Array.of_list (List.map Yojson.Safe.Util.to_number l)
  | _ ->
      failf "%s: %s is not a list" case.name key

let int_array (case : Tutils.Golden.case) key =
  match Tutils.Golden.param case key with
  | `List l ->
      Array.of_list (List.map Yojson.Safe.Util.to_int l)
  | _ ->
      failf "%s: %s is not a list" case.name key

let int_opt case key =
  if has case key then Some (Tutils.Golden.int_param case key) else None

let float_opt case key =
  if has case key then Some (Tutils.Golden.float_param case key) else None

let selection (case : Tutils.Golden.case) =
  if not (has case "method") then `Greedy
  else
    match Tutils.Golden.string_param case "method" with
    | "greedy" ->
        `Greedy
    | "dp_count" ->
        `Dp_count
    | "dp_value" ->
        `Dp_value
    | other ->
        failf "%s: unknown method %s" case.name other

type run = {run: 'a. (float, 'a) Nx.dtype -> unit}

let by_dtype (case : Tutils.Golden.case) {run} =
  match Tutils.Golden.string_param case "dtype" with
  | "float64" ->
      run Nx.float64
  | "float32" ->
      run Nx.float32
  | other ->
      failf "golden case %s: unknown dtype %s" case.name other

let tensor dtype values = Nx.create dtype [|Array.length values|] values

(* {1 Signals} *)

(* [peak_form name n] is the generator's peak_forms entry [name]. *)
let peak_form name n =
  let v = lcg ~offset:1 n in
  let a = Array.map Float.abs v in
  match name with
  | "abs" ->
      a
  | "cube" ->
      Array.map (fun x -> x *. x *. x) a
  | "quarters" ->
      Array.map (fun x -> Float.floor (4. *. x) /. 4.) a
  | "signed" ->
      v
  | other ->
      failf "unknown peak form %s" other

(* [spiky n] is the generator's LCG envelope, [10 |v| ** 3]. *)
let spiky n =
  Array.map
    (fun v ->
      let a = Float.abs v in
      a *. a *. a *. 10. )
    (lcg ~offset:5 n)

(* The generator's audio recipes, operation for operation; products are
   let-bound so that no multiply-add fuses where numpy rounds twice. *)

let fade n k =
  let k = Int.min k n in
  Array.init n (fun i ->
      if i < n - k then 1.
      else
        let j = i - (n - k) + 1 in
        0.5 *. (1. +. Float.cos (Float.pi *. Float.of_int j /. Float.of_int k)) )

let clicks n sr amps ~period ~first =
  let y = Array.make n 0. in
  let srf = Float.of_int sr in
  let step = Float.to_int (period *. srf) in
  let p = ref (Float.to_int (first *. srf)) and k = ref 0 in
  while !p < n do
    y.(!p) <- y.(!p) +. amps.(!k mod Array.length amps) ;
    p := !p + step ;
    incr k
  done ;
  y

let bursts n sr kind starts ~dur =
  let y = Array.make n 0. in
  let noise = lcg ~offset:1 n in
  let srf = Float.of_int sr in
  List.iteri
    (fun k s ->
      let a = Float.to_int (s *. srf) in
      let b = Int.min n (a + Float.to_int (dur *. srf)) in
      let g = fade (b - a) (Float.to_int (0.02 *. srf)) in
      let f = 220. *. Float.pow 2. (Float.of_int (k mod 4) /. 4.) in
      let ta = Float.of_int a /. srf in
      for i = a to b - 1 do
        let tt = (Float.of_int i /. srf) -. ta in
        let env = Float.exp (-.tt /. (dur /. 3.)) *. g.(i - a) in
        let src =
          match kind with
          | `Tone ->
              Float.sin (2. *. Float.pi *. f *. tt)
          | `Noise ->
              noise.(i)
        in
        let v = 0.5 *. env *. src in
        y.(i) <- y.(i) +. v
      done )
    starts ;
  y

(* [audio kind sr] is two seconds of the generator's signal [kind] at [sr]. *)
let audio kind sr =
  let n = 2 * sr in
  let floor = Array.map (fun v -> 1e-3 *. v) (lcg n) in
  let add a b = Array.mapi (fun i x -> x +. b.(i)) a in
  match kind with
  | "clicks" ->
      add floor (clicks n sr [|0.8; 0.25|] ~period:0.25 ~first:0.1)
  | "tones" ->
      add floor (bursts n sr `Tone [0.1; 0.45; 0.8; 1.2; 1.55] ~dur:0.25)
  | "noise" ->
      add floor (bursts n sr `Noise [0.12; 0.5; 0.9; 1.33; 1.7] ~dur:0.15)
  | "mix" ->
      let tones = bursts n sr `Tone [0.15; 0.62; 1.1; 1.55] ~dur:0.35 in
      let noise =
        Array.map
          (fun v -> 0.6 *. v)
          (bursts n sr `Noise [0.4; 0.95; 1.32] ~dur:0.1)
      in
      let clicks = clicks n sr [|0.3; 0.1|] ~period:0.5 ~first:0.27 in
      add (add (add floor tones) noise) clicks
  | other ->
      failf "unknown signal %s" other

let analysis ~sample_rate ~fft_size ~hop ~n_mels =
  ( Stft.Config.create ~fft_size ~hop ~pad:(`Constant 0.) ()
  , Mel.Config.create ~n_mels ~sample_rate ~fft_size () )

(* {1 Peak picking}

   Each greedy case is also streamed through [peak_pick_stage] whole, frame by
   frame and in chunks of odd sizes: the stage against librosa directly. *)

let streamed p chunks =
  let s =
    Pipeline.Stream.prepare p
      ~source:(Pipeline.Format.audio Nx.float64 ~sample_rate:100 ~channels:1)
      ~max_chunk:(List.fold_left (fun m c -> Int.max m (Nx.numel c)) 1 chunks)
  in
  let pushed = List.filter_map (Pipeline.Stream.push s) chunks in
  Array.to_list (Array.concat (pushed @ Pipeline.Stream.flush s))

let pieces x sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        Nx.slice [Nx.R (off, off + s)] x :: go (off + s) rest
  in
  go 0 sizes

let rec odd_sizes k n =
  if n <= 0 then [] else Int.min n k :: odd_sizes (k + 2) (n - k)

let peak_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let values =
        if has case "input" then floats case "input"
        else
          peak_form
            (Tutils.Golden.string_param case "form")
            (Tutils.Golden.int_param case "length")
      in
      let int key = Tutils.Golden.int_param case key in
      let pre_max = int "pre_max"
      and post_max = int "post_max"
      and pre_avg = int "pre_avg"
      and post_avg = int "post_avg"
      and wait = int "wait"
      and delta = Tutils.Golden.float_param case "delta" in
      let selection = selection case in
      by_dtype case
        { run=
            (fun dtype ->
              let x = tensor dtype values in
              let expected = ints case.values in
              equal ~msg:"peak_pick" frames_t expected
                (Array.to_list
                   (peak_pick ~selection ~pre_max ~post_max ~pre_avg ~post_avg
                      ~delta ~wait x ) ) ;
              if selection = `Greedy then begin
                let p =
                  peak_pick_stage ~pre_max ~post_max ~pre_avg ~post_avg ~delta
                    ~wait
                in
                let n = Array.length values in
                List.iter
                  (fun (pname, sizes) ->
                    equal ~msg:pname frames_t expected
                      (streamed p (pieces x sizes)) )
                  [ ("whole", [n])
                  ; ("ones", List.init n (fun _ -> 1))
                  ; ("odd", odd_sizes 1 n) ]
              end ) } )

(* {1 Backtracking} *)

let backtrack_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let energy = floats case "energy" and events = int_array case "events" in
      by_dtype case
        { run=
            (fun dtype ->
              equal ~msg:"onset_backtrack" frames_t (ints case.values)
                (Array.to_list (onset_backtrack events (tensor dtype energy))) )
        } )

(* {1 Onset detection} *)

(* [with_options case f] applies [f] to the case's options. *)
let with_options (case : Tutils.Golden.case) f =
  f
    ?normalize:(Some (Tutils.Golden.bool_param case "normalize"))
    ?selection:(Some (selection case))
    ?pre_max:(int_opt case "pre_max") ?post_max:(int_opt case "post_max")
    ?pre_avg:(int_opt case "pre_avg") ?post_avg:(int_opt case "post_avg")
    ?delta:(float_opt case "delta") ?wait:(int_opt case "wait")

let envelope_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let values =
        if has case "envelope" then floats case "envelope"
        else spiky (Tutils.Golden.int_param case "length")
      in
      let backtrack =
        match Tutils.Golden.string_param case "backtrack" with
        | "none" ->
            None
        | "envelope" ->
            Some `Envelope
        | "energy" ->
            Some (`Energy (tensor Nx.float64 (floats case "energy")))
        | other ->
            failf "%s: unknown backtrack %s" case.name other
      in
      let sample_rate = Tutils.Golden.int_param case "sample_rate"
      and hop = Tutils.Golden.int_param case "hop" in
      by_dtype case
        { run=
            (fun dtype ->
              let env = tensor dtype values in
              equal ~msg:"onset_detect_of_envelope" frames_t (ints case.values)
                (Array.to_list
                   (with_options case
                      (onset_detect_of_envelope ~hop ?backtrack ~sample_rate)
                      env ) ) ) } )

let audio_case (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let int key = Tutils.Golden.int_param case key in
      let sample_rate = int "sample_rate"
      and fft_size = int "fft_size"
      and hop = int "hop" in
      let stft, mel =
        analysis ~sample_rate ~fft_size ~hop ~n_mels:(int "n_mels")
      in
      let y = audio (Tutils.Golden.string_param case "signal") sample_rate in
      by_dtype case
        { run=
            (fun dtype ->
              let x = tensor dtype y in
              let got =
                match Tutils.Golden.string_param case "backtrack" with
                | "none" ->
                    with_options case (onset_detect stft mel ?backtrack:None) x
                | "envelope" ->
                    with_options case
                      (onset_detect stft mel ~backtrack:`Envelope)
                      x
                | "energy" ->
                    let energy = rms ~frame_length:fft_size ~hop x in
                    with_options case
                      (onset_detect stft mel ~backtrack:(`Energy energy))
                      x
                | other ->
                    failf "%s: unknown backtrack %s" case.name other
              in
              equal ~msg:"onset_detect" frames_t (ints case.values)
                (Array.to_list got) ) } )

(* {1 The agreement the audio margins cover}

   SoundML's envelope of each rebuilt mixture, normalised, and its RMS curve
   against librosa's for the same signal, as stored for the envelope cases.
   Measured, as fractions of the curve's maximum: the envelope within 2.2e-15
   from float64 audio and 2.9e-7 from float32 audio (the float32 analysis), the
   RMS curve within 2.6e-15. The bounds keep a factor of ten over that; the
   generator's margins (1e-3 for a pick, 1e-4 and 1e-6 for an envelope and an
   RMS comparison) lie more than two orders of magnitude beyond them. *)

let agreement64 = 2e-14

let agreement32 = 3e-6

let rms_agreement = 3e-14

let normalised values =
  let low = Array.fold_left Float.min values.(0) values in
  let y = Array.map (fun a -> a -. low) values in
  let d = Array.fold_left Float.max y.(0) y +. Float.min_float in
  Array.map (fun a -> a /. d) y

(* [worst_difference ~msg ~bound expected got] checks that [got] has the length
   of [expected] and lies within [bound] of it everywhere. *)
let worst_difference ~msg ~bound expected got =
  equal ~msg:(msg ^ " frames") int (Array.length expected) (Array.length got) ;
  let worst = ref 0. in
  Array.iteri
    (fun i e -> worst := Float.max !worst (Float.abs (e -. got.(i))))
    expected ;
  if !worst > bound then failf "%s: differs by %g (bound %g)" msg !worst bound

let agreement_case (case : Tutils.Golden.case) =
  test ("agreement, " ^ case.name) (fun () ->
      let sample_rate = Tutils.Golden.int_param case "sample_rate"
      and hop = Tutils.Golden.int_param case "hop" in
      let fft_size, n_mels =
        if sample_rate = 22050 then (2048, 128) else (512, 64)
      in
      let stft, mel = analysis ~sample_rate ~fft_size ~hop ~n_mels in
      let y = audio "mix" sample_rate in
      let doubles t = Nx.to_array (Nx.cast Nx.float64 t) in
      let envelope (type a) (dtype : (float, a) Nx.dtype) =
        normalised (doubles (onset_strength stft mel (tensor dtype y)))
      in
      let expected = normalised (floats case "envelope") in
      worst_difference
        ~msg:(case.name ^ " envelope float64")
        ~bound:agreement64 expected (envelope Nx.float64) ;
      worst_difference
        ~msg:(case.name ^ " envelope float32")
        ~bound:agreement32 expected (envelope Nx.float32) ;
      if has case "energy" then begin
        let energy = floats case "energy" in
        let peak = Array.fold_left Float.max 0. energy in
        let scaled a = Array.map (fun v -> v /. peak) a in
        worst_difference ~msg:(case.name ^ " rms") ~bound:rms_agreement
          (scaled energy)
          (scaled
             (doubles (rms ~frame_length:fft_size ~hop (tensor Nx.float64 y))) )
      end )

let suite =
  let files = Tutils.Golden.load_dir vectors_dir in
  let file name =
    match
      List.find_opt
        (fun (f : Tutils.Golden.file) ->
          Filename.basename f.path = name ^ ".json" )
        files
    with
    | Some f ->
        f
    | None ->
        failf "missing golden file %s.json" name
  in
  let cases name kind = List.map kind (file name).cases in
  [ group "peak_pick.json" (cases "peak_pick" peak_case)
  ; group "onset_backtrack.json" (cases "onset_backtrack" backtrack_case)
  ; group "onset_detect_envelope.json"
      (cases "onset_detect_envelope" envelope_case)
  ; group "onset_detect.json" (cases "onset_detect" audio_case)
  ; group "agreement"
      (List.filter_map
         (fun (case : Tutils.Golden.case) ->
           if
             String.ends_with ~suffix:"_backtrack_rms_float64" case.name
             && String.starts_with ~prefix:"mix_" case.name
           then Some (agreement_case case)
           else None )
         (file "onset_detect_envelope").cases ) ]
