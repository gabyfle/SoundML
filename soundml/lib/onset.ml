(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2025                                                       *)
(*    Gabriel Santamaria                                                     *)
(*                                                                           *)
(*                                                                           *)
(*  Licensed under the Apache License, Version 2.0 (the "License");          *)
(*  you may not use this file except in compliance with the License.         *)
(*  You may obtain a copy of the License at                                  *)
(*                                                                           *)
(*    http://www.apache.org/licenses/LICENSE-2.0                             *)
(*                                                                           *)
(*  Unless required by applicable law or agreed to in writing, software      *)
(*  distributed under the License is distributed on an "AS IS" BASIS,        *)
(*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *)
(*  See the License for the specific language governing permissions and      *)
(*  limitations under the License.                                           *)
(*                                                                           *)
(*****************************************************************************)

(* Spectral-flux onset strength on the log-power mel chain: the log-power mel
   spectrogram, the lagged positive difference along the frame axis, the mean
   over the bin axis, and — for centered analysis — the [fft_size / (2 *
   hop)]-frame compensation shift. The difference itself is the Mealy kernel
   below ([lag] frames of carried state); the flat function runs it on one
   chunk.

   Onset detection on the envelope, as librosa.onset.onset_detect runs it: the
   peak picker of librosa.util.peak_pick, whose greedy rule is causal and is a
   Mealy kernel of its own (the flat greedy picker is its one-chunk instance),
   its dynamic-programming rules (a backward pass, offline), the min-max
   normalisation and the window defaults the detector derives from the frame
   rate (whole-signal, offline), and the backtracking of each onset to the
   energy minimum before it (offline). *)

let check_lag fn lag =
  if lag < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot difference frames at a lag of %d (lag must be at least 1)"
         fn lag )

(* {1 Shape helpers} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
    t

let concat_last parts = Nx.concatenate ~axis:(-1) parts

(* [collapsed_shape t frames] is the shape of the aggregated envelope: the shape
   of the spectral tensor [t] with its bin axis dropped and [frames] frames.
   Requires rank >= 2. *)
let collapsed_shape t frames =
  let nd = Nx.ndim t in
  let shape = Nx.shape t in
  Array.init (nd - 1) (fun i -> if i = nd - 2 then frames else shape.(i))

(* {1 The lagged-difference kernel}

   State of one flux stream: [carry] holds the last [min lag seen] input frames,
   kernel-owned copies — incoming chunks are borrowed. Output frame [t] is the
   mean over bins of [max 0 (s t - s (t - lag))], and [0] for the first [lag]
   frames of the stream (the lag-compensation zero prefix); every input frame
   yields exactly one output frame, so there is no tail to drain. *)

type 'a state = {lag: int; mutable carry: (float, 'a) Nx.t option}

let state_create lag = {lag; carry= None}

let state_reset s = s.carry <- None

(* [flux dtype cur prev] is the mean over bins of [max 0 (cur - prev)], in
   double and in {!Ordered}'s fixed association, so a frame's flux is the same
   bits whatever the chunk it arrives in; rounded to [dtype] once at the
   boundary. Float32 and float64 chunks take {!Loops.flux}, the same values in
   one pass. *)
let flux dtype cur prev =
  let mean =
    Ordered.mean
      (Nx.maximum_s
         (Nx.sub (Nx.cast Nx.float64 cur) (Nx.cast Nx.float64 prev))
         0. )
  in
  Nx.cast dtype (Nx.squeeze ~axes:[Nx.ndim mean - 2] mean)

(* [state_step fn s chunk] feeds [chunk] ([[...; bins; frames]], borrowed) and
   is the envelope of exactly its frames. The differences run batched over the
   chunk's whole frame axis against the carried context; the returned tensor
   aliases neither [chunk] nor kernel state. *)
let state_step fn s chunk =
  let nd = Nx.ndim chunk in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot aggregate a rank-%d tensor (the onset flux needs [...; \
          bins; frames])"
         fn nd ) ;
  let shape = Nx.shape chunk in
  if Array.exists (fun d -> d = 0) (Array.sub shape 0 (nd - 1)) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot aggregate a chunk with a zero-size axis (bins and \
          channels must be at least 1)"
         fn ) ;
  let m = last_dim chunk in
  let dtype = Nx.dtype chunk in
  if m = 0 then Nx.zeros dtype (collapsed_shape chunk 0)
  else begin
    let ext =
      match s.carry with None -> chunk | Some t -> concat_last [t; chunk]
    in
    let l = last_dim ext in
    let count = Int.min m (Int.max 0 (l - s.lag)) in
    let out =
      if count = 0 then Nx.zeros dtype (collapsed_shape chunk m)
      else
        let values =
          match Loops.flux ~lag:s.lag ~first:(l - count) ~count ext with
          | Some mean ->
              Nx.cast dtype mean
          | None ->
              flux dtype
                (shrink_last ext (l - count) l)
                (shrink_last ext (l - count - s.lag) (l - s.lag))
        in
        if count = m then values
        else
          concat_last
            [Nx.zeros dtype (collapsed_shape chunk (m - count)); values]
    in
    let keep = Int.min s.lag l in
    s.carry <- Some (Nx.copy (shrink_last ext (l - keep) l)) ;
    out
  end

(* {1 Pipeline stage} *)

let stage ?(lag = 1) () =
  check_lag "onset_strength_stage" lag ;
  (* The element dtype is only witnessed by incoming chunks; [step] records it
     so [concat []] (reached only through [run], where a step always precedes)
     can build the empty envelope. One stage value is monomorphic in its element
     type, so the shared cell is coherent across prepares. *)
  let witness = ref None in
  Pipeline.kernel ~reset:state_reset
    ~concat:(function
      | [] -> (
        match !witness with
        | Some dtype ->
            Nx.zeros dtype [|0|]
        | None ->
            invalid_arg
              "onset_strength_stage: cannot concatenate zero chunks before any \
               chunk fixed the element dtype" )
      | parts ->
          concat_last parts )
    ~prepare:(fun _format -> state_create lag)
    ~step:(fun s chunk ->
      witness := Some (Nx.dtype chunk) ;
      Some (state_step "onset_strength_stage" s chunk) )
    ()

(* {1 The flat function} *)

let check_sizes fn stft_config mel_config =
  let stft_size = Stft.Config.fft_size stft_config in
  let mel_size = Mel.Config.fft_size mel_config in
  if stft_size <> mel_size then
    invalid_arg
      (Printf.sprintf
         "%s: cannot project a %d-point STFT through a filterbank built for an \
          FFT of size %d (the two configurations must agree on fft_size)"
         fn stft_size mel_size )

let onset_strength stft_config mel_config ?(lag = 1) x =
  let fn = "onset_strength" in
  check_lag fn lag ;
  check_sizes fn stft_config mel_config ;
  let stft_size = Stft.Config.fft_size stft_config in
  if Nx.ndim x < 1 then
    invalid_arg
      (fn ^ ": cannot analyse a rank-zero tensor (the time axis must exist)") ;
  let mel =
    match Mel.power_spectrogram stft_config mel_config x with
    | Some mel ->
        mel
    | None ->
        Mel.apply mel_config (Stft.power_spectrum stft_config x)
  in
  let dtype = Nx.dtype x in
  let m = last_dim mel in
  if Nx.numel mel = 0 then
    (* No frames, or no signals at all: nothing to difference — produce the
       broadcast-consistent empty envelope directly. *)
    Nx.zeros dtype (collapsed_shape mel m)
  else begin
    (* The log-power mel: power_to_db with its defaults — reference 1, floor
       1e-10 and the 80 dB clamp under each channel's maximum — in double
       precision. *)
    let db =
      match
        Loops.decibels64
          ~scale:(10. /. Float.log 10.)
          ~offset:0. ~amin:1e-10 ~magnitude:false ~range:(Some 80.) mel
      with
      | Some db ->
          db
      | None ->
          Convert.power_to_db ~top_db:80. (Nx.cast Nx.float64 mel)
    in
    (* The flat flux is the kernel on one chunk. *)
    let z = state_step fn (state_create lag) db in
    (* Centered analysis reads [fft_size / 2] samples ahead of each grid
       position; the envelope compensates by shifting right by [fft_size / (2 *
       hop)] frames, trimmed back to the frame grid. *)
    let shift =
      match Stft.Config.alignment stft_config with
      | `Centered ->
          stft_size / (2 * Stft.Config.hop stft_config)
      | `Left | `Right ->
          0
    in
    let out64 =
      if shift = 0 then z
      else if shift >= m then Nx.zeros Nx.float64 (Nx.shape z)
      else
        concat_last
          [ Nx.zeros Nx.float64 (collapsed_shape mel shift)
          ; shrink_last z 0 (m - shift) ]
    in
    Nx.cast dtype out64
  end

(* {1 Peak picking}

   librosa.util.peak_pick as its numba kernels run it. Frame [n] is a local
   maximum when it equals numba's [np.max] of frames [[n - pre_max, n +
   post_max)], and it clears the threshold when it is at least the mean of
   frames [[n - pre_avg, n + post_avg)] plus [delta]; both windows are cut short
   at the ends of the signal, never padded, and a mean divides by the frames its
   window holds. numba's [np.max] returns the first NaN it meets, so a NaN in
   the maximum's window fails the greedy rule's equality and passes the
   dynamic-programming rule's [x < max] skip, both as the reference has them.

   Values are read as doubles — float64 as they are, float32 widened exactly —
   and every sum is zero plus its terms in ascending order: numba's float64
   [np.mean] and [np.cumsum] term for term. No multiply-add appears, so the
   decisions are the same on every platform. *)

type selection = [`Greedy | `Dp_count | `Dp_value]

type windows =
  { pre_max: int
  ; post_max: int
  ; pre_avg: int
  ; post_avg: int
  ; delta: float
  ; wait: int }

let check_windows fn w =
  if w.pre_max < 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot look back %d frames for the local maximum (pre_max must \
          be non-negative)"
         fn w.pre_max ) ;
  if w.post_max < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot end the local maximum's window %d frames after its frame \
          (post_max must be at least 1)"
         fn w.post_max ) ;
  if w.pre_avg < 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot look back %d frames for the local mean (pre_avg must be \
          non-negative)"
         fn w.pre_avg ) ;
  if w.post_avg < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot end the local mean's window %d frames after its frame \
          (post_avg must be at least 1)"
         fn w.post_avg ) ;
  if not (Float.is_finite w.delta && w.delta >= 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot threshold at %g above the local mean (delta must be \
          finite and non-negative)"
         fn w.delta ) ;
  if w.wait < 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot wait %d frames after a peak (wait must be non-negative)" fn
         w.wait )

(* [check_values fn ~verb ~axis x] refuses a tensor no function here reads: rank
   zero, or a dtype other than float32 and float64. *)
let check_values (type a) fn ~verb ~axis (x : (float, a) Nx.t) =
  if Nx.ndim x < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot %s a rank-zero tensor (the %s axis must exist)" fn verb
         axis ) ;
  match Nx.dtype x with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      invalid_arg
        (Printf.sprintf "%s: cannot %s this dtype (float32 and float64 only)" fn
           verb )

let pp_shape shape =
  String.concat "; " (Array.to_list (Array.map string_of_int shape))

(* [check_signal fn ~verb ~axis x] is {!check_values} for the functions that
   return one list of frames: [x] must also hold exactly one signal, every
   leading axis of size 1. *)
let check_signal fn ~verb ~axis x =
  check_values fn ~verb ~axis x ;
  let shape = Nx.shape x in
  let lead = Array.sub shape 0 (Array.length shape - 1) in
  if Array.exists (fun d -> d <> 1) lead then
    invalid_arg
      (Printf.sprintf
         "%s: cannot read a tensor of leading shape [%s] as one signal (every \
          leading axis must have size 1)"
         fn (pp_shape lead) )

(* [doubles x] is the elements of [x] in C order, as doubles: float32 widens
   exactly. *)
let doubles (type a) (x : (float, a) Nx.t) : float array =
  let n = Nx.numel x in
  if n = 0 then [||]
  else
    match Nx.dtype x with
    | Nx.Float32 ->
        Store.read32 x (fun a -> Array.init n (Bigarray.Array1.get a))
    | Nx.Float64 ->
        Store.read64 x (fun a -> Array.init n (Bigarray.Array1.get a))
    | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
        Nx.to_array (Nx.cast Nx.float64 x)

(* [first n k] is [max 0 (n - k)], the start of a window reaching [k] frames
   before frame [n]; [stop ~total n k] is [min total (n + k)], the end of one
   reaching [k - 1] frames after it; [free n wait] is [n + wait + 1], the first
   frame a peak at [n] leaves free, saturated at [max_int]. Each is written so
   that no window parameter, however large, overflows. *)
let first n k = if k >= n then 0 else n - k

let stop ~total n k = if k >= total - n then total else n + k

let free n wait = if wait >= max_int - n then max_int else n + wait + 1

(* {2 The greedy rule}

   The candidate test has no side effect, so the reference's scan — test a
   frame, step one frame on a failure and [wait + 1] frames on a peak — keeps
   exactly the candidates [n] with [n >= next], [next] moved to [free n wait] by
   each peak: one integer of state, which is what lets the rule stream. *)

(* [candidate w v ~base ~total n] is whether frame [n] of a signal of [total]
   frames is a local maximum clearing the threshold, [v.(i - base)] holding
   frame [i] for every frame its windows read. The mean is taken for local
   maxima only, as the reference takes it. *)
let candidate w (v : float array) ~base ~total n =
  let x = v.(n - base) in
  let hi = stop ~total n w.post_max in
  let rec top i =
    i >= hi
    ||
    let a = v.(i - base) in
    (not (Float.is_nan a || a > x)) && top (i + 1)
  in
  top (first n w.pre_max)
  &&
  let lo = first n w.pre_avg and hi = stop ~total n w.post_avg in
  let s = ref 0. in
  for i = lo to hi - 1 do
    s := !s +. v.(i - base)
  done ;
  x >= (!s /. Float.of_int (hi - lo)) +. w.delta

(* State of one picking stream: [hist] holds frames [[base, seen)] as doubles, a
   kernel-owned copy (incoming chunks are borrowed); frames below [decided] are
   decided, and [next] is the first frame a peak may take. *)
type picker =
  { mutable hist: float array
  ; mutable base: int
  ; mutable seen: int
  ; mutable decided: int
  ; mutable next: int }

let picker_create () = {hist= [||]; base= 0; seen= 0; decided= 0; next= 0}

let picker_reset s =
  s.hist <- [||] ;
  s.base <- 0 ;
  s.seen <- 0 ;
  s.decided <- 0 ;
  s.next <- 0

(* [lookahead w] is the frames a decision reads past its own frame. *)
let lookahead w = Int.max w.post_max w.post_avg - 1

(* [decide w s ~upto] decides the frames from [s.decided] up to [upto]
   (excluded), as frames of a signal of [s.seen] frames, and is the peaks among
   them, ascending. *)
let decide w s ~upto =
  let peaks = ref [] in
  for n = s.decided to upto - 1 do
    if n >= s.next && candidate w s.hist ~base:s.base ~total:s.seen n then begin
      peaks := n :: !peaks ;
      s.next <- free n w.wait
    end
  done ;
  s.decided <- Int.max s.decided upto ;
  Array.of_list (List.rev !peaks)

(* [picker_feed w s v] appends the frames [v] and decides every frame whose
   windows they complete: frame [n] once frame [n + lookahead w] has arrived,
   its windows then ending within the frames seen, where the end of the signal
   cannot cut them. No later decision reads a frame more than [max pre_max
   pre_avg] frames before the first undecided one, so older history is
   dropped. *)
let picker_feed w s v =
  let m = Array.length v in
  if m > 0 then begin
    let keep = Int.max s.base (first s.decided (Int.max w.pre_max w.pre_avg)) in
    s.hist <- Array.append (Array.sub s.hist (keep - s.base) (s.seen - keep)) v ;
    s.base <- keep ;
    s.seen <- s.seen + m
  end ;
  decide w s ~upto:(s.seen - lookahead w)

(* [picker_flush w s] decides the frames still undecided, their windows cut at
   the end of the stream as the reference cuts them at the end of the signal. *)
let picker_flush w s = decide w s ~upto:s.seen

(* [greedy w v] is the greedy rule on the whole signal [v]: the kernel's
   one-chunk instance, the frames one feed decides and then those the flush
   decides. *)
let greedy w v =
  let s = picker_create () in
  let fed = picker_feed w s v in
  Array.append fed (picker_flush w s)

(* {2 The dynamic-programming rules}

   librosa's [__peak_pick_dp]: a backward pass in which [value.(n)] is the best
   total — the number of peaks, or the sum of their values — over frames [[n,
   total)], a frame being taken only when that strictly improves on skipping it,
   and a forward trace from frame [0] along the choices. The means come from
   prefix sums, as the reference computes them, which round differently from the
   greedy rule's sums. *)

let dynamic ~count w (v : float array) =
  let total = Array.length v in
  let cum = Array.make total 0. in
  let c = ref 0. in
  for i = 0 to total - 1 do
    c := !c +. v.(i) ;
    cum.(i) <- !c
  done ;
  let value = Array.make (total + 1) 0. in
  let pointer = Array.make (total + 1) (-1) in
  let taken = Array.make (total + 1) false in
  for n = total - 1 downto 0 do
    value.(n) <- value.(n + 1) ;
    pointer.(n) <- n + 1 ;
    let x = v.(n) in
    (* [x < np.max window] skips the frame; a NaN in the window makes the
       maximum NaN and the comparison false *)
    let hi = stop ~total n w.post_max in
    let nan = ref false and above = ref false in
    let i = ref (first n w.pre_max) in
    while (not !nan) && !i < hi do
      let a = v.(!i) in
      if Float.is_nan a then nan := true else if a > x then above := true ;
      incr i
    done ;
    if !nan || not !above then begin
      let lo = first n w.pre_avg and hi = stop ~total n w.post_avg in
      let mean =
        if lo = 0 then cum.(hi - 1) /. Float.of_int hi
        else (cum.(hi - 1) -. cum.(lo - 1)) /. Float.of_int (hi - lo)
      in
      let gain = if count then 1. else x in
      let next = Int.min total (free n w.wait) in
      if x >= mean +. w.delta && value.(next) +. gain > value.(n + 1) then begin
        value.(n) <- value.(next) +. gain ;
        pointer.(n) <- next ;
        taken.(n) <- true
      end
    end
  done ;
  let peaks = ref [] and n = ref 0 in
  while pointer.(!n) >= 0 do
    if taken.(!n) then peaks := !n :: !peaks ;
    n := pointer.(!n)
  done ;
  Array.of_list (List.rev !peaks)

let pick ~selection w v =
  match selection with
  | `Greedy ->
      greedy w v
  | `Dp_count ->
      dynamic ~count:true w v
  | `Dp_value ->
      dynamic ~count:false w v

let peak_pick ?(selection = `Greedy) ~pre_max ~post_max ~pre_avg ~post_avg
    ~delta ~wait x =
  let fn = "peak_pick" in
  let w = {pre_max; post_max; pre_avg; post_avg; delta; wait} in
  check_windows fn w ;
  check_signal fn ~verb:"pick peaks in" ~axis:"frame" x ;
  pick ~selection w (doubles x)

(* {2 The peak-picking stage} *)

let peak_pick_stage ~pre_max ~post_max ~pre_avg ~post_avg ~delta ~wait =
  let fn = "peak_pick_stage" in
  let w = {pre_max; post_max; pre_avg; post_avg; delta; wait} in
  check_windows fn w ;
  let emitted = function [||] -> None | peaks -> Some peaks in
  Pipeline.kernel ~latency:(lookahead w) ~reset:picker_reset
    ~flush:(fun s -> Option.to_list (emitted (picker_flush w s)))
    ~concat:Array.concat
    ~prepare:(fun _format -> picker_create ())
    ~step:(fun s chunk ->
      check_signal fn ~verb:"pick peaks in" ~axis:"frame" chunk ;
      emitted (picker_feed w s (doubles chunk)) )
    ()

(* {1 Backtracking}

   librosa.onset.onset_backtrack: the minima are frame 0 and every interior
   frame no higher than the one before it and strictly lower than the one after
   it, and each event moves to the latest minimum at or before it — an event
   past the end of the energy to its last minimum. The comparisons are IEEE, so
   a NaN is never a minimum, and they are exact at either dtype. *)

let minima (e : float array) =
  let found = ref [0] in
  for i = 1 to Array.length e - 2 do
    if e.(i) <= e.(i - 1) && e.(i) < e.(i + 1) then found := i :: !found
  done ;
  Array.of_list (List.rev !found)

(* [backtrack_on e events] maps each of the non-negative [events] to the latest
   minimum of [e] at or before it. *)
let backtrack_on e events =
  let minima = minima e in
  (* the largest [minima.(j) <= k]; [minima.(0) = 0 <= k] *)
  let latest k =
    let lo = ref 0 and hi = ref (Array.length minima) in
    while !hi - !lo > 1 do
      let mid = (!lo + !hi) / 2 in
      if minima.(mid) <= k then lo := mid else hi := mid
    done ;
    minima.(!lo)
  in
  Array.map latest events

let check_events fn events =
  Array.iter
    (fun k ->
      if k < 0 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot backtrack an event at frame %d (events must be \
              non-negative)"
             fn k ) )
    events

let onset_backtrack events energy =
  let fn = "onset_backtrack" in
  check_events fn events ;
  check_signal fn ~verb:"backtrack along" ~axis:"frame" energy ;
  if Array.length events = 0 then [||] else backtrack_on (doubles energy) events

(* {1 Onset detection}

   librosa.onset.onset_detect over an envelope: the min-max normalisation, the
   exit on an envelope with no nonzero value or with a value that is not finite,
   the picker at windows derived from the frame rate, and the optional
   backtracking. The arithmetic is the reference's float64 arithmetic, so a
   float32 envelope is detected as librosa detects its float64 image. *)

(* [windows fn ... ~sample_rate ~hop ()] is the picking parameters, each window
   absent replaced by librosa's default: [c * sr // hop] frames for [c] seconds,
   the floor of the double [c *. sample_rate] divided by [hop] — for an integer
   rate, [3 * sample_rate / 100 / hop] frames at 30 ms and [sample_rate / 10 /
   hop] at 100 ms in integer division, the double nearest [0.03] lying too
   little below it to cross an integer. *)
let windows fn ?pre_max ?(post_max = 1) ?pre_avg ?post_avg ?(delta = 0.07) ?wait
    ~sample_rate ~hop () =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot derive the windows at %d Hz (sample_rate must be at least \
          1)"
         fn sample_rate ) ;
  if hop < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot advance frames by %d samples (hop must be at least 1)" fn
         hop ) ;
  let frames seconds =
    Float.to_int (Float.floor (seconds *. Float.of_int sample_rate)) / hop
  in
  let w =
    { pre_max= Option.value pre_max ~default:(frames 0.03)
    ; post_max
    ; pre_avg= Option.value pre_avg ~default:(frames 0.1)
    ; post_avg= Option.value post_avg ~default:(frames 0.1 + 1)
    ; delta
    ; wait= Option.value wait ~default:(frames 0.03) }
  in
  check_windows fn w ; w

let check_backtrack fn = function
  | Some (`Energy energy) ->
      check_signal fn ~verb:"backtrack along" ~axis:"frame" energy
  | Some `Envelope | None ->
      ()

(* [normalised v] is librosa's [(v - min v) / (max (v - min v) + tiny)] at
   float64: the minimum subtracted, then one division by the shifted maximum
   plus the smallest positive normal double, [util.tiny]. [Float.min] and
   [Float.max] propagate NaN, as [np.min] and [np.max] do. *)
let normalised v =
  if Array.length v = 0 then v
  else begin
    let low = Array.fold_left Float.min v.(0) v in
    let y = Array.map (fun a -> a -. low) v in
    let d = Array.fold_left Float.max y.(0) y +. Float.min_float in
    Array.map (fun a -> a /. d) y
  end

let detect ~selection ~normalize ~backtrack w env =
  let v = doubles env in
  let e = if normalize then normalised v else v in
  if
    (not (Array.exists (fun a -> a <> 0.) e))
    || Array.exists (fun a -> not (Float.is_finite a)) e
  then [||]
  else
    let onsets = pick ~selection w e in
    if Array.length onsets = 0 then onsets
    else
      match backtrack with
      | None ->
          onsets
      | Some `Envelope ->
          backtrack_on e onsets
      | Some (`Energy energy) ->
          backtrack_on (doubles energy) onsets

let onset_detect_of_envelope ?(hop = 512) ?(normalize = true) ?backtrack
    ?(selection = `Greedy) ?pre_max ?post_max ?pre_avg ?post_avg ?delta ?wait
    ~sample_rate env =
  let fn = "onset_detect_of_envelope" in
  let w =
    windows fn ?pre_max ?post_max ?pre_avg ?post_avg ?delta ?wait ~sample_rate
      ~hop ()
  in
  check_backtrack fn backtrack ;
  check_signal fn ~verb:"pick onsets in" ~axis:"frame" env ;
  detect ~selection ~normalize ~backtrack w env

let onset_detect stft_config mel_config ?(normalize = true) ?backtrack
    ?(selection = `Greedy) ?pre_max ?post_max ?pre_avg ?post_avg ?delta ?wait x
    =
  let fn = "onset_detect" in
  check_sizes fn stft_config mel_config ;
  let w =
    windows fn ?pre_max ?post_max ?pre_avg ?post_avg ?delta ?wait
      ~sample_rate:(Mel.Config.sample_rate mel_config)
      ~hop:(Stft.Config.hop stft_config)
      ()
  in
  check_backtrack fn backtrack ;
  check_signal fn ~verb:"analyse" ~axis:"time" x ;
  detect ~selection ~normalize ~backtrack w
    (onset_strength stft_config mel_config x)
