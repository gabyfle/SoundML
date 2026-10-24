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

(* Tempo and beat, as librosa 1.0 computes them from an onset envelope: the
   autocorrelation tempogram (librosa.feature.tempogram) and the Fourier one
   (fourier_tempogram), the tempo read off the first (librosa.feature.tempo),
   Ellis' dynamic-programming beat tracker (librosa.beat.beat_track) and the
   predominant local pulse of Grosche and Müller (librosa.beat.plp).

   Every value is computed in double precision from the envelope's values and
   rounded to its dtype once, so a float32 envelope gives the float64 result on
   its own values, rounded. The autocorrelation of a frame is a direct sum in
   ascending order, where librosa transforms; the two agree to a few units in
   the last place of the frame's maximum. The decisions of the tempo estimate
   and of the beat tracker are librosa's, comparison for comparison, and the
   quirks of its code are kept on purpose: the onset envelope's first value
   never enters the local score, frames per beat and half of them round half to
   even, the tightness is rounded to single precision, the trim threshold
   averages the beats' smoothed scores over two more terms than there are beats,
   and the pulse divides each frame by the modulus of its lexicographically
   greatest coefficient. The products of every multiply-add whose bits feed a
   decision are let-bound, so that no platform contracts them and the
   reference's own uncontracted arithmetic is what runs. *)

type norm = [`Inf | `P of float | `None]

type aggregate = [`Mean | `Median | `None]

type kernels = [`Reference | `Unit]

(* {1 Checks} *)

let check_rate fn ~sample_rate ~hop =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse at %d Hz (sample_rate must be at least 1)" fn
         sample_rate ) ;
  if hop < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse at a hop of %d (hop must be at least 1)" fn hop )

let check_win fn win =
  if win < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse over a window of %d frames (win_length must be at \
          least 1)"
         fn win )

let check_positive fn name v =
  if not (Float.is_finite v && v > 0.) then
    invalid_arg
      (Printf.sprintf "%s: cannot use %s = %g (%s must be finite and positive)"
         fn name v name )

let check_norm fn = function
  | `Inf | `None ->
      ()
  | `P p ->
      if not (Float.is_finite p && p > 0.) then
        invalid_arg
          (Printf.sprintf
             "%s: cannot normalise in the %g-norm (the exponent must be finite \
              and positive)"
             fn p )

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let numel shape = Array.fold_left ( * ) 1 shape

let shape_string a =
  String.concat "; " (Array.to_list (Array.map string_of_int a))

(* [values fn x] is the onset envelope [x] as double-precision values in C
   order, after the checks every face shares: a frame axis, a float32 or float64
   dtype, and finite values. *)
let values (type a) fn (x : (float, a) Nx.t) =
  if Nx.ndim x < 1 then
    invalid_arg
      (fn ^ ": cannot analyse a rank-zero tensor (the frame axis must exist)") ;
  ( match Nx.dtype x with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      invalid_arg (fn ^ ": cannot analyse this dtype (float32 and float64 only)")
  ) ;
  let v : float array = Nx.to_array (Nx.cast Nx.float64 x) in
  Array.iter
    (fun e ->
      if not (Float.is_finite e) then
        invalid_arg
          ( fn
          ^ ": cannot analyse an onset envelope holding a non-finite value \
             (every value must be finite)" ) )
    v ;
  v

(* [round_half_even x] is [x] rounded to the nearest integer with ties to even,
   the rounding of numpy's [round] and of numba's: adding and subtracting 2^52
   in round-to-nearest is exactly that rounding for every argument of smaller
   magnitude, and every argument of larger magnitude is already an integer. *)
let round_half_even =
  let magic = 0x1p52 in
  fun x ->
    if Float.abs x >= magic then x
    else if x >= 0. then x +. magic -. magic
    else x -. magic +. magic

(* {1 Tempo frequencies} *)

(* [lag_bpms ~sample_rate ~hop n] is the tempo of each of [n] lag bins: the zero
   lag is infinitely fast, lag [k] is [60 sample_rate / (hop k)] beats per
   minute, evaluated in librosa's order. *)
let lag_bpms ~sample_rate ~hop n =
  let num = 60. *. Float.of_int sample_rate in
  Array.init n (fun k ->
      if k = 0 then Float.infinity
      else num /. (Float.of_int hop *. Float.of_int k) )

(* [fourier_bpms ~sample_rate ~hop win] is the tempo of each bin of a
   [win]-frame Fourier tempogram: numpy's [rfftfreq] at a frame rate of [60
   sample_rate / hop] frames per minute, the bin step [1 / (win d)] with [d] the
   frame period, times the bin index. *)
let fourier_bpms ~sample_rate ~hop win =
  let rate = Float.of_int (sample_rate * 60) /. Float.of_int hop in
  let d = 1. /. rate in
  let step = 1. /. (Float.of_int win *. d) in
  Array.init ((win / 2) + 1) (fun k -> Float.of_int k *. step)

let tempo_frequencies dtype ?(hop = 512) ~sample_rate n =
  let fn = "tempo_frequencies" in
  check_rate fn ~sample_rate ~hop ;
  if n < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot describe a tempogram of %d lag bins (at least one is \
          needed)"
         fn n ) ;
  Nx.create dtype [|n|] (lag_bpms ~sample_rate ~hop n)

let fourier_tempo_frequencies dtype ?(hop = 512) ~sample_rate win =
  let fn = "fourier_tempo_frequencies" in
  check_rate fn ~sample_rate ~hop ;
  check_win fn win ;
  Nx.create dtype [|(win / 2) + 1|] (fourier_bpms ~sample_rate ~hop win)

(* {1 The autocorrelation tempogram}

   Frame [t] of a row autocorrelates the [win] envelope values from [t - left],
   [left] being [win / 2] for centred frames and [0] otherwise, under the
   periodic window. Centred frames read past the ends of the envelope into
   numpy's linear ramps to zero, [j (x0 / left)] at padded position [j] on the
   left and, on the right, the same ramp towards the last value, reversed; numpy
   evaluates both in the envelope's dtype and, for a batch holding a row that
   starts or ends at zero, in another association for every row, both
   differences of one rounding that the double ramps here do not follow. The
   values are read through the padding by index, never materialised. *)

type plan = {win: int; left: int; w: Float.Array.t; norm: norm}

let plan ~win ~alignment ~window ~norm =
  let w = Nx.to_array (Window.make Nx.float64 ~periodic:true window win) in
  { win
  ; left= (match alignment with `Centered -> win / 2 | `Left -> 0)
  ; w= Float.Array.init win (Array.get w)
  ; norm }

(* [normalise norm r] divides the lags [r] by their length in [norm], librosa's
   [util.normalize] along the lag axis: a length below the smallest normal
   double is replaced by [1], which leaves a silent frame at zero. *)
let normalise norm (r : Float.Array.t) =
  let win = Float.Array.length r in
  let divide m =
    let m = if m < Float.min_float then 1. else m in
    for k = 0 to win - 1 do
      Float.Array.unsafe_set r k (Float.Array.unsafe_get r k /. m)
    done
  in
  match norm with
  | `None ->
      ()
  | `Inf ->
      let m = ref 0. in
      for k = 0 to win - 1 do
        let a = Float.abs (Float.Array.unsafe_get r k) in
        if a > !m then m := a
      done ;
      divide !m
  | `P p ->
      let s = ref 0. in
      for k = 0 to win - 1 do
        let a = Float.abs (Float.Array.unsafe_get r k) in
        let term =
          if Float.equal p 1. then a
          else if Float.equal p 2. then a *. a
          else Float.pow a p
        in
        s := !s +. term
      done ;
      divide
        ( if Float.equal p 1. then !s
          else if Float.equal p 2. then Float.sqrt !s
          else Float.pow !s (1. /. p) )

(* [frame plan ~data ~off ~base ~total ~x0 ~xlast t f r] is the normalised
   autocorrelation of frame [t] of one row, left in [r], with [f] as scratch.
   [data] holds the row's values [base ...] from [off]; [total] is the row's
   length, beyond which the right ramp towards [xlast] is read, and [x0] its
   first value. *)
let frame plan ~(data : float array) ~off ~base ~total ~x0 ~xlast t
    (f : Float.Array.t) (r : Float.Array.t) =
  let win = plan.win and p = plan.left in
  let lstep = if p > 0 then x0 /. Float.of_int p else 0.
  and rstep = if p > 0 then xlast /. Float.of_int p else 0. in
  for i = 0 to win - 1 do
    let j = t - p + i in
    let v =
      if j < 0 then Float.of_int (j + p) *. lstep
      else if j >= total then Float.of_int (p - 1 - (j - total)) *. rstep
      else Array.unsafe_get data (off + j - base)
    in
    Float.Array.unsafe_set f i (v *. Float.Array.unsafe_get plan.w i)
  done ;
  for k = 0 to win - 1 do
    let acc = ref 0. in
    for i = 0 to win - 1 - k do
      let a = Float.Array.unsafe_get f i in
      let b = Float.Array.unsafe_get f (i + k) in
      let prod = a *. b in
      acc := !acc +. prod
    done ;
    Float.Array.unsafe_set r k !acc
  done ;
  normalise plan.norm r

(* [emit dtype plan ~lead ~data ~row_len ~base ~total ~x0 ~xlast ~lo ~hi] is
   frames [lo, hi) of every row, shaped [lead @ [win; hi - lo]]: row [r] of
   [data] holds that row's values [base, base + row_len) from [r * row_len],
   every value those frames read inside the row. Frames run across domains; each
   is computed alone, in a fixed order, so the split changes no bit, and neither
   does which call computes a frame. *)
let emit (type a) (dtype : (float, a) Nx.dtype) plan ~lead ~data ~row_len ~base
    ~total ~(x0 : float array) ~(xlast : float array) ~lo ~hi : (float, a) Nx.t
    =
  let rows = numel lead and count = hi - lo and win = plan.win in
  let shape = Array.append lead [|win; count|] in
  let grain = Int.max 1 (32_768 / Int.max 1 (win * (win + 1) / 2)) in
  let run write =
    Par.ranges ~n:(rows * count) ~grain (fun a b ->
        let f = Float.Array.create win and r = Float.Array.create win in
        for idx = a to b - 1 do
          let row = idx / count and t = lo + (idx mod count) in
          frame plan ~data ~off:(row * row_len) ~base ~total ~x0:x0.(row)
            ~xlast:xlast.(row) t f r ;
          write ((row * win * count) + t - lo) r
        done )
  in
  match dtype with
  | Nx.Float32 ->
      Store.make32 shape (fun dst ->
          run (fun o r ->
              for k = 0 to win - 1 do
                Bigarray.Array1.unsafe_set dst
                  (o + (k * count))
                  (Float.Array.unsafe_get r k)
              done ) )
  | Nx.Float64 ->
      Store.make64 shape (fun dst ->
          run (fun o r ->
              for k = 0 to win - 1 do
                Bigarray.Array1.unsafe_set dst
                  (o + (k * count))
                  (Float.Array.unsafe_get r k)
              done ) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      assert false

(* [ends data ~rows ~n] is the first and the last value of each row. *)
let ends (data : float array) ~rows ~n =
  if n = 0 then (Array.make rows 0., Array.make rows 0.)
  else
    ( Array.init rows (fun r -> data.(r * n))
    , Array.init rows (fun r -> data.((r * n) + n - 1)) )

let frame_count fn ~alignment ~win n =
  match alignment with
  | `Centered ->
      n
  | `Left ->
      if n < win then
        invalid_arg
          (Printf.sprintf
             "%s: cannot frame %d onset values with a window of %d (`Left \
              alignment needs at least win_length values)"
             fn n win ) ;
      n - win + 1

(* [autocorrelation dtype fn plan ~alignment x] is the tempogram of [x]. *)
let autocorrelation dtype fn plan ~alignment x =
  let data = values fn x in
  let n = last_dim x and lead = leading_shape x in
  let count = frame_count fn ~alignment ~win:plan.win n in
  let x0, xlast = ends data ~rows:(numel lead) ~n in
  emit dtype plan ~lead ~data ~row_len:n ~base:0 ~total:n ~x0 ~xlast ~lo:0
    ~hi:count

let tempogram ?(win_length = 384) ?(alignment = `Centered)
    ?(window = Window.Hann) ?(norm = `Inf) x =
  let fn = "tempogram" in
  check_win fn win_length ;
  check_norm fn norm ;
  let plan = plan ~win:win_length ~alignment ~window ~norm in
  autocorrelation (Nx.dtype x) fn plan ~alignment x

(* {2 Pipeline stage}

   State of one stream: the first value of each row, which fixes the left ramp;
   the last [win - 1] values of each row, every value a frame not yet emitted
   reads inside the row; and the counts of values in and frames out. Frame [t]
   reads values up to [t - left + win - 1]; while that index lies inside the
   stream the frame needs nothing past it, so it is emitted once that value has
   arrived, [win - 1 - left] values late, and the flush emits the frames that
   read the right ramp, which only the stream's end fixes. The frames are those
   {!emit} computes offline, one at a time. *)

type 'a stream =
  { mutable lead: int array option
  ; mutable dtype: (float, 'a) Nx.dtype option
  ; mutable x0: float array
  ; mutable hist: float array
  ; mutable hlen: int
  ; mutable seen: int
  ; mutable emitted: int
  ; mutable drained: bool }

let stream_create () =
  { lead= None
  ; dtype= None
  ; x0= [||]
  ; hist= [||]
  ; hlen= 0
  ; seen= 0
  ; emitted= 0
  ; drained= false }

let stream_reset s =
  s.lead <- None ;
  s.dtype <- None ;
  s.x0 <- [||] ;
  s.hist <- [||] ;
  s.hlen <- 0 ;
  s.seen <- 0 ;
  s.emitted <- 0 ;
  s.drained <- false

let tempogram_stage ?(win_length = 384) ?(alignment = `Centered)
    ?(window = Window.Hann) ?(norm = `Inf) () =
  let fn = "tempogram_stage" in
  check_win fn win_length ;
  check_norm fn norm ;
  let plan = plan ~win:win_length ~alignment ~window ~norm in
  let win = win_length in
  let latency =
    match alignment with `Centered -> win - 1 - (win / 2) | `Left -> win - 1
  in
  let step s chunk =
    if s.drained then
      invalid_arg
        ( fn
        ^ ": cannot feed a drained stream (flush consumed the tail; reset \
           before reusing)" ) ;
    let data = values fn chunk in
    let dtype = Nx.dtype chunk and lead = leading_shape chunk in
    ( match s.lead with
    | Some l when l = lead ->
        ()
    | Some l ->
        invalid_arg
          (Printf.sprintf
             "%s: cannot continue a stream of leading shape [%s] with a chunk \
              of leading shape [%s]"
             fn (shape_string l) (shape_string lead) )
    | None ->
        s.lead <- Some lead ;
        s.dtype <- Some dtype ) ;
    let rows = numel lead and m = last_dim chunk in
    if s.seen = 0 && m > 0 then s.x0 <- Array.init rows (fun r -> data.(r * m)) ;
    let len = s.hlen + m in
    let ext = Array.make (rows * len) 0. in
    for r = 0 to rows - 1 do
      Array.blit s.hist (r * s.hlen) ext (r * len) s.hlen ;
      Array.blit data (r * m) ext ((r * len) + s.hlen) m
    done ;
    let base = s.seen - s.hlen in
    s.seen <- s.seen + m ;
    let ready =
      match alignment with
      | `Centered ->
          Int.max 0 (s.seen - latency)
      | `Left ->
          Int.max 0 (s.seen - win + 1)
    in
    let lo = s.emitted in
    let out =
      if ready > lo then begin
        s.emitted <- ready ;
        (* No frame emitted here reads past the values seen, so the stream's
           length is not needed yet. *)
        Some
          (emit dtype plan ~lead ~data:ext ~row_len:len ~base ~total:max_int
             ~x0:s.x0 ~xlast:s.x0 ~lo ~hi:ready )
      end
      else None
    in
    let keep = Int.min (win - 1) len in
    s.hist <-
      Array.init (rows * keep) (fun i ->
          ext.((i / keep * len) + len - keep + (i mod keep)) ) ;
    s.hlen <- keep ;
    out
  in
  let flush s =
    match alignment with
    | `Left ->
        (* every frame was emitted as its last value arrived *)
        ignore (frame_count fn ~alignment ~win s.seen) ;
        s.drained <- true ;
        []
    | `Centered -> (
        let pending = (not s.drained) && s.emitted < s.seen in
        s.drained <- true ;
        match (s.lead, s.dtype) with
        | Some lead, Some dtype when pending ->
            let rows = numel lead in
            let xlast =
              Array.init rows (fun r -> s.hist.((r * s.hlen) + s.hlen - 1))
            in
            let out =
              emit dtype plan ~lead ~data:s.hist ~row_len:s.hlen
                ~base:(s.seen - s.hlen) ~total:s.seen ~x0:s.x0 ~xlast
                ~lo:s.emitted ~hi:s.seen
            in
            s.emitted <- s.seen ;
            [out]
        | _ ->
            [] )
  in
  (* The element dtype and the leading shape are only witnessed by incoming
     chunks; [step] records them so [concat []] (reached only through [run],
     where a step always precedes) can build the empty tempogram. *)
  let witness = ref None in
  Pipeline.kernel ~latency ~flush ~reset:stream_reset
    ~concat:(function
      | [] -> (
        match !witness with
        | Some (dtype, lead) ->
            Nx.zeros dtype (Array.append lead [|win; 0|])
        | None ->
            invalid_arg
              ( fn
              ^ ": cannot concatenate zero chunks before any chunk fixed the \
                 element dtype" ) )
      | parts ->
          Nx.concatenate ~axis:(-1) parts )
    ~prepare:(fun _format -> stream_create ())
    ~step:(fun s chunk ->
      let out = step s chunk in
      witness := Some (Nx.dtype chunk, leading_shape chunk) ;
      out )
    ()

(* {1 The Fourier tempogram}

   librosa's Fourier tempogram is its STFT of the envelope at a hop of one
   frame, centred over zero padding: exactly {!Stft.transform} on that geometry,
   whose stage streams it. *)

let fourier_config fn ~win ~alignment ~window =
  check_win fn win ;
  Stft.Config.create ~window ~hop:1
    ~alignment:(alignment :> [`Centered | `Left | `Right])
    ~pad:(`Constant 0.) ~fft_size:win ()

let fourier_tempogram cdtype ?(win_length = 384) ?(alignment = `Centered)
    ?(window = Window.Hann) x =
  let fn = "fourier_tempogram" in
  let c = fourier_config fn ~win:win_length ~alignment ~window in
  ignore (values fn x) ;
  Stft.transform cdtype c x

let fourier_tempogram_stage cdtype ?(win_length = 384) ?(alignment = `Centered)
    ?(window = Window.Hann) () =
  Stft.stage cdtype
    (fourier_config "fourier_tempogram_stage" ~win:win_length ~alignment ~window)

(* {1 Tempo} *)

(* [window_frames fn ~ac_size ~sample_rate ~hop] is librosa's [time_to_frames
   ac_size]: the samples truncated to an integer, then floor-divided by the
   hop. *)
let window_frames fn ~ac_size ~sample_rate ~hop =
  let samples = ac_size *. Float.of_int sample_rate in
  let win = if samples < 0x1p62 then Float.to_int samples / hop else max_int in
  if win < 1 || win = max_int then
    invalid_arg
      (Printf.sprintf
         "%s: cannot autocorrelate over %g seconds at %d Hz and a hop of %d \
          (ac_size * sample_rate must span at least one hop)"
         fn ac_size sample_rate hop ) ;
  win

(* [argmax score n] is the first index of the greatest of [score 0 .. score (n -
   1)], or the first index whose score is NaN: numpy's [argmax]. *)
let argmax score n =
  let rec go k best value =
    if k = n then best
    else
      let s = score k in
      if Float.is_nan s then k
      else if s > value then go (k + 1) k s
      else go (k + 1) best value
  in
  let s0 = score 0 in
  if Float.is_nan s0 then 0 else go 1 0 s0

(* [median v] is numpy's median of the values [v], which it sorts: the middle
   one, or the mean of the two middle ones. *)
let median (v : float array) =
  let v = Array.copy v in
  Array.sort Float.compare v ;
  let c = Array.length v in
  if c mod 2 = 1 then v.(c / 2) else (v.((c / 2) - 1) +. v.(c / 2)) /. 2.

(* [estimate fn ... ~lead ~data ~n] is the tempo of each row of [data]: one per
   row when aggregated, one per frame under [`None], row-major. *)
let estimate fn ~hop ~start_bpm ~std_bpm ~ac_size ~tempo_max ~aggregate ~prior
    ~sample_rate ~lead ~data ~n =
  let win = window_frames fn ~ac_size ~sample_rate ~hop in
  let rows = numel lead in
  let bpms = lag_bpms ~sample_rate ~hop win in
  let logprior =
    Array.map
      (fun b ->
        match prior with
        | Some f ->
            f b
        | None ->
            let d = (Float.log2 b -. Float.log2 start_bpm) /. std_bpm in
            -0.5 *. (d *. d) )
      bpms
  in
  (* [tempo_max] masks the bins faster than it, up to the first that is not;
     when no bin is slower, numpy's [argmax] of an all-false mask is [0] and
     nothing is masked. An infinite bound masks nothing at all, librosa's
     [None]. *)
  if Float.is_finite tempo_max then (
    let first = ref 0 in
    ( try
        Array.iteri
          (fun k b ->
            if b < tempo_max then begin
              first := k ;
              raise Exit
            end )
          bpms
      with Exit -> () ) ;
    for k = 0 to !first - 1 do
      logprior.(k) <- Float.neg_infinity
    done ) ;
  let plan = plan ~win ~alignment:`Centered ~window:Window.Hann ~norm:`Inf in
  let x0, xlast = ends data ~rows ~n in
  let g =
    emit Nx.float64 plan ~lead ~data ~row_len:n ~base:0 ~total:n ~x0 ~xlast
      ~lo:0 ~hi:n
  in
  let pick strength =
    bpms.(argmax (fun k -> Float.log1p (1e6 *. strength k) +. logprior.(k)) win)
  in
  Store.read64 g (fun (src : Store.f64) ->
      let at r k t =
        Bigarray.Array1.unsafe_get src ((((r * win) + k) * n) + t)
      in
      match aggregate with
      | `Mean ->
          Array.init rows (fun r ->
              let means =
                Array.init win (fun k ->
                    let s = ref 0. in
                    for t = 0 to n - 1 do
                      s := !s +. at r k t
                    done ;
                    !s /. Float.of_int n )
              in
              pick (Array.get means) )
      | `Median ->
          Array.init rows (fun r ->
              let medians =
                Array.init win (fun k -> median (Array.init n (at r k)))
              in
              pick (Array.get medians) )
      | `None ->
          Array.init (rows * n) (fun i ->
              let r = i / n and t = i mod n in
              pick (fun k -> at r k t) ) )

let check_tempo fn ~sample_rate ~hop ~start_bpm ~std_bpm ~ac_size ~tempo_max =
  check_rate fn ~sample_rate ~hop ;
  check_positive fn "start_bpm" start_bpm ;
  check_positive fn "std_bpm" std_bpm ;
  check_positive fn "ac_size" ac_size ;
  if not (tempo_max > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot bound the tempo at %g BPM (tempo_max must be positive)" fn
         tempo_max ) ;
  ignore (window_frames fn ~ac_size ~sample_rate ~hop)

let tempo ?(hop = 512) ?(start_bpm = 120.) ?(std_bpm = 1.) ?(ac_size = 8.)
    ?(tempo_max = 320.) ?(aggregate = `Mean) ?prior ~sample_rate x =
  let fn = "tempo" in
  check_tempo fn ~sample_rate ~hop ~start_bpm ~std_bpm ~ac_size ~tempo_max ;
  let data = values fn x in
  let n = last_dim x and lead = leading_shape x in
  ( match aggregate with
  | (`Mean | `Median) when n = 0 ->
      invalid_arg
        (fn ^ ": cannot aggregate the tempogram of an empty onset envelope")
  | `Mean | `Median | `None ->
      () ) ;
  let bpm =
    estimate fn ~hop ~start_bpm ~std_bpm ~ac_size ~tempo_max ~aggregate ~prior
      ~sample_rate ~lead ~data ~n
  in
  let cols = match aggregate with `Mean | `Median -> 1 | `None -> n in
  Nx.create (Nx.dtype x) (Array.append lead [|cols|]) bpm

(* {1 Beat tracking}

   librosa's tracker, step for step: the envelope divided by its standard
   deviation, smoothed into a local score by a Gaussian of a beat's width; the
   dynamic program that scores the best beat sequence ending at each frame; the
   last beat, the backtrace from it, and the trim of weak beats at both ends. *)

(* [normalized o] is [o] divided by its sample standard deviation (one degree of
   freedom), plus the smallest normal double; the sums run in ascending
   order. *)
let normalized (o : float array) =
  let n = Array.length o in
  let s = ref 0. in
  Array.iter (fun v -> s := !s +. v) o ;
  let mean = !s /. Float.of_int n in
  let q = ref 0. in
  Array.iter
    (fun v ->
      let d = v -. mean in
      let sq = d *. d in
      q := !q +. sq )
    o ;
  let den = Float.sqrt (!q /. Float.of_int (n - 1)) +. Float.min_float in
  Array.map (fun v -> v /. den) o

(* [local_score o ~fpb] is the same-mode convolution of [o] with the Gaussian
   [exp (-0.5 (32 d / fpb) ^ 2)] over the offsets [d] of [-fpb .. fpb], in
   librosa's loop: the terms of frame [i] in ascending window index, which reads
   [o] from frame [min (i + fpb) (n - 1)] down to frame [max 1 (i - fpb)]. The
   loop's upper bound excludes frame [0]: the first envelope value never enters
   the score. The window is tabulated only over the offsets the envelope can
   reach. *)
let local_score (o : float array) ~fpb =
  let n = Array.length o in
  let f = if fpb >= Float.of_int n then n else Float.to_int fpb in
  let reach = Int.min f (n - 1) in
  let g =
    Array.init
      ((2 * reach) + 1)
      (fun i ->
        let u = Float.of_int (i - reach) *. 32. /. fpb in
        Float.exp (-0.5 *. (u *. u)) )
  in
  Array.init n (fun i ->
      let acc = ref 0. in
      for m = Int.min (i + f) (n - 1) downto Int.max 1 (i - f) do
        let prod = g.(i - m + reach) *. o.(m) in
        acc := !acc +. prod
      done ;
      !acc )

(* [dynamic ls ~fpb ~tight] is the backlinks and cumulative scores of the
   dynamic program: each frame's best predecessor between [round (fpb / 2)] and
   [2 fpb] frames back, the latest of equal scores, under the penalty [tight
   (log gap - log fpb) ^ 2]. Frames before the first local score of at least a
   hundredth of the maximum link to nothing. *)
let dynamic (ls : float array) ~fpb ~tight =
  let n = Array.length ls in
  let thresh = 0.01 *. Array.fold_left Float.max Float.neg_infinity ls in
  let half = round_half_even (fpb /. 2.) and log_fpb = Float.log fpb in
  let back = Array.make n (-1) and cum = Array.make n 0. in
  let first = ref true in
  for i = 0 to n - 1 do
    let start = if half > Float.of_int i then -1 else i - Float.to_int half in
    let stop =
      if 2. *. fpb >= Float.of_int i then 0 else i - Float.to_int (2. *. fpb)
    in
    let best = ref Float.neg_infinity and loc_b = ref (-1) in
    for loc = start downto stop do
      let d = Float.log (Float.of_int (i - loc)) -. log_fpb in
      let pen = tight *. (d *. d) in
      let score = cum.(loc) -. pen in
      if score > !best then begin
        best := score ;
        loc_b := loc
      end
    done ;
    cum.(i) <- (if !loc_b >= 0 then ls.(i) +. !best else ls.(i)) ;
    if !first && ls.(i) < thresh then back.(i) <- -1
    else begin
      back.(i) <- !loc_b ;
      first := false
    end
  done ;
  (back, cum)

(* [last_beat cum] is the last local maximum of the cumulative score that
   reaches half the median of the local maxima, or the last frame when none
   does. Frame [0] is never a local maximum; the last is one when it exceeds its
   predecessor. *)
let last_beat (cum : float array) =
  let n = Array.length cum in
  let is_max i =
    if i = 0 then false
    else if i = n - 1 then cum.(i) > cum.(i - 1)
    else cum.(i) > cum.(i - 1) && cum.(i) >= cum.(i + 1)
  in
  let maxima =
    Array.of_list
      (List.filter_map
         (fun i -> if is_max i then Some cum.(i) else None)
         (List.init n Fun.id) )
  in
  if Array.length maxima = 0 then n - 1
  else
    let threshold = 0.5 *. median maxima in
    let rec find i =
      if i < 0 then n - 1
      else if is_max i && cum.(i) >= threshold then i
      else find (i - 1)
    in
    find (n - 1)

(* [trimmed ~trim ls beats] drops the beats of the leading and trailing frames
   whose local score does not exceed half the root mean square of the beats'
   scores smoothed by the 5-point Hann window [0, 1/2, 1, 1/2, 0] — librosa's
   full convolution cut to the frame count, which averages over two more terms
   than there are beats when the frames allow — or, without [trim], does not
   exceed zero. librosa scans the frames without bounds; the scans here stop at
   the ends, so a signal whose every score sits at or below the threshold keeps
   no beat. *)
let trimmed ~trim (ls : float array) (beats : bool array) =
  let n = Array.length ls in
  let threshold =
    if not trim then 0.
    else begin
      let a =
        Array.of_list
          (List.filter_map
             (fun i -> if beats.(i) then Some ls.(i) else None)
             (List.init n Fun.id) )
      in
      let nb = Array.length a in
      let count = Int.min n (nb + 2) in
      let sum = ref 0. in
      for j = 0 to count - 1 do
        let s = ref 0. in
        if j >= 1 && j - 1 < nb then s := !s +. (0.5 *. a.(j - 1)) ;
        if j < nb then s := !s +. a.(j) ;
        if j + 1 < nb then s := !s +. (0.5 *. a.(j + 1)) ;
        let sq = !s *. !s in
        sum := !sum +. sq
      done ;
      0.5 *. Float.pow (!sum /. Float.of_int count) 0.5
    end
  in
  let keep = Array.copy beats in
  let i = ref 0 in
  while !i < n && ls.(!i) <= threshold do
    keep.(!i) <- false ;
    incr i
  done ;
  let i = ref (n - 1) in
  while !i >= 0 && ls.(!i) <= threshold do
    keep.(!i) <- false ;
    decr i
  done ;
  Array.of_list (List.filter (fun i -> keep.(i)) (List.init n Fun.id))

let beat_track ?(hop = 512) ?(start_bpm = 120.) ?(tightness = 100.)
    ?(trim = true) ?bpm ?prior ~sample_rate x =
  let fn = "beat_track" in
  check_rate fn ~sample_rate ~hop ;
  check_positive fn "start_bpm" start_bpm ;
  check_positive fn "tightness" tightness ;
  (* librosa's dynamic program takes the tightness in single precision *)
  let tight = Int32.float_of_bits (Int32.bits_of_float tightness) in
  if not (Float.is_finite tight) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use tightness = %g (tightness must be finite in single \
          precision)"
         fn tightness ) ;
  Option.iter (check_positive fn "bpm") bpm ;
  let shape = Nx.shape x in
  let nd = Array.length shape in
  if nd = 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot track beats in a rank-zero tensor (the time axis must \
          exist)"
         fn ) ;
  let lead = Array.sub shape 0 (nd - 1) in
  if Array.exists (fun d -> d <> 1) lead then
    invalid_arg
      (Printf.sprintf
         "%s: cannot read a tensor of leading shape [%s] as one onset envelope \
          (every leading axis must have size 1)"
         fn
         (String.concat "; " (Array.to_list (Array.map string_of_int lead))) ) ;
  let o = values fn x in
  let n = Array.length o in
  if n < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot track beats in %d onset frames (at least 2 are needed)" fn
         n ) ;
  if Array.for_all (fun v -> v = 0.) o then (0., [||])
  else begin
    let bpm =
      match bpm with
      | Some b ->
          b
      | None ->
          (estimate fn ~hop ~start_bpm ~std_bpm:1. ~ac_size:8. ~tempo_max:320.
             ~aggregate:`Mean ~prior ~sample_rate ~lead:[||] ~data:o ~n ).(0)
    in
    let frame_rate = Float.of_int sample_rate /. Float.of_int hop in
    let fpb = round_half_even (frame_rate *. 60. /. bpm) in
    if not (fpb >= 1.) then
      invalid_arg
        (Printf.sprintf
           "%s: cannot track a tempo of %g BPM at %g frames per second (a beat \
            must span at least one frame)"
           fn bpm frame_rate ) ;
    let ls = local_score (normalized o) ~fpb in
    let back, cum = dynamic ls ~fpb ~tight in
    let beats = Array.make n false in
    let rec trace i =
      if i >= 0 then begin
        beats.(i) <- true ;
        trace back.(i)
      end
    in
    trace (last_beat cum) ;
    (bpm, trimmed ~trim ls beats)
  end

(* {1 Predominant local pulse}

   The Fourier tempogram is restricted to the tempo range, each frame keeps the
   coefficients of its strongest tempo and is divided by a modulus, and the
   least-squares inverse of the result is the pulse, clipped at zero and
   normalised to a maximum of one. librosa divides by the modulus of the frame's
   greatest coefficient in numpy's lexicographic order of complex values, real
   part first: a frame whose peak has a negative real part finds one of the
   zeroed coefficients greater, and is divided by the square root of the
   smallest normal double alone, which makes it outweigh every other by about
   1e154 — [`Reference]. [`Unit] divides by the peak's own modulus, so every
   frame contributes a kernel of unit amplitude, the method as Grosche and
   Müller publish it. numpy divides by a real value as by a complex one of zero
   imaginary part, multiplying both components by its reciprocal. *)

let sqrt_tiny = 0x1p-511

let plp ?(hop = 512) ?(win_length = 384) ?(tempo_min = 30.) ?(tempo_max = 300.)
    ?prior ?(kernels = `Reference) ~sample_rate x =
  let fn = "plp" in
  check_rate fn ~sample_rate ~hop ;
  check_win fn win_length ;
  if not (Float.is_finite tempo_min && tempo_min >= 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot bound the tempo below at %g BPM (tempo_min must be finite \
          and non-negative)"
         fn tempo_min ) ;
  if not (tempo_max > tempo_min) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot search tempi in [%g, %g] BPM (tempo_max must exceed \
          tempo_min)"
         fn tempo_min tempo_max ) ;
  let data = values fn x in
  let n = last_dim x and lead = leading_shape x in
  let rows = numel lead in
  let c =
    fourier_config fn ~win:win_length ~alignment:`Centered ~window:Window.Hann
  in
  let x64 = Nx.create Nx.float64 (Nx.shape x) data in
  let z = Stft.transform Nx.complex128 c x64 in
  let bins = (win_length / 2) + 1 and frames = last_dim z in
  let freqs = fourier_bpms ~sample_rate ~hop win_length in
  let inside =
    Array.map (fun f -> not (f < tempo_min || f > tempo_max)) freqs
  in
  let logprior =
    match prior with Some p -> Some (Array.map p freqs) | None -> None
  in
  let grain = Int.max 1 (8_192 / bins) in
  let kept =
    Store.read_parts64 z (fun (src : Store.f64) ->
        Store.make_c128 (Nx.shape z) (fun dst ->
            Par.ranges ~n:(rows * frames) ~grain (fun lo hi ->
                let re = Array.make bins 0. and im = Array.make bins 0. in
                let mag = Array.make bins 0. in
                for col = lo to hi - 1 do
                  let row = col / frames and t = col mod frames in
                  let at k = 2 * ((((row * bins) + k) * frames) + t) in
                  let peak = ref Float.neg_infinity and nan = ref false in
                  for k = 0 to bins - 1 do
                    if inside.(k) then begin
                      re.(k) <- Bigarray.Array1.unsafe_get src (at k) ;
                      im.(k) <- Bigarray.Array1.unsafe_get src (at k + 1)
                    end
                    else begin
                      re.(k) <- 0. ;
                      im.(k) <- 0.
                    end ;
                    let m = Float.log1p (1e6 *. Float.hypot re.(k) im.(k)) in
                    let m =
                      match logprior with Some lp -> m +. lp.(k) | None -> m
                    in
                    mag.(k) <- m ;
                    if Float.is_nan m then nan := true
                    else if m > !peak then peak := m
                  done ;
                  (* numpy's maximum propagates NaN, and nothing compares below
                     it *)
                  let peak = if !nan then Float.nan else !peak in
                  for k = 0 to bins - 1 do
                    if mag.(k) < peak then begin
                      re.(k) <- 0. ;
                      im.(k) <- 0.
                    end
                  done ;
                  let modulus =
                    match kernels with
                    | `Reference ->
                        let b = ref 0 in
                        for k = 1 to bins - 1 do
                          if
                            re.(k) > re.(!b)
                            || (re.(k) = re.(!b) && im.(k) > im.(!b))
                          then b := k
                        done ;
                        Float.hypot re.(!b) im.(!b)
                    | `Unit ->
                        let m = ref 0. in
                        for k = 0 to bins - 1 do
                          m := Float.max !m (Float.hypot re.(k) im.(k))
                        done ;
                        !m
                  in
                  let scale = 1. /. (sqrt_tiny +. modulus) in
                  for k = 0 to bins - 1 do
                    Bigarray.Array1.unsafe_set dst (at k) (re.(k) *. scale) ;
                    Bigarray.Array1.unsafe_set dst (at k + 1) (im.(k) *. scale)
                  done
                done ) ) )
  in
  let pulse = Stft.invert Nx.float64 c ~length:n kept in
  let out =
    Store.read64 pulse (fun (src : Store.f64) ->
        Store.make64 (Nx.shape x) (fun dst ->
            for r = 0 to rows - 1 do
              let m = ref 0. in
              for t = 0 to n - 1 do
                let v =
                  Float.max 0. (Bigarray.Array1.unsafe_get src ((r * n) + t))
                in
                Bigarray.Array1.unsafe_set dst ((r * n) + t) v ;
                if v > !m then m := v
              done ;
              let m = if !m < Float.min_float then 1. else !m in
              for t = 0 to n - 1 do
                let i = (r * n) + t in
                Bigarray.Array1.unsafe_set dst i
                  (Bigarray.Array1.unsafe_get dst i /. m)
              done
            done ) )
  in
  Nx.cast (Nx.dtype x) out
