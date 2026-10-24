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

(* Fundamental frequency estimation: YIN (de Cheveigné and Kawahara, 2002) and
   probabilistic YIN (Mauch and Dixon, 2014), as librosa.yin and librosa.pyin
   compute them.

   Both share one front end per frame, spelled in librosa's order of operations:

   - the autocorrelation [r] of the frame, through one zero-padded real FFT of
   the 5-smooth length scipy's [next_fast_len] picks, the power [re * re + im *
   im] and one inverse FFT, every frame transformed on its own line;

   - the difference [d k = 2 (r 0 - r k) - e (k - 1)] for lags [1 ..
   max_period], [e m] the running sum of the squared samples [0 .. m] — except
   that librosa zeroes [e 0] before reading it, so [d 1 = 2 (r 0 - r 1)] lacks
   [y 0 ^ 2]: a quirk the parity contract reproduces;

   - the cumulative mean normalized difference [d k / (mean (d 1 .. d k) +
   tiny)] over the lags [min_period .. max_period], one fused loop per frame,
   then its troughs and their parabolic shifts.

   The frames are read in place by {!Loops.frames} (an all-ones window, which
   multiplies exactly), every product added to something is let-bound first —
   but the doublings, exact, which round alike fused or not — so no platform
   fuses a multiply-add numpy rounds twice, and a frame's values depend on its
   own samples alone: the batched FFTs transform each line on its own and the
   per-frame loops run over disjoint rows, so blocks, chunks and worker domains
   change no bit. Offline functions frame the whole signal, its border read by
   index; the stage frames the same padded stream materialized chunk by chunk,
   with the [Stft] carry's install-and-tail handling of reflection. *)

module A1 = Bigarray.Array1

type pad = [`Constant of float | `Edge | `Reflect]

type 'a pyin =
  { f0: (float, 'a) Nx.t
  ; voiced: (bool, Nx.bool_elt) Nx.t
  ; voiced_probability: (float, 'a) Nx.t }

(* {1 Shape helpers} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let zero_leading t = Array.exists (fun d -> d = 0) (leading_shape t)

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
    t

let concat_last parts = Nx.concatenate ~axis:(-1) parts

let take_last t indices =
  let indices =
    Nx.create Nx.int64 [|Array.length indices|] (Array.map Int64.of_int indices)
  in
  Nx.take ~axis:(-1) ~indices t

let shape_string shape =
  String.concat "; " (List.map string_of_int (Array.to_list shape))

let ceil_div a b = (a + b - 1) / b

(* {1 Preconditions and the analysis geometry} *)

type geometry =
  { sample_rate: int
  ; frame_length: int
  ; hop: int
  ; center: bool
  ; pad: pad
  ; fmin: float
  ; fmax: float
  ; min_period: int
  ; max_period: int
  ; n_fft: int (* the autocorrelation's FFT length *) }

(* [next_fast_len n] is the least 5-smooth integer at least [n]: the length
   scipy.fft.next_fast_len picks for a real transform. *)
let next_fast_len n =
  let rec strip v p = if v mod p = 0 then strip (v / p) p else v in
  let smooth v = strip (strip (strip v 2) 3) 5 = 1 in
  let rec go v = if smooth v then v else go (v + 1) in
  go (Int.max 1 n)

(* [centered a] is whether frames of alignment [a] are centered on their grid
   position, librosa's [center=True]. *)
let centered = function `Centered -> true | `Left -> false

let geometry fn ~frame_length ~hop ~center ~pad ~fmin ~fmax ~sample_rate =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot estimate pitch at %d Hz (sample_rate must be at least 1)"
         fn sample_rate ) ;
  if frame_length < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse frames of %d samples (frame_length must be at \
          least 1)"
         fn frame_length ) ;
  if not (Float.is_finite fmin && Float.is_finite fmax) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot search between fmin = %g Hz and fmax = %g Hz (both must \
          be finite)"
         fn fmin fmax ) ;
  let sr = Float.of_int sample_rate in
  if fmax > sr /. 2. then
    invalid_arg
      (Printf.sprintf
         "%s: cannot search up to fmax = %g Hz at %d Hz (fmax must not exceed \
          the Nyquist frequency, %g Hz)"
         fn fmax sample_rate (sr /. 2.) ) ;
  if fmin >= fmax then
    invalid_arg
      (Printf.sprintf
         "%s: cannot search between fmin = %g Hz and fmax = %g Hz (fmin must \
          be below fmax)"
         fn fmin fmax ) ;
  if fmin <= 0. then
    invalid_arg
      (Printf.sprintf
         "%s: cannot search from fmin = %g Hz (fmin must be positive)" fn fmin ) ;
  if sr /. fmin >= Float.of_int (frame_length - 1) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fit a period of fmin = %g Hz in frames of %d samples at \
          %d Hz (sample_rate / fmin must be below frame_length - 1: raise fmin \
          above %g Hz or frame_length to %d)"
         fn fmin frame_length sample_rate
         (sr /. Float.of_int (frame_length - 1))
         (Float.to_int (Float.floor (sr /. fmin)) + 2) ) ;
  let hop = Option.value hop ~default:(frame_length / 4) in
  if hop < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot advance frames by %d samples (hop must be at least 1)" fn
         hop ) ;
  ( match pad with
  | `Constant v when not (Float.is_finite v) ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot pad with the constant %g (the padding value must be \
            finite)"
           fn v )
  | `Constant _ | `Edge | `Reflect ->
      () ) ;
  { sample_rate
  ; frame_length
  ; hop
  ; center
  ; pad
  ; fmin
  ; fmax
  ; min_period= Float.to_int (Float.floor (sr /. fmax))
  ; max_period=
      Int.min (Float.to_int (Float.ceil (sr /. fmin))) (frame_length - 1)
  ; n_fft= next_fast_len ((2 * frame_length) - 1) }

let check_threshold fn threshold =
  if not (Float.is_finite threshold) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot select troughs below %g (trough_threshold must be finite)"
         fn threshold )

let check_rank fn x =
  if Nx.ndim x < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-zero tensor (the time axis must exist)" fn )

(* [all_finite x] is [true] iff every sample of [x] is finite. *)
let all_finite (type a) (x : (float, a) Nx.t) =
  let n = Nx.numel x in
  let scan32 (a : Store.f32) =
    let rec go i =
      i = n || (Float.is_finite (A1.unsafe_get a i) && go (i + 1))
    in
    go 0
  and scan64 (a : Store.f64) =
    let rec go i =
      i = n || (Float.is_finite (A1.unsafe_get a i) && go (i + 1))
    in
    go 0
  in
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.read32 x scan32
  | Nx.Float64 ->
      Store.read64 x scan64
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Store.read64 (Nx.cast Nx.float64 x) scan64

let check_finite fn x =
  if not (all_finite x) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot estimate the pitch of a signal holding a non-finite sample"
         fn )

(* {1 The frame grid}

   Centered analysis extends the signal by [frame_length / 2] positions on each
   side; frame [p] covers padded positions [p * hop, p * hop + frame_length). An
   empty signal, and an uncentered one shorter than a frame, produce no frames —
   the [Stft] grid convention, a documented deviation on the public face. *)

let border g = if g.center then g.frame_length / 2 else 0

let count_frames g n =
  if g.center then
    if n = 0 then 0
    else 1 + ((n + (2 * (g.frame_length / 2)) - g.frame_length) / g.hop)
  else if n < g.frame_length then 0
  else 1 + ((n - g.frame_length) / g.hop)

(* [extension g dtype] is the boundary extension of [g]; a constant is first
   stored at the signal's dtype, as padding the signal would store it. *)
let extension g dtype =
  match g.pad with
  | `Reflect ->
      Loops.Reflect
  | `Edge ->
      Loops.Edge
  | `Constant v ->
      Loops.Constant (Nx.item [] (Nx.cast Nx.float64 (Nx.scalar dtype v)))

(* {1 The difference function} *)

(* [power spectrum] is [re * re + im * im] of every bin, as complex128 with a
   zero imaginary part, the input of the inverse transform; each square is
   let-bound, so no platform fuses the sum. *)
let power (spectrum : (Complex.t, Nx.complex64_elt) Nx.t) =
  let n = Nx.numel spectrum in
  Store.read_parts64 spectrum (fun src ->
      Store.make_c128 (Nx.shape spectrum) (fun dst ->
          Par.ranges ~n ~grain:32_768 (fun lo hi ->
              for i = lo to hi - 1 do
                let re = A1.unsafe_get src (2 * i)
                and im = A1.unsafe_get src ((2 * i) + 1) in
                let re2 = re *. re in
                let im2 = im *. im in
                A1.unsafe_set dst (2 * i) (re2 +. im2) ;
                A1.unsafe_set dst ((2 * i) + 1) 0.
              done ) ) )

(* [autocorrelation g frames] is the unscaled circular autocorrelation of every
   frame of [frames] ([[...; count; frame_length]], float64) zero-padded to
   [n_fft] samples: [[...; count; n_fft]], lag [k] at index [k], [n_fft] times
   the linear autocorrelation over the lags the difference reads. *)
let autocorrelation g frames =
  let spectrum = Nx.rfft Nx.complex128 ~n:g.n_fft frames in
  Nx.irfft Nx.float64 ~norm:`Forward ~n:g.n_fft (power spectrum)

(* [cmnd g fr ac r c] writes the cumulative mean normalized difference of row
   [r] — its samples in [fr], its unscaled autocorrelation in [ac] — to [c], lag
   [min_period + i] at index [i]. The terms run in librosa's order: the energies
   are the running sum of the let-bound squares, the difference is [2 (r 0 - r
   k)] less the energy of the samples before lag [k] (none at lag 1, the quirk),
   and each value divides by the running mean of the differences plus the
   smallest normal double. *)
let cmnd g (fr : Store.f64) (ac : Store.f64) r (c : Float.Array.t) =
  let fb = r * g.frame_length and ab = r * g.n_fft in
  let scale = 1. /. Float.of_int g.n_fft in
  let a0 = A1.unsafe_get ac ab *. scale in
  let y0 = A1.unsafe_get fr fb in
  let energy = ref (y0 *. y0) in
  let before = ref 0. in
  let total = ref 0. in
  for k = 1 to g.max_period do
    let ak = A1.unsafe_get ac (ab + k) *. scale in
    let d = (2. *. (a0 -. ak)) -. !before in
    total := !total +. d ;
    if k >= g.min_period then
      Float.Array.unsafe_set c (k - g.min_period)
        (d /. ((!total /. Float.of_int k) +. Float.min_float)) ;
    if k < g.max_period then begin
      let yk = A1.unsafe_get fr (fb + k) in
      let sq = yk *. yk in
      energy := !energy +. sq ;
      before := !energy
    end
  done

(* [is_trough c l i] is librosa's trough test at index [i] of the [l] values of
   [c]: below its left neighbour and not above its right one, the ends compared
   with their single neighbour. *)
let is_trough c l i =
  let x = Float.Array.unsafe_get c i in
  if i = 0 then x < Float.Array.unsafe_get c 1
  else if i = l - 1 then x < Float.Array.unsafe_get c (l - 2)
  else
    x < Float.Array.unsafe_get c (i - 1) && x <= Float.Array.unsafe_get c (i + 1)

(* [shift c l i] is the parabolic interpolation offset at index [i]: the vertex
   of the parabola through the three values around it, [0.] at the ends and
   wherever the vertex would lie a full step away or more. *)
let shift c l i =
  if i = 0 || i = l - 1 then 0.
  else
    let lo = Float.Array.unsafe_get c (i - 1)
    and mid = Float.Array.unsafe_get c i
    and hi = Float.Array.unsafe_get c (i + 1) in
    let a = hi +. lo -. (2. *. mid) in
    let b = (hi -. lo) /. 2. in
    if Float.abs b >= Float.abs a then 0. else -.b /. a

let argmin c l =
  let best = ref 0 in
  for i = 1 to l - 1 do
    if Float.Array.unsafe_get c i < Float.Array.unsafe_get c !best then
      best := i
  done ;
  !best

(* [frequency g c i] is the frequency of the period at index [i] of [c], refined
   by its parabolic shift. *)
let frequency g c l i =
  let period = Float.of_int (g.min_period + i) +. shift c l i in
  Float.of_int g.sample_rate /. period

(* {1 Framing and analysis} *)

(* Frames are analysed in blocks of about [block_elements] FFT samples, so the
   transform buffers stay bounded whatever the signal's length. *)
let block_elements = 1 lsl 20

(* [each_frame g ~left ~ext ~count ~local x f] analyses frames [0, count) of the
   padded stream of the signals [x] — padded position [q] being sample [q -
   left], extended by [ext] outside them — and calls [f st o p c] for plane [o],
   frame [p] and its cumulative mean normalized difference [c], from several
   domains at once, [st] being [local ()] made once per range of consecutive
   frames: [f] must write disjoint memory, and must neither keep nor write
   [c]. *)
let each_frame g ~left ~ext ~count ~local x f =
  let n = last_dim x in
  let planes = if n = 0 then 0 else Nx.numel x / n in
  if count > 0 && planes > 0 then begin
    let window = Float.Array.make g.frame_length 1. in
    let l = g.max_period - g.min_period + 1 in
    let block = Int.max 1 (block_elements / (planes * g.n_fft)) in
    let grain = Int.max 1 (8192 / g.max_period) in
    let first = ref 0 in
    while !first < count do
      let base = !first in
      let cnt = Int.min block (count - base) in
      let frames =
        Loops.frames ~window ~hop:g.hop ~left ~ext ~first:base ~count:cnt x
      in
      let acf = autocorrelation g frames in
      Store.read64 frames (fun fr ->
          Store.read64 acf (fun ac ->
              Par.ranges ~n:(planes * cnt) ~grain (fun lo hi ->
                  let c = Float.Array.create l and st = local () in
                  for r = lo to hi - 1 do
                    cmnd g fr ac r c ;
                    f st (r / cnt) (base + (r mod cnt)) c
                  done ) ) ) ;
      first := base + cnt
    done
  end

(* [tensor dtype shape values] is [values] as a tensor, each rounded once into
   [dtype]. *)
let tensor (type a) (dtype : (float, a) Nx.dtype) shape (values : Float.Array.t)
    : (float, a) Nx.t =
  let n = Float.Array.length values in
  match dtype with
  | Nx.Float64 ->
      Store.make64 shape (fun dst ->
          for i = 0 to n - 1 do
            A1.unsafe_set dst i (Float.Array.unsafe_get values i)
          done )
  | Nx.Float32 ->
      Store.make32 shape (fun dst ->
          for i = 0 to n - 1 do
            A1.unsafe_set dst i (Float.Array.unsafe_get values i)
          done )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast dtype
        (Nx.create Nx.float64 shape (Array.init n (Float.Array.get values)))

(* {1 YIN} *)

(* [yin_value g ~threshold c] is YIN's estimate for one frame: the first trough
   of [c] below [threshold], or else its global minimum, at the frequency of its
   refined period. *)
let yin_value g ~threshold c =
  let l = Float.Array.length c in
  let rec first i =
    if i = l then argmin c l
    else if Float.Array.unsafe_get c i < threshold && is_trough c l i then i
    else first (i + 1)
  in
  frequency g c l (first 0)

(* [yin_frames g ~threshold ~left ~ext ~count x] is the estimate of every frame,
   [[...; count]] in the dtype of [x]. *)
let yin_frames g ~threshold ~left ~ext ~count x =
  let lead = leading_shape x in
  let planes = Array.fold_left ( * ) 1 lead in
  let out = Float.Array.make (planes * count) 0. in
  each_frame g ~left ~ext ~count ~local:ignore x (fun () o p c ->
      Float.Array.unsafe_set out ((o * count) + p) (yin_value g ~threshold c) ) ;
  tensor (Nx.dtype x) (Array.append lead [|count|]) out

(* [yin_offline g ~threshold x] frames the whole signal, its border read in
   place: the frames the stage emits on any partitioning. *)
let yin_offline g ~threshold x =
  yin_frames g ~threshold ~left:(border g)
    ~ext:(extension g (Nx.dtype x))
    ~count:(count_frames g (last_dim x))
    x

let yin ?(frame_length = 2048) ?hop ?(trough_threshold = 0.1)
    ?(alignment = `Centered) ?(pad = `Constant 0.) ~fmin ~fmax ~sample_rate x =
  let fn = "yin" in
  let g =
    geometry fn ~frame_length ~hop ~center:(centered alignment) ~pad ~fmin ~fmax
      ~sample_rate
  in
  check_threshold fn trough_threshold ;
  check_rank fn x ;
  check_finite fn x ;
  yin_offline g ~threshold:trough_threshold x

(* {1 The streaming carry}

   The padded stream is the signal with its boundary extension; [pending_rev]
   holds its unconsumed suffix (in reverse) and [skip] counts padded positions
   still to drop when the hop outruns the data. Before the left extension is
   computable — reflection reads the first [border + 1] samples, a constant or
   an edge at most the first — raw chunks accumulate in [prelude_rev]; [tail]
   keeps the last [border + 1] raw samples for the right extension. Retained
   tensors are kernel-owned copies; incoming chunks are borrowed. *)

type 'a carry =
  { g: geometry
  ; threshold: float
  ; border: int
  ; install_at: int
  ; mutable started: bool
  ; mutable drained: bool
  ; mutable received: int
  ; mutable prelude_rev: (float, 'a) Nx.t list
  ; mutable pending_rev: (float, 'a) Nx.t list
  ; mutable pending_len: int
  ; mutable tail: (float, 'a) Nx.t option
  ; mutable skip: int
  ; mutable lead: int array option }

let carry_create g ~threshold =
  let border = border g in
  { g
  ; threshold
  ; border
  ; install_at=
      (match g.pad with `Reflect when border > 0 -> border + 1 | _ -> 1)
  ; started= false
  ; drained= false
  ; received= 0
  ; prelude_rev= []
  ; pending_rev= []
  ; pending_len= 0
  ; tail= None
  ; skip= 0
  ; lead= None }

let carry_reset s =
  s.started <- false ;
  s.drained <- false ;
  s.received <- 0 ;
  s.prelude_rev <- [] ;
  s.pending_rev <- [] ;
  s.pending_len <- 0 ;
  s.tail <- None ;
  s.skip <- 0 ;
  s.lead <- None

(* [process s extra extra_len] appends [extra] (stream order, possibly borrowed)
   to the pending padded stream and emits every frame that became complete; the
   leftover suffix is copied out. *)
let process s extra extra_len =
  let w = s.g.frame_length and hop = s.g.hop in
  let total = s.pending_len + extra_len in
  let count = if total < w then 0 else 1 + ((total - w) / hop) in
  if count = 0 then begin
    List.iter
      (fun t ->
        if last_dim t > 0 then s.pending_rev <- Nx.copy t :: s.pending_rev )
      extra ;
    s.pending_len <- total ;
    None
  end
  else begin
    let parts = List.rev_append s.pending_rev extra in
    let samples = match parts with [one] -> one | many -> concat_last many in
    let out =
      yin_frames s.g ~threshold:s.threshold ~left:0 ~ext:Loops.Edge ~count
        samples
    in
    let next_start = count * hop in
    if next_start >= total then begin
      s.skip <- s.skip + next_start - total ;
      s.pending_rev <- [] ;
      s.pending_len <- 0
    end
    else begin
      s.pending_rev <- [Nx.copy (shrink_last samples next_start total)] ;
      s.pending_len <- total - next_start
    end ;
    Some out
  end

(* [update_tail s chunk] keeps [tail] equal to the last [border + 1] raw
   samples, or every raw sample while fewer have arrived. *)
let update_tail s chunk =
  let keep = s.border + 1 and m = last_dim chunk in
  let joined =
    if m >= keep then chunk
    else match s.tail with None -> chunk | Some t -> concat_last [t; chunk]
  in
  let len = last_dim joined in
  s.tail <-
    Some
      (Nx.copy
         (if len > keep then shrink_last joined (len - keep) len else joined) )

(* [constant_border s like] is [border] positions of the constant extension, in
   the leading shape and dtype of [like]. *)
let constant_border s like v =
  Nx.full (Nx.dtype like) (Array.append (leading_shape like) [|s.border|]) v

(* [left_border s x] is the left extension, read from the first raw samples [x]
   (at least [install_at] of them). *)
let left_border s x =
  if s.border = 0 then None
  else
    Some
      ( match s.g.pad with
      | `Constant v ->
          constant_border s x v
      | `Edge ->
          take_last x (Array.make s.border 0)
      | `Reflect ->
          take_last x (Array.init s.border (fun j -> s.border - j)) )

(* [right_border s] is the right extension, read from the raw tail; defined once
   started, with [border > 0]. *)
let right_border s =
  let tail = Option.get s.tail in
  let tl = last_dim tail in
  match s.g.pad with
  | `Constant v ->
      constant_border s tail v
  | `Edge ->
      take_last tail (Array.make s.border (tl - 1))
  | `Reflect ->
      take_last tail (Array.init s.border (fun i -> tl - 2 - i))

(* [consume s t] feeds [t], less the positions [skip] still drops, to the
   pending stream: [None] when they cover it. *)
let consume s t =
  let m = last_dim t in
  if s.skip >= m then begin
    s.skip <- s.skip - m ;
    None
  end
  else begin
    let dropped = s.skip in
    s.skip <- 0 ;
    process s
      [(if dropped = 0 then t else shrink_last t dropped m)]
      (m - dropped)
  end

let carry_step fn s chunk =
  if s.drained then
    invalid_arg
      (Printf.sprintf
         "%s: cannot feed a drained stream (flush consumed the tail; reset \
          before reusing)"
         fn ) ;
  check_rank fn chunk ;
  if zero_leading chunk then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a chunk with a zero-size leading axis (channels \
          must be at least 1)"
         fn ) ;
  let lead = leading_shape chunk in
  ( match s.lead with
  | None ->
      s.lead <- Some lead
  | Some l ->
      if l <> lead then
        invalid_arg
          (Printf.sprintf
             "%s: cannot continue a stream of leading shape [%s] with a chunk \
              of leading shape [%s]"
             fn (shape_string l) (shape_string lead) ) ) ;
  check_finite fn chunk ;
  let m = last_dim chunk in
  if m = 0 then None
  else begin
    s.received <- s.received + m ;
    if s.border > 0 then update_tail s chunk ;
    if not s.started then
      if s.received >= s.install_at then begin
        let parts = List.rev (chunk :: s.prelude_rev) in
        let x = match parts with [one] -> one | many -> concat_last many in
        s.started <- true ;
        s.prelude_rev <- [] ;
        match left_border s x with
        | None ->
            process s [x] (last_dim x)
        | Some lp ->
            process s [lp; x] (s.border + last_dim x)
      end
      else begin
        s.prelude_rev <- Nx.copy chunk :: s.prelude_rev ;
        None
      end
    else consume s chunk
  end

let carry_flush s =
  if s.drained then None
  else begin
    s.drained <- true ;
    let out =
      if not s.started then
        if s.received = 0 then None
        else begin
          (* a stream too short to reflect: frame it as the offline function
             does, reflecting as many times as it takes *)
          let parts = List.rev s.prelude_rev in
          let x = match parts with [one] -> one | many -> concat_last many in
          if count_frames s.g (last_dim x) = 0 then None
          else Some (yin_offline s.g ~threshold:s.threshold x)
        end
      else if s.border = 0 then None
      else consume s (right_border s)
    in
    s.prelude_rev <- [] ;
    s.pending_rev <- [] ;
    s.pending_len <- 0 ;
    out
  end

(* {1 The pipeline stage} *)

(* [stage_latency g] is the lookahead the stage declares: a centered frame
   reaches [frame_length / 2] samples past its grid position — reflection's
   install threshold, [frame_length / 2 + 1] raw samples, adds nothing — and an
   uncentered frame, anchored at its first sample, [frame_length - 1]. *)
let stage_latency g =
  if g.center then g.frame_length / 2 else g.frame_length - 1

let frame_bound g b = ceil_div (b + stage_latency g) g.hop + 1

let split_frames bound t =
  match bound with
  | None ->
      [t]
  | Some bound ->
      let total = last_dim t in
      if total <= bound then [t]
      else
        let rec go start acc =
          if start >= total then List.rev acc
          else
            let stop = Int.min total (start + bound) in
            go stop (shrink_last t start stop :: acc)
        in
        go 0 []

type 'a stage_state = {carry: 'a carry; bound: int option}

let stage ?(frame_length = 2048) ?hop ?(trough_threshold = 0.1)
    ?(alignment = `Centered) ?(pad = `Constant 0.) ~fmin ~fmax ~sample_rate () =
  let fn = "yin_stage" in
  let g =
    geometry fn ~frame_length ~hop ~center:(centered alignment) ~pad ~fmin ~fmax
      ~sample_rate
  in
  check_threshold fn trough_threshold ;
  let rate = {Pipeline.Rate.num= 1; den= g.hop} in
  let bound fmt = Option.map (frame_bound g) (Pipeline.Format.max_items fmt) in
  let out_format fmt =
    let ips = Pipeline.Rate.(Pipeline.Format.items_per_second fmt * rate) in
    fmt
    |> Pipeline.Format.with_items_per_second ips
    |> Pipeline.Format.with_max_items (bound fmt)
  in
  (* The element dtype and the leading shape are only witnessed by incoming
     chunks; [step] records them so [concat []] can build the empty chunk. *)
  let witness = ref None in
  Pipeline.kernel ~latency:(stage_latency g) ~rate ~out_format
    ~flush:(fun s ->
      match carry_flush s.carry with
      | None ->
          []
      | Some out ->
          split_frames s.bound out )
    ~reset:(fun s -> carry_reset s.carry)
    ~concat:(function
      | [] -> (
        match !witness with
        | Some (dtype, lead) ->
            Nx.zeros dtype (Array.append lead [|0|])
        | None ->
            invalid_arg
              ( fn
              ^ ": cannot concatenate zero chunks before any chunk fixed the \
                 element dtype" ) )
      | parts ->
          concat_last parts )
    ~prepare:(fun fmt ->
      let ips = Pipeline.Format.items_per_second fmt in
      if not (Pipeline.Rate.equal ips {num= sample_rate; den= 1}) then
        invalid_arg
          (Stdlib.Format.asprintf
             "%s: cannot estimate the pitch of a stream at %a items/s \
              (sample_rate is %d Hz)"
             fn Pipeline.Rate.pp ips sample_rate ) ;
      {carry= carry_create g ~threshold:trough_threshold; bound= bound fmt} )
    ~step:(fun s chunk ->
      if Nx.ndim chunk >= 1 then
        witness := Some (Nx.dtype chunk, leading_shape chunk) ;
      carry_step fn s.carry chunk )
    ()

(* {1 Probabilistic YIN} *)

(* [pairwise a lo n] is numpy's pairwise sum of the [n] values of [a] from [lo]:
   below eight values a plain sum, up to 128 eight interleaved partial sums
   combined as a tree then the remainder, and above that the two halves — the
   first a multiple of eight long — summed apart. *)
let rec pairwise (a : Float.Array.t) lo n =
  if n < 8 then begin
    let r = ref 0. in
    for i = lo to lo + n - 1 do
      r := !r +. Float.Array.unsafe_get a i
    done ;
    !r
  end
  else if n <= 128 then begin
    let r = Array.init 8 (fun k -> Float.Array.unsafe_get a (lo + k)) in
    let i = ref 8 in
    while !i < n - (n mod 8) do
      for k = 0 to 7 do
        r.(k) <- r.(k) +. Float.Array.unsafe_get a (lo + !i + k)
      done ;
      i := !i + 8
    done ;
    let res =
      ref
        ( r.(0) +. r.(1)
        +. (r.(2) +. r.(3))
        +. (r.(4) +. r.(5) +. (r.(6) +. r.(7))) )
    in
    while !i < n do
      res := !res +. Float.Array.unsafe_get a (lo + !i) ;
      incr i
    done ;
    !res
  end
  else
    let n2 = n / 2 in
    let n2 = n2 - (n2 mod 8) in
    pairwise a lo n2 +. pairwise a (lo + n2) (n - n2)

(* [round_half_even x] is [x] rounded to the nearest integer, ties to even:
   numpy's [round] and Python's. *)
let round_half_even x =
  let f = Float.floor x in
  let d = x -. f in
  if d < 0.5 then f
  else if d > 0.5 then f +. 1.
  else if Float.rem f 2. = 0. then f
  else f +. 1.

(* The frame-independent tables of one pyin call. *)
type model =
  { thresholds: Float.Array.t (* the upper thresholds [t 1 .. t N] *)
  ; beta: Float.Array.t (* the prior mass between consecutive thresholds *)
  ; below_sums: Float.Array.t
        (* [m] -> the pairwise sum of the first [m] masses, [m = 0 .. N] *)
  ; decay: Float.Array.t (* [k] -> exp (-lambda k) *)
  ; scale: Float.Array.t (* [n] -> the Boltzmann normalisation over n troughs *)
  ; no_trough: float
  ; bins_per_octave: float (* [12 * bins per semitone] *)
  ; n_bins: int }

let model g ~n_thresholds ~beta_parameters:(a, b) ~boltzmann_parameter
    ~no_trough_prob ~bins_per_semitone ~n_bins =
  let n = n_thresholds in
  let step = 1. /. Float.of_int n in
  let points =
    Array.init (n + 1) (fun j -> if j = n then 1. else Float.of_int j *. step)
  in
  let cdf =
    Nx.to_array
      (Nx.betainc
         (Nx.full Nx.float64 [|n + 1|] a)
         (Nx.full Nx.float64 [|n + 1|] b)
         (Nx.create Nx.float64 [|n + 1|] points) )
  in
  (* the distribution's support ends, exactly as scipy's wrapper sets them *)
  cdf.(0) <- 0. ;
  cdf.(n) <- 1. ;
  let beta = Float.Array.init n (fun j -> cdf.(j + 1) -. cdf.(j)) in
  let l = g.max_period - g.min_period + 1 in
  let lambda = boltzmann_parameter in
  let head = 1. -. Float.exp (-.lambda) in
  { thresholds= Float.Array.init n (fun j -> points.(j + 1))
  ; beta
  ; below_sums= Float.Array.init (n + 1) (fun m -> pairwise beta 0 m)
  ; decay=
      Float.Array.init (l + 1) (fun k -> Float.exp (-.lambda *. Float.of_int k))
  ; scale=
      Float.Array.init (l + 1) (fun k ->
          head /. (1. -. Float.exp (-.lambda *. Float.of_int k)) )
  ; no_trough= no_trough_prob
  ; bins_per_octave= Float.of_int (12 * bins_per_semitone)
  ; n_bins }

(* The observation of one frame: the pitch bins that received a candidate,
   ascending, with their probabilities, and the voiced probability. *)
type observation = {bins: int array; probs: Float.Array.t; voicing: float}

let silent = {bins= [||]; probs= Float.Array.create 0; voicing= 0.}

(* Scratch space for {!observe}, one per worker range. *)
type scratch =
  { index: int array (* trough indices *)
  ; height: Float.Array.t (* trough values *)
  ; mass: Float.Array.t (* candidate probabilities *)
  ; slot: int array (* bin -> its candidate, or -1 *)
  ; found_bins: int array
  ; found_probs: Float.Array.t }

let scratch l n_bins =
  { index= Array.make l 0
  ; height= Float.Array.make l 0.
  ; mass= Float.Array.make l 0.
  ; slot= Array.make n_bins (-1)
  ; found_bins= Array.make l 0
  ; found_probs= Float.Array.make l 0. }

let clip01 x = if x < 0. then 0. else if x > 1. then 1. else x

(* [observe g m s c] is the observation of the frame whose cumulative mean
   normalized difference is [c], as librosa's pyin helper computes it: every
   threshold spreads its prior mass over the troughs below it by the Boltzmann
   law on their rank, the global minimum takes the mass of the thresholds no
   trough reaches times [no_trough_prob], and each trough left with a nonzero
   probability lands on the pitch bin of its refined frequency, a later
   (longer-period) candidate replacing an earlier one in the same bin and a
   candidate past the top bin dropped. The mass sums run in ascending threshold
   order and the voiced probability in ascending bin order, each from [0.]. *)
let observe g m s c =
  let l = Float.Array.length c in
  let nt = ref 0 in
  for i = 0 to l - 1 do
    if is_trough c l i then begin
      s.index.(!nt) <- i ;
      Float.Array.unsafe_set s.height !nt (Float.Array.unsafe_get c i) ;
      incr nt
    end
  done ;
  let nt = !nt in
  if nt = 0 then silent
  else begin
    Float.Array.fill s.mass 0 nt 0. ;
    let n_thresholds = Float.Array.length m.thresholds in
    for j = 0 to n_thresholds - 1 do
      let t = Float.Array.unsafe_get m.thresholds j in
      let count = ref 0 in
      for i = 0 to nt - 1 do
        if Float.Array.unsafe_get s.height i < t then incr count
      done ;
      if !count > 0 then begin
        let fact = Float.Array.unsafe_get m.scale !count
        and mass = Float.Array.unsafe_get m.beta j in
        let rank = ref 0 in
        for i = 0 to nt - 1 do
          if Float.Array.unsafe_get s.height i < t then begin
            let e = Float.Array.unsafe_get m.decay !rank in
            let prior = fact *. e in
            let term = clip01 prior *. mass in
            Float.Array.unsafe_set s.mass i
              (Float.Array.unsafe_get s.mass i +. term) ;
            incr rank
          end
        done
      end
    done ;
    (* the global minimum also takes the thresholds no trough falls below *)
    let gmin = argmin s.height nt in
    let lowest = Float.Array.unsafe_get s.height gmin in
    let above = ref 0 in
    for j = 0 to n_thresholds - 1 do
      if not (lowest < Float.Array.unsafe_get m.thresholds j) then incr above
    done ;
    let extra = m.no_trough *. Float.Array.unsafe_get m.below_sums !above in
    Float.Array.unsafe_set s.mass gmin
      (Float.Array.unsafe_get s.mass gmin +. extra) ;
    let found = ref 0 in
    for i = 0 to nt - 1 do
      let p = Float.Array.unsafe_get s.mass i in
      if p <> 0. then begin
        let f0 = frequency g c l s.index.(i) in
        let b =
          round_half_even (m.bins_per_octave *. Float.log2 (f0 /. g.fmin))
        in
        let b =
          if b < 0. then 0
          else if b > Float.of_int m.n_bins then m.n_bins
          else Float.to_int b
        in
        if b < m.n_bins then
          if s.slot.(b) >= 0 then
            Float.Array.unsafe_set s.found_probs s.slot.(b) p
          else begin
            s.slot.(b) <- !found ;
            s.found_bins.(!found) <- b ;
            Float.Array.unsafe_set s.found_probs !found p ;
            incr found
          end
      end
    done ;
    let found = !found in
    let order = Array.init found Fun.id in
    Array.sort (fun x y -> Int.compare s.found_bins.(x) s.found_bins.(y)) order ;
    let bins = Array.map (fun k -> s.found_bins.(k)) order in
    let probs =
      Float.Array.map_from_array
        (fun k -> Float.Array.unsafe_get s.found_probs k)
        order
    in
    Array.iter (fun b -> s.slot.(b) <- -1) bins ;
    let total = ref 0. in
    for k = 0 to found - 1 do
      total := !total +. Float.Array.unsafe_get probs k
    done ;
    {bins; probs; voicing= clip01 !total}
  end

(* {2 Transitions}

   librosa's transition matrix is the Kronecker product of the voicing switch
   [[1 - p; p]; [p; 1 - p]] (its off-diagonal computed as [(1 - (1 - p)) / 1])
   with the banded triangular pitch transition: row [i] is the symmetric
   triangle of [width] points centered on bin [i], cut to the [width / 2] bins
   on either side and normalised by its numpy pairwise sum over the whole row.
   Only the band is kept. *)

type transitions =
  { n: int (* pitch bins *)
  ; half: int (* [width / 2] *)
  ; band_lo: int array (* row [i] -> the first column of its band *)
  ; log_band: Float.Array.t array
        (* [(2 a + b) * n + i] -> the log transitions from plane [a], row [i] to
           plane [b], over the columns of the band from [band_lo i] *)
  ; log_eps: float (* the log of a zero transition *)
  ; exact: bool (* every transition is a predecessor *)
  ; preds: int array array (* state -> its predecessors, ascending *)
  ; pred_logs: Float.Array.t array (* their log transitions *) }

(* [triangle w] is scipy's symmetric triangular window of [w] points. *)
let triangle w =
  if w <= 1 then Float.Array.make (Int.max w 0) 1.
  else
    let half = (w + 1) / 2 in
    let rising =
      Array.init half (fun k ->
          let k = Float.of_int (k + 1) in
          if w mod 2 = 0 then ((2. *. k) -. 1.) /. Float.of_int w
          else 2. *. k /. (Float.of_int w +. 1.) )
    in
    Float.Array.init w (fun m ->
        if m < half then rising.(m) else rising.(w - 1 - m) )

let eps = Float.min_float

let transitions fn ~n ~width ~switch_prob ~transition_min_prob =
  let window = triangle width in
  let lpad = (n - width) / 2 in
  let half = width / 2 in
  let row = Float.Array.make n 0. in
  let band_lo = Array.init n (fun i -> Int.max 0 (i - half)) in
  let band =
    Array.init n (fun i ->
        Float.Array.fill row 0 n 0. ;
        let lo = band_lo.(i) and hi = Int.min n (i + half + 1) in
        for m = 0 to width - 1 do
          let col = (lpad + m + (n / 2) + i + 1) mod n in
          if col >= lo && col < hi then
            Float.Array.unsafe_set row col (Float.Array.unsafe_get window m)
        done ;
        let sum = pairwise row 0 n in
        Float.Array.init (hi - lo) (fun k ->
            Float.Array.unsafe_get row (lo + k) /. sum ) )
  in
  let stay = 1. -. switch_prob in
  let switch = (1. -. stay) /. 1. in
  let log_eps = Float.log eps in
  let log_band =
    Array.init (4 * n) (fun ab ->
        let a = ab / n / 2 and b = ab / n mod 2 and i = ab mod n in
        let ts = if a = b then stay else switch in
        Float.Array.map
          (fun t ->
            let p = ts *. t in
            Float.log (p +. eps) )
          band.(i) )
  in
  let threshold =
    if transition_min_prob > 0. then Float.log (transition_min_prob +. eps)
    else Float.neg_infinity
  in
  let exact = log_eps >= threshold in
  let states = 2 * n in
  let preds = Array.make (if exact then 0 else states) [||]
  and pred_logs =
    Array.make (if exact then 0 else states) (Float.Array.create 0)
  in
  if not exact then
    for target = 0 to states - 1 do
      let b = target / n and j = target mod n in
      let ks = ref [] and ls = ref [] in
      for a = 1 downto 0 do
        for i = Int.min (n - 1) (j + half) downto Int.max 0 (j - half) do
          let v =
            Float.Array.unsafe_get
              log_band.((((2 * a) + b) * n) + i)
              (j - band_lo.(i))
          in
          if v >= threshold then begin
            ks := ((a * n) + i) :: !ks ;
            ls := v :: !ls
          end
        done
      done ;
      if !ks = [] then
        invalid_arg
          (Printf.sprintf
             "%s: cannot decode with transition_min_prob = %g (no transition \
              into state %d reaches it; lower it)"
             fn transition_min_prob target ) ;
      preds.(target) <- Array.of_list !ks ;
      pred_logs.(target) <- Float.Array.of_list !ls
    done ;
  {n; half; band_lo; log_band; log_eps; exact; preds; pred_logs}

(* [pointers frames states] is a store for [frames * states] back-pointers, two
   bytes each while the states fit, as its setter and getter. *)
let pointers frames states =
  if states <= 65536 then
    let a =
      A1.create Bigarray.int16_unsigned Bigarray.c_layout (frames * states)
    in
    ((fun i v -> A1.unsafe_set a i v), fun i -> A1.unsafe_get a i)
  else
    let a = A1.create Bigarray.int32 Bigarray.c_layout (frames * states) in
    ( (fun i v -> A1.unsafe_set a i (Int32.of_int v))
    , fun i -> Int32.to_int (A1.unsafe_get a i) )

(* [viterbi tr obs] is the most likely state sequence of the observations [obs],
   decoded exactly as librosa's Viterbi kernel: log observations [log (p +
   tiny)], the uniform initial distribution over every state, predecessors
   scanned in ascending order with the first maximum kept, the first maximum of
   the last frame, then the back-pointers. *)
let viterbi tr (obs : observation array) =
  let n = tr.n in
  let states = 2 * n in
  let frames = Array.length obs in
  let log_obs = Float.Array.make states 0. in
  let load t =
    let o = obs.(t) in
    Float.Array.fill log_obs 0 n tr.log_eps ;
    Array.iteri
      (fun k b ->
        Float.Array.unsafe_set log_obs b
          (Float.log (Float.Array.unsafe_get o.probs k +. eps)) )
      o.bins ;
    let unvoiced = Float.log (((1. -. o.voicing) /. Float.of_int n) +. eps) in
    Float.Array.fill log_obs n n unvoiced
  in
  let log_init = Float.log ((1. /. Float.of_int states) +. eps) in
  let value = ref (Float.Array.make states 0.)
  and next = ref (Float.Array.make states 0.) in
  load 0 ;
  for s = 0 to states - 1 do
    Float.Array.unsafe_set !value s
      (Float.Array.unsafe_get log_obs s +. log_init)
  done ;
  let set, get = pointers (Int.max 0 (frames - 1)) states in
  for t = 1 to frames - 1 do
    load t ;
    let v = !value and nv = !next in
    let row = (t - 1) * states in
    for j = 0 to states - 1 do
      let best = ref Float.neg_infinity and arg = ref 0 in
      if tr.exact then begin
        (* every state is a predecessor: the band's own transitions, and the log
           of a zero transition everywhere else *)
        let b = j / n and jj = j mod n in
        let ilo = Int.max 0 (jj - tr.half)
        and ihi = Int.min (n - 1) (jj + tr.half) in
        for a = 0 to 1 do
          let base = a * n and bands = ((2 * a) + b) * n in
          for i = 0 to n - 1 do
            let lt =
              if i < ilo || i > ihi then tr.log_eps
              else
                Float.Array.unsafe_get
                  tr.log_band.(bands + i)
                  (jj - tr.band_lo.(i))
            in
            let cost = Float.Array.unsafe_get v (base + i) +. lt in
            if cost > !best then begin
              best := cost ;
              arg := base + i
            end
          done
        done
      end
      else begin
        let ks = tr.preds.(j) and ls = tr.pred_logs.(j) in
        for q = 0 to Array.length ks - 1 do
          let k = Array.unsafe_get ks q in
          let cost =
            Float.Array.unsafe_get v k +. Float.Array.unsafe_get ls q
          in
          if cost > !best then begin
            best := cost ;
            arg := k
          end
        done
      end ;
      set (row + j) !arg ;
      Float.Array.unsafe_set nv j (Float.Array.unsafe_get log_obs j +. !best)
    done ;
    value := nv ;
    next := v
  done ;
  let path = Array.make frames 0 in
  if frames > 0 then begin
    let v = !value in
    let best = ref 0 in
    for s = 1 to states - 1 do
      if Float.Array.unsafe_get v s > Float.Array.unsafe_get v !best then
        best := s
    done ;
    path.(frames - 1) <- !best ;
    for t = frames - 2 downto 0 do
      path.(t) <- get ((t * states) + path.(t + 1))
    done
  end ;
  path

let check_probability fn name v =
  if not (v >= 0. && v <= 1.) then
    invalid_arg
      (Printf.sprintf "%s: cannot use %s = %g (%s must lie in [0, 1])" fn name v
         name )

let check_positive fn name v =
  if not (Float.is_finite v && v > 0.) then
    invalid_arg
      (Printf.sprintf "%s: cannot use %s = %g (%s must be finite and positive)"
         fn name v name )

let check_non_negative fn name v =
  if not (Float.is_finite v && v >= 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use %s = %g (%s must be finite and non-negative)" fn name v
         name )

let pyin ?(frame_length = 2048) ?hop ?(n_thresholds = 100)
    ?(beta_parameters = (2., 18.)) ?(boltzmann_parameter = 2.)
    ?(resolution = 0.1) ?(max_transition_rate = 35.92) ?(switch_prob = 0.01)
    ?(no_trough_prob = 0.01) ?(fill = `Value Float.nan) ?(alignment = `Centered)
    ?(pad = `Constant 0.) ?(transition_min_prob = 1e-4) ~fmin ~fmax ~sample_rate
    x =
  let fn = "pyin" in
  let g =
    geometry fn ~frame_length ~hop ~center:(centered alignment) ~pad ~fmin ~fmax
      ~sample_rate
  in
  if n_thresholds < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use %d thresholds (n_thresholds must be at least 1)" fn
         n_thresholds ) ;
  let a, b = beta_parameters in
  check_positive fn "the first beta parameter" a ;
  check_positive fn "the second beta parameter" b ;
  check_positive fn "boltzmann_parameter" boltzmann_parameter ;
  check_positive fn "resolution" resolution ;
  check_non_negative fn "max_transition_rate" max_transition_rate ;
  check_probability fn "switch_prob" switch_prob ;
  check_probability fn "no_trough_prob" no_trough_prob ;
  check_non_negative fn "transition_min_prob" transition_min_prob ;
  let bins_per_semitone = Float.to_int (Float.ceil (1. /. resolution)) in
  let n_bins =
    Float.to_int
      (Float.floor
         (Float.of_int (12 * bins_per_semitone) *. Float.log2 (fmax /. fmin)) )
    + 1
  in
  if n_bins < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot decode %d pitch bin between fmin = %g Hz and fmax = %g Hz \
          at a resolution of %g semitones (widen the range or refine the \
          resolution to reach two)"
         fn n_bins fmin fmax resolution ) ;
  let per_frame =
    Float.to_int
      (round_half_even
         ( max_transition_rate *. 12. *. Float.of_int g.hop
         /. Float.of_int sample_rate ) )
  in
  let width = (per_frame * bins_per_semitone) + 1 in
  if width > n_bins then
    invalid_arg
      (Printf.sprintf
         "%s: cannot spread pitch transitions over %d bins of %d (lower \
          max_transition_rate or hop, or widen the pitch range)"
         fn width n_bins ) ;
  check_rank fn x ;
  check_finite fn x ;
  let tr = transitions fn ~n:n_bins ~width ~switch_prob ~transition_min_prob in
  let m =
    model g ~n_thresholds ~beta_parameters ~boltzmann_parameter ~no_trough_prob
      ~bins_per_semitone ~n_bins
  in
  let lead = leading_shape x in
  let planes = Array.fold_left ( * ) 1 lead in
  let count = count_frames g (last_dim x) in
  let obs = Array.make (planes * count) silent in
  let l = g.max_period - g.min_period + 1 in
  each_frame g ~left:(border g)
    ~ext:(extension g (Nx.dtype x))
    ~count
    ~local:(fun () -> scratch l n_bins)
    x
    (fun st o p c -> obs.((o * count) + p) <- observe g m st c) ;
  let freqs =
    Float.Array.init n_bins (fun k ->
        fmin
        *. Float.pow 2. (Float.of_int k /. Float.of_int (12 * bins_per_semitone)) )
  in
  let f0 = Float.Array.make (planes * count) 0.
  and voiced = Array.make (planes * count) false
  and probability = Float.Array.make (planes * count) 0. in
  if count > 0 then
    Par.run ~chunks:planes (fun o ->
        let path = viterbi tr (Array.sub obs (o * count) count) in
        Array.iteri
          (fun p s ->
            let i = (o * count) + p in
            let is_voiced = s < n_bins in
            let pitch = Float.Array.get freqs (s mod n_bins) in
            Float.Array.set f0 i
              ( match fill with
              | `Value v when not is_voiced ->
                  v
              | `Value _ | `Decoded ->
                  pitch ) ;
            voiced.(i) <- is_voiced ;
            Float.Array.set probability i obs.(i).voicing )
          path ) ;
  let shape = Array.append lead [|count|] in
  let dtype = Nx.dtype x in
  { f0= tensor dtype shape f0
  ; voiced= Nx.create Nx.bool shape voiced
  ; voiced_probability= tensor dtype shape probability }
