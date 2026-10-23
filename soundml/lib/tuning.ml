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

(* Pitch tracking and tuning estimation, as librosa.piptrack, pitch_tuning and
   estimate_tuning compute them.

   piptrack reads one spectrogram frame at a time: the frame's reference (its
   maximum times the threshold, or a fixed magnitude), the bins above it, their
   local maxima along the bin axis, and at each maximum the vertex of the
   parabola through it and its two neighbours. Nothing crosses a frame, so the
   stage is the flat function on each chunk and agrees with it on every
   chunking. Each frame is widened to double into a scratch column, analysed in
   librosa's operation order, and its pitches and magnitudes rounded once into
   the dtype on the way back; frames run across domains, each writing its own
   column, so the split changes no bit.

   The tuning is a histogram mode: every frequency's distance to the nearest
   step of the equal-tempered ladder through 27.5 Hz, in fractions of a step,
   counted in [ceil (1 / resolution)] bins over [-0.5, 0.5], and the left edge
   of the fullest bin returned. Ties go to the lowest bin, as numpy's argmax
   breaks them. estimate_tuning feeds it the pitches piptrack finds, keeping
   those whose magnitude reaches the median of all detected magnitudes. Both are
   whole-signal reductions, computed in double whatever the input dtype; the
   detections they read are never rounded. *)

type reference = [`Maximum | `Value of float]

type params =
  { sample_rate: int
  ; fmin: float (* raised to 0 Hz *)
  ; fmax: float (* lowered to the Nyquist frequency *)
  ; threshold: float
  ; reference: reference
  ; fft_size: int option }

let params fn ~fmin ~fmax ~threshold ~reference ~fft_size ~sample_rate =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot track pitches at %d Hz (sample_rate must be at least 1)" fn
         sample_rate ) ;
  if Float.is_nan fmin then
    invalid_arg
      (fn ^ ": cannot bound the band at fmin = nan (fmin must be a number)") ;
  if Float.is_nan fmax then
    invalid_arg
      (fn ^ ": cannot bound the band at fmax = nan (fmax must be a number)") ;
  if not (Float.is_finite threshold && threshold >= 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use threshold = %g (threshold must be finite and \
          non-negative)"
         fn threshold ) ;
  ( match reference with
  | `Value v when not (Float.is_finite v && v >= 0.) ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot compare magnitudes with `Value %g (a fixed reference \
            must be finite and non-negative)"
           fn v )
  | `Value _ | `Maximum ->
      () ) ;
  Option.iter
    (fun n ->
      if n < 2 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot track pitches with an FFT of size %d (fft_size must \
              be at least 2)"
             fn n ) )
    fft_size ;
  { sample_rate
  ; fmin= Float.max fmin 0.
  ; fmax= Float.min fmax (Float.of_int sample_rate /. 2.)
  ; threshold
  ; reference
  ; fft_size }

let check_resolution fn resolution =
  if not (resolution >= 1e-9 && resolution < 1.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot estimate the tuning at a resolution of %g (resolution \
          must lie in [1e-9, 1))"
         fn resolution )

let check_bins_per_octave fn bins_per_octave =
  if bins_per_octave < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot divide the octave into %d bins (bins_per_octave must be \
          at least 1)"
         fn bins_per_octave )

(* [geometry fn p s] validates the spectrogram [s] against [p] and is its
   leading block count, bin count, frame count and FFT size: [p.fft_size] when
   given, which must imply the bin count, or the even size [2 * (bins - 1)] the
   bin count implies. *)
let geometry fn p s =
  let nd = Nx.ndim s in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-%d tensor (a spectrogram is [...; bins; \
          frames])"
         fn nd ) ;
  let shape = Nx.shape s in
  let bins = shape.(nd - 2) and frames = shape.(nd - 1) in
  if bins < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot track pitches in a %d-bin spectrogram (a spectrogram has \
          at least 2 bins)"
         fn bins ) ;
  let fft_size =
    match p.fft_size with
    | None ->
        2 * (bins - 1)
    | Some n ->
        if (n / 2) + 1 <> bins then
          invalid_arg
            (Printf.sprintf
               "%s: cannot track %d bins with an FFT of size %d (fft_size / 2 \
                + 1 must be the bin count)"
               fn bins n ) ;
        n
  in
  let blocks = Array.fold_left ( * ) 1 (Array.sub shape 0 (nd - 2)) in
  (blocks, bins, frames, fft_size)

let negative fn =
  invalid_arg
    (Printf.sprintf
       "%s: cannot analyse a spectrogram with negative or NaN values (a \
        magnitude spectrogram is non-negative)"
       fn )

(* [band p ~fft_size ~bins] is the bins [\[lo, hi)] whose frequency [f] has
   [fmin <= f < fmax], each frequency evaluated as numpy's rfftfreq evaluates
   it, [k] times the reciprocal of [fft_size / sample_rate]: the mask is a
   discrete decision, so its arithmetic is the reference's. The frequencies do
   not decrease, so the bins below [fmin] and those below [fmax] are both
   prefixes and the band is the range between their ends — empty when [fmax <=
   fmin]. *)
let band p ~fft_size ~bins =
  let step =
    1.0 /. (Float.of_int fft_size *. (1.0 /. Float.of_int p.sample_rate))
  in
  let below v =
    let c = ref 0 in
    for k = 0 to bins - 1 do
      if Float.of_int k *. step < v then incr c
    done ;
    !c
  in
  (below p.fmin, below p.fmax)

(* {1 One frame} *)

(* The column of one frame, widened to double, and its detections: bin [ks.(i)]
   at pitch [ps.(i)] and magnitude [ms.(i)], in ascending bin order. *)
type scratch =
  {x: Float.Array.t; ks: int array; ps: Float.Array.t; ms: Float.Array.t}

let scratch bins =
  { x= Float.Array.create bins
  ; ks= Array.make bins 0
  ; ps= Float.Array.create bins
  ; ms= Float.Array.create bins }

(* [reference_value p ~bins x] is the magnitude the frame [x] must exceed:
   [threshold] times its maximum over every bin, or the fixed magnitude. *)
let reference_value p ~bins x =
  match p.reference with
  | `Value v ->
      v
  | `Maximum ->
      let m = ref (Float.Array.unsafe_get x 0) in
      for k = 1 to bins - 1 do
        let v = Float.Array.unsafe_get x k in
        if v > !m then m := v
      done ;
      p.threshold *. !m

(* [peaks p ~fft_size ~bins ~lo ~hi sc] records the detections of the frame
   [sc.x] in [sc] and is their count. A bin of the band is detected when the
   frame with every bin at or below the reference zeroed has a local maximum
   there — above its lower neighbour, at least its upper one, the neighbours
   read even outside the band; never bin 0; the top bin against its lower
   neighbour alone. At an interior maximum the parabola through the
   unthresholded bins [xm], [x0], [xp] has its vertex at [shift = -b / a] bins,
   [a = (xp + xm) - 2 x0] and [b = (xp - xm) / 2] (no shift when [|b| >= |a|]),
   and the magnitude there is [x0 + (0.5 b) shift] — the product let-bound, so
   that no platform fuses the sum; the top bin has no shift. The pitch is [((k +
   shift) * sample_rate) / fft_size], the direct formula librosa uses, not the
   reciprocal grid of the band. *)
let peaks p ~fft_size ~bins ~lo ~hi sc =
  let x = sc.x in
  let r = reference_value p ~bins x in
  let sr = Float.of_int p.sample_rate and n = Float.of_int fft_size in
  let count = ref 0 in
  for k = Int.max lo 1 to hi - 1 do
    let x0 = Float.Array.unsafe_get x k in
    if x0 > r then begin
      let xm = Float.Array.unsafe_get x (k - 1) in
      let below = if xm > r then xm else 0. in
      if x0 > below then
        if k = bins - 1 then begin
          Array.unsafe_set sc.ks !count k ;
          Float.Array.unsafe_set sc.ps !count (Float.of_int k *. sr /. n) ;
          Float.Array.unsafe_set sc.ms !count x0 ;
          incr count
        end
        else
          let xp = Float.Array.unsafe_get x (k + 1) in
          let above = if xp > r then xp else 0. in
          if x0 >= above then begin
            let a = xp +. xm -. (2. *. x0) in
            let b = (xp -. xm) /. 2. in
            let shift = if Float.abs b >= Float.abs a then 0. else -.b /. a in
            let half = 0.5 *. b in
            let dskew = half *. shift in
            Array.unsafe_set sc.ks !count k ;
            Float.Array.unsafe_set sc.ps !count
              ((Float.of_int k +. shift) *. sr /. n) ;
            Float.Array.unsafe_set sc.ms !count (x0 +. dskew) ;
            incr count
          end
    end
  done ;
  !count

(* [GATHER src ~base ~stride ~bins x] widens the column of [src] that starts at
   [base] and advances by [stride] into [x], and is whether every value is
   non-negative; [SCATTER] writes a frame's detections back, zero elsewhere. One
   copy per element type, so the loads and stores stay unboxed. *)
let gather32 (src : Store.f32) ~base ~stride ~bins x =
  let ok = ref true in
  for k = 0 to bins - 1 do
    let v = Bigarray.Array1.unsafe_get src (base + (k * stride)) in
    if not (v >= 0.) then ok := false ;
    Float.Array.unsafe_set x k v
  done ;
  !ok

let gather64 (src : Store.f64) ~base ~stride ~bins x =
  let ok = ref true in
  for k = 0 to bins - 1 do
    let v = Bigarray.Array1.unsafe_get src (base + (k * stride)) in
    if not (v >= 0.) then ok := false ;
    Float.Array.unsafe_set x k v
  done ;
  !ok

let scatter32 (dp : Store.f32) (dm : Store.f32) ~base ~stride ~bins sc count =
  let next = ref 0 in
  for k = 0 to bins - 1 do
    let o = base + (k * stride) in
    if !next < count && Array.unsafe_get sc.ks !next = k then begin
      Bigarray.Array1.unsafe_set dp o (Float.Array.unsafe_get sc.ps !next) ;
      Bigarray.Array1.unsafe_set dm o (Float.Array.unsafe_get sc.ms !next) ;
      incr next
    end
    else begin
      Bigarray.Array1.unsafe_set dp o 0. ;
      Bigarray.Array1.unsafe_set dm o 0.
    end
  done

let scatter64 (dp : Store.f64) (dm : Store.f64) ~base ~stride ~bins sc count =
  let next = ref 0 in
  for k = 0 to bins - 1 do
    let o = base + (k * stride) in
    if !next < count && Array.unsafe_get sc.ks !next = k then begin
      Bigarray.Array1.unsafe_set dp o (Float.Array.unsafe_get sc.ps !next) ;
      Bigarray.Array1.unsafe_set dm o (Float.Array.unsafe_get sc.ms !next) ;
      incr next
    end
    else begin
      Bigarray.Array1.unsafe_set dp o 0. ;
      Bigarray.Array1.unsafe_set dm o 0.
    end
  done

(* [columns ~blocks ~bins ~frames f] calls [f sc lo hi] on consecutive ranges of
   the [blocks * frames] columns, across domains, each range with a scratch of
   its own; column [c] starts at element [base ~bins ~frames c]. *)
let columns ~blocks ~bins ~frames f =
  let grain = Int.max 1 (8_192 / bins) in
  Par.ranges ~n:(blocks * frames) ~grain (fun lo hi -> f (scratch bins) lo hi)

let base ~bins ~frames c = (c / frames * bins * frames) + (c mod frames)

(* {1 Pitch tracking} *)

let rec track : type a.
    string -> params -> (float, a) Nx.t -> (float, a) Nx.t * (float, a) Nx.t =
 fun fn p s ->
  let blocks, bins, frames, fft_size = geometry fn p s in
  let lo, hi = band p ~fft_size ~bins in
  let valid = Atomic.make true in
  let each gather scatter sc c0 c1 =
    for c = c0 to c1 - 1 do
      let base = base ~bins ~frames c in
      if not (gather ~base sc.x) then Atomic.set valid false ;
      scatter ~base sc (peaks p ~fft_size ~bins ~lo ~hi sc)
    done
  in
  let result : (float, a) Nx.t * (float, a) Nx.t =
    match Nx.dtype s with
    | Nx.Float32 ->
        Store.make32_pair (Nx.shape s) (fun dp dm ->
            Store.read32 s (fun src ->
                columns ~blocks ~bins ~frames
                  (each
                     (gather32 src ~stride:frames ~bins)
                     (scatter32 dp dm ~stride:frames ~bins) ) ) )
    | Nx.Float64 ->
        Store.make64_pair (Nx.shape s) (fun dp dm ->
            Store.read64 s (fun src ->
                columns ~blocks ~bins ~frames
                  (each
                     (gather64 src ~stride:frames ~bins)
                     (scatter64 dp dm ~stride:frames ~bins) ) ) )
    | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
        let dtype = Nx.dtype s in
        let pitches, mags = track fn p (Nx.cast Nx.float64 s) in
        (Nx.cast dtype pitches, Nx.cast dtype mags)
  in
  if not (Atomic.get valid) then negative fn ;
  result

let piptrack ?(fmin = 150.) ?(fmax = 4000.) ?(threshold = 0.1)
    ?(reference = `Maximum) ?fft_size ~sample_rate s =
  let fn = "piptrack" in
  track fn
    (params fn ~fmin ~fmax ~threshold ~reference ~fft_size ~sample_rate)
    s

(* The stage is the flat function on each chunk, its parameters validated once
   when it is built; the chunk's shape and values are checked chunk by chunk,
   since the stream format carries no bin count. *)
let stage ?(fmin = 150.) ?(fmax = 4000.) ?(threshold = 0.1)
    ?(reference = `Maximum) ?fft_size ~sample_rate () =
  let fn = "piptrack_stage" in
  let p = params fn ~fmin ~fmax ~threshold ~reference ~fft_size ~sample_rate in
  Pipeline.stateless (fun s -> track fn p s)

(* {1 Tuning} *)

(* A growable array of doubles. *)
type floats = {mutable data: Float.Array.t; mutable length: int}

let floats () = {data= Float.Array.create 64; length= 0}

let push g v =
  if g.length = Float.Array.length g.data then begin
    let d = Float.Array.create (2 * g.length) in
    Float.Array.blit g.data 0 d 0 g.length ;
    g.data <- d
  end ;
  Float.Array.unsafe_set g.data g.length v ;
  g.length <- g.length + 1

(* [residual ~bins_per_octave f] is the distance of the frequency [f] to the
   nearest step of the [bins_per_octave]-tone ladder through 27.5 Hz, in
   fractions of a step, in [\[-0.5, 0.5)] — librosa's [mod (bins_per_octave *
   log2 (f / 27.5), 1)], numpy's [mod] adding the divisor to a negative
   remainder and making a zero one positive, folded down by one at [0.5] and
   above. An infinite frequency has a NaN residual. *)
let residual ~bins_per_octave f =
  let o = Float.log2 (f /. 27.5) in
  let t = Float.of_int bins_per_octave *. o in
  let m = Float.rem t 1.0 in
  let m = if m < 0. then m +. 1.0 else if m = 0. then 0. else m in
  if m >= 0.5 then m -. 1.0 else m

(* [tuning ~resolution ~bins_per_octave f] is librosa's pitch_tuning over the
   first [f.length] values of [f.data]: the positive ones (NaN and non-positive
   frequencies are dropped, and none left gives [0.]) have their residuals
   counted in the [div = ceil (1 / resolution)] bins between numpy's [linspace
   (-0.5, 0.5, div + 1)] edges — edge [i] is [i * (1 / div)] rounded and then
   [-0.5] added, rounded again, and the last edge is [0.5] — bin [i] holding
   [\[e i, e (i + 1))], the last bin closed. The result is the left edge of the
   first bin with the largest count; a set whose residuals are all NaN has every
   count zero and gives the first edge, [-0.5]. The bins are counted by sorting
   the residuals' bin indices, so no [div]-sized array is built whatever the
   resolution. *)
let tuning ~resolution ~bins_per_octave (f : floats) =
  let kept = ref 0 in
  for i = 0 to f.length - 1 do
    if Float.Array.unsafe_get f.data i > 0. then incr kept
  done ;
  if !kept = 0 then 0.
  else begin
    let div = Float.to_int (Float.ceil (1.0 /. resolution)) in
    let step = 1.0 /. Float.of_int div in
    let edge i =
      if i = div then 0.5
      else
        let p = Float.of_int i *. step in
        p +. -0.5
    in
    let indices = Array.make !kept 0 in
    let binned = ref 0 in
    for i = 0 to f.length - 1 do
      let v = Float.Array.unsafe_get f.data i in
      if v > 0. then begin
        let r = residual ~bins_per_octave v in
        if not (Float.is_nan r) then begin
          (* a guess within a bin or so of the answer, then the edges themselves
             decide *)
          let j =
            ref
              (Int.max 0
                 (Int.min (div - 1)
                    (Float.to_int (Float.floor ((r +. 0.5) /. step))) ) )
          in
          while !j > 0 && r < edge !j do
            decr j
          done ;
          while !j < div - 1 && r >= edge (!j + 1) do
            incr j
          done ;
          indices.(!binned) <- !j ;
          incr binned
        end
      end
    done ;
    let indices = Array.sub indices 0 !binned in
    Array.sort Int.compare indices ;
    let best = ref 0 and best_count = ref 0 in
    let i = ref 0 in
    while !i < !binned do
      let b = indices.(!i) in
      let j = ref !i in
      while !j < !binned && indices.(!j) = b do
        incr j
      done ;
      if !j - !i > !best_count then begin
        best := b ;
        best_count := !j - !i
      end ;
      i := !j
    done ;
    edge !best
  end

let pitch_tuning ?(resolution = 0.01) ?(bins_per_octave = 12) frequencies =
  let fn = "pitch_tuning" in
  check_resolution fn resolution ;
  check_bins_per_octave fn bins_per_octave ;
  let values = Nx.to_array (Nx.cast Nx.float64 frequencies) in
  tuning ~resolution ~bins_per_octave
    {data= Float.Array.map_from_array Fun.id values; length= Array.length values}

(* [detections fn p s] is every detection of {!track} over [s] in double
   precision, unrounded: the pitches and the magnitudes, in an order that
   depends on the split across domains — the median and the histogram that
   consume them do not. Only the detections are kept, never a spectrogram-sized
   output. *)
let detections (type a) fn p (s : (float, a) Nx.t) =
  let blocks, bins, frames, fft_size = geometry fn p s in
  let lo, hi = band p ~fft_size ~bins in
  let valid = Atomic.make true in
  let lock = Mutex.create () in
  let parts = ref [] in
  let collect gather sc c0 c1 =
    let ps = floats () and ms = floats () in
    for c = c0 to c1 - 1 do
      if not (gather ~base:(base ~bins ~frames c) sc.x) then
        Atomic.set valid false ;
      let count = peaks p ~fft_size ~bins ~lo ~hi sc in
      for i = 0 to count - 1 do
        push ps (Float.Array.unsafe_get sc.ps i) ;
        push ms (Float.Array.unsafe_get sc.ms i)
      done
    done ;
    Mutex.protect lock (fun () -> parts := (ps, ms) :: !parts)
  in
  if Nx.numel s > 0 then
    begin match Nx.dtype s with
    | Nx.Float32 ->
        Store.read32 s (fun src ->
            columns ~blocks ~bins ~frames
              (collect (gather32 src ~stride:frames ~bins)) )
    | Nx.Float64 ->
        Store.read64 s (fun src ->
            columns ~blocks ~bins ~frames
              (collect (gather64 src ~stride:frames ~bins)) )
    | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
        Store.read64 (Nx.cast Nx.float64 s) (fun src ->
            columns ~blocks ~bins ~frames
              (collect (gather64 src ~stride:frames ~bins)) )
    end ;
  if not (Atomic.get valid) then negative fn ;
  let ps = floats () and ms = floats () in
  List.iter
    (fun (p, m) ->
      for i = 0 to p.length - 1 do
        push ps (Float.Array.unsafe_get p.data i) ;
        push ms (Float.Array.unsafe_get m.data i)
      done )
    !parts ;
  (ps, ms)

(* [median v] is numpy's median of the values [v]: the middle value, or the mean
   of the two middle values, their sum halved. *)
let median (v : Float.Array.t) =
  let v = Float.Array.copy v in
  Float.Array.sort Float.compare v ;
  let n = Float.Array.length v in
  if n mod 2 = 1 then Float.Array.get v (n / 2)
  else (Float.Array.get v ((n / 2) - 1) +. Float.Array.get v (n / 2)) /. 2.

(* librosa's estimate_tuning over the detections: the magnitudes of the positive
   pitches (every detection's, the pitch of a peak being positive, but the test
   is librosa's) have their median taken, and the pitches whose magnitude
   reaches it go to the histogram. *)
let estimate fn p ~resolution ~bins_per_octave s =
  let ps, ms = detections fn p s in
  let positive = floats () in
  for i = 0 to ps.length - 1 do
    if Float.Array.unsafe_get ps.data i > 0. then
      push positive (Float.Array.unsafe_get ms.data i)
  done ;
  let threshold =
    if positive.length = 0 then 0.
    else median (Float.Array.sub positive.data 0 positive.length)
  in
  let chosen = floats () in
  for i = 0 to ps.length - 1 do
    let pitch = Float.Array.unsafe_get ps.data i in
    if pitch > 0. && Float.Array.unsafe_get ms.data i >= threshold then
      push chosen pitch
  done ;
  tuning ~resolution ~bins_per_octave chosen

let estimate_tuning_of_spectrogram ?(resolution = 0.01) ?(bins_per_octave = 12)
    ?(fmin = 150.) ?(fmax = 4000.) ?(threshold = 0.1) ?(reference = `Maximum)
    ?fft_size ~sample_rate s =
  let fn = "estimate_tuning_of_spectrogram" in
  check_resolution fn resolution ;
  check_bins_per_octave fn bins_per_octave ;
  let p = params fn ~fmin ~fmax ~threshold ~reference ~fft_size ~sample_rate in
  estimate fn p ~resolution ~bins_per_octave s

(* The analysis is double precision whatever the dtype of [x]: the audio is
   widened, exactly, before the transform, so a float32 clip is analysed as the
   float64 clip of the same samples. *)
let estimate_tuning c ?(resolution = 0.01) ?(bins_per_octave = 12)
    ?(fmin = 150.) ?(fmax = 4000.) ?(threshold = 0.1) ?(reference = `Maximum)
    ~sample_rate x =
  let fn = "estimate_tuning" in
  check_resolution fn resolution ;
  check_bins_per_octave fn bins_per_octave ;
  let p =
    params fn ~fmin ~fmax ~threshold ~reference
      ~fft_size:(Some (Stft.Config.fft_size c))
      ~sample_rate
  in
  if Nx.ndim x < 1 then
    invalid_arg
      (fn ^ ": cannot analyse a rank-zero tensor (the time axis must exist)") ;
  estimate fn p ~resolution ~bins_per_octave
    (Stft.power_spectrum ~power:1. c (Nx.cast Nx.float64 x))
