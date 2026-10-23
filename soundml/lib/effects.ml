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

(* Time-scale and pitch modification by the phase vocoder.

   The analysis is an STFT at hop [H]; the synthesis reads the same frames back
   on a resampled time axis and hands them to an inverse STFT at the same hop.
   Output frame [i] sits at fractional analysis position [t_i = i * r]: its
   magnitudes are the linear interpolation of the two analysis frames it lies
   between, and its phase is an accumulator advanced, once per output frame, by
   the instantaneous frequency measured between those two frames.

   Writing [X] for the analysis spectrum, [i0 = floor t_i], [a = t_i - i0], [N]
   for the transform size and [w_k = 2 pi k / N] for the centre frequency of bin
   [k] in radians per sample,

   mag_i[k] = (1 - a) |X[k, i0]| + a |X[k, i0 + 1]|

   D_i[k] = pv (arg X[k, i0 + 1] - arg X[k, i0] - w_k H)

   phi_0 = arg X[., 0] phi_{i+1} = phi_i + (w_k H + D_i)

   Y[k, i] = mag_i[k] (cos phi_i[k] + j sin phi_i[k])

   over the [M] analysis frames but the last; over the final interval, [i0 = M -
   1], the magnitudes extrapolate the line through the last two frames and the
   accumulator holds, as librosa 1.0 does by clamping the next frame to the last
   one:

   mag_i[k] = (1 + a) |X[k, M - 1]| - a |X[k, M - 2]| phi_{i+1} = phi_i

   and a single analysis frame is held as it is.

   with [pv x = x - 2 pi round (x / 2 pi)] the principal value, ties to even.
   [w_k H + D_i] is the heterodyned phase increment of Flanagan & Golden (1966)
   and Portnoff (1976): the deviation of the measured inter-frame phase advance
   from the one the bin centre predicts, unwrapped into (-pi, pi], is the
   estimate of how far the partial in that bin sits from the bin centre, and
   adding it back to the expected advance propagates that estimate at the
   synthesis hop. See Dolson (1986) for the tutorial derivation and Ellis (2002)
   for this interpolating formulation.

   Identity phase locking (Laroche & Dolson 1999) replaces the per-bin
   accumulator of the bins around a spectral peak by the peak's own: a bin in a
   peak's region is given the peak's accumulated phase plus the phase difference
   the two bins have in the analysis frame, which is the relationship a single
   windowed sinusoid imposes on the bins its main lobe covers. Propagating the
   bins of one partial independently lets that relationship drift, which is the
   mechanism behind the loss of waveform shape the classical algorithm is known
   for.

   The interior is float64 throughout and rounds once, at the boundary, into the
   caller's dtype. *)

let two_pi = 2. *. Float.pi

(* [round_half_even x] is [x] rounded to the nearest integer with ties to even.
   Adding and subtracting 2^52 in round-to-nearest is exactly that rounding for
   every argument of smaller magnitude, and every argument of larger magnitude
   is already an integer. *)
let round_half_even =
  let magic = 0x1p52 in
  fun x ->
    if Float.abs x >= magic then x
    else if x >= 0. then x +. magic -. magic
    else x -. magic +. magic

(* [advance c] is the phase [2 pi k H / N] a bin centre advances over one hop,
   one entry per bin. *)
let advance c =
  let hop = Float.of_int (Stft.Config.hop c) in
  let step = 1. /. (Float.of_int (Stft.Config.fft_size c) *. (1. /. two_pi)) in
  Array.init (Stft.Config.bins c) (fun k -> hop *. (Float.of_int k *. step))

(* [out_frames ~frames ~rate] is the number of output frames: the positions [i *
   rate] strictly below [frames]. *)
let out_frames ~frames ~rate =
  if frames = 0 then 0
  else Float.to_int (Float.ceil (Float.of_int frames /. rate))

(* {1 Validation} *)

let check_rate op rate =
  if not (Float.is_finite rate && rate > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot stretch by a rate of %g (the rate must be finite and \
          positive)"
         op rate )

let check_spectrum op (c : Stft.Config.t) z =
  if Nx.ndim z < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot vocode a rank-%d tensor (the bin and frame axes must \
          exist)"
         op (Nx.ndim z) ) ;
  let bins = Nx.dim (Nx.ndim z - 2) z in
  if bins <> Stft.Config.bins c then
    invalid_arg
      (Printf.sprintf
         "%s: cannot vocode %d frequency bins of a %d-point transform (the bin \
          axis must hold fft_size / 2 + 1 = %d values)"
         op bins (Stft.Config.fft_size c) (Stft.Config.bins c) )

let check_rank op t =
  if Nx.ndim t < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot process a rank-zero tensor (the time axis must exist)" op )

(* {1 The core recurrence}

   With phase locking, one traversal of the output frames per signal
   interpolates the magnitudes, advances the accumulator and locks it, and the
   phases are materialised into a dense matrix so the phasor that closes the
   computation runs over the whole matrix at once; without, the fused traversal
   further below computes the same values band by band. The traversal is
   frame-major — the frame axis is outermost — so every row it reads and writes
   is contiguous.

   Both traversals are resumable: they compute any run [[lo, hi)] of output
   frames from the analysis frames that run reads and the accumulator of its
   first frame, and leave the accumulator of the frame after it behind. Output
   frame [i] depends on nothing else — its position is [i * r] recomputed by
   multiplication, never a running sum — so cutting the output axis anywhere and
   resuming there computes every frame from the same operands in the same order,
   and the offline traversal, one run over every output frame, is the streaming
   traversal's one-chunk instance. *)

(* Flat float64 storage. The kind is named rather than inferred: a polymorphic
   Bigarray element type turns every access in the traversal below into a
   dispatch. *)
type doubles =
  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

let doubles n = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout n

(* [reading t f] is [f] applied to the elements of [t] in C order, read in
   place: valid only during the call, and never written. *)
let reading (t : (float, Nx.float64_elt) Nx.t) (f : doubles -> 'a) : 'a =
  Rig.Buffer.read (Nx.to_buffer t) Bigarray.float64 f

(* [peaks_of m base bins into] fills [into] with the bins whose magnitude
   strictly exceeds every one of the four neighbours it has, and is their
   count. *)
let peaks_of (m : doubles) base bins into =
  let count = ref 0 in
  for k = 0 to bins - 1 do
    let v = Bigarray.Array1.unsafe_get m (base + k) in
    if
      (k < 2 || v > Bigarray.Array1.unsafe_get m (base + k - 2))
      && (k < 1 || v > Bigarray.Array1.unsafe_get m (base + k - 1))
      && (k + 1 >= bins || v > Bigarray.Array1.unsafe_get m (base + k + 1))
      && (k + 2 >= bins || v > Bigarray.Array1.unsafe_get m (base + k + 2))
    then (
      Array.unsafe_set into !count k ;
      incr count )
  done ;
  !count

(* [lock phi po ang ao bins peaks n] rewrites the accumulated phases of frame
   row [po] in place: each peak keeps its own, and every other bin of a peak's
   region takes the peak's phase offset by the phase difference the two bins
   have in the analysis frame at [ao]. Regions are split halfway between
   consecutive peaks, a bin equidistant from both going to the upper one. *)
let lock (phi : doubles) po (ang : doubles) ao bins peaks n =
  let start = ref 0 in
  for p = 0 to n - 1 do
    let kp = Array.unsafe_get peaks p in
    let stop =
      if p = n - 1 then bins else (kp + Array.unsafe_get peaks (p + 1) + 1) / 2
    in
    let base = Bigarray.Array1.unsafe_get phi (po + kp) in
    let reference = Bigarray.Array1.unsafe_get ang (ao + kp) in
    for k = !start to stop - 1 do
      if k <> kp then
        Bigarray.Array1.unsafe_set phi (po + k)
          (base +. (Bigarray.Array1.unsafe_get ang (ao + k) -. reference))
    done ;
    start := stop
  done

(* [propagate] is one signal's traversal over output frames [[lo, hi)]: [mag]
   and [ang] hold its magnitudes and arguments over the analysis frames [base]
   to [last], rows of [bins], and [amp] and [phi] receive the interpolated
   magnitudes and the accumulated phases, one row per output frame. The
   accumulator of output frame [lo] is read from [state] — from the argument of
   analysis frame [0] when [lo = 0] — and [state] receives the accumulator of
   output frame [hi]. Over the final interval, from the last analysis frame on,
   the magnitudes extrapolate the line through the last two frames and the
   accumulator holds; a single frame is held as it is. *)
let propagate ~locked ~bins ~base ~last ~rate ~lo ~hi ~omega ~(mag : doubles)
    ~mb ~(ang : doubles) ~ab ~(amp : doubles) ~qb ~(phi : doubles) ~pb
    ~(state : doubles) ~sb =
  let peaks = if locked then Array.make bins 0 else [||] in
  let first = if lo = 0 then ang else state
  and fo = if lo = 0 then ab else sb in
  for k = 0 to bins - 1 do
    Bigarray.Array1.unsafe_set phi (pb + k)
      (Bigarray.Array1.unsafe_get first (fo + k))
  done ;
  for i = lo to hi - 1 do
    let position = Float.of_int i *. rate in
    let i0 = Int.min (Float.to_int position) last in
    let alpha = position -. Float.of_int i0 in
    let m0 = mb + ((i0 - base) * bins) in
    let a0 = ab + ((i0 - base) * bins) in
    let po = pb + ((i - lo) * bins) in
    let qo = qb + ((i - lo) * bins) in
    if i0 < last then begin
      let m1 = m0 + bins in
      for k = 0 to bins - 1 do
        Bigarray.Array1.unsafe_set amp (qo + k)
          ( ((1. -. alpha) *. Bigarray.Array1.unsafe_get mag (m0 + k))
          +. (alpha *. Bigarray.Array1.unsafe_get mag (m1 + k)) )
      done
    end
    else if last > 0 then begin
      let mp = m0 - bins in
      for k = 0 to bins - 1 do
        Bigarray.Array1.unsafe_set amp (qo + k)
          ( ((1. +. alpha) *. Bigarray.Array1.unsafe_get mag (m0 + k))
          -. (alpha *. Bigarray.Array1.unsafe_get mag (mp + k)) )
      done
    end
    else
      for k = 0 to bins - 1 do
        Bigarray.Array1.unsafe_set amp (qo + k)
          (Bigarray.Array1.unsafe_get mag (m0 + k))
      done ;
    if locked then begin
      let n = peaks_of mag m0 bins peaks in
      if n > 0 then lock phi po ang a0 bins peaks n
    end ;
    (* the next output frame's accumulator: its own row, or [state] past the
       last row of the run *)
    let next = if i + 1 < hi then phi else state in
    let pn = if i + 1 < hi then po + bins else sb in
    if i0 < last then begin
      let a1 = a0 + bins in
      for k = 0 to bins - 1 do
        let om = Array.unsafe_get omega k in
        let x =
          Bigarray.Array1.unsafe_get ang (a1 + k)
          -. Bigarray.Array1.unsafe_get ang (a0 + k)
          -. om
        in
        (* the principal value [x - 2 pi round_half_even (x / 2 pi)], written
           out so that no float is boxed *)
        let q = x /. two_pi in
        let r =
          if Float.abs q >= 0x1p52 then q
          else if q >= 0. then q +. 0x1p52 -. 0x1p52
          else q -. 0x1p52 +. 0x1p52
        in
        let t = two_pi *. r in
        let deviation = x -. t in
        Bigarray.Array1.unsafe_set next (pn + k)
          (Bigarray.Array1.unsafe_get phi (po + k) +. (om +. deviation))
      done
    end
    else
      for k = 0 to bins - 1 do
        Bigarray.Array1.unsafe_set next (pn + k)
          (Bigarray.Array1.unsafe_get phi (po + k))
      done
  done

(* {1 The independent traversal, fused}

   Without phase locking every bin evolves alone, so the traversal splits into
   bands of [band] bins, each carried through every output frame of the run by
   one parallel unit ({!Par}) that computes everything on the way: the
   magnitudes and arguments of the analysis frames it reaches (each once, into
   two slots indexed by the frame's parity, which keep the two frames an
   interval reads apart), the interpolated magnitude, the phasor, written
   straight into the complex result, and the accumulator, carried in the unit's
   own memory and read from and written back to the band's slice of [state] at
   the ends of the run, so no two domains write one cache line while it runs.
   The values are those of {!propagate} and the whole-matrix phasor, operation
   for operation: [hypot] and [atan2] as [Nx.magnitude] and [Nx.angle] take
   them, each product bound to a name before it is added so no platform
   contracts the pair, and the principal value with [round_half_even] written
   out. *)

let band = 32

let vocode_band ~bins ~base ~frames ~last ~rate ~lo ~hi ~omega (src : doubles)
    (dst : doubles) (state : doubles) s b0 b1 mags angs amps phis loaded =
  let w = b1 - b0 in
  let sbase = s * frames * bins and dbase = s * (hi - lo) * bins in
  let pb = (s * bins) + b0 in
  Array.unsafe_set loaded 0 (-1) ;
  Array.unsafe_set loaded 1 (-1) ;
  let load f =
    let slot = f land 1 in
    if Array.unsafe_get loaded slot <> f then begin
      Array.unsafe_set loaded slot f ;
      let row = 2 * (sbase + ((f - base) * bins) + b0) and o = slot * w in
      for k = 0 to w - 1 do
        let re = Bigarray.Array1.unsafe_get src (row + (2 * k))
        and im = Bigarray.Array1.unsafe_get src (row + (2 * k) + 1) in
        Float.Array.unsafe_set mags (o + k) (Float.hypot re im) ;
        Float.Array.unsafe_set angs (o + k) (Float.atan2 im re)
      done
    end
  in
  if lo = 0 then begin
    load 0 ;
    for k = 0 to w - 1 do
      Float.Array.unsafe_set phis k (Float.Array.unsafe_get angs k)
    done
  end
  else
    for k = 0 to w - 1 do
      Float.Array.unsafe_set phis k (Bigarray.Array1.unsafe_get state (pb + k))
    done ;
  for i = lo to hi - 1 do
    let position = Float.of_int i *. rate in
    let i0 = Int.min (Float.to_int position) last in
    let alpha = position -. Float.of_int i0 in
    load i0 ;
    let o0 = i0 land 1 * w and o1 = (1 - (i0 land 1)) * w in
    let out = 2 * (dbase + ((i - lo) * bins) + b0) in
    (* the interpolated magnitudes go to [amps], then become phasors *)
    if i0 < last then begin
      load (i0 + 1) ;
      for k = 0 to w - 1 do
        let p0 = (1. -. alpha) *. Float.Array.unsafe_get mags (o0 + k) in
        let p1 = alpha *. Float.Array.unsafe_get mags (o1 + k) in
        Float.Array.unsafe_set amps k (p0 +. p1)
      done
    end
    else if last > 0 then begin
      load (i0 - 1) ;
      for k = 0 to w - 1 do
        let p0 = (1. +. alpha) *. Float.Array.unsafe_get mags (o0 + k) in
        let p1 = alpha *. Float.Array.unsafe_get mags (o1 + k) in
        Float.Array.unsafe_set amps k (p0 -. p1)
      done
    end
    else
      for k = 0 to w - 1 do
        Float.Array.unsafe_set amps k (Float.Array.unsafe_get mags (o0 + k))
      done ;
    for k = 0 to w - 1 do
      let amp = Float.Array.unsafe_get amps k
      and ph = Float.Array.unsafe_get phis k in
      Bigarray.Array1.unsafe_set dst (out + (2 * k)) (amp *. Float.cos ph) ;
      Bigarray.Array1.unsafe_set dst (out + (2 * k) + 1) (amp *. Float.sin ph)
    done ;
    if i0 < last then
      for k = 0 to w - 1 do
        let om = Array.unsafe_get omega (b0 + k) in
        let x =
          Float.Array.unsafe_get angs (o1 + k)
          -. Float.Array.unsafe_get angs (o0 + k)
          -. om
        in
        let q = x /. two_pi in
        let r =
          if Float.abs q >= 0x1p52 then q
          else if q >= 0. then q +. 0x1p52 -. 0x1p52
          else q -. 0x1p52 +. 0x1p52
        in
        let t = two_pi *. r in
        let deviation = x -. t in
        Float.Array.unsafe_set phis k
          (Float.Array.unsafe_get phis k +. (om +. deviation))
      done
  done ;
  for k = 0 to w - 1 do
    Bigarray.Array1.unsafe_set state (pb + k) (Float.Array.unsafe_get phis k)
  done

(* [traverse ~locked ~rate ~batch ~base ~last ~lo ~hi ~state framed] is output
   frames [[lo, hi)] of the stretched spectrum, frames-major ([[...; hi - lo;
   bins]]) at complex128, in float64 arithmetic. [framed] holds the analysis
   frames [base] to [last] of [batch]-shaped signals, frames-major at
   complex128, and they must cover every frame the run reads; [state] holds one
   accumulator per signal and bin, that of output frame [lo] on entry — unless
   [lo = 0], when analysis frame [0] gives it — and that of output frame [hi] on
   return. *)
let traverse ~bins ~omega ~locked ~rate ~batch ~base ~last ~lo ~hi ~state framed
    =
  let signals = Array.fold_left ( * ) 1 batch in
  let count = hi - lo in
  let frames = last + 1 - base in
  let out = Array.append batch [|count; bins|] in
  if signals = 0 || count = 0 then Nx.zeros Nx.complex128 out
  else if not locked then begin
    let bands = (bins + band - 1) / band in
    (* a unit costs [count * band] phasors: a run of a frame or two, as a stream
       pushes, gathers its bands into few chunks *)
    let grain = Int.max 1 (1024 / (count * band)) in
    Store.make_c128 out (fun dst ->
        Store.read_parts64 framed (fun src ->
            Par.ranges ~n:(signals * bands) ~grain (fun l h ->
                let mags = Float.Array.make (2 * band) 0.
                and angs = Float.Array.make (2 * band) 0.
                and amps = Float.Array.make band 0.
                and phis = Float.Array.make band 0.
                and loaded = Array.make 2 (-1) in
                for u = l to h - 1 do
                  let s = u / bands and b = u mod bands in
                  vocode_band ~bins ~base ~frames ~last ~rate ~lo ~hi ~omega src
                    dst state s (b * band)
                    (Int.min bins ((b + 1) * band))
                    mags angs amps phis loaded
                done ) ) )
  end
  else begin
    let span = frames * bins in
    let stride = count * bins in
    (* the magnitudes and arguments, [hypot] and [atan2] of every component pair
       as [Nx.magnitude] and [Nx.angle] take them, in one parallel pass *)
    let magnitudes, arguments =
      Store.make64_pair (Nx.shape framed) (fun mag ang ->
          Store.read_parts64 framed (fun src ->
              Par.ranges ~n:(signals * span) ~grain:8192 (fun l h ->
                  for i = l to h - 1 do
                    let re = Bigarray.Array1.unsafe_get src (2 * i)
                    and im = Bigarray.Array1.unsafe_get src ((2 * i) + 1) in
                    Bigarray.Array1.unsafe_set mag i (Float.hypot re im) ;
                    Bigarray.Array1.unsafe_set ang i (Float.atan2 im re)
                  done ) ) )
    in
    (* every element of both is written by the traversal *)
    let amplitudes, phases =
      Store.make64_pair out (fun amp phi ->
          reading magnitudes (fun mag ->
              reading arguments (fun ang ->
                  for s = 0 to signals - 1 do
                    propagate ~locked ~bins ~base ~last ~rate ~lo ~hi ~omega
                      ~mag ~mb:(s * span) ~ang ~ab:(s * span) ~amp
                      ~qb:(s * stride) ~phi ~pb:(s * stride) ~state
                      ~sb:(s * bins)
                  done ) ) )
    in
    (* the phasor, as the whole-matrix [mul amp (cos phi)] and [mul amp (sin
       phi)] compute it, written straight into the complex result *)
    Store.make_c128 out (fun dst ->
        reading amplitudes (fun amp ->
            reading phases (fun phi ->
                Par.ranges ~n:(signals * stride) ~grain:8192 (fun l h ->
                    for i = l to h - 1 do
                      let a = Bigarray.Array1.unsafe_get amp i
                      and ph = Bigarray.Array1.unsafe_get phi i in
                      Bigarray.Array1.unsafe_set dst (2 * i) (a *. Float.cos ph) ;
                      Bigarray.Array1.unsafe_set dst
                        ((2 * i) + 1)
                        (a *. Float.sin ph)
                    done ) ) ) )
  end

(* [vocode c ~locked ~rate z] is the stretched spectrum of the complex128
   spectrum [z], in float64 arithmetic and shaped [[...; bins; count]]: one run
   over every output frame. *)
let vocode (c : Stft.Config.t) ~locked ~rate z =
  let bins = Stft.Config.bins c in
  let rank = Nx.ndim z in
  let frames = Nx.dim (rank - 1) z in
  let batch = Array.sub (Nx.shape z) 0 (rank - 2) in
  let state = doubles (Array.fold_left ( * ) 1 batch * bins) in
  Nx.swapaxes (-1) (-2)
    (traverse ~bins ~omega:(advance c) ~locked ~rate ~batch ~base:0
       ~last:(frames - 1) ~lo:0 ~hi:(out_frames ~frames ~rate) ~state
       (Nx.contiguous (Nx.swapaxes (-1) (-2) z)) )

let to_complex128 : type c.
    (Complex.t, c) Nx.t -> (Complex.t, Nx.complex64_elt) Nx.t =
 fun t ->
  match Nx.dtype t with Nx.Complex128 -> t | _ -> Nx.cast Nx.complex128 t

let locked_of_phase = function `Locked -> true | `Independent -> false

(* {1 Offline entry points} *)

let phase_vocoder ?(phase = `Independent) c ~rate z =
  check_rate "phase_vocoder" rate ;
  check_spectrum "phase_vocoder" c z ;
  Nx.cast (Nx.dtype z)
    (vocode c ~locked:(locked_of_phase phase) ~rate (to_complex128 z))

let time_stretch ?(phase = `Independent) c ~rate x =
  check_rate "time_stretch" rate ;
  check_rank "time_stretch" x ;
  let n = Nx.dim (Nx.ndim x - 1) x in
  let length = Float.to_int (round_half_even (Float.of_int n /. rate)) in
  let z = Stft.transform Nx.complex128 c x in
  Stft.invert (Nx.dtype x) c ~length
    (vocode c ~locked:(locked_of_phase phase) ~rate z)

(* [fix_length n y] is [y] cut or zero-extended to exactly [n] samples on its
   time axis. *)
let fix_length n y =
  let rank = Nx.ndim y in
  let have = Nx.dim (rank - 1) y in
  if have = n then y
  else if have > n then
    Nx.slice
      (List.init rank (fun i -> if i = rank - 1 then Nx.R (0, n) else Nx.A))
      y
  else
    let widths = Array.make rank (0, 0) in
    widths.(rank - 1) <- (0, n - have) ;
    Nx.pad widths 0. y

let check_ratio op (ratio : Pipeline.Rate.t) =
  if ratio.num < 1 || ratio.den < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot shift by a frequency ratio of %d/%d (both terms must be \
          at least 1)"
         op ratio.num ratio.den )

let pitch_shift ?(phase = `Independent) ?quality c ~(ratio : Pipeline.Rate.t) x
    =
  check_ratio "pitch_shift" ratio ;
  check_rank "pitch_shift" x ;
  let n = Nx.dim (Nx.ndim x - 1) x in
  let rate = Float.of_int ratio.den /. Float.of_int ratio.num in
  let stretched = time_stretch ~phase c ~rate x in
  let resampler =
    Resample.Config.cached ?quality ~sample_rate:ratio.num ~target:ratio.den ()
  in
  fix_length n (Resample.apply resampler stretched)

(* [semitones] searches every denominator up to the cap and keeps the ratio
   closest to [2 ** (n / bins_per_octave)] in the logarithmic sense, which is
   the best rational approximation of the frequency ratio under that bound: for
   a fixed denominator the nearest numerator is the rounded product, so the
   search is exhaustive. The cap bounds the number of polyphase phases the
   resampler builds from the ratio, which is what the ratio costs in memory and
   in arithmetic; the residual interval error it leaves is four orders below the
   smallest audible one. *)

let ratio_cap = 512

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

let semitones ?(bins_per_octave = 12) n =
  if bins_per_octave < 1 then
    invalid_arg
      (Printf.sprintf
         "semitones: cannot divide the octave into %d steps (bins_per_octave \
          must be at least 1)"
         bins_per_octave ) ;
  if not (Float.is_finite n) then
    invalid_arg
      (Printf.sprintf
         "semitones: cannot shift by %g steps (the step count must be finite)" n ) ;
  let target = Float.pow 2. (n /. Float.of_int bins_per_octave) in
  let best = ref None in
  for den = 1 to ratio_cap do
    let num = Float.to_int (round_half_even (target *. Float.of_int den)) in
    if num >= 1 && num <= ratio_cap then begin
      let error =
        Float.abs
          ( Float.log2 (Float.of_int num /. Float.of_int den)
          -. Float.log2 target )
      in
      match !best with
      | Some (e, _, _) when e <= error ->
          ()
      | _ ->
          best := Some (error, num, den)
    end
  done ;
  match !best with
  | None ->
      invalid_arg
        (Printf.sprintf
           "semitones: cannot represent a frequency ratio of %g within %d (the \
            step count is too far from unity)"
           target ratio_cap )
  | Some (_, num, den) ->
      let d = gcd num den in
      {Pipeline.Rate.num= num / d; den= den / d}

(* {1 Streaming}

   The vocoder as a Mealy kernel over analysis frames. Output frame [i] reads
   analysis frames [floor (i * r)] and the one after it, so it is final the
   moment that second frame arrives: a step that brings the stream to [R] frames
   emits every pending output whose position [i * r] lies below [R - 1] —
   compared as the same float product the traversal computes — and runs the
   traversal over exactly those, with the frames it has. The drain emits the
   rest, up to [ceil (M / r)] for the [M] frames of the whole stream: every one
   of them sits on the final interval and reads the last two frames. Positions
   are monotone in [i], so the cut between the two never reorders anything, and
   a run read on the interior during a step is read on the interior offline too,
   where the last frame lies at least as far. The state is the accumulator, the
   last two frames the extrapolation may still read, kept as copies, and the two
   counters.

   A run's first output reads frames from the last one the previous step
   received on, because it was not emitted then; the drain reads the last two.
   No frame further back is ever read again. *)

let ceil_div a b = (a + b - 1) / b

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
    t

let concat_last parts = Nx.concatenate ~axis:(-1) parts

let shape_string shape =
  String.concat "; " (List.map string_of_int (Array.to_list shape))

(* [continue op held lead] fixes the leading shape of a stream at its first
   chunk and holds every later chunk to it. *)
let continue op held lead =
  match !held with
  | None ->
      held := Some lead
  | Some l when l = lead ->
      ()
  | Some l ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot continue a stream of leading shape [%s] with a chunk of \
            leading shape [%s]"
           op (shape_string l) (shape_string lead) )

let drained op =
  invalid_arg
    (Printf.sprintf
       "%s: cannot feed a drained stream (flush consumed the tail; reset \
        before reusing)"
       op )

let check_stage_rate op (rate : Pipeline.Rate.t) =
  if rate.num < 1 || rate.den < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot stretch by a rate of %d/%d (both terms must be at least 1)"
         op rate.num rate.den )

(* [quotient rate] is the double a stage stretches by: [num / den] correctly
   rounded, the double the decimal literal of the same value denotes. *)
let quotient (rate : Pipeline.Rate.t) =
  Float.of_int rate.num /. Float.of_int rate.den

(* [join a b] is the concatenation of two optional runs of samples or frames
   along the last axis. *)
let join a b =
  match (a, b) with
  | None, r | r, None ->
      r
  | Some a, Some b ->
      Some (concat_last [a; b])

type vocoder =
  { vbins: int
  ; vrate: float
  ; vlocked: bool
  ; vomega: float array
  ; vlead: int array option ref
  ; mutable vstate: doubles
        (* the accumulator of output frame [vemitted], signals by bins *)
  ; mutable vhist: (Complex.t, Nx.complex64_elt) Nx.t option
        (* the last two frames received, frames-major *)
  ; mutable vreceived: int
  ; mutable vemitted: int
  ; mutable vdrained: bool }

let vocoder_create ~bins ~rate ~locked ~omega =
  { vbins= bins
  ; vrate= rate
  ; vlocked= locked
  ; vomega= omega
  ; vlead= ref None
  ; vstate= doubles 0
  ; vhist= None
  ; vreceived= 0
  ; vemitted= 0
  ; vdrained= false }

let vocoder_reset v =
  v.vlead := None ;
  v.vhist <- None ;
  v.vreceived <- 0 ;
  v.vemitted <- 0 ;
  v.vdrained <- false

let vocoder_run v ~batch ~base ~hi framed =
  let lo = v.vemitted in
  v.vemitted <- hi ;
  Nx.swapaxes (-1) (-2)
    (traverse ~bins:v.vbins ~omega:v.vomega ~locked:v.vlocked ~rate:v.vrate
       ~batch ~base ~last:(v.vreceived - 1) ~lo ~hi ~state:v.vstate framed )

(* [vocoder_step v op z] feeds the complex128 frames [z], shaped [[...; bins;
   frames]], and is the output frames they completed, if any. *)
let vocoder_step v op z =
  if v.vdrained then drained op ;
  let rank = Nx.ndim z in
  let batch = Array.sub (Nx.shape z) 0 (rank - 2) in
  if Option.is_none !(v.vlead) then
    v.vstate <- doubles (Array.fold_left ( * ) 1 batch * v.vbins) ;
  continue op v.vlead batch ;
  let k = Nx.dim (rank - 1) z in
  if k = 0 then None
  else begin
    let rows = Nx.swapaxes (-1) (-2) z in
    let framed, kept =
      match v.vhist with
      | None ->
          (rows, 0)
      | Some h ->
          (Nx.concatenate ~axis:(-2) [h; rows], Nx.dim (rank - 2) h)
    in
    let base = v.vreceived - kept in
    v.vreceived <- v.vreceived + k ;
    let total = kept + k in
    v.vhist <-
      Some
        (Nx.copy
           (Nx.slice
              (List.init rank (fun i ->
                   if i = rank - 2 then Nx.R (total - Int.min 2 total, total)
                   else Nx.A ) )
              framed ) ) ;
    let last = Float.of_int (v.vreceived - 1) in
    let hi = ref v.vemitted in
    while Float.of_int !hi *. v.vrate < last do
      incr hi
    done ;
    if !hi = v.vemitted then None
    else Some (vocoder_run v ~batch ~base ~hi:!hi framed)
  end

(* [vocoder_flush v] drains [v]: the output frames on the final interval. *)
let vocoder_flush v =
  if v.vdrained then None
  else begin
    v.vdrained <- true ;
    match (v.vhist, !(v.vlead)) with
    | Some h, Some batch ->
        v.vhist <- None ;
        let frames = v.vreceived in
        let count = out_frames ~frames ~rate:v.vrate in
        if count <= v.vemitted then None
        else
          Some
            (vocoder_run v ~batch
               ~base:(frames - Nx.dim (Nx.ndim h - 2) h)
               ~hi:count h )
    | _ ->
        None
  end

(* [vocoder_bound rate b] dominates the output frames one step over [b] frames,
   or the drain, emits: the step moves the emission limit [b] frames on, and a
   run of [b] frames holds at most [ceil (b * den / num)] positions, the float
   product one more; the drain's final interval is one frame long, which the [b
   + 1] covers at every [b]. *)
let vocoder_bound (rate : Pipeline.Rate.t) b =
  ceil_div ((b + 1) * rate.den) rate.num + 2

let check_lead_axes op lead =
  if Array.exists (fun d -> d = 0) lead then
    invalid_arg
      (Printf.sprintf
         "%s: cannot stretch a chunk with a zero-size leading axis (channels \
          must be at least 1)"
         op )

let phase_vocoder_stage ?(phase = `Independent) c ~(rate : Pipeline.Rate.t) =
  let op = "phase_vocoder_stage" in
  check_stage_rate op rate ;
  let bins = Stft.Config.bins c and omega = advance c in
  let locked = locked_of_phase phase and rf = quotient rate in
  let frame_rate = {Pipeline.Rate.num= rate.den; den= rate.num} in
  let out_format fmt =
    let ips =
      Pipeline.Rate.(Pipeline.Format.items_per_second fmt * frame_rate)
    in
    fmt
    |> Pipeline.Format.with_items_per_second ips
    |> Pipeline.Format.with_max_items
         (Option.map (vocoder_bound rate) (Pipeline.Format.max_items fmt))
  in
  (* The element dtype and the leading axes are only witnessed by incoming
     chunks; [step] records them so [concat []] and [flush] can build and cast.
     One stage value is monomorphic in its element type, so the shared cell is
     coherent across prepares. *)
  let witness = ref None in
  Pipeline.kernel ~latency:1 ~rate:frame_rate ~out_format
    ~flush:(fun v ->
      match (vocoder_flush v, !witness) with
      | Some y, Some (dtype, _) ->
          [Nx.cast dtype y]
      | _ ->
          [] )
    ~reset:vocoder_reset
    ~concat:(function
      | [] -> (
        match !witness with
        | Some (dtype, lead) ->
            Nx.zeros dtype (Array.append lead [|bins; 0|])
        | None ->
            invalid_arg
              "phase_vocoder_stage: cannot concatenate zero chunks before any \
               chunk fixed the element dtype" )
      | parts ->
          concat_last parts )
    ~prepare:(fun _ -> vocoder_create ~bins ~rate:rf ~locked ~omega)
    ~step:(fun v z ->
      check_spectrum op c z ;
      let rank = Nx.ndim z in
      witness := Some (Nx.dtype z, Array.sub (Nx.shape z) 0 (rank - 2)) ;
      Option.map (Nx.cast (Nx.dtype z)) (vocoder_step v op (to_complex128 z)) )
    ()

(* {2 Time stretching}

   The stretch drives the analysis kernel, the vocoder and the synthesis kernel
   in turn, each step feeding what the one before it completed to the next — the
   composition [Stft.stage >> phase_vocoder_stage >> Stft.synthesis_stage] — and
   adds the one thing a composition cannot know: the length. The offline result
   has [round (n / r)] samples for [n] input samples, a length the synthesis
   stream alone does not end at, so the stretch never releases a sample past
   [round (seen / r)] for the [seen] samples it has been given — a bound that
   only grows and never passes the final length — and holds the rest back until
   more input moves the bound or the drain ends the stream exactly where
   {!Stft.invert}[ ~length] ends it, through {!Stft.Synthesis.flush}[ ~length].
   Under centered frames the synthesis never gets ahead of the bound and nothing
   is held. *)

type 'a stretch =
  { tcfg: Stft.Config.t
  ; trate: float
  ; tvoc: vocoder
  ; tchannels: int
  ; tblock: int option (* the most samples a chunk holds *)
  ; tframes: int option (* the most frames the vocoder emits at once *)
  ; tlead: int array option ref
  ; mutable tdtype: (float, 'a) Nx.dtype option
  ; mutable tanalysis: ('a, Nx.complex64_elt) Stft.Kernel.t option
  ; mutable tsynthesis: ('a, Nx.complex64_elt) Stft.Synthesis.t option
  ; mutable tseen: int
  ; mutable temitted: int
  ; mutable theld: (float, 'a) Nx.t option
  ; mutable tdrained: bool }

(* [analysis_latency c] is the latency {!Stft.stage} declares, in samples. *)
let analysis_latency c = (Pipeline.latency (Stft.stage Nx.complex128 c)).num

(* [frame_bound c ~lst b] is the most frames the analysis emits for a chunk of
   [b] samples, the bound {!Stft.stage} threads. *)
let frame_bound c ~lst b = ceil_div (b + lst) (Stft.Config.hop c) + 1

(* [stretch_bound c rate ~lst b] dominates the samples one step of the stretch
   releases for a chunk of [b] samples: the synthesis of every frame the vocoder
   emits, or, when samples were held, the growth of the length bound. *)
let stretch_bound c (rate : Pipeline.Rate.t) ~lst b =
  let frames = vocoder_bound rate (frame_bound c ~lst b) in
  Int.max (frames * Stft.Config.hop c) (ceil_div (b * rate.den) rate.num + 2)

let stretch_create c ~rate ~locked ~omega ~frames fmt =
  { tcfg= c
  ; trate= quotient rate
  ; tvoc=
      vocoder_create ~bins:(Stft.Config.bins c) ~rate:(quotient rate) ~locked
        ~omega
  ; tchannels= Pipeline.Format.channels fmt
  ; tblock= Pipeline.Format.max_items fmt
  ; tframes= Option.map frames (Pipeline.Format.max_items fmt)
  ; tlead= ref None
  ; tdtype= None
  ; tanalysis= None
  ; tsynthesis= None
  ; tseen= 0
  ; temitted= 0
  ; theld= None
  ; tdrained= false }

let stretch_reset st =
  Option.iter Stft.Kernel.reset st.tanalysis ;
  vocoder_reset st.tvoc ;
  Option.iter Stft.Synthesis.reset st.tsynthesis ;
  st.tlead := None ;
  st.tseen <- 0 ;
  st.temitted <- 0 ;
  st.theld <- None ;
  st.tdrained <- false

(* [stretch_length st] is the length the input so far stretches to — the
   expression {!time_stretch} computes. *)
let stretch_length st =
  Float.to_int (round_half_even (Float.of_int st.tseen /. st.trate))

(* [stretch_synthesise st frames] feeds the vocoded [frames] to the synthesis
   kernel, prepared at the first of them, and is the samples they completed. *)
let stretch_synthesise st = function
  | None ->
      None
  | Some z ->
      let k =
        match st.tsynthesis with
        | Some k ->
            k
        | None ->
            let max_block = Option.value st.tframes ~default:(last_dim z) in
            let k =
              Stft.Synthesis.prepare (Option.get st.tdtype) st.tcfg
                Nx.complex128 ~channels:st.tchannels
                ~max_block:(Int.max 1 max_block)
            in
            st.tsynthesis <- Some k ;
            k
      in
      Stft.Synthesis.step k z

(* [capped ~allowed held samples] is the part of [held] then [samples] that may
   be released, [allowed] samples at most, and the part that must be held. *)
let capped ~allowed held samples =
  match join held samples with
  | None ->
      (None, None)
  | Some p ->
      let have = last_dim p in
      if have <= allowed then (Some p, None)
      else
        ( (if allowed = 0 then None else Some (shrink_last p 0 allowed))
        , Some (Nx.copy (shrink_last p allowed have)) )

(* [fit dtype lead want pending] is [pending] cut or zero-extended to exactly
   [want] samples, or [None] when [want] is [0]. *)
let fit dtype lead want pending =
  if want = 0 then None
  else
    match pending with
    | None ->
        Some (Nx.zeros dtype (Array.append lead [|want|]))
    | Some p ->
        let have = last_dim p in
        if have = want then Some p
        else if have > want then Some (shrink_last p 0 want)
        else
          Some
            (concat_last
               [p; Nx.zeros dtype (Array.append lead [|want - have|])] )

let stretch_step st op chunk =
  if st.tdrained then drained op ;
  check_rank op chunk ;
  let lead = leading_shape chunk in
  continue op st.tlead lead ;
  check_lead_axes op lead ;
  st.tdtype <- Some (Nx.dtype chunk) ;
  st.tseen <- st.tseen + last_dim chunk ;
  let analysis =
    match st.tanalysis with
    | Some k ->
        k
    | None ->
        let max_block = Option.value st.tblock ~default:(last_dim chunk) in
        let k =
          Stft.Kernel.prepare Nx.complex128 st.tcfg (Nx.dtype chunk)
            ~channels:st.tchannels ~max_block:(Int.max 1 max_block)
        in
        st.tanalysis <- Some k ;
        k
  in
  let samples =
    stretch_synthesise st
      (Option.bind (Stft.Kernel.step analysis chunk) (vocoder_step st.tvoc op))
  in
  let released, held =
    capped ~allowed:(stretch_length st - st.temitted) st.theld samples
  in
  st.theld <- held ;
  Option.iter (fun r -> st.temitted <- st.temitted + last_dim r) released ;
  released

(* [stretch_flush st op] drains [st] to exactly the length of its input
   stretched, as one run. *)
let stretch_flush st op =
  if st.tdrained then None
  else begin
    st.tdrained <- true ;
    match (st.tdtype, !(st.tlead)) with
    | Some dtype, Some lead ->
        let length = stretch_length st in
        let frames =
          Option.bind
            (Option.bind st.tanalysis Stft.Kernel.flush)
            (vocoder_step st.tvoc op)
        in
        let s1 = stretch_synthesise st frames in
        let s2 = stretch_synthesise st (vocoder_flush st.tvoc) in
        let s3 =
          Option.bind st.tsynthesis (fun k -> Stft.Synthesis.flush ~length k)
        in
        let pending = List.fold_left join st.theld [s1; s2; s3] in
        let want = length - st.temitted in
        st.theld <- None ;
        st.temitted <- length ;
        fit dtype lead want pending
    | _ ->
        None
  end

(* [split bound t] is [t] cut along its last axis into chunks of at most [bound]
   samples. *)
let split bound t =
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

let concat_samples op witness = function
  | [] -> (
    match !witness with
    | Some (dtype, lead) ->
        Nx.zeros dtype (Array.append lead [|0|])
    | None ->
        invalid_arg
          (Printf.sprintf
             "%s: cannot concatenate zero chunks before any chunk fixed the \
              element dtype"
             op ) )
  | parts ->
      concat_last parts

type 'a bounded = {run: 'a; bound: int option}

let time_stretch_stage ?(phase = `Independent) c ~(rate : Pipeline.Rate.t) =
  let op = "time_stretch_stage" in
  check_stage_rate op rate ;
  Stft.check_invertible op c ;
  let lst = analysis_latency c in
  let locked = locked_of_phase phase and omega = advance c in
  let out_rate = {Pipeline.Rate.num= rate.den; den= rate.num} in
  let bound fmt =
    Option.map (stretch_bound c rate ~lst) (Pipeline.Format.max_items fmt)
  in
  let out_format fmt =
    let ips = Pipeline.Rate.(Pipeline.Format.items_per_second fmt * out_rate) in
    fmt
    |> Pipeline.Format.with_items_per_second ips
    |> Pipeline.Format.with_max_items (bound fmt)
  in
  let frames b = vocoder_bound rate (frame_bound c ~lst b) in
  let witness = ref None in
  Pipeline.kernel
    ~latency:(lst + Stft.Config.hop c)
    ~output_latency:(Stft.Config.synthesis_latency c)
    ~rate:out_rate ~out_format
    ~flush:(fun s ->
      match stretch_flush s.run op with None -> [] | Some y -> split s.bound y )
    ~reset:(fun s -> stretch_reset s.run)
    ~concat:(concat_samples op witness)
    ~prepare:(fun fmt ->
      {run= stretch_create c ~rate ~locked ~omega ~frames fmt; bound= bound fmt} )
    ~step:(fun s chunk ->
      check_rank op chunk ;
      witness := Some (Nx.dtype chunk, leading_shape chunk) ;
      stretch_step s.run op chunk )
    ()

(* {2 Pitch shifting}

   The stretch at [den / num], its released samples fed to the resampler from
   [num] to [den] in pieces no longer than the kernel was prepared for — the
   partition law of {!Resample.Kernel} makes the pieces invisible — and the same
   length discipline as the stretch's, the bound now the input length itself:
   nothing past the samples seen is released, and the drain cuts or zero-extends
   the converted stream to exactly the input's length. *)

type 'a pitch =
  { pstretch: 'a stretch
  ; pcfg: Resample.Config.t
  ; pblock: int option (* the most samples one step of the stretch releases *)
  ; mutable presampler: ('a Resample.Kernel.t * int) option
  ; mutable pemitted: int
  ; mutable pheld: (float, 'a) Nx.t option }

let pitch_reset p =
  stretch_reset p.pstretch ;
  Option.iter (fun (k, _) -> Resample.Kernel.reset k) p.presampler ;
  p.pemitted <- 0 ;
  p.pheld <- None

(* [pitch_resample p x] feeds [x] to the resampler, prepared at the first
   samples it is given, and is what it emitted. Offline the stretch releases its
   whole result at once, and the pieces are then as long as that run, or 4096
   samples if it is shorter. *)
let pitch_resample p x =
  let k, block =
    match p.presampler with
    | Some kb ->
        kb
    | None ->
        let block =
          match p.pblock with
          | Some b ->
              Int.max 1 b
          | None ->
              Int.max 4096 (last_dim x)
        in
        let channels = Array.fold_left ( * ) 1 (leading_shape x) in
        let k =
          Resample.Kernel.prepare p.pcfg (Nx.dtype x) ~channels ~max_block:block
        in
        p.presampler <- Some (k, block) ;
        (k, block)
  in
  let n = last_dim x in
  let rec go start acc =
    if start >= n then acc
    else
      let stop = Int.min n (start + block) in
      go stop
        ( match Resample.Kernel.step k (shrink_last x start stop) with
        | None ->
            acc
        | Some y ->
            y :: acc )
  in
  match go 0 [] with
  | [] ->
      None
  | [y] ->
      Some y
  | ys ->
      Some (concat_last (List.rev ys))

let pitch_step p op chunk =
  let st = p.pstretch in
  let resampled = Option.bind (stretch_step st op chunk) (pitch_resample p) in
  let released, held =
    capped ~allowed:(st.tseen - p.pemitted) p.pheld resampled
  in
  p.pheld <- held ;
  Option.iter (fun r -> p.pemitted <- p.pemitted + last_dim r) released ;
  released

let pitch_flush p op =
  let st = p.pstretch in
  if st.tdrained then None
  else
    let tail = Option.bind (stretch_flush st op) (pitch_resample p) in
    match (st.tdtype, !(st.tlead)) with
    | Some dtype, Some lead ->
        let drained =
          Option.bind p.presampler (fun (k, _) -> Resample.Kernel.flush k)
        in
        let pending = join (join p.pheld tail) drained in
        let want = st.tseen - p.pemitted in
        p.pheld <- None ;
        p.pemitted <- st.tseen ;
        fit dtype lead want pending
    | _ ->
        None

let check_resampled_dtype op fmt =
  match Pipeline.Format.dtype fmt with
  | Pipeline.Format.Dtype Nx.Float32 | Pipeline.Format.Dtype Nx.Float64 ->
      ()
  | Pipeline.Format.Dtype dt ->
      invalid_arg
        (Stdlib.Format.asprintf
           "%s: cannot shift %a audio (the resampler carries float32 and \
            float64)"
           op Nx.pp_dtype dt )

let pitch_shift_stage ?(phase = `Independent) ?quality c
    ~(ratio : Pipeline.Rate.t) =
  let op = "pitch_shift_stage" in
  check_ratio op ratio ;
  Stft.check_invertible op c ;
  let rate = {Pipeline.Rate.num= ratio.den; den= ratio.num} in
  let resampler =
    Resample.Config.cached ?quality ~sample_rate:ratio.num ~target:ratio.den ()
  in
  let lst = analysis_latency c in
  let locked = locked_of_phase phase and omega = advance c in
  (* what the synthesis and the conversion withhold, in output samples of the
     stretch, carried back to the input's rate and rounded up to a sample *)
  let held =
    ceil_div
      ( (Stft.Config.synthesis_latency c + Resample.Config.latency resampler)
      * ratio.den )
      ratio.num
  in
  let stretched fmt =
    Option.map (stretch_bound c rate ~lst) (Pipeline.Format.max_items fmt)
  in
  let bound fmt =
    Option.map
      (fun b ->
        Int.max b
          (Resample.output_bound resampler ~n:(stretch_bound c rate ~lst b)) )
      (Pipeline.Format.max_items fmt)
  in
  let frames b = vocoder_bound rate (frame_bound c ~lst b) in
  let witness = ref None in
  Pipeline.kernel
    ~latency:(lst + Stft.Config.hop c)
    ~output_latency:held
    ~out_format:(fun fmt -> Pipeline.Format.with_max_items (bound fmt) fmt)
    ~flush:(fun s ->
      match pitch_flush s.run op with None -> [] | Some y -> split s.bound y )
    ~reset:(fun s -> pitch_reset s.run)
    ~concat:(concat_samples op witness)
    ~prepare:(fun fmt ->
      check_resampled_dtype op fmt ;
      { run=
          { pstretch= stretch_create c ~rate ~locked ~omega ~frames fmt
          ; pcfg= resampler
          ; pblock= stretched fmt
          ; presampler= None
          ; pemitted= 0
          ; pheld= None }
      ; bound= bound fmt } )
    ~step:(fun s chunk ->
      check_rank op chunk ;
      witness := Some (Nx.dtype chunk, leading_shape chunk) ;
      pitch_step s.run op chunk )
    ()

(* {1 Silence}

   librosa's non-silence test: the RMS energy of each frame (constant-zero
   centered framing, {!Energy.rms}), in decibels against the reference as
   [amplitude_to_db] computes them — squared and floored at [1e-10] on both
   sides — reduced over the leading axes by their maximum, and compared with
   [-top_db]. The decibels are double precision whatever the dtype of the
   signal.

   Once the reference is fixed the verdicts are frame-local, so a [`Value]
   reference streams: {!split_stage} judges the frames of the RMS carry as they
   complete, through the same {!loud_frames} and the same interval {!tracker}
   the offline functions use, and the carry's frames are those of {!Energy.rms}
   on every partitioning. *)

type reference = [`Maximum | `Value of float]

let check_silence op ~top_db ~reference ~frame_length ~hop =
  if frame_length < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse frames of %d samples (frame_length must be at \
          least 1)"
         op frame_length ) ;
  if hop < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot advance frames by %d samples (hop must be at least 1)" op
         hop ) ;
  if not (Float.is_finite top_db) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot threshold at %g dB below the reference (top_db must be \
          finite)"
         op top_db ) ;
  match reference with
  | `Value r when not (Float.is_finite r && r > 0.) ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot measure decibels against %g (the reference must be \
            finite and positive)"
           op r )
  | `Value _ | `Maximum ->
      ()

let amin = 1e-5 *. 1e-5

(* [offset_of r] is the decibels of the reference amplitude [r]. *)
let offset_of r = 10. *. Float.log10 (Float.max amin (r *. r))

(* [energies rms] is the frame count of the energies [rms] ([[...; 1; frames]])
   and their values in double precision, row after row. *)
let energies rms =
  (Nx.dim (Nx.ndim rms - 1) rms, Nx.to_array (Nx.cast Nx.float64 rms))

(* [loud_frames ~offset ~top_db ~frames mse] is, for each of the [frames] frames
   whose energies [mse] holds row after row, whether its loudest row lies above
   [-top_db] decibels relative to the reference of decibels [offset]. The
   offline and the streaming faces both decide through this one function, so a
   frame's verdict is the same arithmetic on both. *)
let loud_frames ~offset ~top_db ~frames mse =
  let rows = if frames = 0 then 0 else Array.length mse / frames in
  Array.init frames (fun t ->
      let loudest = ref Float.neg_infinity in
      for r = 0 to rows - 1 do
        let v = mse.((r * frames) + t) in
        let db = (10. *. Float.log10 (Float.max amin (v *. v))) -. offset in
        if db > !loudest then loudest := db
      done ;
      !loudest > -.top_db )

(* [nonsilent op ~top_db ~reference ~frame_length ~hop x] is, for each frame of
   [x], whether it is above the threshold. *)
let nonsilent op ~top_db ~reference ~frame_length ~hop x =
  check_rank op x ;
  check_silence op ~top_db ~reference ~frame_length ~hop ;
  let frames, mse = energies (Energy.rms ~frame_length ~hop x) in
  let reference =
    match reference with
    | `Value r ->
        r
    | `Maximum ->
        Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. mse
  in
  loud_frames ~offset:(offset_of reference) ~top_db ~frames mse

(* An interval tracker over the verdicts of the frames in stream order: [frame]
   counts the verdicts read, and [opened] is the first frame of the run of
   non-silent frames in progress, or [-1]. The run of frames [f] to [g - 1] is
   the samples [(f * hop, g * hop)], librosa's [frames_to_samples] of its
   edges. *)
type tracker = {hop: int; mutable frame: int; mutable opened: int}

let tracker hop = {hop; frame= 0; opened= -1}

let tracker_reset t =
  t.frame <- 0 ;
  t.opened <- -1

(* [advance t loud closed] reads the verdicts [loud] and is [closed] with the
   intervals they close added in front, newest first. An interval closes at the
   first silent frame [g] after its run, and its end [g * hop] is not clamped:
   frame [g] exists, so [g * hop <= (frames - 1) * hop <= n - frame_length mod
   2] for the [frames] frames of [n] samples, and librosa's clamp to [n] cannot
   bind — nor on a start, for the same reason. *)
let advance t loud closed =
  Array.fold_left
    (fun closed l ->
      let f = t.frame in
      t.frame <- f + 1 ;
      if l then begin
        if t.opened < 0 then t.opened <- f ;
        closed
      end
      else if t.opened < 0 then closed
      else begin
        let a = t.opened in
        t.opened <- -1 ;
        (a * t.hop, f * t.hop) :: closed
      end )
    closed loud

(* [finish t ~n closed] closes the run still open at the end of the [n]-sample
   signal, at [min n (frames * hop)] — librosa's clamp, which binds here and
   only here: [frames * hop >= n] always, so the run ends at [n]. *)
let finish t ~n closed =
  if t.opened < 0 then closed
  else begin
    let a = t.opened in
    t.opened <- -1 ;
    (a * t.hop, Int.min n (t.frame * t.hop)) :: closed
  end

let trim ?(top_db = 60.) ?(reference = `Maximum) ?(frame_length = 2048)
    ?(hop = 512) x =
  let loud = nonsilent "trim" ~top_db ~reference ~frame_length ~hop x in
  let n = Nx.dim (Nx.ndim x - 1) x in
  let first = ref (-1) and last = ref (-1) in
  Array.iteri
    (fun t l ->
      if l then begin
        if !first < 0 then first := t ;
        last := t
      end )
    loud ;
  let start, stop =
    if !first < 0 then (0, 0) else (!first * hop, Int.min n ((!last + 1) * hop))
  in
  let nd = Nx.ndim x in
  ( Nx.slice
      (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
      x
  , (start, stop) )

let split ?(top_db = 60.) ?(reference = `Maximum) ?(frame_length = 2048)
    ?(hop = 512) x =
  let loud = nonsilent "split" ~top_db ~reference ~frame_length ~hop x in
  let n = Nx.dim (Nx.ndim x - 1) x in
  let t = tracker hop in
  Array.of_list (List.rev (finish t ~n (advance t loud [])))

(* {2 The streaming face}

   One stream's state: the RMS carry, the interval tracker over the verdicts of
   its frames, the samples seen — the [n] the final interval is clamped to — and
   the leading shape of the first chunk, which every later chunk must keep. A
   stream with a zero-size leading axis holds no signal, so none of its frames
   is ever loud: it bypasses the carry, which refuses such chunks, and closes
   nothing, as offline {!split} finds nothing in it. *)

type 'a split_state =
  { carry: 'a Energy.Rms_kernel.t
  ; edges: tracker
  ; mutable samples: int
  ; mutable lead: int array option }

let shape_string a =
  String.concat "; " (Array.to_list (Array.map string_of_int a))

(* [emitted closed] is the chunk of the intervals [closed], newest first. *)
let emitted = function
  | [] ->
      None
  | closed ->
      Some (Array.of_list (List.rev closed))

let split_stage ?(top_db = 60.) ?(frame_length = 2048) ?(hop = 512) ~reference
    () =
  let fn = "split_stage" in
  check_silence fn ~top_db
    ~reference:(reference :> reference)
    ~frame_length ~hop ;
  let (`Value r) = reference in
  let offset = offset_of r in
  let judge rms =
    let frames, mse = energies rms in
    loud_frames ~offset ~top_db ~frames mse
  in
  let step s chunk =
    check_rank fn chunk ;
    let rank = Nx.ndim chunk in
    let lead = Array.sub (Nx.shape chunk) 0 (rank - 1) in
    ( match s.lead with
    | None ->
        s.lead <- Some lead
    | Some l when l = lead ->
        ()
    | Some l ->
        invalid_arg
          (Printf.sprintf
             "%s: cannot continue a stream of leading shape [%s] with a chunk \
              of leading shape [%s]"
             fn (shape_string l) (shape_string lead) ) ) ;
    s.samples <- s.samples + Nx.dim (rank - 1) chunk ;
    if Array.exists (fun d -> d = 0) lead then None
    else
      match Energy.Rms_kernel.step s.carry chunk with
      | None ->
          None
      | Some rms ->
          emitted (advance s.edges (judge rms) [])
  in
  (* the drained frames may close runs, and the run still open ends with the
     stream; a frame closes at most one interval and the final one needs a loud
     last frame, so the tail is at most one interval per drained frame — or the
     final one alone — and fits the threaded bound as one chunk *)
  let flush s =
    let closed =
      match Energy.Rms_kernel.flush s.carry with
      | None ->
          []
      | Some rms ->
          advance s.edges (judge rms) []
    in
    Option.to_list (emitted (finish s.edges ~n:s.samples closed))
  in
  let reset s =
    Energy.Rms_kernel.reset s.carry ;
    tracker_reset s.edges ;
    s.samples <- 0 ;
    s.lead <- None
  in
  Pipeline.kernel
    ~latency:(Energy.Rms_kernel.latency ~frame_length)
    ~rate:(Energy.Rms_kernel.rate ~hop)
    ~out_format:(Energy.Rms_kernel.out_format ~frame_length ~hop)
    ~flush ~reset ~concat:Array.concat
    ~prepare:(fun _ ->
      { carry= Energy.Rms_kernel.create ~frame_length ~hop
      ; edges= tracker hop
      ; samples= 0
      ; lead= None } )
    ~step ()
