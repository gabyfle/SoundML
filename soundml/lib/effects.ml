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
   is contiguous. *)

(* Flat float64 storage. The kind is named rather than inferred: a polymorphic
   Bigarray element type turns every access in the traversal below into a
   dispatch. *)
type doubles =
  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

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

(* [propagate] is one signal's traversal: [mag] and [ang] hold its magnitudes
   and arguments over [frames] rows of [bins], and [amp] and [phi] receive the
   interpolated magnitudes and the accumulated phases over [count] rows. The
   accumulator starts at the argument of the first analysis frame. Over the
   final interval, from the last analysis frame on, the magnitudes extrapolate
   the line through the last two frames and the accumulator holds; a single
   frame is held as it is. *)
let propagate ~locked ~bins ~frames ~rate ~count ~omega ~(mag : doubles) ~mb
    ~(ang : doubles) ~ab ~(amp : doubles) ~qb ~(phi : doubles) ~pb =
  let peaks = if locked then Array.make bins 0 else [||] in
  let last = frames - 1 in
  for k = 0 to bins - 1 do
    Bigarray.Array1.unsafe_set phi (pb + k)
      (Bigarray.Array1.unsafe_get ang (ab + k))
  done ;
  for i = 0 to count - 1 do
    let position = Float.of_int i *. rate in
    let i0 = Int.min (Float.to_int position) last in
    let alpha = position -. Float.of_int i0 in
    let m0 = mb + (i0 * bins) in
    let a0 = ab + (i0 * bins) in
    let po = pb + (i * bins) in
    let qo = qb + (i * bins) in
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
    if i + 1 < count then begin
      let pn = po + bins in
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
          Bigarray.Array1.unsafe_set phi (pn + k)
            (Bigarray.Array1.unsafe_get phi (po + k) +. (om +. deviation))
        done
      end
      else
        for k = 0 to bins - 1 do
          Bigarray.Array1.unsafe_set phi (pn + k)
            (Bigarray.Array1.unsafe_get phi (po + k))
        done
    end
  done

(* {1 The independent traversal, fused}

   Without phase locking every bin evolves alone, so the traversal splits into
   bands of [band] bins, each carried through every output frame by one parallel
   unit ({!Par}) that computes everything on the way: the magnitudes and
   arguments of the analysis frames it reaches (each once, into two slots
   indexed by the frame's parity, which keep the two frames an interval reads
   apart), the interpolated magnitude, the phasor, written straight into the
   complex result, and the accumulator. The values are those of {!propagate} and
   the whole-matrix phasor, operation for operation: [hypot] and [atan2] as
   [Nx.magnitude] and [Nx.angle] take them, each product bound to a name before
   it is added so no platform contracts the pair, and the principal value with
   [round_half_even] written out. *)

let band = 32

let vocode_band ~bins ~frames ~rate ~count ~omega (src : doubles)
    (dst : doubles) s b0 b1 mags angs amps phis loaded =
  let w = b1 - b0 in
  let last = frames - 1 in
  let sbase = s * frames * bins and dbase = s * count * bins in
  Array.unsafe_set loaded 0 (-1) ;
  Array.unsafe_set loaded 1 (-1) ;
  let load f =
    let slot = f land 1 in
    if Array.unsafe_get loaded slot <> f then begin
      Array.unsafe_set loaded slot f ;
      let row = 2 * (sbase + (f * bins) + b0) and o = slot * w in
      for k = 0 to w - 1 do
        let re = Bigarray.Array1.unsafe_get src (row + (2 * k))
        and im = Bigarray.Array1.unsafe_get src (row + (2 * k) + 1) in
        Float.Array.unsafe_set mags (o + k) (Float.hypot re im) ;
        Float.Array.unsafe_set angs (o + k) (Float.atan2 im re)
      done
    end
  in
  load 0 ;
  for k = 0 to w - 1 do
    Float.Array.unsafe_set phis k (Float.Array.unsafe_get angs k)
  done ;
  for i = 0 to count - 1 do
    let position = Float.of_int i *. rate in
    let i0 = Int.min (Float.to_int position) last in
    let alpha = position -. Float.of_int i0 in
    load i0 ;
    let o0 = i0 land 1 * w and o1 = (1 - (i0 land 1)) * w in
    let out = 2 * (dbase + (i * bins) + b0) in
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
    if i + 1 < count && i0 < last then
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
  done

(* [vocode_independent c ~rate z] is [vocode c ~locked:false ~rate z], over
   [signals] signals of [frames] frames, [count] output frames. *)
let vocode_independent (c : Stft.Config.t) ~rate ~signals ~frames ~count ~out z
    =
  let bins = Stft.Config.bins c in
  let omega = advance c in
  let framed = Nx.contiguous (Nx.swapaxes (-1) (-2) z) in
  let bands = (bins + band - 1) / band in
  Nx.swapaxes (-1) (-2)
    (Store.make_c128 out (fun dst ->
         Store.read_parts64 framed (fun src ->
             Par.ranges ~n:(signals * bands) ~grain:1 (fun lo hi ->
                 let mags = Float.Array.make (2 * band) 0.
                 and angs = Float.Array.make (2 * band) 0.
                 and amps = Float.Array.make band 0.
                 and phis = Float.Array.make band 0.
                 and loaded = Array.make 2 (-1) in
                 for u = lo to hi - 1 do
                   let s = u / bands and b = u mod bands in
                   vocode_band ~bins ~frames ~rate ~count ~omega src dst s
                     (b * band)
                     (Int.min bins ((b + 1) * band))
                     mags angs amps phis loaded
                 done ) ) ) )

(* [vocode c ~locked ~rate z] is the stretched spectrum of the complex128
   spectrum [z], in float64 arithmetic and shaped [[...; bins; count]]. *)
let vocode (c : Stft.Config.t) ~locked ~rate z =
  let bins = Stft.Config.bins c in
  let rank = Nx.ndim z in
  let frames = Nx.dim (rank - 1) z in
  let count = out_frames ~frames ~rate in
  let batch = Array.sub (Nx.shape z) 0 (rank - 2) in
  let signals = Array.fold_left ( * ) 1 batch in
  let out = Array.append batch [|count; bins|] in
  if signals = 0 || count = 0 then
    Nx.swapaxes (-1) (-2) (Nx.zeros Nx.complex128 out)
  else if not locked then
    vocode_independent c ~rate ~signals ~frames ~count ~out z
  else begin
    let framed = Nx.contiguous (Nx.swapaxes (-1) (-2) z) in
    let span = frames * bins in
    let stride = count * bins in
    (* the magnitudes and arguments, [hypot] and [atan2] of every component pair
       as [Nx.magnitude] and [Nx.angle] take them, in one parallel pass *)
    let magnitudes, arguments =
      Store.make64_pair (Nx.shape framed) (fun mag ang ->
          Store.read_parts64 framed (fun src ->
              Par.ranges ~n:(signals * span) ~grain:8192 (fun lo hi ->
                  for i = lo to hi - 1 do
                    let re = Bigarray.Array1.unsafe_get src (2 * i)
                    and im = Bigarray.Array1.unsafe_get src ((2 * i) + 1) in
                    Bigarray.Array1.unsafe_set mag i (Float.hypot re im) ;
                    Bigarray.Array1.unsafe_set ang i (Float.atan2 im re)
                  done ) ) )
    in
    let omega = advance c in
    (* every element of both is written by the traversal *)
    let amplitudes, phases =
      Store.make64_pair out (fun amp phi ->
          reading magnitudes (fun mag ->
              reading arguments (fun ang ->
                  for s = 0 to signals - 1 do
                    propagate ~locked ~bins ~frames ~rate ~count ~omega ~mag
                      ~mb:(s * span) ~ang ~ab:(s * span) ~amp ~qb:(s * stride)
                      ~phi ~pb:(s * stride)
                  done ) ) )
    in
    (* the phasor, as the whole-matrix [mul amp (cos phi)] and [mul amp (sin
       phi)] compute it, written straight into the complex result *)
    Nx.swapaxes (-1) (-2)
      (Store.make_c128 out (fun dst ->
           reading amplitudes (fun amp ->
               reading phases (fun phi ->
                   Par.ranges ~n:(signals * stride) ~grain:8192 (fun lo hi ->
                       for i = lo to hi - 1 do
                         let a = Bigarray.Array1.unsafe_get amp i
                         and ph = Bigarray.Array1.unsafe_get phi i in
                         Bigarray.Array1.unsafe_set dst (2 * i)
                           (a *. Float.cos ph) ;
                         Bigarray.Array1.unsafe_set dst
                           ((2 * i) + 1)
                           (a *. Float.sin ph)
                       done ) ) ) ) )
  end

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
