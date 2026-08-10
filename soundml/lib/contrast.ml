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

(* Octave-band spectral contrast, band by band. Each of the [n_bands + 1] octave
   bands is a contiguous FFT-bin range: band [k] collects the bins whose center
   frequency lies in [edge k, edge (k + 1)] inclusive (edge 0 is 0 Hz, edge j is
   [f_min * 2^(j-1)] beyond), extended one bin below for [k > 0], to the top of
   the spectrum for [k = n_bands], and trimmed of its top bin — returned to the
   next band — for [k < n_bands]. Per band and frame, the valley is the mean of
   the [take] smallest magnitudes and the peak the mean of the [take] largest,
   where [take] is the quantile of the extended band's bin count, rounded half
   to even ([rint]) and floored at one bin; a [take] beyond the trimmed band
   clamps to cover the whole band. *)

type band =
  { lo: int  (** First bin of the trimmed band. *)
  ; sub_len: int  (** Bins in the trimmed band. *)
  ; take: int  (** Bins averaged per side, already clamped to [sub_len]. *) }

type plan =
  { fft_size: int
  ; bins: int
  ; n_bands: int
  ; linear: bool
  ; clamped: bool
        (* The flat function applies [power_to_db] with its defaults — including
           the 80 dB whole-tensor clamp — on the log path; the per-frame stage
           cannot (the clamp is a whole-signal reduction) and leaves it out,
           documented. *)
  ; bands: band list }

(* [rint x] is [x] rounded to the nearest integer, ties to even — the rounding
   that sizes the quantile. *)
let rint x =
  let fl = Float.floor x in
  let d = x -. fl in
  if d > 0.5 then fl +. 1.
  else if d < 0.5 then fl
  else if Float.rem fl 2. = 0. then fl
  else fl +. 1.

let plan ~fn ~clamped ~n_bands ~f_min ~quantile ~linear ~sample_rate ~fft_size =
  if n_bands < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot divide the spectrum into %d octave bands (n_bands must be \
          at least 1)"
         fn n_bands ) ;
  if not (Float.is_finite f_min && f_min > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot start the first octave band at %g Hz (f_min must be \
          finite and positive)"
         fn f_min ) ;
  if not (quantile > 0. && quantile < 1.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot average the %g quantile of each band (quantile must lie \
          strictly between 0 and 1)"
         fn quantile ) ;
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use a sample rate of %d Hz (sample_rate must be at least \
          1)"
         fn sample_rate ) ;
  (* Band edges: 0, f_min, 2 f_min, ..., f_min * 2^n_bands. *)
  let edge j =
    if j = 0 then 0. else f_min *. Float.pow 2. (Float.of_int (j - 1))
  in
  let nyquist = 0.5 *. Float.of_int sample_rate in
  if edge n_bands >= nyquist then
    invalid_arg
      (Printf.sprintf
         "%s: cannot start the top octave band at %g Hz at a sample rate of %d \
          Hz (every band must start below the Nyquist frequency %g; lower \
          f_min or n_bands)"
         fn (edge n_bands) sample_rate nyquist ) ;
  let bins = (fft_size / 2) + 1 in
  (* Bin center frequencies, evaluated as one reciprocal and one multiply per
     bin — the reference arithmetic the parity contract fixes. *)
  let step =
    1.0 /. (Float.of_int fft_size *. (1.0 /. Float.of_int sample_rate))
  in
  (* Bands are built in ascending order — [List.init] does not fix its
     application order, and validation errors must name the lowest offending
     band deterministically. *)
  let band k =
    let f_low = edge k and f_high = edge (k + 1) in
    let lo0 = ref (-1) and hi0 = ref (-1) in
    for i = 0 to bins - 1 do
      let f = Float.of_int i *. step in
      if f >= f_low && f <= f_high then begin
        if !lo0 < 0 then lo0 := i ;
        hi0 := i
      end
    done ;
    if !lo0 < 0 then
      invalid_arg
        (Printf.sprintf
           "%s: cannot resolve the octave band [%g, %g] Hz with an FFT of size \
            %d at a sample rate of %d Hz (the band spans no FFT bin; raise \
            fft_size or lower n_bands)"
           fn f_low f_high fft_size sample_rate ) ;
    (* [lo0 >= 1] whenever [k > 0]: bin 0 sits at 0 Hz, below every positive
       lower edge, so the one-bin extension never underflows. *)
    let lo = if k > 0 then !lo0 - 1 else !lo0 in
    let hi = if k = n_bands then bins - 1 else !hi0 in
    let count = hi - lo + 1 in
    let take =
      Stdlib.max 1 (Float.to_int (rint (quantile *. Float.of_int count)))
    in
    let sub_len = (if k = n_bands then hi else hi - 1) - lo + 1 in
    if sub_len < 1 then
      invalid_arg
        (Printf.sprintf
           "%s: cannot resolve the octave band [%g, %g] Hz with an FFT of size \
            %d at a sample rate of %d Hz (the band spans no FFT bin below its \
            top edge; raise fft_size or lower n_bands)"
           fn f_low f_high fft_size sample_rate ) ;
    {lo; sub_len; take= Stdlib.min take sub_len}
  in
  let rec bands k = if k > n_bands then [] else band k :: bands (k + 1) in
  {fft_size; bins; n_bands; linear; clamped; bands= bands 0}

let shrink_rows t start stop =
  let nd = Nx.ndim t in
  Nx.shrink
    (Array.init nd (fun i ->
         if i = nd - 2 then (start, stop) else (0, Nx.dim i t) ) )
    t

(* [apply fn plan s] is the contrast of the magnitude spectrum [s], [[...; bins;
   frames]] to [[...; n_bands + 1; frames]]. One sort, two row means per band,
   batched over frames and leading axes; the interior runs in double and rounds
   to the dtype of [s] once, at the boundary. *)
let apply fn (p : plan) s =
  let nd = Nx.ndim s in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-%d tensor (spectral contrast needs [...; \
          bins; frames])"
         fn nd ) ;
  let shape = Nx.shape s in
  if shape.(nd - 2) <> p.bins then
    invalid_arg
      (Printf.sprintf
         "%s: cannot band %d frequency bins with a plan built for an FFT of \
          size %d (%d bins)"
         fn
         shape.(nd - 2)
         p.fft_size p.bins ) ;
  let dtype = Nx.dtype s in
  if Array.exists (fun d -> d = 0) shape then begin
    (* No frames, or no signals at all: nothing to sort — produce the
       broadcast-consistent empty result directly. *)
    let out = Array.copy shape in
    out.(nd - 2) <- p.n_bands + 1 ;
    Nx.zeros dtype out
  end
  else
    let s64 = Nx.cast Nx.float64 s in
    let valleys, peaks =
      List.split
        (List.map
           (fun b ->
             let sorted =
               fst
                 (Nx.sort ~axis:(-2) (shrink_rows s64 b.lo (b.lo + b.sub_len)))
             in
             let mean_rows start stop =
               Nx.mean ~axes:[-2] ~keepdims:true (shrink_rows sorted start stop)
             in
             (mean_rows 0 b.take, mean_rows (b.sub_len - b.take) b.sub_len) )
           p.bands )
    in
    let valley = Nx.concatenate ~axis:(-2) valleys
    and peak = Nx.concatenate ~axis:(-2) peaks in
    let out64 =
      if p.linear then Nx.sub peak valley
      else if p.clamped then
        Nx.sub
          (Convert.power_to_db ~top_db:80. peak)
          (Convert.power_to_db ~top_db:80. valley)
      else Nx.sub (Convert.power_to_db peak) (Convert.power_to_db valley)
    in
    Nx.cast dtype out64

let spectral_contrast stft_config ?(n_bands = 6) ?(f_min = 200.)
    ?(quantile = 0.02) ?(linear = false) ~sample_rate x =
  let fn = "spectral_contrast" in
  let p =
    plan ~fn ~clamped:true ~n_bands ~f_min ~quantile ~linear ~sample_rate
      ~fft_size:(Stft.Config.fft_size stft_config)
  in
  if Nx.ndim x < 1 then
    invalid_arg
      (fn ^ ": cannot analyse a rank-zero tensor (the time axis must exist)") ;
  apply fn p (Stft.power_spectrum ~power:1. stft_config x)

(* [spectral_contrast_of_spectrogram] is the spectrogram-input mode: the same
   band plan and contrast over an already-computed magnitude spectrogram, the
   FFT size implied by the bin count. *)
let spectral_contrast_of_spectrogram ?(n_bands = 6) ?(f_min = 200.)
    ?(quantile = 0.02) ?(linear = false) ~sample_rate s =
  let fn = "spectral_contrast_of_spectrogram" in
  let nd = Nx.ndim s in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-%d tensor (spectral contrast needs [...; \
          bins; frames])"
         fn nd ) ;
  let bins = Nx.dim (nd - 2) s in
  if bins < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot derive bin frequencies for a %d-bin spectrogram (the \
          implied FFT size is %d; a magnitude spectrogram holds fft_size / 2 + \
          1 bins)"
         fn bins
         (2 * (bins - 1)) ) ;
  let p =
    plan ~fn ~clamped:true ~n_bands ~f_min ~quantile ~linear ~sample_rate
      ~fft_size:(2 * (bins - 1))
  in
  if Nx.numel s > 0 && not (Nx.item [] (Nx.all (Nx.greater_equal_s s 0.))) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a spectrogram with negative or NaN values (a \
          magnitude spectrogram is non-negative)"
         fn ) ;
  apply fn p s

let stage stft_config ?(n_bands = 6) ?(f_min = 200.) ?(quantile = 0.02)
    ?(linear = false) ~sample_rate () =
  let fn = "spectral_contrast_stage" in
  let p =
    plan ~fn ~clamped:false ~n_bands ~f_min ~quantile ~linear ~sample_rate
      ~fft_size:(Stft.Config.fft_size stft_config)
  in
  Pipeline.stateless (apply fn p)
