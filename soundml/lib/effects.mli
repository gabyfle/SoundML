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

(** Time-scale and pitch modification.

    The phase vocoder reads a signal's {!Stft} at hop [H] and writes it back at
    the same hop from a resampled time axis: output frame [i] is assembled from
    the analysis frames around position [i * rate], carrying their interpolated
    magnitudes and a phase advanced by the instantaneous frequency measured
    between them. Playing the result at the original rate stretches the signal
    in time and leaves its spectrum where it was.

    {!phase_vocoder} is that map on spectra, {!time_stretch} wraps it in the
    analysis and the least-squares synthesis of {!Stft}, and {!pitch_shift}
    composes {!time_stretch} with {!Resample} at the reciprocal ratio, which
    moves the spectrum and restores the duration. {!semitones} names a ratio in
    equal temperament. The analysis geometry is an {!Stft.Config.t}, used
    whole: the window, its length, the hop, the transform size, the alignment
    and the normalisation are the analysis's, and the padding mode is the
    boundary extension it reads.

    All four entry points are dtype-preserving — complex to complex, float to
    float — and never expose a complex dtype the caller did not already hold:
    the interior is float64 throughout and rounds once, at the boundary, into
    the dtype of the argument. The time or frame axis is the last axis; leading
    axes broadcast, so a batch of clips is one call.

    {2 Phase locking}

    [phase] selects how the accumulated phases of neighbouring bins relate and
    defaults to [`Independent], the classical algorithm: every bin propagates
    its own estimate. A single windowed partial spreads over the several bins
    its main lobe covers, and those bins carry a fixed phase relationship that
    independent propagation lets drift — the mechanism behind the loss of
    waveform shape ("phasiness") the classical algorithm is known for.

    [`Locked] is identity phase locking (Laroche & Dolson 1999): each frame's
    spectral peaks propagate normally, every other bin is assigned to the
    nearest peak, and a bin in a peak's region takes the peak's accumulated
    phase offset by the phase difference the two bins have in the analysis
    frame — the relationship a single windowed sinusoid imposes. The locked
    phases, not the independent ones, are what the next frame accumulates
    from.

    Locking is measurably but not uniformly better, which is why it is an
    option and not the default. Over a thirty-cell grid of three signal kinds
    (a swept sine in coloured noise, a vibrato-laden harmonic mixture, a
    filtered pulse train), five analysis geometries and two rates, the
    consistency of the produced spectrum —
    [20 log10 (‖transform (invert Y) - Y‖ / ‖Y‖)], how nearly the frames are
    the transform of an actual signal — improves by a median of 8.2 dB and by
    as much as 20.2 dB, and degrades by at most 4.0 dB on the two cells where
    it loses; the energy the synthesis retains rises from 0.596 to 0.873 of the
    input's. On a pure tone both paths hold the frequency to within 0.03 cents
    of the original at every rate, and the energy locking keeps out of the main
    lobe never rises and falls by up to 24 dB.

    {2 Parity}

    Numerical parity with librosa 0.11 ([librosa.phase_vocoder],
    [librosa.effects.time_stretch] and [librosa.effects.pitch_shift]) is
    enforced against committed golden vectors, and is pinned at matching
    explicit settings: [`Independent], [~pad:(`Constant 0.)] (its default
    boundary extension, where this library's is [`Reflect]), and the same
    transform size, hop, window and alignment. The deviations that remain are
    these, and each is a contract rather than an accident.

    - The interior is float64 at both dtypes, including the phase accumulator,
      where the reference propagates float32 phases for float32 audio. That
      accumulator is a recurrence, so the difference is not a rounding but a
      divergence: the reference's own float32-against-float64 spread is
      [1.0e-3] to [9.4e-2] of peak on two seconds of music, four to six orders
      above what separates this implementation from its float64 path. The
      deviation is on the accurate side, and the float32 golden vectors
      therefore quantize the input to float32 and compute the reference in
      float64, as every suite in this library does.
    - The principal-value reduction rounds ties to even, matching the
      reference's rounding rather than the round-half-away-from-zero of
      {!Float.round}. Ties are unreachable in practice; the rule is fixed so
      that they are not a source of drift.
    - {!phase_vocoder} of a spectrum with no frames is a spectrum with no
      frames; the reference raises there.
    - {!pitch_shift} resamples through this library's {!Resample} rather than
      through the reference's soxr binding, at the exact rational ratio
      {!semitones} names — where the reference converts between a float rate
      and an integer one, and so does not resample at the ratio it stretched
      by. Both resamplers are windowed-sinc designs at the same published
      specification, and they may differ arbitrarily above the passband both
      keep flat; they also start and flush their filters differently, so the
      substitution is not spread evenly over the output and is stated here in
      three parts. On two seconds of swept sine in noise, over the tested
      ratios, substituting one for the other moves the shifted signal by at
      most [8.1e-4] of peak between the ends of the signal, by [4.2e-3] over
      the first five output samples, and by [5.1e-2] in the final one —
      against the [3.3e-4] to [6.3e-4] the reference moves itself by switching
      its own soxr from its default HQ tier to VHQ. On the full-band noise of
      the golden vectors, which puts a fifth of its energy in the transition
      band, the same three figures are [6.8e-3], [2.6e-2] and [7.7e-3],
      against a tier switch of [6.5e-3] to [9.0e-3]. Between the ends the
      substitution is of the order of the reference's own choice of tier — two
      and a half times it at worst — which is the sense in which the pitch
      vectors pin parity, and the stretch stage of those same cells is pinned
      exactly.

    Citations. Flanagan, J. L. and Golden, R. M., "Phase Vocoder", {e Bell
    System Technical Journal} 45(9), 1966. Portnoff, M. R., "Implementation of
    the Digital Phase Vocoder Using the Fast Fourier Transform", {e IEEE
    Transactions on Acoustics, Speech, and Signal Processing} 24(3), 1976.
    Dolson, M., "The Phase Vocoder: A Tutorial", {e Computer Music Journal}
    10(4), 1986. Laroche, J. and Dolson, M., "Improved Phase Vocoder Time-Scale
    Modification of Audio", {e IEEE Transactions on Speech and Audio
    Processing} 7(3), 1999. Ellis, D. P. W., "A Phase Vocoder in Matlab", 2002.
    Griffin, D. W. and Lim, J. S., "Signal Estimation from Modified Short-Time
    Fourier Transform", {e IEEE Transactions on Acoustics, Speech, and Signal
    Processing} 32(2), 1984. *)

(** {1 Spectra} *)

val phase_vocoder :
     ?phase:[`Independent | `Locked]
  -> Stft.Config.t
  -> rate:float
  -> (Complex.t, 'c) Nx.t
  -> (Complex.t, 'c) Nx.t
(** [phase_vocoder c ~rate z] is the spectrum [z] — shaped
    [[...; bins; frames]], the shape {!Stft.transform} produces on the geometry
    of [c] — resampled onto a time axis [rate] times as fast, shaped
    [[...; bins; ceil (frames / rate)]] and in the dtype of [z]. A [rate] above
    [1.] shortens, below [1.] lengthens.

    Output frame [i] reads analysis position [i * rate] — computed by
    multiplication, so the grid carries no accumulated error — and is
    assembled from the analysis frames on either side of it: their magnitudes
    linearly interpolated, and a phase accumulator advanced once per output
    frame by the expected advance of the bin centre plus the principal value of
    the measured deviation from it. The accumulator starts at the phases of
    analysis frame [0] verbatim and is never reduced modulo [2 pi]. Positions
    past the last analysis frame read silence.

    Only the transform size and the hop of [c] are read: the vocoder maps
    frames to frames, and the window, alignment and padding of [c] are the
    analysis's business. [rate = 1.] is not a shortcut — the recurrence runs
    in full — but it is the identity to within the arithmetic, which returns
    the input spectrum to a few units in the last place.

    Raises [Invalid_argument] if [rate] is not finite and positive, if [z] has
    rank below two, or if its bin axis is not [Stft.Config.bins c] long. *)

(** {1 Signals} *)

val time_stretch :
     ?phase:[`Independent | `Locked]
  -> Stft.Config.t
  -> rate:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [time_stretch c ~rate x] is [x] played [rate] times as fast with its
    spectrum left where it was, in the dtype of [x]: {!Stft.transform} on the
    geometry of [c], {!phase_vocoder} at [rate], and {!Stft.invert} back at the
    same geometry.

    The result has exactly [round (n / rate)] samples for an input of [n], ties
    to even — the length {!Stft.invert} is asked for, so the frames that open
    past it are not synthesised and a request past the frames is zero-filled.
    An empty input gives an empty result.

    [rate = 1.] is not a shortcut either: the full analysis, recurrence and
    synthesis run, and the round trip is what {!Stft.invert} documents — the
    positions no nonzero window tap reaches come back as [0], and everything
    else close to it: measured [1.9e-12] of peak at [fft_size 2048] and
    [9.3e-12] at [fft_size 64], the residual growing as the transform shrinks
    because a shorter transform runs the recurrence over more frames. At
    float32 the storage rounding dominates and the round trip is exact at
    [fft_size 2048].

    Raises [Invalid_argument] if [rate] is not finite and positive, if [x] has
    rank zero, or under the conditions of {!Stft.invert} — a configuration
    whose squared window does not overlap-add to something the synthesis can
    divide by is not one this function can use. *)

val pitch_shift :
     ?phase:[`Independent | `Locked]
  -> ?quality:Resample.quality
  -> Stft.Config.t
  -> ratio:Pipeline.Rate.t
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [pitch_shift c ~ratio x] is [x] with its spectrum scaled by the frequency
    ratio [ratio] and its duration unchanged, in the dtype of [x]:
    [time_stretch c ~rate:(den / num)] followed by {!Resample.apply} from
    [num] to [den] hertz, cut or zero-extended back to the length of [x].
    [ratio = {num = 3; den = 2}] raises the pitch by a just fifth;
    {!semitones} names the ratios of equal temperament.

    The stretch and the conversion run at the same exact rational ratio, which
    is what makes the composition well defined: the vocoder is chaotic in its
    rate — five ten-thousandths of a cent moves the output by [1.8e-2] of peak
    — so a stretch by a float rate followed by a conversion at a nearby
    rational drifts against the stretch it was meant to undo, by as much as
    [0.2] of peak over eight seconds.

    [quality] is the resampler preset and defaults to [`High], the tier the
    cap of {!semitones} is sized against. [`Best] designs a longer prototype
    from the same ratio, and a ratio with large terms — one built by hand
    rather than by {!semitones} — may exceed the coefficient budget
    {!Resample.Config.create} documents, which raises from there. The terms of
    the ratio, not its value, are what the conversion costs: they are the
    number of polyphase phases the bank holds.

    [ratio = {num = 1; den = 1}] is not a shortcut: the vocoder runs at rate
    [1.] as above and the identity resampler passes its result through
    untouched, so the residual is {!time_stretch}'s.

    Raises [Invalid_argument] if either term of [ratio] is below [1], if [x]
    has rank zero, under the conditions of {!time_stretch}, or under those of
    {!Resample.Config.create} for the ratio and preset given. *)

val semitones : ?bins_per_octave:int -> float -> Pipeline.Rate.t
(** [semitones n] is the frequency ratio [2 ** (n / bins_per_octave)] as an
    exact rational, for {!pitch_shift}: [semitones 12.] is [2/1],
    [semitones (-12.)] is [1/2], and [semitones 4.] — a major third — is
    [349/277]. [bins_per_octave] defaults to [12] and [n] may be fractional.

    The ratio is the best rational approximation of that power of two with
    neither term above [512]. That bound is what the interval costs to play:
    the terms are the polyphase phases {!Resample} builds its bank from, and a
    ratio ten times finer makes the conversion tens of times slower for an
    accuracy no ear reaches. The approximation is exact at whole octaves and
    within [0.027] cents everywhere in twelve-tone equal temperament and [0.09]
    cents over the finer divisions tested, two orders below the finest
    published just-noticeable difference; a caller who wants a particular
    ratio and will pay for it passes it to {!pitch_shift} directly.

    Raises [Invalid_argument] if [bins_per_octave < 1], if [n] is not finite,
    or if the resulting ratio is beyond the cap — nine octaves from unity,
    where no admissible rational approximates it. *)
