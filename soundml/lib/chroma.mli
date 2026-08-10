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

(** Chroma: energy folded onto the twelve pitch classes.

    A chromagram discards octave identity and keeps pitch class — the
    representation Fujishima's pitch-class profile introduced for chord
    recognition, and the one every key- and chord-tracking front end still
    uses. Two projections are on offer, one for each kind of spectrum:

    - From a linear-frequency spectrum, {!Config} precomputes a
      [[n_chroma; bins]] matrix of Gaussian bumps — each FFT bin contributes to
      the pitch class its frequency lands on, with a width set by the local bin
      spacing — and {!apply} projects a power spectrum through it.
    - From a constant-Q spectrum, {!cqt_projection} is the exact 0/1 assignment
      of each {!Cqt} bin to its pitch class, and {!of_cqt} projects a magnitude
      transform through it. Constant-Q bins already sit on the equal-tempered
      ladder, so nothing has to be interpolated.

    Both end in the same per-frame normalisation, along the chroma axis. The
    projection matrix is built in double precision and every projection runs in
    a double interior, rounding once into the dtype of the spectrum.

    Numerical parity with the reference implementation
    is enforced against committed golden vectors in the test suite; the
    compatibility contract, and its one
    deliberate departure, live under {!section:parity}. Chroma energy
    normalisation ([chroma_cens]) and the variable-Q chromagram are deliberately
    outside this module. *)

(** The type for the per-frame normalisation applied along the chroma axis.

    - [`Inf] — divide each frame by its largest entry, so the strongest pitch
      class reads one. The default everywhere.
    - [`P p] — divide by the [p]-norm of the frame; [p] must be finite and
      positive. [`P 1.] gives a frame that sums to one, [`P 2.] a unit vector.
    - [`None] — leave the raw projection.

    A frame whose length underflows the smallest normal of the spectrum's dtype
    is left untouched rather than amplified: dividing a silent frame by its own
    rounding noise would manufacture a pitch profile. Zero-norm and
    negative-infinity-norm options are deliberately not offered — neither is a
    norm. *)
type norm = [`Inf | `P of float | `None]

(** {1 Configuration} *)

module Config : sig
  (** The type for validated, immutable chroma-filterbank configurations.
      Creation precomputes the [[n_chroma; bins]] matrix in double precision;
      configurations are cheap to share and compare, and the config-owned
      matrix is never exposed — {!filterbank} copies. *)
  type t

  val create :
       ?n_chroma:int
    -> ?tuning:float
    -> ?ctroct:float
    -> ?octwidth:float option
    -> ?base_c:bool
    -> sample_rate:int
    -> fft_size:int
    -> unit
    -> t
  (** [create ~sample_rate ~fft_size ()] is a chroma filterbank projecting the
      [fft_size / 2 + 1] bins of an [fft_size]-point spectrum at [sample_rate]
      onto [n_chroma] pitch classes.

      [n_chroma] defaults to [12], one band per semitone. [tuning] shifts the
      reference pitch by that many bands ([0.] by default, A440); it is always
      explicit and never estimated from the signal.

      [ctroct] and [octwidth] describe the octave-dominance envelope that damps
      bins far from the centre of the useful range: [ctroct] is the centre in
      octaves above the reference ([5.] by default) and [octwidth] the Gaussian
      half-width in octaves ([Some 2.]). [~octwidth:None] weights every octave
      equally.

      [base_c] defaults to [true] and puts C in row zero; [false] puts A there.

      Raises [Invalid_argument] if [n_chroma], [sample_rate] or [fft_size] is
      smaller than [1], if [tuning] or [ctroct] is not finite, or if [octwidth]
      is not finite and positive. *)

  val n_chroma : t -> int
  (** [n_chroma c] is the number of pitch classes. *)

  val sample_rate : t -> int
  (** [sample_rate c] is the sample rate in hertz the filterbank was built
      for. *)

  val fft_size : t -> int
  (** [fft_size c] is the FFT length in samples the filterbank was built
      for. *)

  val bins : t -> int
  (** [bins c] is the number of frequency bins the filterbank consumes,
      [fft_size / 2 + 1]. *)

  val tuning : t -> float
  (** [tuning c] is the reference-pitch shift in chroma bands. *)

  val ctroct : t -> float
  (** [ctroct c] is the centre of the octave-dominance envelope, in octaves
      above the reference. *)

  val octwidth : t -> float option
  (** [octwidth c] is the half-width of the octave-dominance envelope in
      octaves, or [None] for flat weighting. *)

  val base_c : t -> bool
  (** [base_c c] is [true] iff row zero is C. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt c] prints [c] on [fmt] in a compact single-line form. *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff [a] and [b] were created from the same
      parameters; float parameters are compared with [Float.equal]. *)
end

(** {1 From a linear-frequency spectrum} *)

val filterbank : (float, 'a) Nx.dtype -> Config.t -> (float, 'a) Nx.t
(** [filterbank dtype c] is the [[n_chroma; bins]] weight matrix of [c] in
    [dtype]: entry [(p, k)] is the weight FFT bin [k] carries into pitch class
    [p].

    A bin's weight is a Gaussian in the distance — wrapped around the octave —
    between the bin's position on the chroma scale and the band's, with a
    standard deviation set by the local spacing of the bins and floored at one
    band. Columns are normalised to unit euclidean length, then damped by the
    octave-dominance envelope. Bin zero carries no frequency and is placed an
    octave and a half below bin one, far enough out that its bump vanishes.

    The matrix is precomputed once, in double precision, inside
    {!Config.create}; every call returns a {e fresh} tensor cast from the
    config-owned matrix, so mutating the result never affects [c] or any other
    caller. *)

val apply : ?norm:norm -> Config.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [apply c s] is the power spectrum [s] projected onto the pitch classes of
    [c] — the batched matrix product [filterbank ⋅ s], normalised per frame —
    mapping [[...; bins; frames]] to [[...; n_chroma; frames]]. Leading axes
    broadcast, so a batch of spectrograms is one call. [norm] defaults to
    [`Inf]. The product is computed in double precision against the
    config-owned matrix and rounded to the dtype of [s] once, at the boundary.

    Raises [Invalid_argument] if [s] has rank below two, if its
    second-to-last axis does not hold [Config.bins c] bins, or if [norm] is
    [`P p] with [p] not finite and positive. *)

(** {1 From a constant-Q spectrum} *)

val cqt_projection :
  (float, 'a) Nx.dtype -> ?n_chroma:int -> Cqt.Config.t -> (float, 'a) Nx.t
(** [cqt_projection dtype c] is the [[n_chroma; Cqt.Config.n_bins c]]
    assignment matrix folding the constant-Q ladder of [c] onto pitch classes,
    in [dtype]. [n_chroma] defaults to [12].

    Every entry is [0.] or [1.]: constant-Q bins already sit on the
    equal-tempered ladder, so folding is an assignment, not an interpolation.
    The [bins_per_octave / n_chroma] bins of one chroma step are merged, the
    merge window is centred on the step rather than starting at it, and the
    whole assignment is rotated so that the ladder's lowest bin lands on the
    pitch class of [Cqt.Config.fmin] — untuned: the [tuning] shift plays no
    part in the rotation.

    Raises [Invalid_argument] if [n_chroma] is smaller than [1] or if
    [Cqt.Config.bins_per_octave c] is not an integer multiple of it. *)

val of_cqt :
     ?n_chroma:int
  -> ?norm:norm
  -> Cqt.Config.t
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [of_cqt c m] is the {e magnitude} constant-Q transform [m] — the output of
    {!Cqt.power_spectrum}[ ~power:1.] — folded onto pitch classes and
    normalised per frame, mapping [[...; n_bins; frames]] to
    [[...; n_chroma; frames]]. [n_chroma] defaults to [12] and [norm] to
    [`Inf].

    Raises [Invalid_argument] if [m] has rank below two, if its
    second-to-last axis does not hold [Cqt.Config.n_bins c] bins, if
    [n_chroma] does not divide [Cqt.Config.bins_per_octave c], or if [norm] is
    [`P p] with [p] not finite and positive. *)

(** {1:parity Parity with librosa 0.11}

    The filterbanks match [librosa.filters.chroma] and
    [librosa.filters.cq_to_chroma] to double-precision rounding, and the
    projections match [librosa.feature.chroma_stft] and
    [librosa.feature.chroma_cqt] at matching explicit settings — with
    [dtype=numpy.float64] passed through to the filterbank, since librosa
    builds it in single precision by default. One departure:

    - {b The constant-Q chromagram is taken from the exact transform.}
      [librosa.feature.chroma_cqt] called on audio computes its own constant-Q
      transform with librosa's default sparsified filter basis, which moves
      chroma values by up to 13% of the frame peak (measured on music);
      parity here is against its [C=] path, given a transform computed without
      sparsification. Everything {!Cqt} itself names under its own parity
      section applies underneath.

    librosa's [threshold] parameter is deliberately absent: its default of
    [0.] zeroes nothing in a projection of non-negative magnitudes. *)
