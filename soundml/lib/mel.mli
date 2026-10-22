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

(** Mel filterbanks and their application to power spectra.

    A {!Config.t} fixes the filterbank geometry once and precomputes the
    triangular weight matrix at creation, in double precision; {!apply}
    projects a power spectrum onto the mel bands, each band adding its bins in
    a fixed order, and {!stage} is the same projection as a memoryless
    {!Pipeline} stage consuming {!Stft.power_stage} output.

    The weights follow librosa 1.0 ([librosa.filters.mel]): [n_mels]
    triangular filters with breakpoints equally spaced on the mel scale
    between [f_min] and [f_max], on the Slaney or HTK variant of the scale
    ({!Convert.hz_to_mel}), optionally area-normalised. Parity is enforced
    against committed golden vectors in the test suite. *)

(** {1 Configuration} *)

module Config : sig
  (** The type for validated, immutable mel-filterbank configurations.
      Creation precomputes the [[n_mels; bins]] weight matrix in double
      precision; configurations are cheap to share and compare, and the
      config-owned matrix is immutable — {!filterbank} hands it out itself at
      float64 and a cast of it otherwise. *)
  type t

  val create :
       ?f_min:float
    -> ?f_max:float
    -> ?scale:[`Slaney | `Htk]
    -> ?norm:[`Slaney | `None]
    -> n_mels:int
    -> sample_rate:int
    -> fft_size:int
    -> unit
    -> t
  (** [create ~n_mels ~sample_rate ~fft_size ()] is a mel-filterbank
      configuration projecting the [fft_size / 2 + 1] bins of an [fft_size]-
      point spectrum at [sample_rate] onto [n_mels] mel bands.

      [f_min] and [f_max] bound the filterbank in hertz and default to [0.]
      and [sample_rate / 2.]. [scale] selects the mel scale variant of
      {!Convert.hz_to_mel} and defaults to [`Slaney]. [norm]
      defaults to [`Slaney], which divides each filter by half the width of
      its band in hertz so it integrates to approximately one; [`None] keeps
      unit peak gain.

      Raises [Invalid_argument] if [n_mels], [sample_rate] or [fft_size] is
      smaller than [1], if [f_min] is not finite and non-negative, if [f_max]
      is not finite and greater than [f_min] or exceeds the Nyquist frequency
      [sample_rate / 2.], if adjacent mel breakpoints collapse in double
      precision, or if some filter spans no FFT bin — raise [fft_size] or
      lower [n_mels] instead. The last three checks are stricter than the
      reference implementation, which builds past Nyquist, degrades on
      collapsed breakpoints and only warns on empty filters. *)

  val n_mels : t -> int
  (** [n_mels c] is the number of mel bands. *)

  val sample_rate : t -> int
  (** [sample_rate c] is the sample rate in hertz the filterbank was built
      for. *)

  val fft_size : t -> int
  (** [fft_size c] is the FFT length in samples the filterbank was built
      for. *)

  val bins : t -> int
  (** [bins c] is the number of frequency bins the filterbank consumes,
      [fft_size / 2 + 1]. *)

  val f_min : t -> float
  (** [f_min c] is the lower frequency bound in hertz. *)

  val f_max : t -> float
  (** [f_max c] is the upper frequency bound in hertz. *)

  val scale : t -> [`Slaney | `Htk]
  (** [scale c] is the mel scale variant. *)

  val norm : t -> [`Slaney | `None]
  (** [norm c] is the filter normalisation mode. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt c] prints [c] on [fmt] in a compact single-line form. *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff [a] and [b] were created from the same
      parameters; the frequency bounds are compared with [Float.equal]. *)
end

(** {1 The filterbank} *)

val filterbank : (float, 'a) Nx.dtype -> Config.t -> (float, 'a) Nx.t
(** [filterbank dtype c] is the [[n_mels; bins]] weight matrix of [c] in
    [dtype]: row [m] holds the triangular filter of mel band [m] over the
    FFT bins.

    The matrix is precomputed once, in double precision, inside
    {!Config.create}; every call returns the config-owned matrix cast to
    [dtype] (the matrix itself at float64). Tensors are immutable, so no
    caller can affect [c] or any other caller through the result. *)

val apply : Config.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [apply c s] is the power spectrum [s] projected onto the mel bands of
    [c] — the product [filterbank ⋅ s] — mapping [[...; bins; frames]] to
    [[...; n_mels; frames]]. Leading axes broadcast, so a batch of
    spectrograms is one call. The product is computed in double precision
    and rounded to the dtype of [s] once, at the boundary.

    Each output adds its filter's bins one at a time in ascending order, so a
    frame projects to the same bits alone or among any number of others:
    {!stage} agrees with [apply] on every chunking, at every dtype. Bins
    outside a filter's support contribute nothing, even where [s] is not
    finite — the one way the result departs from a dense product.

    Raises [Invalid_argument] if [s] has rank below two or if its
    second-to-last axis does not hold [Config.bins c] bins. *)

(** {1 Pipeline stage} *)

val stage : Config.t -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage c] is {!apply} as a pipeline stage: memoryless, rate 1:1 in
    frames, latency zero and format-preserving — channels, items per second
    and the per-chunk bound all pass through unchanged, since the projection
    maps each spectral frame's bins to mel bands without touching the frame
    axis. It consumes {!Stft.power_stage} output ([[...; bins; frames]]
    chunks) and is polymorphic in its capability, so one value drives both
    {!Pipeline.run} and {!Pipeline.Stream}. An empty chunk maps to an empty
    chunk, [[...; n_mels; 0]].

    The stream {!Pipeline.Format} does not carry a bin count, so a bins
    mismatch is reported by {!apply}'s [Invalid_argument] at the first
    chunk, not at prepare time. *)

(**/**)

val power_spectrogram :
  Stft.Config.t -> Config.t -> (float, 'a) Nx.t -> (float, 'a) Nx.t option
(* Library-internal: [apply c (Stft.power_spectrum stft x)] at float32 and
   float64, bit for bit, without materializing the power spectrum; [None]
   otherwise. *)
