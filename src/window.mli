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

(** Window functions for spectral analysis and filter design.

    A window is described by a {!t} specification and instantiated with
    {!make} once a dtype and a length are chosen. Values match librosa 0.11
    (which delegates to the scipy signal windows) to double-precision
    rounding error; parity is enforced against committed golden vectors in
    the test suite. *)

(** The type for window specifications. A specification is data: it can be
    compared, printed, and stored in configs before any dtype or length is
    chosen. Shape parameters are validated when the specification is used,
    not when it is built. *)
type t =
  | Hann  (** The raised-cosine window; zero at both symmetric endpoints. *)
  | Hamming
      (** The Hamming window, coefficients [0.54]/[0.46]; endpoints do not
          reach zero. *)
  | Blackman  (** The classic three-term Blackman window. *)
  | Blackman_harris
      (** The minimum four-term Blackman-Harris window (92 dB sidelobes). *)
  | Nuttall  (** Nuttall's four-term window with continuous first derivative. *)
  | Bartlett  (** The triangular window; zero at both symmetric endpoints. *)
  | Kaiser of float
      (** [Kaiser beta] is the Kaiser window with shape parameter [beta].
          Larger [beta] narrows the main lobe trade-off toward lower
          sidelobes; [beta = 0.] is rectangular. *)
  | Gaussian of float
      (** [Gaussian std] is the Gaussian window with standard deviation
          [std], in samples. *)
  | Tukey of float
      (** [Tukey taper] is the tapered-cosine window with taper fraction
          [taper] in [\[0, 1\]]: [0.] is rectangular, [1.] is {!Hann}. *)
  | Flat_top
      (** The five-term flat-top window for amplitude measurement; takes
          small negative values. *)
  | Rectangular  (** The all-ones (boxcar) window. *)

val make :
  (float, 'a) Nx.dtype -> ?periodic:bool -> t -> int -> (float, 'a) Nx.t
(** [make dtype ?periodic w n] is the [n]-point window [w] as a rank-one
    tensor of dtype [dtype].

    [periodic] defaults to [true], the DFT-even form used for spectral
    analysis: for [n >= 2] it is the symmetric window of length [n + 1]
    with the last sample dropped. Symmetric windows, the form used for
    filter design, use [~periodic:false]. A one-point window is [[1.]] in
    both forms.

    Raises [Invalid_argument] if [n < 1], or if the shape parameter of [w]
    is invalid: [Kaiser beta] requires a finite non-negative [beta],
    [Gaussian std] a finite positive [std], and [Tukey taper] a [taper] in
    [\[0, 1\]]. *)

val cola : t -> length:int -> hop:int -> bool
(** [cola w ~length ~hop] is [true] iff the [length]-point periodic window
    [w] satisfies constant overlap-add at hop [hop], the invertibility
    precondition of inverse STFT overlap-add.

    The check is numerical: the window is overlap-added at every shift that
    is a multiple of [hop], and the resulting per-sample sums over one hop
    period must be positive and stay within a relative tolerance of [1e-10]
    of their mean.

    Raises [Invalid_argument] if [length < 1], if [hop] is not in
    [\[1, length\]], or if the shape parameter of [w] is invalid as
    documented for {!make}. *)

val pp : Format.formatter -> t -> unit
(** [pp fmt w] prints [w] on [fmt] in a compact lowercase form, e.g.
    [hann] or [kaiser(8.6)]. *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] are the same specification; shape
    parameters are compared with [Float.equal]. *)
