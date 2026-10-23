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

(** Fused element loops over tensor storage.

    Each function here replaces a chain of whole-tensor Nx operations — each
    a pass over memory and a fresh allocation — with one pass that reads its
    operands in place and writes a fresh result, split across domains by
    {!Par}. Every output element is computed from its own inputs alone, in a
    fixed order, so the split never changes a bit.

    The loops exist for float32 and float64 storage, the dtypes the library's
    hot paths run at; callers keep their Nx formulation for the others. *)

val spectrum_power :
     (float, 'a) Nx.dtype
  -> power:float
  -> (Complex.t, 'c) Nx.t
  -> (float, 'a) Nx.t option
(** [spectrum_power dtype ~power z] is [|z| ^ power] transposed over the last
    two axes, [[...; frames; bins]] to [[...; bins; frames]], in [dtype]:
    [Some] when [dtype] is float32 and [z] complex64, or [dtype] float64 and
    [z] complex128, [None] otherwise.

    Each element is computed in double precision and rounded once: [power =
    2.] is [re * re + im * im], [power = 1.] its square root (at float64,
    [Float.hypot re im], which cannot overflow early), and any other power
    that root raised to [power]. *)

(** The boundary extension of a signal: positions before the start and past
    the end read the signal reflected about its end samples (multiple
    reflections for a short signal), the nearest end sample, or a constant. *)
type extension = Reflect | Edge | Constant of float

val frames :
     ?into:Store.buffer
  -> window:Float.Array.t
  -> hop:int
  -> left:int
  -> ext:extension
  -> first:int
  -> count:int
  -> (float, 'a) Nx.t
  -> (float, Nx.float64_elt) Nx.t
(** [frames ~window ~hop ~left ~ext ~first ~count x] is frames [\[first, first
    + count)] of the signals [x] ([[...; n]]), each multiplied by [window], in
    double precision: [[...; count; fft]] with [fft] the window's length.
    Frame [p] covers positions [\[p * hop, p * hop + fft)] of the padded
    stream, whose position [q] is sample [q - left] of the signal, extended
    by [ext] outside it. Element [j] of frame [p] is the padded sample widened
    to double times [window.(j)]: the product [Nx.stft] forms.

    Requires [n >= 1] whenever a frame reaches outside the signal. [into] is
    {!Store.make64}'s. *)

(** A frame energy: the root mean square, or the zero-crossing rate with a
    threshold. *)
type energy = Rms | Zcr of float

val energy :
     energy
  -> frame_length:int
  -> hop:int
  -> count:int
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [energy kind ~frame_length ~hop ~count x] is the energy of frames
    [\[0, count)] of the padded-stream segments [x] ([[...; n]], [n] at least
    [(count - 1) * hop + frame_length]): [[...; 1; count]] in [x]'s dtype,
    frame [p] covering samples [\[p * hop, p * hop + frame_length)].

    [Rms] is the root of the mean square, the squares summed in ascending
    order from [0.] in double precision. [Zcr threshold] is the fraction of
    the frame's [frame_length - 1] consecutive-sample pairs whose signs
    differ, divided by [frame_length], a sample being negative iff it lies
    strictly below [-threshold]. Each value is rounded once into [x]'s
    dtype. *)

(** {1 Spectral shape}

    Each function reduces every frame of a non-empty spectrogram [[...; k;
    n]] (any float dtype) to [[...; 1; n]] in its dtype, adding the frame's
    bins in ascending order from [0.] in double precision and rounding once.
    [fq] holds the [k] bin frequencies. When some value of the spectrogram is
    negative or NaN, [invalid ()] is called once the frames are computed, and
    is expected to raise. *)

val centroid :
     fq:Float.Array.t
  -> invalid:(unit -> unit)
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [centroid] is the sum over bins of [fq.(j) *. (s /. l)], [l] the frame's
    bin sum, or [1.] where that sum is below the smallest positive normal
    double. *)

val bandwidth :
     fq:Float.Array.t
  -> p:float
  -> ?centroid:(float, Nx.float64_elt) Nx.t
  -> invalid:(unit -> unit)
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [bandwidth] is the sum over bins of [(s /. l) *. |c -. fq.(j)| ** p],
    raised to [1 /. p] ([sqrt] and squares at [p = 2.]), [l] as for
    {!centroid} and [c] the frame's centroid: [centroid]'s, [[...; 1; n]], when
    given, the frame's own otherwise. *)

val rolloff :
     fq:Float.Array.t
  -> roll:float
  -> invalid:(unit -> unit)
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [rolloff] is the least [fq.(j)] whose running bin sum reaches [roll] times
    the frame's total. *)

val flatness :
     amin:float
  -> power:float
  -> invalid:(unit -> unit)
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [flatness] is [exp (mean (log v)) /. mean v] over the frame's bins, [v]
    being each bin raised to [power] (squared at [power = 2.]) and floored at
    [amin]. *)

(** {1 Synthesis} *)

val widen_rows :
  (Complex.t, 'c) Nx.t -> (Complex.t, Nx.complex64_elt) Nx.t option
(** [widen_rows z] is the spectrum [z] ([[...; bins; frames]]) widened to
    complex128 and transposed over its last two axes, [[...; frames; bins]]:
    [Some] when [z] is complex64, [None] otherwise. The widening is exact. A
    [z] that views frames-major storage transposed, as the analysis returns
    it, is read in place. *)

val overlap_add :
     ?into:Store.buffer
  -> window:Float.Array.t
  -> scale:float
  -> hop:int
  -> period:Float.Array.t
  -> partial:(int -> float)
  -> head:int
  -> stop:int
  -> left:int
  -> out_len:int
  -> valid:int
  -> (float, Nx.float64_elt) Nx.t
  -> (float, Nx.float64_elt) Nx.t
(** [overlap_add ~window ~scale ~hop ~period ~partial ~head ~stop ~left
    ~out_len ~valid frames] is the least-squares synthesis of the unscaled
    inverse-transformed frames [frames] ([[...; count; fft]]): [[...;
    out_len]], output [i] being padded position [left + i]. Its numerator adds
    the taps of the frames reaching the position, the frame opening latest
    first, a tap being the frame's sample times [scale], then times the
    window — the values [Nx.mul] forms from a transform scaled by [scale] and
    the window, in that order; its denominator is the envelope,
    [period.(q mod hop)] at padded positions [q] in [\[head, stop)] and
    [partial q] elsewhere. Outputs from [valid] on are zero. [into] is
    {!Store.make64}'s. *)

val overlap_add32 :
     window:Float.Array.t
  -> scale:float
  -> hop:int
  -> period:Float.Array.t
  -> partial:(int -> float)
  -> head:int
  -> stop:int
  -> left:int
  -> out_len:int
  -> valid:int
  -> (float, Nx.float64_elt) Nx.t
  -> (float, Nx.float32_elt) Nx.t
(** [overlap_add32] is {!overlap_add} rounded once into float32: the values
    [Nx.cast Nx.float32] makes of {!overlap_add}'s, without the float64
    intermediate. *)

val gl_update :
     ?into:Store.buffer
  -> beta:float
  -> magnitudes:(float, Nx.float64_elt) Nx.t
  -> ?previous:(Complex.t, Nx.complex64_elt) Nx.t
  -> (Complex.t, Nx.complex64_elt) Nx.t
  -> (Complex.t, Nx.complex64_elt) Nx.t
(** [gl_update ~beta ~magnitudes ?previous rebuilt] is one fast Griffin-Lim
    spectrum, elementwise over tensors of one shape and layout: [e] is
    [rebuilt], less [beta] times [previous] when given, and the result is
    [magnitudes *. e /. (|e| +. min_float)], [|e|] computed by
    [Float.hypot]. [into] is {!Store.make64}'s. *)

(** {1 Band extremes} *)

val extremes :
     bands:int array
  -> (float, 'a) Nx.t
  -> (float, Nx.float64_elt) Nx.t * (float, Nx.float64_elt) Nx.t
(** [extremes ~bands s] is the valleys and peaks of the non-empty spectrogram
    [s] ([[...; k; n]]): for each band [b] — rows [\[lo, lo + len)] with
    [bands.(3b) = lo], [bands.(3b + 1) = len] and [bands.(3b + 2) = take], [1
    <= take <= len] — and each frame, the mean of the [take] least values and
    the mean of the [take] greatest, each [take] values added in ascending
    order from [0.] in double precision, NaN ordering after every number.
    Both are [[...; nb; n]] at float64, [nb] the number of bands. *)

(** {1 Decibels} *)

val decibels :
     scale:float
  -> offset:float
  -> amin:float
  -> magnitude:bool
  -> range:float option
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t option
(** [decibels ~scale ~offset ~amin ~magnitude ~range x] is, elementwise,
    [log (max amin v) *. scale -. offset] with [v] the element ([abs] of it
    when [magnitude]), the product and the difference each rounded in double
    precision and the result rounded once into [x]'s dtype; [max] keeps a NaN.
    When [range] is given, every value of a channel — the whole tensor up to
    rank two, each index of the leading axes beyond — is then floored at the
    channel's greatest value less [range], rounded into the dtype; a channel
    holding a NaN becomes NaN throughout. [Some] at float32 and float64,
    [None] otherwise. *)

val decibels64 :
     scale:float
  -> offset:float
  -> amin:float
  -> magnitude:bool
  -> range:float option
  -> (float, 'a) Nx.t
  -> (float, Nx.float64_elt) Nx.t option
(** [decibels64] is {!decibels} at float64 over a float32 or float64 [x] —
    the float64 decibels of [Nx.cast Nx.float64 x] — and [None] otherwise. *)

(** {1 Onset flux} *)

val flux :
     lag:int
  -> first:int
  -> count:int
  -> (float, 'a) Nx.t
  -> (float, Nx.float64_elt) Nx.t option
(** [flux ~lag ~first ~count x] is the spectral flux of frames [\[first, first
    + count)] of the spectrogram [x] ([[...; k; n]], [k >= 1], [first >= lag]
    and [first + count <= n]): [[...; count]] at float64, frame [t] being the
    mean over bins of [max 0 (x.(j, first + t) -. x.(j, first + t - lag))],
    computed in double precision, the bins added in ascending order from [0.]
    and the sum divided by [k]; [max] keeps a NaN. [Some] at float32 and
    float64, [None] otherwise. *)
