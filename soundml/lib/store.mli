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

(** Tensor storage for the library's own loops.

    Results are allocated as rig host buffers, the memory nx allocates its
    own results in. rig paces the collector by the memory a program holds
    and hands a dropped buffer's pages back to the next allocation of its
    size, where a bigarray made by [Bigarray.Array1.create] makes the runtime
    run a major cycle for nearly every multi-megabyte result and faults in
    fresh pages for each. *)

type f32 = (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t

type f64 = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

val make32 : int array -> (f32 -> unit) -> (float, Nx.float32_elt) Nx.t
(** [make32 shape fill] is the float32 tensor of [shape] whose C-order
    elements [fill] wrote; [fill] is not called when [shape] holds no
    element, and must write every element otherwise. *)

(** The type for host buffers that successive results reuse. *)
type buffer

val buffer : int -> buffer
(** [buffer bytes] is a fresh host buffer of [bytes] bytes. *)

val make64 :
  ?into:buffer -> int array -> (f64 -> unit) -> (float, Nx.float64_elt) Nx.t
(** [make64] is {!make32} at float64. With [into], the tensor lies in that
    buffer's memory instead of fresh memory, which every tensor made in it
    before must no longer be used to read: it is overwritten. Raises
    [Invalid_argument] if the buffer is too small. *)

val make_c128 :
     ?into:buffer
  -> int array
  -> (f64 -> unit)
  -> (Complex.t, Nx.complex64_elt) Nx.t
(** [make_c128 shape fill] is the complex128 tensor of [shape] whose
    components [fill] wrote: element [2i] of the array is the real part of
    element [i] in C order, element [2i + 1] its imaginary part. [into] is
    {!make64}'s. *)

val make64_pair :
     int array
  -> (f64 -> f64 -> unit)
  -> (float, Nx.float64_elt) Nx.t * (float, Nx.float64_elt) Nx.t
(** [make64_pair shape fill] is two float64 tensors of [shape] that one call
    of [fill] wrote, in the order given. *)

val read32 : (float, Nx.float32_elt) Nx.t -> (f32 -> 'r) -> 'r
(** [read32 x f] is [f a] with [a] [x]'s elements in C order — [x]'s own
    storage when it holds them so, a copy otherwise — lent for the duration
    of the call: [f] must neither write nor keep [a]. *)

val read64 : (float, Nx.float64_elt) Nx.t -> (f64 -> 'r) -> 'r
(** [read64] is {!read32} at float64. *)

val read_parts32 : (Complex.t, Nx.complex32_elt) Nx.t -> (f32 -> 'r) -> 'r
(** [read_parts32 z f] is {!read32} over [z]'s components: element [2i] of
    [a] is the real part of [z]'s element [i], element [2i + 1] its
    imaginary part. *)

val read_parts64 : (Complex.t, Nx.complex64_elt) Nx.t -> (f64 -> 'r) -> 'r
(** [read_parts64] is {!read_parts32} at complex128. *)
