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

(** Double-precision reductions in a fixed association.

    Nx sums in an association that depends on the shape and layout of its
    operand, and {!Nx.matmul} hands large products to Accelerate on macOS, so
    one frame reduced alone and the same frame reduced among others can round
    differently. A stage's output must not depend on how its input was
    chunked, so every per-frame reduction of the library goes through this
    module instead: each output is [0.] plus its terms, added one at a time in
    ascending index along the reduced axis, whatever the shape around it. The
    bits of an output therefore depend on its own terms alone, never on how
    many frames or signals share the call.

    Every function reduces the second-to-last axis of a [[...; k; n]] tensor,
    the bin axis of the library's spectral layout, and maps the leading axes
    independently. Operands are read in place when they lie in C order (a
    contiguous copy otherwise); results are fresh. The work is spread over
    domains by {!Par}, by planes and column tiles, which never changes an
    output's terms or their order. *)

val sum : (float, Nx.float64_elt) Nx.t -> (float, Nx.float64_elt) Nx.t
(** [sum x] is [x] summed along its second-to-last axis, kept with size [1]:
    [[...; k; n]] to [[...; 1; n]], output [t] of a plane being
    [(((0. +. x.(0, t)) +. x.(1, t)) +. ...) +. x.(k - 1, t)]. A sum of nothing
    is [0.], and an exactly-zero sum is [0.], never [-0.].

    Raises [Invalid_argument] if [x] has rank below [2]. *)

val mean : (float, Nx.float64_elt) Nx.t -> (float, Nx.float64_elt) Nx.t
(** [mean x] is [sum x] divided by the length [k] of the reduced axis.

    Raises [Invalid_argument] if [x] has rank below [2] or [k = 0]. *)

(** {1 Projections} *)

(** The type for [[m; k]] matrices prepared for {!project}: the weights, and
    for each row the range of columns outside which it is zero. *)
type matrix

val matrix : (float, Nx.float64_elt) Nx.t -> matrix
(** [matrix w] prepares the rank-2 matrix [w] for {!project}.

    Raises [Invalid_argument] if [w] is not rank 2. *)

val rows : matrix -> int
(** [rows w] is [m], the number of rows of [w]. *)

val columns : matrix -> int
(** [columns w] is [k], the number of columns of [w]. *)

val project : matrix -> (float, 'a) Nx.t -> (float, Nx.float64_elt) Nx.t
(** [project w x] is the product of [w] with [x], [[...; k; n]] to
    [[...; m; n]]: output [(i, t)] of a plane is [0.] plus
    [w.(i, j) *. x.(j, t)] added in ascending [j] over the columns between row
    [i]'s first and last nonzero weight. Columns outside that range contribute
    nothing, even where [x] is not finite: that is the only way the result
    departs from the dense product. Each term is one multiply-add, which
    rounds once where OCaml contracts it into a fused instruction (arm64) and
    twice elsewhere (x86_64): an output's bits depend on its terms and the
    platform, never on its batch.

    [x] may have any float dtype: float32 and float64 are read in place,
    other dtypes are widened to float64 first.

    Raises [Invalid_argument] if [x] has rank below [2] or its second-to-last
    axis is not [k] long. *)

val project_cast : matrix -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [project_cast w x] is [Nx.cast (Nx.dtype x) (project w x)], each output
    rounded once from its double-precision fold, without the float64
    intermediate when [x] is float32 or float64.

    Raises [Invalid_argument] as {!project} does. *)
