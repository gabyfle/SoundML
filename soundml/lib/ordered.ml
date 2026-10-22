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

(* The kernels live in ordered_stubs.c: each output accumulates its terms in
   ascending row order, starting from [0.], over a [[outer; k; n]] view of a
   C-order operand, with the column loop innermost. The stub file is compiled
   with -ffp-contract=off, so no multiply feeding an add is fused and the bits
   are the same on every machine. *)

type array1 = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t

external sum_stub : array1 -> array1 -> int array -> unit
  = "soundml_ordered_sum"

external cumsum_stub : array1 -> array1 -> int array -> unit
  = "soundml_ordered_cumsum"

external project_stub :
  array1 -> array1 -> array1 -> int array -> int array -> unit
  = "soundml_ordered_project"

let geometry op x =
  let nd = Nx.ndim x in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "Ordered.%s: cannot reduce a rank-%d tensor (the reduction needs \
          [...; k; n])"
         op nd ) ;
  let shape = Nx.shape x in
  let k = shape.(nd - 2) and n = shape.(nd - 1) in
  let outer = Array.fold_left ( * ) 1 (Array.sub shape 0 (nd - 2)) in
  (shape, outer, k, n)

let fresh len : array1 =
  Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout len

let wrap (a : array1) shape =
  Nx.of_bigarray (Bigarray.reshape (Bigarray.genarray_of_array1 a) shape)

(* [reading x f] lends [x]'s C-order elements to [f] for the duration of the
   call; an empty [x] has no storage worth reading. *)
let reading x f =
  if Nx.numel x = 0 then f (fresh 0)
  else Rig.Buffer.read (Nx.to_buffer x) Bigarray.float64 f

let sum x =
  let shape, outer, k, n = geometry "sum" x in
  let out = fresh (outer * n) in
  reading x (fun src -> sum_stub src out [|outer; k; n|]) ;
  let shape = Array.copy shape in
  shape.(Array.length shape - 2) <- 1 ;
  wrap out shape

let mean x =
  let _, _, k, _ = geometry "mean" x in
  if k = 0 then
    invalid_arg "Ordered.mean: cannot average over an axis of length 0" ;
  Nx.div_s (sum x) (Float.of_int k)

let cumsum x =
  let shape, outer, k, n = geometry "cumsum" x in
  let out = fresh (outer * k * n) in
  reading x (fun src -> cumsum_stub src out [|outer; k; n|]) ;
  wrap out shape

(* {1 Projections} *)

type matrix =
  { m: int
  ; k: int
  ; weights: array1 (* [m * k], row-major *)
  ; bounds: int array
        (* per row [i], at [2i] and [2i + 1]: the first nonzero column and one
           past the last, an empty range for an all-zero row *) }

let matrix w =
  if Nx.ndim w <> 2 then
    invalid_arg
      (Printf.sprintf
         "Ordered.matrix: cannot prepare a rank-%d tensor (a matrix is rank 2)"
         (Nx.ndim w) ) ;
  let m = Nx.dim 0 w and k = Nx.dim 1 w in
  let values = Nx.to_array (Nx.reshape [|m * k|] w) in
  let weights = fresh (m * k) in
  Array.iteri (fun i v -> Bigarray.Array1.unsafe_set weights i v) values ;
  let zero i j = Float.equal (Float.abs values.((i * k) + j)) 0. in
  let bounds = Array.make (2 * m) 0 in
  for i = 0 to m - 1 do
    let first = ref 0 in
    while !first < k && zero i !first do
      incr first
    done ;
    let last = ref k in
    while !last > !first && zero i (!last - 1) do
      decr last
    done ;
    bounds.(2 * i) <- !first ;
    bounds.((2 * i) + 1) <- !last
  done ;
  {m; k; weights; bounds}

let rows w = w.m

let columns w = w.k

let project w x =
  let shape, outer, k, n = geometry "project" x in
  if k <> w.k then
    invalid_arg
      (Printf.sprintf
         "Ordered.project: cannot project %d rows through a matrix of %d \
          columns"
         k w.k ) ;
  let out = fresh (outer * w.m * n) in
  reading x (fun src ->
      project_stub src out w.weights w.bounds [|outer; w.m; k; n|] ) ;
  let shape = Array.copy shape in
  shape.(Array.length shape - 2) <- w.m ;
  wrap out shape
