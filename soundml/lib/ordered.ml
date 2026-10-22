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

(* Every kernel walks a [[outer; k; n]] view of a C-order operand: [outer]
   planes of [k] rows of [n] columns. Each output accumulates its terms in
   ascending row order, starting from [0.], with the column loop innermost. The
   columns are walked in tiles of [tile]: the rows of a tile stay in cache while
   every output of the tile reads them, and each (plane, tile) pair is one unit
   of parallel work. Tiling and the split across domains change which outputs
   are computed together, never the order of any output's terms.

   The row loops are unrolled by four, so an accumulator is loaded and stored
   once per four terms: [(((d +. a) +. b) +. c) +. e] is evaluated left to
   right, the same additions as four separate steps. OCaml never contracts a
   multiply into an add, so the bits do not depend on the machine.

   The loops are written once per storage kind, float32 and float64, so that
   every element access compiles to a plain load; other dtypes are widened to
   float64 first. *)

open Bigarray

type f64 = Store.f64

type f32 = Store.f32

let tile = 64

let tiles n = (n + tile - 1) / tile

(* Roughly the number of additions worth one parallel chunk. *)
let grain_terms = 65_536

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

(* [units ~outer ~n ~cost f] runs [f o t0 t1] over every (plane, column tile)
   pair, [cost] being the additions one tile column costs. *)
let units ~outer ~n ~cost f =
  let per_plane = tiles n in
  let grain = Stdlib.max 1 (grain_terms / Stdlib.max 1 (cost * tile)) in
  Par.ranges ~n:(outer * per_plane) ~grain (fun lo hi ->
      for u = lo to hi - 1 do
        let o = u / per_plane and t0 = u mod per_plane * tile in
        f o t0 (Stdlib.min n (t0 + tile))
      done )

(* {1 Sums} *)

let sum_tile (src : f64) (dst : f64) ~k ~n o t0 t1 =
  let plane = o * k * n and d0 = o * n in
  for t = t0 to t1 - 1 do
    Array1.unsafe_set dst (d0 + t) 0.
  done ;
  let j = ref 0 in
  while !j + 3 < k do
    let r0 = plane + (!j * n) in
    let r1 = r0 + n in
    let r2 = r1 + n in
    let r3 = r2 + n in
    for t = t0 to t1 - 1 do
      Array1.unsafe_set dst (d0 + t)
        ( Array1.unsafe_get dst (d0 + t)
        +. Array1.unsafe_get src (r0 + t)
        +. Array1.unsafe_get src (r1 + t)
        +. Array1.unsafe_get src (r2 + t)
        +. Array1.unsafe_get src (r3 + t) )
    done ;
    j := !j + 4
  done ;
  while !j < k do
    let r = plane + (!j * n) in
    for t = t0 to t1 - 1 do
      Array1.unsafe_set dst (d0 + t)
        (Array1.unsafe_get dst (d0 + t) +. Array1.unsafe_get src (r + t))
    done ;
    incr j
  done

let sum x =
  let shape, outer, k, n = geometry "sum" x in
  let shape = Array.copy shape in
  shape.(Array.length shape - 2) <- 1 ;
  Store.make64 shape (fun out ->
      Store.read64 x (fun src ->
          units ~outer ~n ~cost:k (sum_tile src out ~k ~n) ) )

let mean x =
  let _, _, k, _ = geometry "mean" x in
  if k = 0 then
    invalid_arg "Ordered.mean: cannot average over an axis of length 0" ;
  Nx.div_s (sum x) (Float.of_int k)

(* {1 Projections} *)

type matrix =
  { m: int
  ; k: int
  ; weights: f64 (* [m * k], row-major *)
  ; bounds: int array
        (* per row [i], at [2i] and [2i + 1]: the first nonzero column and one
           past the last, an empty range for an all-zero row *)
  ; terms: int (* the multiply-adds of one column: the ranges' total *) }

let matrix w =
  if Nx.ndim w <> 2 then
    invalid_arg
      (Printf.sprintf
         "Ordered.matrix: cannot prepare a rank-%d tensor (a matrix is rank 2)"
         (Nx.ndim w) ) ;
  let m = Nx.dim 0 w and k = Nx.dim 1 w in
  let values = Nx.to_array (Nx.reshape [|m * k|] w) in
  let weights = Array1.create float64 c_layout (m * k) in
  Array.iteri (fun i v -> Array1.unsafe_set weights i v) values ;
  let zero i j = Float.equal (Float.abs values.((i * k) + j)) 0. in
  let bounds = Array.make (2 * m) 0 in
  let terms = ref 0 in
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
    bounds.((2 * i) + 1) <- !last ;
    terms := !terms + (!last - !first)
  done ;
  {m; k; weights; bounds; terms= !terms}

let rows w = w.m

let columns w = w.k

(* [fold64 mx src ~n acc ~plane ~i ~t0 ~width] and [fold32] add row [i]'s terms
   over one tile into the scratch [acc] (whose slot [t] is column [t0 + t]), in
   ascending column order, four terms per load and store of a slot. *)
let fold64 (mx : matrix) (src : f64) ~n acc ~plane ~i ~t0 ~width =
  let first = mx.bounds.(2 * i) and last = mx.bounds.((2 * i) + 1) in
  let wbase = i * mx.k in
  let j = ref first in
  while !j + 3 < last do
    let j0 = !j in
    let w0 = Array1.unsafe_get mx.weights (wbase + j0)
    and w1 = Array1.unsafe_get mx.weights (wbase + j0 + 1)
    and w2 = Array1.unsafe_get mx.weights (wbase + j0 + 2)
    and w3 = Array1.unsafe_get mx.weights (wbase + j0 + 3) in
    let r0 = plane + (j0 * n) + t0 in
    let r1 = r0 + n in
    let r2 = r1 + n in
    let r3 = r2 + n in
    for t = 0 to width - 1 do
      Float.Array.unsafe_set acc t
        ( Float.Array.unsafe_get acc t
        +. (w0 *. Array1.unsafe_get src (r0 + t))
        +. (w1 *. Array1.unsafe_get src (r1 + t))
        +. (w2 *. Array1.unsafe_get src (r2 + t))
        +. (w3 *. Array1.unsafe_get src (r3 + t)) )
    done ;
    j := j0 + 4
  done ;
  while !j < last do
    let wj = Array1.unsafe_get mx.weights (wbase + !j) in
    let r = plane + (!j * n) + t0 in
    for t = 0 to width - 1 do
      Float.Array.unsafe_set acc t
        (Float.Array.unsafe_get acc t +. (wj *. Array1.unsafe_get src (r + t)))
    done ;
    incr j
  done

let fold32 (mx : matrix) (src : f32) ~n acc ~plane ~i ~t0 ~width =
  let first = mx.bounds.(2 * i) and last = mx.bounds.((2 * i) + 1) in
  let wbase = i * mx.k in
  let j = ref first in
  while !j + 3 < last do
    let j0 = !j in
    let w0 = Array1.unsafe_get mx.weights (wbase + j0)
    and w1 = Array1.unsafe_get mx.weights (wbase + j0 + 1)
    and w2 = Array1.unsafe_get mx.weights (wbase + j0 + 2)
    and w3 = Array1.unsafe_get mx.weights (wbase + j0 + 3) in
    let r0 = plane + (j0 * n) + t0 in
    let r1 = r0 + n in
    let r2 = r1 + n in
    let r3 = r2 + n in
    for t = 0 to width - 1 do
      Float.Array.unsafe_set acc t
        ( Float.Array.unsafe_get acc t
        +. (w0 *. Array1.unsafe_get src (r0 + t))
        +. (w1 *. Array1.unsafe_get src (r1 + t))
        +. (w2 *. Array1.unsafe_get src (r2 + t))
        +. (w3 *. Array1.unsafe_get src (r3 + t)) )
    done ;
    j := j0 + 4
  done ;
  while !j < last do
    let wj = Array1.unsafe_get mx.weights (wbase + !j) in
    let r = plane + (!j * n) + t0 in
    for t = 0 to width - 1 do
      Float.Array.unsafe_set acc t
        (Float.Array.unsafe_get acc t +. (wj *. Array1.unsafe_get src (r + t)))
    done ;
    incr j
  done

(* [project_units mx ~outer ~n fold store] runs every (plane, tile) unit: for
   each output row, [fold acc ~plane ~i ~t0 ~width] fills the zeroed scratch and
   [store acc ~row ~width] writes it out at output offset [row]. *)
let project_units mx ~outer ~n fold store =
  let k = mx.k and m = mx.m in
  units ~outer ~n ~cost:(Stdlib.max 1 mx.terms) (fun o t0 t1 ->
      let width = t1 - t0 in
      let acc = Float.Array.create tile in
      let plane = o * k * n in
      for i = 0 to m - 1 do
        Float.Array.fill acc 0 width 0. ;
        fold acc ~plane ~i ~t0 ~width ;
        store acc ~row:((((o * m) + i) * n) + t0) ~width
      done )

let store64 (out : f64) acc ~row ~width =
  for t = 0 to width - 1 do
    Array1.unsafe_set out (row + t) (Float.Array.unsafe_get acc t)
  done

let store32 (out : f32) acc ~row ~width =
  for t = 0 to width - 1 do
    Array1.unsafe_set out (row + t) (Float.Array.unsafe_get acc t)
  done

let check_project mx x =
  let shape, outer, k, n = geometry "project" x in
  if k <> mx.k then
    invalid_arg
      (Printf.sprintf
         "Ordered.project: cannot project %d rows through a matrix of %d \
          columns"
         k mx.k ) ;
  let out_shape = Array.copy shape in
  out_shape.(Array.length shape - 2) <- mx.m ;
  (out_shape, outer, n)

(* [project_into mx x store] folds [x] — read in place at float32 or float64,
   widened to float64 otherwise — and hands each finished tile row to
   [store]. *)
let project_into (type a) mx (x : (float, a) Nx.t) ~outer ~n store =
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.read32 x (fun src ->
          project_units mx ~outer ~n (fold32 mx src ~n) store )
  | Nx.Float64 ->
      Store.read64 x (fun src ->
          project_units mx ~outer ~n (fold64 mx src ~n) store )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Store.read64 (Nx.cast Nx.float64 x) (fun src ->
          project_units mx ~outer ~n (fold64 mx src ~n) store )

let project mx x =
  let out_shape, outer, n = check_project mx x in
  Store.make64 out_shape (fun out -> project_into mx x ~outer ~n (store64 out))

let project_cast (type a) mx (x : (float, a) Nx.t) : (float, a) Nx.t =
  let out_shape, outer, n = check_project mx x in
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.make32 out_shape (fun out ->
          project_into mx x ~outer ~n (store32 out) )
  | Nx.Float64 ->
      Store.make64 out_shape (fun out ->
          project_into mx x ~outer ~n (store64 out) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast (Nx.dtype x) (project mx x)
