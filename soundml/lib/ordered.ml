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

   The row loops of the sums are unrolled by four, so an accumulator is loaded
   and stored once per four terms: [(((d +. a) +. b) +. c) +. e] is evaluated
   left to right, the same additions as four separate steps. Projections
   multiply-add (see below).

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
  let grain = Int.max 1 (grain_terms / Int.max 1 (cost * tile)) in
  Par.ranges ~n:(outer * per_plane) ~grain (fun lo hi ->
      for u = lo to hi - 1 do
        let o = u / per_plane and t0 = u mod per_plane * tile in
        f o t0 (Int.min n (t0 + tile))
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

(* {1 Projections}

   Output [(i, t)] of a projection starts from [0.] and takes one multiply-add
   [acc +. w *. x] per column of row [i]'s span, in ascending order. Every loop
   below spells that multiply-add the same way, both operands bound to names
   first: the shape OCaml's arm64 backend contracts into one fused instruction,
   rounded once (x86_64 keeps the multiply and the add apart). An output's bits
   therefore depend on its terms and the platform, never on the path that
   computed it.

   The outputs are computed in register groups, each group's accumulators held
   in registers for the whole span, so that every weight and sample loaded feeds
   several multiply-adds: eight columns of one row, or, where four consecutive
   rows share a long stretch of their spans (a block), four columns of all four
   rows over that stretch. A block's rows add their terms before the shared
   stretch alone, then the stretch together, then their terms after it alone, so
   each output still takes its terms in ascending order. An operand too narrow
   for the eight-column groups takes a kernel of its own (see below). *)

type block =
  { row: int (* the block's first row *)
  ; lo: int
  ; hi: int (* the stretch every row of the block spans *)
  ; packed: f64
        (* [(hi - lo) * 4] weights, column-major: [w.(row + r).(j)] at [4 * (j -
           lo) + r] *) }

type step = Row of int | Block of block

type matrix =
  { m: int
  ; k: int
  ; weights: f64 (* [m * k], row-major *)
  ; bounds: int array
        (* per row [i], at [2i] and [2i + 1]: the first nonzero column and one
           past the last, an empty range for an all-zero row *)
  ; terms: int (* the multiply-adds of one column: the ranges' total *)
  ; steps: step array (* every row once, in ascending order *) }

(* The shortest shared stretch worth a block. *)
let min_stretch = 16

let plan ~k weights bounds m =
  let steps = ref [] and i = ref 0 in
  while !i < m do
    let lo = ref 0 and hi = ref max_int in
    if !i + 4 <= m then
      for r = !i to !i + 3 do
        lo := Int.max !lo bounds.(2 * r) ;
        hi := Int.min !hi bounds.((2 * r) + 1)
      done ;
    if !i + 4 <= m && !hi - !lo >= min_stretch then begin
      let lo = !lo and hi = !hi and row = !i in
      let packed = Array1.create float64 c_layout ((hi - lo) * 4) in
      for j = lo to hi - 1 do
        for r = 0 to 3 do
          Array1.unsafe_set packed
            ((4 * (j - lo)) + r)
            (Array1.unsafe_get weights (((row + r) * k) + j))
        done
      done ;
      steps := Block {row; lo; hi; packed} :: !steps ;
      i := row + 4
    end
    else begin
      steps := Row !i :: !steps ;
      incr i
    end
  done ;
  Array.of_list (List.rev !steps)

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
  {m; k; weights; bounds; terms= !terms; steps= plan ~k weights bounds m}

let rows w = w.m

let columns w = w.k

(* The accumulators of a unit: four rows of a tile, row [r] at [r * tile]. *)
let scratch () = Float.Array.make (4 * tile) 0.

(* [span64 mx src ~n acc ~a ~width ~i ~j0 ~j1 ~x] and [span32] add row [i]'s
   terms for columns [\[j0, j1)] to the accumulators [acc.(a + c)], [c < width],
   of the samples at [x + c]: column [j]'s sample [c] is [src.(x + j * n +
   c)]. *)
let span64 (mx : matrix) (src : f64) ~n acc ~a ~width ~i ~j0 ~j1 ~x =
  let wbase = i * mx.k in
  let c = ref 0 in
  while !c + 7 < width do
    let a = a + !c and x = x + !c in
    let s0 = ref (Float.Array.unsafe_get acc a)
    and s1 = ref (Float.Array.unsafe_get acc (a + 1))
    and s2 = ref (Float.Array.unsafe_get acc (a + 2))
    and s3 = ref (Float.Array.unsafe_get acc (a + 3))
    and s4 = ref (Float.Array.unsafe_get acc (a + 4))
    and s5 = ref (Float.Array.unsafe_get acc (a + 5))
    and s6 = ref (Float.Array.unsafe_get acc (a + 6))
    and s7 = ref (Float.Array.unsafe_get acc (a + 7)) in
    for j = j0 to j1 - 1 do
      let w = Array1.unsafe_get mx.weights (wbase + j) and r = x + (j * n) in
      let x0 = Array1.unsafe_get src r
      and x1 = Array1.unsafe_get src (r + 1)
      and x2 = Array1.unsafe_get src (r + 2)
      and x3 = Array1.unsafe_get src (r + 3)
      and x4 = Array1.unsafe_get src (r + 4)
      and x5 = Array1.unsafe_get src (r + 5)
      and x6 = Array1.unsafe_get src (r + 6)
      and x7 = Array1.unsafe_get src (r + 7) in
      s0 := !s0 +. (w *. x0) ;
      s1 := !s1 +. (w *. x1) ;
      s2 := !s2 +. (w *. x2) ;
      s3 := !s3 +. (w *. x3) ;
      s4 := !s4 +. (w *. x4) ;
      s5 := !s5 +. (w *. x5) ;
      s6 := !s6 +. (w *. x6) ;
      s7 := !s7 +. (w *. x7)
    done ;
    Float.Array.unsafe_set acc a !s0 ;
    Float.Array.unsafe_set acc (a + 1) !s1 ;
    Float.Array.unsafe_set acc (a + 2) !s2 ;
    Float.Array.unsafe_set acc (a + 3) !s3 ;
    Float.Array.unsafe_set acc (a + 4) !s4 ;
    Float.Array.unsafe_set acc (a + 5) !s5 ;
    Float.Array.unsafe_set acc (a + 6) !s6 ;
    Float.Array.unsafe_set acc (a + 7) !s7 ;
    c := !c + 8
  done ;
  while !c < width do
    let a = a + !c and x = x + !c in
    let s = ref (Float.Array.unsafe_get acc a) in
    for j = j0 to j1 - 1 do
      let w = Array1.unsafe_get mx.weights (wbase + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s := !s +. (w *. v)
    done ;
    Float.Array.unsafe_set acc a !s ;
    incr c
  done

let span32 (mx : matrix) (src : f32) ~n acc ~a ~width ~i ~j0 ~j1 ~x =
  let wbase = i * mx.k in
  let c = ref 0 in
  while !c + 7 < width do
    let a = a + !c and x = x + !c in
    let s0 = ref (Float.Array.unsafe_get acc a)
    and s1 = ref (Float.Array.unsafe_get acc (a + 1))
    and s2 = ref (Float.Array.unsafe_get acc (a + 2))
    and s3 = ref (Float.Array.unsafe_get acc (a + 3))
    and s4 = ref (Float.Array.unsafe_get acc (a + 4))
    and s5 = ref (Float.Array.unsafe_get acc (a + 5))
    and s6 = ref (Float.Array.unsafe_get acc (a + 6))
    and s7 = ref (Float.Array.unsafe_get acc (a + 7)) in
    for j = j0 to j1 - 1 do
      let w = Array1.unsafe_get mx.weights (wbase + j) and r = x + (j * n) in
      let x0 = Array1.unsafe_get src r
      and x1 = Array1.unsafe_get src (r + 1)
      and x2 = Array1.unsafe_get src (r + 2)
      and x3 = Array1.unsafe_get src (r + 3)
      and x4 = Array1.unsafe_get src (r + 4)
      and x5 = Array1.unsafe_get src (r + 5)
      and x6 = Array1.unsafe_get src (r + 6)
      and x7 = Array1.unsafe_get src (r + 7) in
      s0 := !s0 +. (w *. x0) ;
      s1 := !s1 +. (w *. x1) ;
      s2 := !s2 +. (w *. x2) ;
      s3 := !s3 +. (w *. x3) ;
      s4 := !s4 +. (w *. x4) ;
      s5 := !s5 +. (w *. x5) ;
      s6 := !s6 +. (w *. x6) ;
      s7 := !s7 +. (w *. x7)
    done ;
    Float.Array.unsafe_set acc a !s0 ;
    Float.Array.unsafe_set acc (a + 1) !s1 ;
    Float.Array.unsafe_set acc (a + 2) !s2 ;
    Float.Array.unsafe_set acc (a + 3) !s3 ;
    Float.Array.unsafe_set acc (a + 4) !s4 ;
    Float.Array.unsafe_set acc (a + 5) !s5 ;
    Float.Array.unsafe_set acc (a + 6) !s6 ;
    Float.Array.unsafe_set acc (a + 7) !s7 ;
    c := !c + 8
  done ;
  while !c < width do
    let a = a + !c and x = x + !c in
    let s = ref (Float.Array.unsafe_get acc a) in
    for j = j0 to j1 - 1 do
      let w = Array1.unsafe_get mx.weights (wbase + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s := !s +. (w *. v)
    done ;
    Float.Array.unsafe_set acc a !s ;
    incr c
  done

(* [stretch64 src ~n b acc ~width ~x] and [stretch32] add block [b]'s shared
   stretch to the accumulators of its four rows, [acc.(r * tile + c)], over the
   samples at [x + c] as {!span64} reads them. *)
let stretch64 (src : f64) ~n b acc ~width ~x =
  let c = ref 0 in
  while !c + 3 < width do
    let a = !c and x = x + !c in
    let s00 = ref (Float.Array.unsafe_get acc a)
    and s01 = ref (Float.Array.unsafe_get acc (a + 1))
    and s02 = ref (Float.Array.unsafe_get acc (a + 2))
    and s03 = ref (Float.Array.unsafe_get acc (a + 3))
    and s10 = ref (Float.Array.unsafe_get acc (tile + a))
    and s11 = ref (Float.Array.unsafe_get acc (tile + a + 1))
    and s12 = ref (Float.Array.unsafe_get acc (tile + a + 2))
    and s13 = ref (Float.Array.unsafe_get acc (tile + a + 3))
    and s20 = ref (Float.Array.unsafe_get acc ((2 * tile) + a))
    and s21 = ref (Float.Array.unsafe_get acc ((2 * tile) + a + 1))
    and s22 = ref (Float.Array.unsafe_get acc ((2 * tile) + a + 2))
    and s23 = ref (Float.Array.unsafe_get acc ((2 * tile) + a + 3))
    and s30 = ref (Float.Array.unsafe_get acc ((3 * tile) + a))
    and s31 = ref (Float.Array.unsafe_get acc ((3 * tile) + a + 1))
    and s32 = ref (Float.Array.unsafe_get acc ((3 * tile) + a + 2))
    and s33 = ref (Float.Array.unsafe_get acc ((3 * tile) + a + 3)) in
    for j = b.lo to b.hi - 1 do
      let r = x + (j * n) and p = 4 * (j - b.lo) in
      let x0 = Array1.unsafe_get src r
      and x1 = Array1.unsafe_get src (r + 1)
      and x2 = Array1.unsafe_get src (r + 2)
      and x3 = Array1.unsafe_get src (r + 3)
      and w0 = Array1.unsafe_get b.packed p
      and w1 = Array1.unsafe_get b.packed (p + 1)
      and w2 = Array1.unsafe_get b.packed (p + 2)
      and w3 = Array1.unsafe_get b.packed (p + 3) in
      s00 := !s00 +. (w0 *. x0) ;
      s01 := !s01 +. (w0 *. x1) ;
      s02 := !s02 +. (w0 *. x2) ;
      s03 := !s03 +. (w0 *. x3) ;
      s10 := !s10 +. (w1 *. x0) ;
      s11 := !s11 +. (w1 *. x1) ;
      s12 := !s12 +. (w1 *. x2) ;
      s13 := !s13 +. (w1 *. x3) ;
      s20 := !s20 +. (w2 *. x0) ;
      s21 := !s21 +. (w2 *. x1) ;
      s22 := !s22 +. (w2 *. x2) ;
      s23 := !s23 +. (w2 *. x3) ;
      s30 := !s30 +. (w3 *. x0) ;
      s31 := !s31 +. (w3 *. x1) ;
      s32 := !s32 +. (w3 *. x2) ;
      s33 := !s33 +. (w3 *. x3)
    done ;
    Float.Array.unsafe_set acc a !s00 ;
    Float.Array.unsafe_set acc (a + 1) !s01 ;
    Float.Array.unsafe_set acc (a + 2) !s02 ;
    Float.Array.unsafe_set acc (a + 3) !s03 ;
    Float.Array.unsafe_set acc (tile + a) !s10 ;
    Float.Array.unsafe_set acc (tile + a + 1) !s11 ;
    Float.Array.unsafe_set acc (tile + a + 2) !s12 ;
    Float.Array.unsafe_set acc (tile + a + 3) !s13 ;
    Float.Array.unsafe_set acc ((2 * tile) + a) !s20 ;
    Float.Array.unsafe_set acc ((2 * tile) + a + 1) !s21 ;
    Float.Array.unsafe_set acc ((2 * tile) + a + 2) !s22 ;
    Float.Array.unsafe_set acc ((2 * tile) + a + 3) !s23 ;
    Float.Array.unsafe_set acc ((3 * tile) + a) !s30 ;
    Float.Array.unsafe_set acc ((3 * tile) + a + 1) !s31 ;
    Float.Array.unsafe_set acc ((3 * tile) + a + 2) !s32 ;
    Float.Array.unsafe_set acc ((3 * tile) + a + 3) !s33 ;
    c := !c + 4
  done ;
  while !c < width do
    let a = !c and x = x + !c in
    let s0 = ref (Float.Array.unsafe_get acc a)
    and s1 = ref (Float.Array.unsafe_get acc (tile + a))
    and s2 = ref (Float.Array.unsafe_get acc ((2 * tile) + a))
    and s3 = ref (Float.Array.unsafe_get acc ((3 * tile) + a)) in
    for j = b.lo to b.hi - 1 do
      let p = 4 * (j - b.lo) in
      let v = Array1.unsafe_get src (x + (j * n))
      and w0 = Array1.unsafe_get b.packed p
      and w1 = Array1.unsafe_get b.packed (p + 1)
      and w2 = Array1.unsafe_get b.packed (p + 2)
      and w3 = Array1.unsafe_get b.packed (p + 3) in
      s0 := !s0 +. (w0 *. v) ;
      s1 := !s1 +. (w1 *. v) ;
      s2 := !s2 +. (w2 *. v) ;
      s3 := !s3 +. (w3 *. v)
    done ;
    Float.Array.unsafe_set acc a !s0 ;
    Float.Array.unsafe_set acc (tile + a) !s1 ;
    Float.Array.unsafe_set acc ((2 * tile) + a) !s2 ;
    Float.Array.unsafe_set acc ((3 * tile) + a) !s3 ;
    incr c
  done

let stretch32 (src : f32) ~n b acc ~width ~x =
  let c = ref 0 in
  while !c + 3 < width do
    let a = !c and x = x + !c in
    let s00 = ref (Float.Array.unsafe_get acc a)
    and s01 = ref (Float.Array.unsafe_get acc (a + 1))
    and s02 = ref (Float.Array.unsafe_get acc (a + 2))
    and s03 = ref (Float.Array.unsafe_get acc (a + 3))
    and s10 = ref (Float.Array.unsafe_get acc (tile + a))
    and s11 = ref (Float.Array.unsafe_get acc (tile + a + 1))
    and s12 = ref (Float.Array.unsafe_get acc (tile + a + 2))
    and s13 = ref (Float.Array.unsafe_get acc (tile + a + 3))
    and s20 = ref (Float.Array.unsafe_get acc ((2 * tile) + a))
    and s21 = ref (Float.Array.unsafe_get acc ((2 * tile) + a + 1))
    and s22 = ref (Float.Array.unsafe_get acc ((2 * tile) + a + 2))
    and s23 = ref (Float.Array.unsafe_get acc ((2 * tile) + a + 3))
    and s30 = ref (Float.Array.unsafe_get acc ((3 * tile) + a))
    and s31 = ref (Float.Array.unsafe_get acc ((3 * tile) + a + 1))
    and s32 = ref (Float.Array.unsafe_get acc ((3 * tile) + a + 2))
    and s33 = ref (Float.Array.unsafe_get acc ((3 * tile) + a + 3)) in
    for j = b.lo to b.hi - 1 do
      let r = x + (j * n) and p = 4 * (j - b.lo) in
      let x0 = Array1.unsafe_get src r
      and x1 = Array1.unsafe_get src (r + 1)
      and x2 = Array1.unsafe_get src (r + 2)
      and x3 = Array1.unsafe_get src (r + 3)
      and w0 = Array1.unsafe_get b.packed p
      and w1 = Array1.unsafe_get b.packed (p + 1)
      and w2 = Array1.unsafe_get b.packed (p + 2)
      and w3 = Array1.unsafe_get b.packed (p + 3) in
      s00 := !s00 +. (w0 *. x0) ;
      s01 := !s01 +. (w0 *. x1) ;
      s02 := !s02 +. (w0 *. x2) ;
      s03 := !s03 +. (w0 *. x3) ;
      s10 := !s10 +. (w1 *. x0) ;
      s11 := !s11 +. (w1 *. x1) ;
      s12 := !s12 +. (w1 *. x2) ;
      s13 := !s13 +. (w1 *. x3) ;
      s20 := !s20 +. (w2 *. x0) ;
      s21 := !s21 +. (w2 *. x1) ;
      s22 := !s22 +. (w2 *. x2) ;
      s23 := !s23 +. (w2 *. x3) ;
      s30 := !s30 +. (w3 *. x0) ;
      s31 := !s31 +. (w3 *. x1) ;
      s32 := !s32 +. (w3 *. x2) ;
      s33 := !s33 +. (w3 *. x3)
    done ;
    Float.Array.unsafe_set acc a !s00 ;
    Float.Array.unsafe_set acc (a + 1) !s01 ;
    Float.Array.unsafe_set acc (a + 2) !s02 ;
    Float.Array.unsafe_set acc (a + 3) !s03 ;
    Float.Array.unsafe_set acc (tile + a) !s10 ;
    Float.Array.unsafe_set acc (tile + a + 1) !s11 ;
    Float.Array.unsafe_set acc (tile + a + 2) !s12 ;
    Float.Array.unsafe_set acc (tile + a + 3) !s13 ;
    Float.Array.unsafe_set acc ((2 * tile) + a) !s20 ;
    Float.Array.unsafe_set acc ((2 * tile) + a + 1) !s21 ;
    Float.Array.unsafe_set acc ((2 * tile) + a + 2) !s22 ;
    Float.Array.unsafe_set acc ((2 * tile) + a + 3) !s23 ;
    Float.Array.unsafe_set acc ((3 * tile) + a) !s30 ;
    Float.Array.unsafe_set acc ((3 * tile) + a + 1) !s31 ;
    Float.Array.unsafe_set acc ((3 * tile) + a + 2) !s32 ;
    Float.Array.unsafe_set acc ((3 * tile) + a + 3) !s33 ;
    c := !c + 4
  done ;
  while !c < width do
    let a = !c and x = x + !c in
    let s0 = ref (Float.Array.unsafe_get acc a)
    and s1 = ref (Float.Array.unsafe_get acc (tile + a))
    and s2 = ref (Float.Array.unsafe_get acc ((2 * tile) + a))
    and s3 = ref (Float.Array.unsafe_get acc ((3 * tile) + a)) in
    for j = b.lo to b.hi - 1 do
      let p = 4 * (j - b.lo) in
      let v = Array1.unsafe_get src (x + (j * n))
      and w0 = Array1.unsafe_get b.packed p
      and w1 = Array1.unsafe_get b.packed (p + 1)
      and w2 = Array1.unsafe_get b.packed (p + 2)
      and w3 = Array1.unsafe_get b.packed (p + 3) in
      s0 := !s0 +. (w0 *. v) ;
      s1 := !s1 +. (w1 *. v) ;
      s2 := !s2 +. (w2 *. v) ;
      s3 := !s3 +. (w3 *. v)
    done ;
    Float.Array.unsafe_set acc a !s0 ;
    Float.Array.unsafe_set acc (tile + a) !s1 ;
    Float.Array.unsafe_set acc ((2 * tile) + a) !s2 ;
    Float.Array.unsafe_set acc ((3 * tile) + a) !s3 ;
    incr c
  done

(* [run_steps mx span stretch store acc ~width ~x ~out] computes one unit's
   outputs, step by step: a row's span, or a block's rows alone before and after
   their shared stretch, accumulate from zero in the scratch [acc] over the
   samples at [x], and [store acc ~a ~row ~width] writes [acc.(a + c)], [c <
   width], out at output offset [out i] for row [i]. *)
let run_steps mx span stretch store acc ~width ~x ~out =
  Array.iter
    (function
      | Row i ->
          Float.Array.fill acc 0 width 0. ;
          span acc ~a:0 ~width ~i
            ~j0:mx.bounds.(2 * i)
            ~j1:mx.bounds.((2 * i) + 1)
            ~x ;
          store acc ~a:0 ~row:(out i) ~width
      | Block b ->
          Float.Array.fill acc 0 (4 * tile) 0. ;
          for r = 0 to 3 do
            span acc ~a:(r * tile) ~width ~i:(b.row + r)
              ~j0:mx.bounds.(2 * (b.row + r))
              ~j1:b.lo ~x
          done ;
          stretch b acc ~width ~x ;
          for r = 0 to 3 do
            let i = b.row + r in
            span acc ~a:(r * tile) ~width ~i ~j0:b.hi
              ~j1:mx.bounds.((2 * i) + 1)
              ~x ;
            store acc ~a:(r * tile) ~row:(out i) ~width
          done )
    mx.steps

(* [project_units mx ~outer ~n span stretch store] runs {!run_steps} over every
   (plane, tile) unit; [span] and [stretch] read the operand. *)
let project_units mx ~outer ~n span stretch store =
  let k = mx.k and m = mx.m in
  let per_plane = tiles n in
  let grain = Int.max 1 (grain_terms / Int.max 1 (Int.max 1 mx.terms * tile)) in
  Par.ranges ~n:(outer * per_plane) ~grain (fun lo hi ->
      let acc = scratch () in
      for u = lo to hi - 1 do
        let o = u / per_plane and t0 = u mod per_plane * tile in
        run_steps mx span stretch store acc
          ~width:(Int.min n (t0 + tile) - t0)
          ~x:((o * k * n) + t0)
          ~out:(fun i -> (((o * m) + i) * n) + t0)
      done )

let store64 (out : f64) acc ~a ~row ~width =
  for t = 0 to width - 1 do
    Array1.unsafe_set out (row + t) (Float.Array.unsafe_get acc (a + t))
  done

let store32 (out : f32) acc ~a ~row ~width =
  for t = 0 to width - 1 do
    Array1.unsafe_set out (row + t) (Float.Array.unsafe_get acc (a + t))
  done

(* {2 Narrow operands}

   An operand narrower than the eight-column groups — a streamed chunk of a
   frame or a few — would leave the steps one chain of dependent multiply-adds
   per output, each behind a scratch fill and closure calls. Such an operand
   takes kernels of its own, one column at a time, which write each output
   straight into the result, rounded once there as the stores round it. A
   block's rows go side by side over their shared stretch as {!stretch64} takes
   them, one sample feeding four multiply-adds; rows outside blocks go four at a
   time too, each from its own first column, term by term for as many terms as
   the shortest of the four spans holds, then each finishes its span alone.
   Either way four independent chains are in flight, and each output still
   starts from [0.] and takes its row's terms in ascending order, spelled as
   above: its bits are the ones the steps give it.

   In the kernels, a plane's operand starts at [x] and its result at [d]: column
   [j]'s sample [c] is [src.(x + j * n + c)], and row [i]'s output [c] goes to
   [d + i * n + c]. *)

(* The narrowest operand the steps take: the width of their column groups. *)
let narrow = 8

(* A result being written, at float32 or float64. *)
type dst = To32 of f32 | To64 of f64

let[@inline] put dst i v =
  match dst with
  | To32 out ->
      Array1.unsafe_set out i v
  | To64 out ->
      Array1.unsafe_set out i v

(* The steps' [store] into [dst]. *)
let store dst =
  match dst with To32 out -> store32 out | To64 out -> store64 out

(* Whether step [s], row [i], starts four steps of a row each. *)
let four_rows steps s i =
  s + 3 < Array.length steps
  && match steps.(s + 3) with Row j -> j = i + 3 | Block _ -> false

(* [alone64 mx src ~n dst ~x ~d i] and [alone32] write row [i]'s outputs. *)
let alone64 (mx : matrix) (src : f64) ~n dst ~x ~d i =
  let wbase = i * mx.k in
  for c = 0 to n - 1 do
    let x = x + c in
    let s = ref 0. in
    for j = mx.bounds.(2 * i) to mx.bounds.((2 * i) + 1) - 1 do
      let w = Array1.unsafe_get mx.weights (wbase + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s := !s +. (w *. v)
    done ;
    put dst (d + (i * n) + c) !s
  done

let alone32 (mx : matrix) (src : f32) ~n dst ~x ~d i =
  let wbase = i * mx.k in
  for c = 0 to n - 1 do
    let x = x + c in
    let s = ref 0. in
    for j = mx.bounds.(2 * i) to mx.bounds.((2 * i) + 1) - 1 do
      let w = Array1.unsafe_get mx.weights (wbase + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s := !s +. (w *. v)
    done ;
    put dst (d + (i * n) + c) !s
  done

(* [rows64 mx src ~n dst ~x ~d i] and [rows32] write the outputs of rows [i] to
   [i + 3], four steps of a row each. *)
let rows64 (mx : matrix) (src : f64) ~n dst ~x ~d i =
  let k = mx.k and weights = mx.weights and bounds = mx.bounds in
  let j0 = bounds.(2 * i)
  and j1 = bounds.((2 * i) + 2)
  and j2 = bounds.((2 * i) + 4)
  and j3 = bounds.((2 * i) + 6) in
  let l0 = bounds.((2 * i) + 1) - j0
  and l1 = bounds.((2 * i) + 3) - j1
  and l2 = bounds.((2 * i) + 5) - j2
  and l3 = bounds.((2 * i) + 7) - j3 in
  let common = Int.min (Int.min l0 l1) (Int.min l2 l3) in
  let p0 = (i * k) + j0
  and p1 = ((i + 1) * k) + j1
  and p2 = ((i + 2) * k) + j2
  and p3 = ((i + 3) * k) + j3 in
  for c = 0 to n - 1 do
    let r0 = x + (j0 * n) + c
    and r1 = x + (j1 * n) + c
    and r2 = x + (j2 * n) + c
    and r3 = x + (j3 * n) + c in
    let s0 = ref 0. and s1 = ref 0. and s2 = ref 0. and s3 = ref 0. in
    for t = 0 to common - 1 do
      let r = t * n in
      let w0 = Array1.unsafe_get weights (p0 + t)
      and v0 = Array1.unsafe_get src (r0 + r)
      and w1 = Array1.unsafe_get weights (p1 + t)
      and v1 = Array1.unsafe_get src (r1 + r)
      and w2 = Array1.unsafe_get weights (p2 + t)
      and v2 = Array1.unsafe_get src (r2 + r)
      and w3 = Array1.unsafe_get weights (p3 + t)
      and v3 = Array1.unsafe_get src (r3 + r) in
      s0 := !s0 +. (w0 *. v0) ;
      s1 := !s1 +. (w1 *. v1) ;
      s2 := !s2 +. (w2 *. v2) ;
      s3 := !s3 +. (w3 *. v3)
    done ;
    for t = common to l0 - 1 do
      let w = Array1.unsafe_get weights (p0 + t)
      and v = Array1.unsafe_get src (r0 + (t * n)) in
      s0 := !s0 +. (w *. v)
    done ;
    for t = common to l1 - 1 do
      let w = Array1.unsafe_get weights (p1 + t)
      and v = Array1.unsafe_get src (r1 + (t * n)) in
      s1 := !s1 +. (w *. v)
    done ;
    for t = common to l2 - 1 do
      let w = Array1.unsafe_get weights (p2 + t)
      and v = Array1.unsafe_get src (r2 + (t * n)) in
      s2 := !s2 +. (w *. v)
    done ;
    for t = common to l3 - 1 do
      let w = Array1.unsafe_get weights (p3 + t)
      and v = Array1.unsafe_get src (r3 + (t * n)) in
      s3 := !s3 +. (w *. v)
    done ;
    put dst (d + (i * n) + c) !s0 ;
    put dst (d + ((i + 1) * n) + c) !s1 ;
    put dst (d + ((i + 2) * n) + c) !s2 ;
    put dst (d + ((i + 3) * n) + c) !s3
  done

let rows32 (mx : matrix) (src : f32) ~n dst ~x ~d i =
  let k = mx.k and weights = mx.weights and bounds = mx.bounds in
  let j0 = bounds.(2 * i)
  and j1 = bounds.((2 * i) + 2)
  and j2 = bounds.((2 * i) + 4)
  and j3 = bounds.((2 * i) + 6) in
  let l0 = bounds.((2 * i) + 1) - j0
  and l1 = bounds.((2 * i) + 3) - j1
  and l2 = bounds.((2 * i) + 5) - j2
  and l3 = bounds.((2 * i) + 7) - j3 in
  let common = Int.min (Int.min l0 l1) (Int.min l2 l3) in
  let p0 = (i * k) + j0
  and p1 = ((i + 1) * k) + j1
  and p2 = ((i + 2) * k) + j2
  and p3 = ((i + 3) * k) + j3 in
  for c = 0 to n - 1 do
    let r0 = x + (j0 * n) + c
    and r1 = x + (j1 * n) + c
    and r2 = x + (j2 * n) + c
    and r3 = x + (j3 * n) + c in
    let s0 = ref 0. and s1 = ref 0. and s2 = ref 0. and s3 = ref 0. in
    for t = 0 to common - 1 do
      let r = t * n in
      let w0 = Array1.unsafe_get weights (p0 + t)
      and v0 = Array1.unsafe_get src (r0 + r)
      and w1 = Array1.unsafe_get weights (p1 + t)
      and v1 = Array1.unsafe_get src (r1 + r)
      and w2 = Array1.unsafe_get weights (p2 + t)
      and v2 = Array1.unsafe_get src (r2 + r)
      and w3 = Array1.unsafe_get weights (p3 + t)
      and v3 = Array1.unsafe_get src (r3 + r) in
      s0 := !s0 +. (w0 *. v0) ;
      s1 := !s1 +. (w1 *. v1) ;
      s2 := !s2 +. (w2 *. v2) ;
      s3 := !s3 +. (w3 *. v3)
    done ;
    for t = common to l0 - 1 do
      let w = Array1.unsafe_get weights (p0 + t)
      and v = Array1.unsafe_get src (r0 + (t * n)) in
      s0 := !s0 +. (w *. v)
    done ;
    for t = common to l1 - 1 do
      let w = Array1.unsafe_get weights (p1 + t)
      and v = Array1.unsafe_get src (r1 + (t * n)) in
      s1 := !s1 +. (w *. v)
    done ;
    for t = common to l2 - 1 do
      let w = Array1.unsafe_get weights (p2 + t)
      and v = Array1.unsafe_get src (r2 + (t * n)) in
      s2 := !s2 +. (w *. v)
    done ;
    for t = common to l3 - 1 do
      let w = Array1.unsafe_get weights (p3 + t)
      and v = Array1.unsafe_get src (r3 + (t * n)) in
      s3 := !s3 +. (w *. v)
    done ;
    put dst (d + (i * n) + c) !s0 ;
    put dst (d + ((i + 1) * n) + c) !s1 ;
    put dst (d + ((i + 2) * n) + c) !s2 ;
    put dst (d + ((i + 3) * n) + c) !s3
  done

(* [block64 mx src ~n dst ~x ~d b] and [block32] write block [b]'s outputs. *)
let block64 (mx : matrix) (src : f64) ~n dst ~x ~d b =
  let k = mx.k and weights = mx.weights and bounds = mx.bounds and i = b.row in
  let p0 = i * k in
  let p1 = p0 + k in
  let p2 = p1 + k in
  let p3 = p2 + k in
  for c = 0 to n - 1 do
    let x = x + c in
    let s0 = ref 0. and s1 = ref 0. and s2 = ref 0. and s3 = ref 0. in
    for j = bounds.(2 * i) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p0 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s0 := !s0 +. (w *. v)
    done ;
    for j = bounds.((2 * i) + 2) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p1 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s1 := !s1 +. (w *. v)
    done ;
    for j = bounds.((2 * i) + 4) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p2 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s2 := !s2 +. (w *. v)
    done ;
    for j = bounds.((2 * i) + 6) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p3 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s3 := !s3 +. (w *. v)
    done ;
    for j = b.lo to b.hi - 1 do
      let p = 4 * (j - b.lo) in
      let v = Array1.unsafe_get src (x + (j * n))
      and w0 = Array1.unsafe_get b.packed p
      and w1 = Array1.unsafe_get b.packed (p + 1)
      and w2 = Array1.unsafe_get b.packed (p + 2)
      and w3 = Array1.unsafe_get b.packed (p + 3) in
      s0 := !s0 +. (w0 *. v) ;
      s1 := !s1 +. (w1 *. v) ;
      s2 := !s2 +. (w2 *. v) ;
      s3 := !s3 +. (w3 *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 1) - 1 do
      let w = Array1.unsafe_get weights (p0 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s0 := !s0 +. (w *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 3) - 1 do
      let w = Array1.unsafe_get weights (p1 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s1 := !s1 +. (w *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 5) - 1 do
      let w = Array1.unsafe_get weights (p2 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s2 := !s2 +. (w *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 7) - 1 do
      let w = Array1.unsafe_get weights (p3 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s3 := !s3 +. (w *. v)
    done ;
    put dst (d + (i * n) + c) !s0 ;
    put dst (d + ((i + 1) * n) + c) !s1 ;
    put dst (d + ((i + 2) * n) + c) !s2 ;
    put dst (d + ((i + 3) * n) + c) !s3
  done

let block32 (mx : matrix) (src : f32) ~n dst ~x ~d b =
  let k = mx.k and weights = mx.weights and bounds = mx.bounds and i = b.row in
  let p0 = i * k in
  let p1 = p0 + k in
  let p2 = p1 + k in
  let p3 = p2 + k in
  for c = 0 to n - 1 do
    let x = x + c in
    let s0 = ref 0. and s1 = ref 0. and s2 = ref 0. and s3 = ref 0. in
    for j = bounds.(2 * i) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p0 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s0 := !s0 +. (w *. v)
    done ;
    for j = bounds.((2 * i) + 2) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p1 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s1 := !s1 +. (w *. v)
    done ;
    for j = bounds.((2 * i) + 4) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p2 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s2 := !s2 +. (w *. v)
    done ;
    for j = bounds.((2 * i) + 6) to b.lo - 1 do
      let w = Array1.unsafe_get weights (p3 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s3 := !s3 +. (w *. v)
    done ;
    for j = b.lo to b.hi - 1 do
      let p = 4 * (j - b.lo) in
      let v = Array1.unsafe_get src (x + (j * n))
      and w0 = Array1.unsafe_get b.packed p
      and w1 = Array1.unsafe_get b.packed (p + 1)
      and w2 = Array1.unsafe_get b.packed (p + 2)
      and w3 = Array1.unsafe_get b.packed (p + 3) in
      s0 := !s0 +. (w0 *. v) ;
      s1 := !s1 +. (w1 *. v) ;
      s2 := !s2 +. (w2 *. v) ;
      s3 := !s3 +. (w3 *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 1) - 1 do
      let w = Array1.unsafe_get weights (p0 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s0 := !s0 +. (w *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 3) - 1 do
      let w = Array1.unsafe_get weights (p1 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s1 := !s1 +. (w *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 5) - 1 do
      let w = Array1.unsafe_get weights (p2 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s2 := !s2 +. (w *. v)
    done ;
    for j = b.hi to bounds.((2 * i) + 7) - 1 do
      let w = Array1.unsafe_get weights (p3 + j)
      and v = Array1.unsafe_get src (x + (j * n)) in
      s3 := !s3 +. (w *. v)
    done ;
    put dst (d + (i * n) + c) !s0 ;
    put dst (d + ((i + 1) * n) + c) !s1 ;
    put dst (d + ((i + 2) * n) + c) !s2 ;
    put dst (d + ((i + 3) * n) + c) !s3
  done

(* [narrow64 mx src ~n dst o] and [narrow32] write plane [o]'s outputs of the
   operand [src], [n < narrow] columns wide, into [dst], step by step. *)
let narrow64 (mx : matrix) (src : f64) ~n dst o =
  let x = o * mx.k * n and d = o * mx.m * n in
  let steps = mx.steps in
  let s = ref 0 in
  while !s < Array.length steps do
    match steps.(!s) with
    | Block b ->
        block64 mx src ~n dst ~x ~d b ;
        incr s
    | Row i when four_rows steps !s i ->
        rows64 mx src ~n dst ~x ~d i ;
        s := !s + 4
    | Row i ->
        alone64 mx src ~n dst ~x ~d i ;
        incr s
  done

let narrow32 (mx : matrix) (src : f32) ~n dst o =
  let x = o * mx.k * n and d = o * mx.m * n in
  let steps = mx.steps in
  let s = ref 0 in
  while !s < Array.length steps do
    match steps.(!s) with
    | Block b ->
        block32 mx src ~n dst ~x ~d b ;
        incr s
    | Row i when four_rows steps !s i ->
        rows32 mx src ~n dst ~x ~d i ;
        s := !s + 4
    | Row i ->
        alone32 mx src ~n dst ~x ~d i ;
        incr s
  done

(* The planes of a narrow operand worth one parallel chunk. *)
let plane_grain mx ~n =
  Int.max 1 (grain_terms / Int.max 1 (Int.max 1 mx.terms * n))

(* [project64 mx src ~outer ~n dst] and [project32] fold the operand [src] into
   [dst]: a narrow one plane by plane, any other unit by unit. *)
let project64 mx (src : f64) ~outer ~n dst =
  if n < narrow then
    Par.ranges ~n:outer ~grain:(plane_grain mx ~n) (fun lo hi ->
        for o = lo to hi - 1 do
          narrow64 mx src ~n dst o
        done )
  else
    project_units mx ~outer ~n (span64 mx src ~n) (stretch64 src ~n) (store dst)

let project32 mx (src : f32) ~outer ~n dst =
  if n < narrow then
    Par.ranges ~n:outer ~grain:(plane_grain mx ~n) (fun lo hi ->
        for o = lo to hi - 1 do
          narrow32 mx src ~n dst o
        done )
  else
    project_units mx ~outer ~n (span32 mx src ~n) (stretch32 src ~n) (store dst)

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

(* [project_into mx x dst] folds [x] — read in place at float32 or float64,
   widened to float64 otherwise — into [dst]. *)
let project_into (type a) mx (x : (float, a) Nx.t) ~outer ~n dst =
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.read32 x (fun src -> project32 mx src ~outer ~n dst)
  | Nx.Float64 ->
      Store.read64 x (fun src -> project64 mx src ~outer ~n dst)
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Store.read64 (Nx.cast Nx.float64 x) (fun src ->
          project64 mx src ~outer ~n dst )

let project mx x =
  let out_shape, outer, n = check_project mx x in
  Store.make64 out_shape (fun out -> project_into mx x ~outer ~n (To64 out))

let project_cast (type a) mx (x : (float, a) Nx.t) : (float, a) Nx.t =
  let out_shape, outer, n = check_project mx x in
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.make32 out_shape (fun out -> project_into mx x ~outer ~n (To32 out))
  | Nx.Float64 ->
      Store.make64 out_shape (fun out -> project_into mx x ~outer ~n (To64 out))
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast (Nx.dtype x) (project mx x)

(* {1 Projections of a spectrum's power}

   The power of a frames-major complex spectrum is projected without being
   materialized: each unit, a tile of frames of one plane, first writes the
   tile's powers bins-major into a scratch the unit's domain keeps, then runs
   the steps over that scratch as over any operand. A power is [re *. re +. im
   *. im] from the components widened to double, spelled as
   {!Loops.spectrum_power} spells it (so each platform contracts it alike) and
   rounded into the scratch's dtype: float32 for a complex64 spectrum, float64
   for complex128 — the power spectrum's own values. *)

(* The frames of a unit: a tile narrower than the projections' own, so a
   spectrogram splits into units enough to keep every domain busy to the end. *)
let power_tile = 16

(* The powers of a tile are written a band of [power_band] bins at a time, so
   the scratch rows a band fills stay in the first-level cache while every frame
   of the tile reaches them. *)
let power_band = 32

let power_fill32 (z : f32) (p : f32) ~k ~n ~o ~t0 ~w =
  let b0 = ref 0 in
  while !b0 < k do
    let b1 = Int.min k (!b0 + power_band) in
    for t = 0 to w - 1 do
      let row = 2 * (((o * n) + t0 + t) * k) in
      let j = ref !b0 in
      while !j + 3 < b1 do
        let i = row + (2 * !j) and d = (!j * w) + t in
        let r0 = Array1.unsafe_get z i and m0 = Array1.unsafe_get z (i + 1) in
        let r1 = Array1.unsafe_get z (i + 2)
        and m1 = Array1.unsafe_get z (i + 3) in
        let r2 = Array1.unsafe_get z (i + 4)
        and m2 = Array1.unsafe_get z (i + 5) in
        let r3 = Array1.unsafe_get z (i + 6)
        and m3 = Array1.unsafe_get z (i + 7) in
        Array1.unsafe_set p d ((r0 *. r0) +. (m0 *. m0)) ;
        Array1.unsafe_set p (d + w) ((r1 *. r1) +. (m1 *. m1)) ;
        Array1.unsafe_set p (d + (2 * w)) ((r2 *. r2) +. (m2 *. m2)) ;
        Array1.unsafe_set p (d + (3 * w)) ((r3 *. r3) +. (m3 *. m3)) ;
        j := !j + 4
      done ;
      while !j < b1 do
        let i = row + (2 * !j) in
        let re = Array1.unsafe_get z i and im = Array1.unsafe_get z (i + 1) in
        Array1.unsafe_set p ((!j * w) + t) ((re *. re) +. (im *. im)) ;
        incr j
      done
    done ;
    b0 := b1
  done

let power_fill64 (z : f64) (p : f64) ~k ~n ~o ~t0 ~w =
  let b0 = ref 0 in
  while !b0 < k do
    let b1 = Int.min k (!b0 + power_band) in
    for t = 0 to w - 1 do
      let row = 2 * (((o * n) + t0 + t) * k) in
      let j = ref !b0 in
      while !j + 3 < b1 do
        let i = row + (2 * !j) and d = (!j * w) + t in
        let r0 = Array1.unsafe_get z i and m0 = Array1.unsafe_get z (i + 1) in
        let r1 = Array1.unsafe_get z (i + 2)
        and m1 = Array1.unsafe_get z (i + 3) in
        let r2 = Array1.unsafe_get z (i + 4)
        and m2 = Array1.unsafe_get z (i + 5) in
        let r3 = Array1.unsafe_get z (i + 6)
        and m3 = Array1.unsafe_get z (i + 7) in
        Array1.unsafe_set p d ((r0 *. r0) +. (m0 *. m0)) ;
        Array1.unsafe_set p (d + w) ((r1 *. r1) +. (m1 *. m1)) ;
        Array1.unsafe_set p (d + (2 * w)) ((r2 *. r2) +. (m2 *. m2)) ;
        Array1.unsafe_set p (d + (3 * w)) ((r3 *. r3) +. (m3 *. m3)) ;
        j := !j + 4
      done ;
      while !j < b1 do
        let i = row + (2 * !j) in
        let re = Array1.unsafe_get z i and im = Array1.unsafe_get z (i + 1) in
        Array1.unsafe_set p ((!j * w) + t) ((re *. re) +. (im *. im)) ;
        incr j
      done
    done ;
    b0 := b1
  done

(* [power_units mx ~outer ~n ~k ~scratch:make fill span stretch store] runs
   every (plane, tile) unit over the powers [fill p ~o ~t0 ~w] writes into the
   unit domain's scratch [p], [make (k * tile)] once per domain. *)
let power_units mx ~outer ~n ~k ~make fill span stretch store =
  let m = mx.m in
  let per_plane = (n + power_tile - 1) / power_tile in
  let grain =
    Int.max 1 (grain_terms / Int.max 1 ((Int.max 1 mx.terms + k) * power_tile))
  in
  Par.ranges ~n:(outer * per_plane) ~grain (fun lo hi ->
      let acc = scratch () and p = make (k * power_tile) in
      for u = lo to hi - 1 do
        let o = u / per_plane and t0 = u mod per_plane * power_tile in
        let width = Int.min n (t0 + power_tile) - t0 in
        fill p ~o ~t0 ~w:width ;
        run_steps mx (span p ~n:width) (stretch p ~n:width) store acc ~width
          ~x:0 ~out:(fun i -> (((o * m) + i) * n) + t0 )
      done )

let check_power mx z =
  let nd = Nx.ndim z in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "Ordered.project_power: cannot project a rank-%d spectrum (the \
          projection needs [...; n; k])"
         nd ) ;
  let shape = Nx.shape z in
  let n = shape.(nd - 2) and k = shape.(nd - 1) in
  if k <> mx.k then
    invalid_arg
      (Printf.sprintf
         "Ordered.project_power: cannot project %d bins through a matrix of %d \
          columns"
         k mx.k ) ;
  let lead = Array.sub shape 0 (nd - 2) in
  (Array.append lead [|mx.m; n|], Array.fold_left ( * ) 1 lead, n, k)

let make32 len = Array1.create float32 c_layout len

let make64 len = Array1.create float64 c_layout len

let project_power (type a c) mx (dtype : (float, a) Nx.dtype)
    (z : (Complex.t, c) Nx.t) : (float, Nx.float64_elt) Nx.t option =
  let shape, outer, n, k = check_power mx z in
  match (dtype, Nx.dtype z) with
  | Nx.Float32, Nx.Complex64 ->
      Some
        (Store.make64 shape (fun out ->
             Store.read_parts32 z (fun src ->
                 power_units mx ~outer ~n ~k ~make:make32
                   (power_fill32 src ~k ~n) (span32 mx) stretch32 (store64 out) ) )
        )
  | Nx.Float64, Nx.Complex128 ->
      Some
        (Store.make64 shape (fun out ->
             Store.read_parts64 z (fun src ->
                 power_units mx ~outer ~n ~k ~make:make64
                   (power_fill64 src ~k ~n) (span64 mx) stretch64 (store64 out) ) )
        )
  | _ ->
      None

let project_power_cast (type a c) mx (dtype : (float, a) Nx.dtype)
    (z : (Complex.t, c) Nx.t) : (float, a) Nx.t option =
  let shape, outer, n, k = check_power mx z in
  match (dtype, Nx.dtype z) with
  | Nx.Float32, Nx.Complex64 ->
      Some
        (Store.make32 shape (fun out ->
             Store.read_parts32 z (fun src ->
                 power_units mx ~outer ~n ~k ~make:make32
                   (power_fill32 src ~k ~n) (span32 mx) stretch32 (store32 out) ) )
        )
  | Nx.Float64, Nx.Complex128 ->
      Some
        (Store.make64 shape (fun out ->
             Store.read_parts64 z (fun src ->
                 power_units mx ~outer ~n ~k ~make:make64
                   (power_fill64 src ~k ~n) (span64 mx) stretch64 (store64 out) ) )
        )
  | _ ->
      None
