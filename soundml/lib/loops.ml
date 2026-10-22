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

(* The loops are written once per storage kind and per element formula, so that
   every element access compiles to a plain load or store and every float stays
   unboxed: without flambda, a formula passed to a shared loop as a function is
   a call that boxes its result. Hot loops are unrolled by four, which OCaml
   does not do on its own. *)

open Bigarray

type f64 = Store.f64

type f32 = Store.f32

(* {1 Spectrum powers}

   A parallel unit is one band of [band] bins of one plane, every frame. Each
   frame's band is read in place, contiguous, four bins per step, and its powers
   are written to the band's [band] output rows, which advance one element per
   frame: every stream is sequential. Without flambda an element function passed
   to a shared loop would box every float, so each power has a loop of its own:
   the square, the root and the general power, at each width. *)

let band = 32

let square32 (src : f32) (dst : f32) ~frames ~bins o b0 b1 =
  let sbase = o * frames * bins * 2 and dbase = o * bins * frames in
  for f = 0 to frames - 1 do
    let row = sbase + (2 * f * bins) in
    let b = ref b0 in
    while !b + 3 < b1 do
      let i = row + (2 * !b) and d = dbase + (!b * frames) + f in
      let r0 = Array1.unsafe_get src i and m0 = Array1.unsafe_get src (i + 1) in
      let r1 = Array1.unsafe_get src (i + 2)
      and m1 = Array1.unsafe_get src (i + 3) in
      let r2 = Array1.unsafe_get src (i + 4)
      and m2 = Array1.unsafe_get src (i + 5) in
      let r3 = Array1.unsafe_get src (i + 6)
      and m3 = Array1.unsafe_get src (i + 7) in
      Array1.unsafe_set dst d ((r0 *. r0) +. (m0 *. m0)) ;
      Array1.unsafe_set dst (d + frames) ((r1 *. r1) +. (m1 *. m1)) ;
      Array1.unsafe_set dst (d + (2 * frames)) ((r2 *. r2) +. (m2 *. m2)) ;
      Array1.unsafe_set dst (d + (3 * frames)) ((r3 *. r3) +. (m3 *. m3)) ;
      b := !b + 4
    done ;
    while !b < b1 do
      let i = row + (2 * !b) in
      let re = Array1.unsafe_get src i and im = Array1.unsafe_get src (i + 1) in
      Array1.unsafe_set dst
        (dbase + (!b * frames) + f)
        ((re *. re) +. (im *. im)) ;
      incr b
    done
  done

let root32 (src : f32) (dst : f32) ~frames ~bins o b0 b1 =
  let sbase = o * frames * bins * 2 and dbase = o * bins * frames in
  for f = 0 to frames - 1 do
    let row = sbase + (2 * f * bins) in
    let b = ref b0 in
    while !b + 3 < b1 do
      let i = row + (2 * !b) and d = dbase + (!b * frames) + f in
      let r0 = Array1.unsafe_get src i and m0 = Array1.unsafe_get src (i + 1) in
      let r1 = Array1.unsafe_get src (i + 2)
      and m1 = Array1.unsafe_get src (i + 3) in
      let r2 = Array1.unsafe_get src (i + 4)
      and m2 = Array1.unsafe_get src (i + 5) in
      let r3 = Array1.unsafe_get src (i + 6)
      and m3 = Array1.unsafe_get src (i + 7) in
      Array1.unsafe_set dst d (Float.sqrt ((r0 *. r0) +. (m0 *. m0))) ;
      Array1.unsafe_set dst (d + frames) (Float.sqrt ((r1 *. r1) +. (m1 *. m1))) ;
      Array1.unsafe_set dst
        (d + (2 * frames))
        (Float.sqrt ((r2 *. r2) +. (m2 *. m2))) ;
      Array1.unsafe_set dst
        (d + (3 * frames))
        (Float.sqrt ((r3 *. r3) +. (m3 *. m3))) ;
      b := !b + 4
    done ;
    while !b < b1 do
      let i = row + (2 * !b) in
      let re = Array1.unsafe_get src i and im = Array1.unsafe_get src (i + 1) in
      Array1.unsafe_set dst
        (dbase + (!b * frames) + f)
        (Float.sqrt ((re *. re) +. (im *. im))) ;
      incr b
    done
  done

let general32 (src : f32) (dst : f32) ~power ~frames ~bins o b0 b1 =
  let sbase = o * frames * bins * 2 and dbase = o * bins * frames in
  for f = 0 to frames - 1 do
    let row = sbase + (2 * f * bins) in
    for b = b0 to b1 - 1 do
      let i = row + (2 * b) in
      let re = Array1.unsafe_get src i and im = Array1.unsafe_get src (i + 1) in
      Array1.unsafe_set dst
        (dbase + (b * frames) + f)
        (Float.pow (Float.sqrt ((re *. re) +. (im *. im))) power)
    done
  done

let square64 (src : f64) (dst : f64) ~frames ~bins o b0 b1 =
  let sbase = o * frames * bins * 2 and dbase = o * bins * frames in
  for f = 0 to frames - 1 do
    let row = sbase + (2 * f * bins) in
    let b = ref b0 in
    while !b + 3 < b1 do
      let i = row + (2 * !b) and d = dbase + (!b * frames) + f in
      let r0 = Array1.unsafe_get src i and m0 = Array1.unsafe_get src (i + 1) in
      let r1 = Array1.unsafe_get src (i + 2)
      and m1 = Array1.unsafe_get src (i + 3) in
      let r2 = Array1.unsafe_get src (i + 4)
      and m2 = Array1.unsafe_get src (i + 5) in
      let r3 = Array1.unsafe_get src (i + 6)
      and m3 = Array1.unsafe_get src (i + 7) in
      Array1.unsafe_set dst d ((r0 *. r0) +. (m0 *. m0)) ;
      Array1.unsafe_set dst (d + frames) ((r1 *. r1) +. (m1 *. m1)) ;
      Array1.unsafe_set dst (d + (2 * frames)) ((r2 *. r2) +. (m2 *. m2)) ;
      Array1.unsafe_set dst (d + (3 * frames)) ((r3 *. r3) +. (m3 *. m3)) ;
      b := !b + 4
    done ;
    while !b < b1 do
      let i = row + (2 * !b) in
      let re = Array1.unsafe_get src i and im = Array1.unsafe_get src (i + 1) in
      Array1.unsafe_set dst
        (dbase + (!b * frames) + f)
        ((re *. re) +. (im *. im)) ;
      incr b
    done
  done

let root64 (src : f64) (dst : f64) ~frames ~bins o b0 b1 =
  let sbase = o * frames * bins * 2 and dbase = o * bins * frames in
  for f = 0 to frames - 1 do
    let row = sbase + (2 * f * bins) in
    for b = b0 to b1 - 1 do
      let i = row + (2 * b) in
      Array1.unsafe_set dst
        (dbase + (b * frames) + f)
        (Float.hypot (Array1.unsafe_get src i) (Array1.unsafe_get src (i + 1)))
    done
  done

let general64 (src : f64) (dst : f64) ~power ~frames ~bins o b0 b1 =
  let sbase = o * frames * bins * 2 and dbase = o * bins * frames in
  for f = 0 to frames - 1 do
    let row = sbase + (2 * f * bins) in
    for b = b0 to b1 - 1 do
      let i = row + (2 * b) in
      Array1.unsafe_set dst
        (dbase + (b * frames) + f)
        (Float.pow
           (Float.hypot (Array1.unsafe_get src i)
              (Array1.unsafe_get src (i + 1)) )
           power )
    done
  done

(* [planes_of z] is [z]'s leading shape, its plane count, frames and bins. *)
let planes_of z =
  let shape = Nx.shape z in
  let nd = Array.length shape in
  let frames = shape.(nd - 2) and bins = shape.(nd - 1) in
  let lead = Array.sub shape 0 (nd - 2) in
  (lead, Array.fold_left ( * ) 1 lead, frames, bins)

let transpose_units ~outer ~frames ~bins f =
  let bands = (bins + band - 1) / band in
  (* about 32 Kelements per chunk *)
  let grain = Int.max 1 (32_768 / Int.max 1 (band * frames)) in
  Par.ranges ~n:(outer * bands) ~grain (fun lo hi ->
      for u = lo to hi - 1 do
        let o = u / bands and b0 = u mod bands * band in
        f o b0 (Int.min bins (b0 + band))
      done )

let spectrum_power (type a c) (dtype : (float, a) Nx.dtype) ~power
    (z : (Complex.t, c) Nx.t) : (float, a) Nx.t option =
  let lead, outer, frames, bins = planes_of z in
  let shape = Array.append lead [|bins; frames|] in
  let units = transpose_units ~outer ~frames ~bins in
  match (dtype, Nx.dtype z) with
  | Nx.Float32, Nx.Complex64 ->
      Some
        (Store.make32 shape (fun dst ->
             Store.read_parts32 z (fun src ->
                 if Float.equal power 2. then
                   units (square32 src dst ~frames ~bins)
                 else if Float.equal power 1. then
                   units (root32 src dst ~frames ~bins)
                 else units (general32 src dst ~power ~frames ~bins) ) ) )
  | Nx.Float64, Nx.Complex128 ->
      Some
        (Store.make64 shape (fun dst ->
             Store.read_parts64 z (fun src ->
                 if Float.equal power 2. then
                   units (square64 src dst ~frames ~bins)
                 else if Float.equal power 1. then
                   units (root64 src dst ~frames ~bins)
                 else units (general64 src dst ~power ~frames ~bins) ) ) )
  | _ ->
      None

(* {1 Windowed frames}

   Frame [p] covers positions [p * hop, p * hop + fft) of the padded stream,
   whose position [q] is signal sample [q - left]; positions outside the signal
   take the boundary extension. A frame's samples inside the signal are one
   contiguous run, multiplied by the window four at a time; only the frames that
   reach a border take the extension's index map. *)

type extension = Reflect | Edge | Constant of float

let reflect n q =
  if n = 1 then 0
  else
    let period = 2 * (n - 1) in
    let m = ((q mod period) + period) mod period in
    if m < n then m else period - m

(* [border get ext ~n s] is padded sample [s] (signal index, outside [0, n)). *)
let border ext get ~n s =
  match ext with
  | Reflect ->
      get (reflect n s)
  | Edge ->
      get (if s < 0 then 0 else n - 1)
  | Constant v ->
      v

let frame32 (src : f32) (dst : f64) (w : Float.Array.t) ~ext ~n ~fft ~hop ~left
    ~first ~count o lo hi =
  let sbase = o * n and dbase = o * count * fft in
  let get i = Array1.unsafe_get src (sbase + i) in
  for r = lo to hi - 1 do
    let s0 = ((first + r) * hop) - left and d = dbase + (r * fft) in
    let jlo = Int.max 0 (Int.min fft (-s0))
    and jhi = Int.max 0 (Int.min fft (n - s0)) in
    let jhi = Int.max jlo jhi in
    for j = 0 to jlo - 1 do
      Array1.unsafe_set dst (d + j)
        (border ext get ~n (s0 + j) *. Float.Array.unsafe_get w j)
    done ;
    let j = ref jlo in
    while !j + 3 < jhi do
      let k = !j in
      let i = sbase + s0 + k in
      Array1.unsafe_set dst (d + k)
        (Array1.unsafe_get src i *. Float.Array.unsafe_get w k) ;
      Array1.unsafe_set dst
        (d + k + 1)
        (Array1.unsafe_get src (i + 1) *. Float.Array.unsafe_get w (k + 1)) ;
      Array1.unsafe_set dst
        (d + k + 2)
        (Array1.unsafe_get src (i + 2) *. Float.Array.unsafe_get w (k + 2)) ;
      Array1.unsafe_set dst
        (d + k + 3)
        (Array1.unsafe_get src (i + 3) *. Float.Array.unsafe_get w (k + 3)) ;
      j := k + 4
    done ;
    for k = !j to jhi - 1 do
      Array1.unsafe_set dst (d + k)
        (Array1.unsafe_get src (sbase + s0 + k) *. Float.Array.unsafe_get w k)
    done ;
    for j = jhi to fft - 1 do
      Array1.unsafe_set dst (d + j)
        (border ext get ~n (s0 + j) *. Float.Array.unsafe_get w j)
    done
  done

let frame64 (src : f64) (dst : f64) (w : Float.Array.t) ~ext ~n ~fft ~hop ~left
    ~first ~count o lo hi =
  let sbase = o * n and dbase = o * count * fft in
  let get i = Array1.unsafe_get src (sbase + i) in
  for r = lo to hi - 1 do
    let s0 = ((first + r) * hop) - left and d = dbase + (r * fft) in
    let jlo = Int.max 0 (Int.min fft (-s0))
    and jhi = Int.max 0 (Int.min fft (n - s0)) in
    let jhi = Int.max jlo jhi in
    for j = 0 to jlo - 1 do
      Array1.unsafe_set dst (d + j)
        (border ext get ~n (s0 + j) *. Float.Array.unsafe_get w j)
    done ;
    let j = ref jlo in
    while !j + 3 < jhi do
      let k = !j in
      let i = sbase + s0 + k in
      Array1.unsafe_set dst (d + k)
        (Array1.unsafe_get src i *. Float.Array.unsafe_get w k) ;
      Array1.unsafe_set dst
        (d + k + 1)
        (Array1.unsafe_get src (i + 1) *. Float.Array.unsafe_get w (k + 1)) ;
      Array1.unsafe_set dst
        (d + k + 2)
        (Array1.unsafe_get src (i + 2) *. Float.Array.unsafe_get w (k + 2)) ;
      Array1.unsafe_set dst
        (d + k + 3)
        (Array1.unsafe_get src (i + 3) *. Float.Array.unsafe_get w (k + 3)) ;
      j := k + 4
    done ;
    for k = !j to jhi - 1 do
      Array1.unsafe_set dst (d + k)
        (Array1.unsafe_get src (sbase + s0 + k) *. Float.Array.unsafe_get w k)
    done ;
    for j = jhi to fft - 1 do
      Array1.unsafe_set dst (d + j)
        (border ext get ~n (s0 + j) *. Float.Array.unsafe_get w j)
    done
  done

let frames (type a) ~window ~hop ~left ~ext ~first ~count (x : (float, a) Nx.t)
    =
  let fft = Float.Array.length window in
  let shape = Nx.shape x in
  let nd = Array.length shape in
  let n = shape.(nd - 1) in
  let lead = Array.sub shape 0 (nd - 1) in
  let outer = Array.fold_left ( * ) 1 lead in
  (* about 64 Kelements per chunk *)
  let grain = Int.max 1 (65_536 / Int.max 1 fft) in
  let units f =
    Par.ranges ~n:(outer * count) ~grain (fun lo hi ->
        (* split [lo, hi) at plane boundaries *)
        let u = ref lo in
        while !u < hi do
          let o = !u / count in
          let stop = Int.min hi ((o + 1) * count) in
          f o (!u - (o * count)) (stop - (o * count)) ;
          u := stop
        done )
  in
  Store.make64
    (Array.append lead [|count; fft|])
    (fun dst ->
      match Nx.dtype x with
      | Nx.Float32 ->
          Store.read32 x (fun src ->
              units
                (frame32 src dst window ~ext ~n ~fft ~hop ~left ~first ~count) )
      | Nx.Float64 ->
          Store.read64 x (fun src ->
              units
                (frame64 src dst window ~ext ~n ~fft ~hop ~left ~first ~count) )
      | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
          Store.read64 (Nx.cast Nx.float64 x) (fun src ->
              units
                (frame64 src dst window ~ext ~n ~fft ~hop ~left ~first ~count) ) )

(* {1 Frame energies}

   Over a padded-stream segment [x] ([[...; n]]), frame [p] covers samples [p *
   hop, p * hop + frame_length). *)

(* Root mean square: each frame's squares are added in ascending order from
   [0.], as {!Ordered.sum} adds them, then divided by the frame length and
   rooted. Four frames are summed side by side, so the four additions of a step
   are independent and the chain of one frame's sum is not the loop's
   latency. *)
let rms32 (src : f32) (dst : f32) ~n ~fl ~hop ~count o lo hi =
  let sbase = o * n and dbase = o * count in
  let len = Float.of_int fl in
  let p = ref lo in
  while !p + 3 < hi do
    let a0 = sbase + (!p * hop) in
    let a1 = a0 + hop in
    let a2 = a1 + hop in
    let a3 = a2 + hop in
    let s0 = ref 0. and s1 = ref 0. and s2 = ref 0. and s3 = ref 0. in
    for j = 0 to fl - 1 do
      let v0 = Array1.unsafe_get src (a0 + j)
      and v1 = Array1.unsafe_get src (a1 + j)
      and v2 = Array1.unsafe_get src (a2 + j)
      and v3 = Array1.unsafe_get src (a3 + j) in
      s0 := !s0 +. (v0 *. v0) ;
      s1 := !s1 +. (v1 *. v1) ;
      s2 := !s2 +. (v2 *. v2) ;
      s3 := !s3 +. (v3 *. v3)
    done ;
    let d = dbase + !p in
    Array1.unsafe_set dst d (Float.sqrt (!s0 /. len)) ;
    Array1.unsafe_set dst (d + 1) (Float.sqrt (!s1 /. len)) ;
    Array1.unsafe_set dst (d + 2) (Float.sqrt (!s2 /. len)) ;
    Array1.unsafe_set dst (d + 3) (Float.sqrt (!s3 /. len)) ;
    p := !p + 4
  done ;
  while !p < hi do
    let a = sbase + (!p * hop) in
    let s = ref 0. in
    for j = 0 to fl - 1 do
      let v = Array1.unsafe_get src (a + j) in
      s := !s +. (v *. v)
    done ;
    Array1.unsafe_set dst (dbase + !p) (Float.sqrt (!s /. len)) ;
    incr p
  done

let rms64 (src : f64) (dst : f64) ~n ~fl ~hop ~count o lo hi =
  let sbase = o * n and dbase = o * count in
  let len = Float.of_int fl in
  let p = ref lo in
  while !p + 3 < hi do
    let a0 = sbase + (!p * hop) in
    let a1 = a0 + hop in
    let a2 = a1 + hop in
    let a3 = a2 + hop in
    let s0 = ref 0. and s1 = ref 0. and s2 = ref 0. and s3 = ref 0. in
    for j = 0 to fl - 1 do
      let v0 = Array1.unsafe_get src (a0 + j)
      and v1 = Array1.unsafe_get src (a1 + j)
      and v2 = Array1.unsafe_get src (a2 + j)
      and v3 = Array1.unsafe_get src (a3 + j) in
      s0 := !s0 +. (v0 *. v0) ;
      s1 := !s1 +. (v1 *. v1) ;
      s2 := !s2 +. (v2 *. v2) ;
      s3 := !s3 +. (v3 *. v3)
    done ;
    let d = dbase + !p in
    Array1.unsafe_set dst d (Float.sqrt (!s0 /. len)) ;
    Array1.unsafe_set dst (d + 1) (Float.sqrt (!s1 /. len)) ;
    Array1.unsafe_set dst (d + 2) (Float.sqrt (!s2 /. len)) ;
    Array1.unsafe_set dst (d + 3) (Float.sqrt (!s3 /. len)) ;
    p := !p + 4
  done ;
  while !p < hi do
    let a = sbase + (!p * hop) in
    let s = ref 0. in
    for j = 0 to fl - 1 do
      let v = Array1.unsafe_get src (a + j) in
      s := !s +. (v *. v)
    done ;
    Array1.unsafe_set dst (dbase + !p) (Float.sqrt (!s /. len)) ;
    incr p
  done

(* Zero-crossing rate: the crossings of frame [p] sit at its positions [1,
   frame_length), so with [c] the running count of sign changes over the
   segment, frame [p] holds [c (p * hop + fl - 1) - c (p * hop)] of them — exact
   integers, so one pass over the samples serves every frame. A sample is
   negative iff it lies strictly below [-threshold]. *)
let zcr32 (src : f32) (dst : f32) ~n ~fl ~hop ~count ~threshold o lo hi =
  let sbase = o * n and dbase = o * count in
  let len = Float.of_int fl in
  let below = -.threshold in
  let first = lo * hop and last = ((hi - 1) * hop) + fl - 1 in
  let starts = Array.make (hi - lo) 0 in
  let running = ref 0 in
  let prev = ref (Array1.unsafe_get src (sbase + first) < below) in
  let s = ref lo and e = ref lo in
  for q = first to last do
    if q > first then begin
      let neg = Array1.unsafe_get src (sbase + q) < below in
      if neg <> !prev then incr running ;
      prev := neg
    end ;
    while !s < hi && !s * hop = q do
      starts.(!s - lo) <- !running ;
      incr s
    done ;
    while !e < hi && (!e * hop) + fl - 1 = q do
      Array1.unsafe_set dst (dbase + !e)
        (Float.of_int (!running - starts.(!e - lo)) /. len) ;
      incr e
    done
  done

let zcr64 (src : f64) (dst : f64) ~n ~fl ~hop ~count ~threshold o lo hi =
  let sbase = o * n and dbase = o * count in
  let len = Float.of_int fl in
  let below = -.threshold in
  let first = lo * hop and last = ((hi - 1) * hop) + fl - 1 in
  let starts = Array.make (hi - lo) 0 in
  let running = ref 0 in
  let prev = ref (Array1.unsafe_get src (sbase + first) < below) in
  let s = ref lo and e = ref lo in
  for q = first to last do
    if q > first then begin
      let neg = Array1.unsafe_get src (sbase + q) < below in
      if neg <> !prev then incr running ;
      prev := neg
    end ;
    while !s < hi && !s * hop = q do
      starts.(!s - lo) <- !running ;
      incr s
    done ;
    while !e < hi && (!e * hop) + fl - 1 = q do
      Array1.unsafe_set dst (dbase + !e)
        (Float.of_int (!running - starts.(!e - lo)) /. len) ;
      incr e
    done
  done

type energy = Rms | Zcr of float

let energy (type a) kind ~frame_length ~hop ~count (x : (float, a) Nx.t) :
    (float, a) Nx.t =
  let shape = Nx.shape x in
  let nd = Array.length shape in
  let n = shape.(nd - 1) in
  let lead = Array.sub shape 0 (nd - 1) in
  let outer = Array.fold_left ( * ) 1 lead in
  let out_shape = Array.append lead [|1; count|] in
  let fl = frame_length in
  (* a frame costs [frame_length] steps; about 64 Ksteps per chunk *)
  let grain = Int.max 1 (65_536 / Int.max 1 fl) in
  let units f =
    Par.ranges ~n:(outer * count) ~grain (fun lo hi ->
        let u = ref lo in
        while !u < hi do
          let o = !u / count in
          let stop = Int.min hi ((o + 1) * count) in
          f o (!u - (o * count)) (stop - (o * count)) ;
          u := stop
        done )
  in
  let run64 x =
    Store.make64 out_shape (fun dst ->
        Store.read64 x (fun src ->
            match kind with
            | Rms ->
                units (rms64 src dst ~n ~fl ~hop ~count)
            | Zcr threshold ->
                units (zcr64 src dst ~n ~fl ~hop ~count ~threshold) ) )
  in
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.make32 out_shape (fun dst ->
          Store.read32 x (fun src ->
              match kind with
              | Rms ->
                  units (rms32 src dst ~n ~fl ~hop ~count)
              | Zcr threshold ->
                  units (zcr32 src dst ~n ~fl ~hop ~count ~threshold) ) )
  | Nx.Float64 ->
      run64 x
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast (Nx.dtype x) (run64 (Nx.cast Nx.float64 x))

(* {1 Spectral shape}

   Over a spectrogram [[...; k; n]], every feature reduces each frame's [k]
   bins, in ascending bin order from [0.]. A parallel unit is one tile of [tile]
   frames of one plane: the tile's rows are read contiguous, one row at a time,
   into per-frame accumulators, so a pass over the bins is a pass over memory in
   storage order. The first pass also checks that every value is non-negative (a
   NaN fails the comparison too), and sets [bad] otherwise.

   Every value is computed in double precision and rounded once into the
   spectrogram's dtype. *)

let tile = 64

(* [sum32 src ~plane ~k ~n ~t0 ~w acc bad] sets [acc.(t)] to frame [t0 + t]'s
   bin sum. *)
let sum32 (src : f32) ~plane ~k ~n ~t0 ~w acc bad =
  Float.Array.fill acc 0 w 0. ;
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 in
    for t = 0 to w - 1 do
      let v = Array1.unsafe_get src (r + t) in
      if not (v >= 0.) then bad := true ;
      Float.Array.unsafe_set acc t (Float.Array.unsafe_get acc t +. v)
    done
  done

(* [guard len ~w] replaces the frame sums below the smallest positive normal
   double by [1.], so a silent frame normalises to itself. *)
let guard len ~w =
  for t = 0 to w - 1 do
    if Float.Array.unsafe_get len t < Float.min_float then
      Float.Array.unsafe_set len t 1.
  done

(* [centroid32 src fq ~plane ~k ~n ~t0 ~w len acc] sets [acc.(t)] to the sum of
   [fq.(j) *. (s /. len.(t))]: the bin frequencies weighted by the normalised
   frame. *)
let centroid32 (src : f32) fq ~plane ~k ~n ~t0 ~w len acc =
  Float.Array.fill acc 0 w 0. ;
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 and f = Float.Array.unsafe_get fq j in
    for t = 0 to w - 1 do
      Float.Array.unsafe_set acc t
        ( Float.Array.unsafe_get acc t
        +. (f *. (Array1.unsafe_get src (r + t) /. Float.Array.unsafe_get len t))
        )
    done
  done

(* [spread32 src fq ~p ~plane ~k ~n ~t0 ~w len c acc] sets [acc.(t)] to the sum
   of [(s /. len.(t)) *. |c.(t) -. fq.(j)| ** p]. *)
let spread32 (src : f32) fq ~p ~plane ~k ~n ~t0 ~w len c acc =
  Float.Array.fill acc 0 w 0. ;
  let square = Float.equal p 2. in
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 and f = Float.Array.unsafe_get fq j in
    for t = 0 to w - 1 do
      let d = Float.abs (Float.Array.unsafe_get c t -. f) in
      let dp = if square then d *. d else Float.pow d p in
      Float.Array.unsafe_set acc t
        ( Float.Array.unsafe_get acc t
        +. (Array1.unsafe_get src (r + t) /. Float.Array.unsafe_get len t *. dp)
        )
    done
  done

(* [rolloff32 src fq ~roll ~plane ~k ~n ~t0 ~w total acc] sets [acc.(t)] to the
   least [fq.(j)] whose running bin sum reaches [roll *. total.(t)]. *)
let rolloff32 (src : f32) fq ~roll ~plane ~k ~n ~t0 ~w total run acc =
  Float.Array.fill acc 0 w Float.infinity ;
  Float.Array.fill run 0 w 0. ;
  for t = 0 to w - 1 do
    Float.Array.unsafe_set total t (Float.Array.unsafe_get total t *. roll)
  done ;
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 and f = Float.Array.unsafe_get fq j in
    for t = 0 to w - 1 do
      let c = Float.Array.unsafe_get run t +. Array1.unsafe_get src (r + t) in
      Float.Array.unsafe_set run t c ;
      if c >= Float.Array.unsafe_get total t && f < Float.Array.unsafe_get acc t
      then Float.Array.unsafe_set acc t f
    done
  done

(* [flatness32 src ~amin ~power ~plane ~k ~n ~t0 ~w logs sums bad] sets
   [logs.(t)] and [sums.(t)] to the sums of [log v] and [v], [v] being each bin
   raised to [power] and floored at [amin]. *)
let flatness32 (src : f32) ~amin ~power ~plane ~k ~n ~t0 ~w logs sums bad =
  Float.Array.fill logs 0 w 0. ;
  Float.Array.fill sums 0 w 0. ;
  let square = Float.equal power 2. in
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 in
    for t = 0 to w - 1 do
      let s = Array1.unsafe_get src (r + t) in
      if not (s >= 0.) then bad := true ;
      let v = Float.max (if square then s *. s else Float.pow s power) amin in
      Float.Array.unsafe_set logs t
        (Float.Array.unsafe_get logs t +. Float.log v) ;
      Float.Array.unsafe_set sums t (Float.Array.unsafe_get sums t +. v)
    done
  done

(* [sum64 src ~plane ~k ~n ~t0 ~w acc bad] sets [acc.(t)] to frame [t0 + t]'s
   bin sum. *)
let sum64 (src : f64) ~plane ~k ~n ~t0 ~w acc bad =
  Float.Array.fill acc 0 w 0. ;
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 in
    for t = 0 to w - 1 do
      let v = Array1.unsafe_get src (r + t) in
      if not (v >= 0.) then bad := true ;
      Float.Array.unsafe_set acc t (Float.Array.unsafe_get acc t +. v)
    done
  done

(* [centroid64 src fq ~plane ~k ~n ~t0 ~w len acc] sets [acc.(t)] to the sum of
   [fq.(j) *. (s /. len.(t))]: the bin frequencies weighted by the normalised
   frame. *)
let centroid64 (src : f64) fq ~plane ~k ~n ~t0 ~w len acc =
  Float.Array.fill acc 0 w 0. ;
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 and f = Float.Array.unsafe_get fq j in
    for t = 0 to w - 1 do
      Float.Array.unsafe_set acc t
        ( Float.Array.unsafe_get acc t
        +. (f *. (Array1.unsafe_get src (r + t) /. Float.Array.unsafe_get len t))
        )
    done
  done

(* [spread64 src fq ~p ~plane ~k ~n ~t0 ~w len c acc] sets [acc.(t)] to the sum
   of [(s /. len.(t)) *. |c.(t) -. fq.(j)| ** p]. *)
let spread64 (src : f64) fq ~p ~plane ~k ~n ~t0 ~w len c acc =
  Float.Array.fill acc 0 w 0. ;
  let square = Float.equal p 2. in
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 and f = Float.Array.unsafe_get fq j in
    for t = 0 to w - 1 do
      let d = Float.abs (Float.Array.unsafe_get c t -. f) in
      let dp = if square then d *. d else Float.pow d p in
      Float.Array.unsafe_set acc t
        ( Float.Array.unsafe_get acc t
        +. (Array1.unsafe_get src (r + t) /. Float.Array.unsafe_get len t *. dp)
        )
    done
  done

(* [rolloff64 src fq ~roll ~plane ~k ~n ~t0 ~w total acc] sets [acc.(t)] to the
   least [fq.(j)] whose running bin sum reaches [roll *. total.(t)]. *)
let rolloff64 (src : f64) fq ~roll ~plane ~k ~n ~t0 ~w total run acc =
  Float.Array.fill acc 0 w Float.infinity ;
  Float.Array.fill run 0 w 0. ;
  for t = 0 to w - 1 do
    Float.Array.unsafe_set total t (Float.Array.unsafe_get total t *. roll)
  done ;
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 and f = Float.Array.unsafe_get fq j in
    for t = 0 to w - 1 do
      let c = Float.Array.unsafe_get run t +. Array1.unsafe_get src (r + t) in
      Float.Array.unsafe_set run t c ;
      if c >= Float.Array.unsafe_get total t && f < Float.Array.unsafe_get acc t
      then Float.Array.unsafe_set acc t f
    done
  done

(* [flatness64 src ~amin ~power ~plane ~k ~n ~t0 ~w logs sums bad] sets
   [logs.(t)] and [sums.(t)] to the sums of [log v] and [v], [v] being each bin
   raised to [power] and floored at [amin]. *)
let flatness64 (src : f64) ~amin ~power ~plane ~k ~n ~t0 ~w logs sums bad =
  Float.Array.fill logs 0 w 0. ;
  Float.Array.fill sums 0 w 0. ;
  let square = Float.equal power 2. in
  for j = 0 to k - 1 do
    let r = plane + (j * n) + t0 in
    for t = 0 to w - 1 do
      let s = Array1.unsafe_get src (r + t) in
      if not (s >= 0.) then bad := true ;
      let v = Float.max (if square then s *. s else Float.pow s power) amin in
      Float.Array.unsafe_set logs t
        (Float.Array.unsafe_get logs t +. Float.log v) ;
      Float.Array.unsafe_set sums t (Float.Array.unsafe_get sums t +. v)
    done
  done

(* [reduce_frames ~invalid x ~tile32 ~tile64] is [[...; 1; n]] in [x]'s dtype,
   frame [t] of each plane being element [t - t0] of the accumulator the tile
   function returns for the tile [[t0, t0 + w)]; float32 and float64 are read in
   place, other dtypes widened first and the result narrowed. [invalid ()] is
   called once every tile has run if a tile set [bad]. *)
let reduce_frames (type a) ~invalid (x : (float, a) Nx.t) ~tile32 ~tile64 :
    (float, a) Nx.t =
  let shape = Nx.shape x in
  let nd = Array.length shape in
  let k = shape.(nd - 2) and n = shape.(nd - 1) in
  let lead = Array.sub shape 0 (nd - 2) in
  let outer = Array.fold_left ( * ) 1 lead in
  let out_shape = Array.append lead [|1; n|] in
  let bad = ref false in
  let tiles = (n + tile - 1) / tile in
  (* a tile costs [k * tile] steps per pass; about 32 Ksteps per chunk *)
  let grain = Int.max 1 (32_768 / Int.max 1 (k * tile)) in
  let units f =
    Par.ranges ~n:(outer * tiles) ~grain (fun lo hi ->
        for u = lo to hi - 1 do
          let o = u / tiles and t0 = u mod tiles * tile in
          f o t0 (Int.min n (t0 + tile) - t0)
        done )
  in
  let run64 x =
    Store.make64 out_shape (fun dst ->
        Store.read64 x (fun src ->
            units (fun o t0 w ->
                let acc = tile64 src ~plane:(o * k * n) ~t0 ~w bad in
                for t = 0 to w - 1 do
                  Array1.unsafe_set dst
                    ((o * n) + t0 + t)
                    (Float.Array.unsafe_get acc t)
                done ) ) )
  in
  let result : (float, a) Nx.t =
    match Nx.dtype x with
    | Nx.Float32 ->
        Store.make32 out_shape (fun dst ->
            Store.read32 x (fun src ->
                units (fun o t0 w ->
                    let acc = tile32 src ~plane:(o * k * n) ~t0 ~w bad in
                    for t = 0 to w - 1 do
                      Array1.unsafe_set dst
                        ((o * n) + t0 + t)
                        (Float.Array.unsafe_get acc t)
                    done ) ) )
    | Nx.Float64 ->
        run64 x
    | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
        Nx.cast (Nx.dtype x) (run64 (Nx.cast Nx.float64 x))
  in
  if !bad then invalid () ;
  result

let dims x =
  let shape = Nx.shape x in
  let nd = Array.length shape in
  (shape.(nd - 2), shape.(nd - 1))

let centroid ~fq ~invalid x =
  let k, n = dims x in
  let tile32 src ~plane ~t0 ~w bad =
    let len = Float.Array.create w and acc = Float.Array.create w in
    sum32 src ~plane ~k ~n ~t0 ~w len bad ;
    guard len ~w ;
    centroid32 src fq ~plane ~k ~n ~t0 ~w len acc ;
    acc
  and tile64 src ~plane ~t0 ~w bad =
    let len = Float.Array.create w and acc = Float.Array.create w in
    sum64 src ~plane ~k ~n ~t0 ~w len bad ;
    guard len ~w ;
    centroid64 src fq ~plane ~k ~n ~t0 ~w len acc ;
    acc
  in
  reduce_frames ~invalid x ~tile32 ~tile64

(* The root closes the sum: a square root for [p = 2.], [1 /. p] otherwise. *)
let close ~p acc ~w =
  let square = Float.equal p 2. in
  for t = 0 to w - 1 do
    let v = Float.Array.unsafe_get acc t in
    Float.Array.unsafe_set acc t
      (if square then Float.sqrt v else Float.pow v (1. /. p))
  done

let bandwidth ~fq ~p ?centroid ~invalid x =
  let k, n = dims x in
  (* [centre ~plane ~t0 ~w len] is the tile's centroids: the given ones, or
     those of the normalised frames. *)
  let given o t0 w =
    match centroid with
    | None ->
        None
    | Some c ->
        let a = Float.Array.create w in
        Store.read64 c (fun cs ->
            for t = 0 to w - 1 do
              Float.Array.unsafe_set a t
                (Array1.unsafe_get cs ((o * n) + t0 + t))
            done ) ;
        Some a
  in
  let tile32 src ~plane ~t0 ~w bad =
    let len = Float.Array.create w and acc = Float.Array.create w in
    sum32 src ~plane ~k ~n ~t0 ~w len bad ;
    guard len ~w ;
    let c =
      match given (plane / (k * n)) t0 w with
      | Some c ->
          c
      | None ->
          let c = Float.Array.create w in
          centroid32 src fq ~plane ~k ~n ~t0 ~w len c ;
          c
    in
    spread32 src fq ~p ~plane ~k ~n ~t0 ~w len c acc ;
    close ~p acc ~w ;
    acc
  and tile64 src ~plane ~t0 ~w bad =
    let len = Float.Array.create w and acc = Float.Array.create w in
    sum64 src ~plane ~k ~n ~t0 ~w len bad ;
    guard len ~w ;
    let c =
      match given (plane / (k * n)) t0 w with
      | Some c ->
          c
      | None ->
          let c = Float.Array.create w in
          centroid64 src fq ~plane ~k ~n ~t0 ~w len c ;
          c
    in
    spread64 src fq ~p ~plane ~k ~n ~t0 ~w len c acc ;
    close ~p acc ~w ;
    acc
  in
  reduce_frames ~invalid x ~tile32 ~tile64

let rolloff ~fq ~roll ~invalid x =
  let k, n = dims x in
  let tile32 src ~plane ~t0 ~w bad =
    let total = Float.Array.create w
    and run = Float.Array.create w
    and acc = Float.Array.create w in
    sum32 src ~plane ~k ~n ~t0 ~w total bad ;
    rolloff32 src fq ~roll ~plane ~k ~n ~t0 ~w total run acc ;
    acc
  and tile64 src ~plane ~t0 ~w bad =
    let total = Float.Array.create w
    and run = Float.Array.create w
    and acc = Float.Array.create w in
    sum64 src ~plane ~k ~n ~t0 ~w total bad ;
    rolloff64 src fq ~roll ~plane ~k ~n ~t0 ~w total run acc ;
    acc
  in
  reduce_frames ~invalid x ~tile32 ~tile64

let flatness ~amin ~power ~invalid x =
  let k, n = dims x in
  let finish logs sums ~w =
    let len = Float.of_int k in
    for t = 0 to w - 1 do
      Float.Array.unsafe_set sums t
        ( Float.exp (Float.Array.unsafe_get logs t /. len)
        /. (Float.Array.unsafe_get sums t /. len) )
    done
  in
  let tile32 src ~plane ~t0 ~w bad =
    let logs = Float.Array.create w and sums = Float.Array.create w in
    flatness32 src ~amin ~power ~plane ~k ~n ~t0 ~w logs sums bad ;
    finish logs sums ~w ;
    sums
  and tile64 src ~plane ~t0 ~w bad =
    let logs = Float.Array.create w and sums = Float.Array.create w in
    flatness64 src ~amin ~power ~plane ~k ~n ~t0 ~w logs sums bad ;
    finish logs sums ~w ;
    sums
  in
  reduce_frames ~invalid x ~tile32 ~tile64

(* {1 Synthesis}

   [ola64] carries one range of output positions of one plane. Output [i] is
   padded position [q = left + i]: block [m = q / hop], residue [r]. Frame [m -
   k] reaches it with its sample [k * hop + r], for [k] from [0] to [blocks -
   1]; the windowed taps are added in that order, a missing frame or a sample
   past the window adding an exact zero, which is the association the
   whole-signal overlap-add of shifted hop blocks performs. The sum is divided
   by the envelope: one period of complete taps in [[head, stop)], the [partial]
   sum of the frames that exist elsewhere. Positions past [valid] are the zero
   extension of a requested length. *)
let ola64 (src : f64) (dst : f64) (w : Float.Array.t) ~fft ~hop ~count ~blocks
    ~period ~partial ~head ~stop ~left ~out_len ~valid o lo hi =
  let sbase = o * count * fft and dbase = o * out_len in
  for i = lo to hi - 1 do
    if i >= valid then Array1.unsafe_set dst (dbase + i) 0.
    else begin
      let q = left + i in
      let m = q / hop in
      let r = q - (m * hop) in
      let acc = ref 0. in
      for k = 0 to blocks - 1 do
        let p = m - k and j = (k * hop) + r in
        let v =
          if p >= 0 && p < count && j < fft then
            Array1.unsafe_get src (sbase + (p * fft) + j)
            *. Float.Array.unsafe_get w j
          else 0.
        in
        if k = 0 then acc := v else acc := !acc +. v
      done ;
      let env =
        if q < head || q >= stop then partial q
        else Float.Array.unsafe_get period (q mod hop)
      in
      Array1.unsafe_set dst (dbase + i) (!acc /. env)
    end
  done

let overlap_add ~window ~hop ~period ~partial ~head ~stop ~left ~out_len ~valid
    frames =
  let fft = Float.Array.length window in
  let shape = Nx.shape frames in
  let nd = Array.length shape in
  let count = shape.(nd - 2) in
  let lead = Array.sub shape 0 (nd - 2) in
  let outer = Array.fold_left ( * ) 1 lead in
  let blocks = (fft + hop - 1) / hop in
  (* a position costs [blocks] taps; about 64 Ktaps per chunk *)
  let grain = Int.max 1 (65_536 / Int.max 1 blocks) in
  Store.make64 (Array.append lead [|out_len|]) (fun dst ->
      Store.read64 frames (fun src ->
          Par.ranges ~n:(outer * out_len) ~grain (fun lo hi ->
              let u = ref lo in
              while !u < hi do
                let o = !u / out_len in
                let stop_u = Int.min hi ((o + 1) * out_len) in
                ola64 src dst window ~fft ~hop ~count ~blocks ~period ~partial
                  ~head ~stop ~left ~out_len ~valid o
                  (!u - (o * out_len))
                  (stop_u - (o * out_len)) ;
                u := stop_u
              done ) ) )

(* [update64] carries one range of elements of the Griffin-Lim step: [e] is the
   rebuilt spectrum less [beta] times the previous one, its unit-modulus value
   [e /. (|e| +. min_float)] is scaled by the magnitude, as the complex product
   and quotient by a real compute it. *)
let update64 (r : f64) (prev : f64 option) (mag : f64) (dst : f64) ~beta lo hi =
  match prev with
  | None ->
      for i = lo to hi - 1 do
        let er = Array1.unsafe_get r (2 * i)
        and ei = Array1.unsafe_get r ((2 * i) + 1) in
        let d = Float.hypot er ei +. Float.min_float in
        let m = Array1.unsafe_get mag i in
        Array1.unsafe_set dst (2 * i) (m *. (er /. d)) ;
        Array1.unsafe_set dst ((2 * i) + 1) (m *. (ei /. d))
      done
  | Some p ->
      for i = lo to hi - 1 do
        let er =
          Array1.unsafe_get r (2 * i) -. (Array1.unsafe_get p (2 * i) *. beta)
        and ei =
          Array1.unsafe_get r ((2 * i) + 1)
          -. (Array1.unsafe_get p ((2 * i) + 1) *. beta)
        in
        let d = Float.hypot er ei +. Float.min_float in
        let m = Array1.unsafe_get mag i in
        Array1.unsafe_set dst (2 * i) (m *. (er /. d)) ;
        Array1.unsafe_set dst ((2 * i) + 1) (m *. (ei /. d))
      done

let gl_update ~beta ~magnitudes ?previous rebuilt =
  let n = Nx.numel rebuilt in
  Store.make_c128 (Nx.shape rebuilt) (fun dst ->
      Store.read64 magnitudes (fun mag ->
          Store.read_parts64 rebuilt (fun r ->
              let run prev =
                Par.ranges ~n ~grain:16_384 (fun lo hi ->
                    update64 r prev mag dst ~beta lo hi )
              in
              match previous with
              | None ->
                  run None
              | Some p ->
                  Store.read_parts64 p (fun p -> run (Some p)) ) ) )

(* [extremes32 src valley peak ~k ~n ~bands o t0 w small large] carries one tile
   of [w] frames of plane [o] through every band. A band is rows [lo, lo + len)
   and keeps, per frame, its [take] least and [take] greatest values in two
   ascending buffers ([small] and [large], [take] slots per frame): the rows a
   full sort would put first and last, NaN ordering after every number as the
   sort orders it. The buffers fill from the band's first [take] rows, then
   admit a value only when it beats their current worst. Each mean adds its
   buffer in ascending order from [0.] and divides by [take]. *)
let extremes32 (src : f32) (valley : f64) (peak : f64) ~k ~n ~bands o t0 w small
    large =
  let nb = Array.length bands / 3 in
  let plane = o * k * n in
  for b = 0 to nb - 1 do
    let lo = bands.(3 * b) and len = bands.((3 * b) + 1) in
    let take = bands.((3 * b) + 2) in
    for r = 0 to len - 1 do
      let row = plane + ((lo + r) * n) + t0 in
      for t = 0 to w - 1 do
        let v = Array1.unsafe_get src (row + t) in
        let base = t * take in
        if r < take then begin
          (* insert into the [r] values held, both buffers *)
          let i = ref r in
          while
            !i > 0
            &&
            let a = Float.Array.unsafe_get small (base + !i - 1) in
            v < a || (a <> a && v = v)
          do
            Float.Array.unsafe_set small (base + !i)
              (Float.Array.unsafe_get small (base + !i - 1)) ;
            decr i
          done ;
          Float.Array.unsafe_set small (base + !i) v ;
          let i = ref r in
          while
            !i > 0
            &&
            let a = Float.Array.unsafe_get large (base + !i - 1) in
            v < a || (a <> a && v = v)
          do
            Float.Array.unsafe_set large (base + !i)
              (Float.Array.unsafe_get large (base + !i - 1)) ;
            decr i
          done ;
          Float.Array.unsafe_set large (base + !i) v
        end
        else begin
          let worst = Float.Array.unsafe_get small (base + take - 1) in
          if v < worst || (worst <> worst && v = v) then begin
            (* drop the greatest held, insert [v] *)
            let i = ref (take - 1) in
            while
              !i > 0
              &&
              let a = Float.Array.unsafe_get small (base + !i - 1) in
              v < a || (a <> a && v = v)
            do
              Float.Array.unsafe_set small (base + !i)
                (Float.Array.unsafe_get small (base + !i - 1)) ;
              decr i
            done ;
            Float.Array.unsafe_set small (base + !i) v
          end ;
          let least = Float.Array.unsafe_get large base in
          if least < v || (v <> v && least = least) then begin
            (* drop the least held, insert [v] *)
            let i = ref 0 in
            while
              !i < take - 1
              &&
              let a = Float.Array.unsafe_get large (base + !i + 1) in
              a < v || (v <> v && a = a)
            do
              Float.Array.unsafe_set large (base + !i)
                (Float.Array.unsafe_get large (base + !i + 1)) ;
              incr i
            done ;
            Float.Array.unsafe_set large (base + !i) v
          end
        end
      done
    done ;
    let count = Float.of_int take in
    let out = (((o * nb) + b) * n) + t0 in
    for t = 0 to w - 1 do
      let base = t * take in
      let sv = ref 0. and sp = ref 0. in
      for i = 0 to take - 1 do
        sv := !sv +. Float.Array.unsafe_get small (base + i) ;
        sp := !sp +. Float.Array.unsafe_get large (base + i)
      done ;
      Array1.unsafe_set valley (out + t) (!sv /. count) ;
      Array1.unsafe_set peak (out + t) (!sp /. count)
    done
  done

let extremes64 (src : f64) (valley : f64) (peak : f64) ~k ~n ~bands o t0 w small
    large =
  let nb = Array.length bands / 3 in
  let plane = o * k * n in
  for b = 0 to nb - 1 do
    let lo = bands.(3 * b) and len = bands.((3 * b) + 1) in
    let take = bands.((3 * b) + 2) in
    for r = 0 to len - 1 do
      let row = plane + ((lo + r) * n) + t0 in
      for t = 0 to w - 1 do
        let v = Array1.unsafe_get src (row + t) in
        let base = t * take in
        if r < take then begin
          (* insert into the [r] values held, both buffers *)
          let i = ref r in
          while
            !i > 0
            &&
            let a = Float.Array.unsafe_get small (base + !i - 1) in
            v < a || (a <> a && v = v)
          do
            Float.Array.unsafe_set small (base + !i)
              (Float.Array.unsafe_get small (base + !i - 1)) ;
            decr i
          done ;
          Float.Array.unsafe_set small (base + !i) v ;
          let i = ref r in
          while
            !i > 0
            &&
            let a = Float.Array.unsafe_get large (base + !i - 1) in
            v < a || (a <> a && v = v)
          do
            Float.Array.unsafe_set large (base + !i)
              (Float.Array.unsafe_get large (base + !i - 1)) ;
            decr i
          done ;
          Float.Array.unsafe_set large (base + !i) v
        end
        else begin
          let worst = Float.Array.unsafe_get small (base + take - 1) in
          if v < worst || (worst <> worst && v = v) then begin
            (* drop the greatest held, insert [v] *)
            let i = ref (take - 1) in
            while
              !i > 0
              &&
              let a = Float.Array.unsafe_get small (base + !i - 1) in
              v < a || (a <> a && v = v)
            do
              Float.Array.unsafe_set small (base + !i)
                (Float.Array.unsafe_get small (base + !i - 1)) ;
              decr i
            done ;
            Float.Array.unsafe_set small (base + !i) v
          end ;
          let least = Float.Array.unsafe_get large base in
          if least < v || (v <> v && least = least) then begin
            (* drop the least held, insert [v] *)
            let i = ref 0 in
            while
              !i < take - 1
              &&
              let a = Float.Array.unsafe_get large (base + !i + 1) in
              a < v || (v <> v && a = a)
            do
              Float.Array.unsafe_set large (base + !i)
                (Float.Array.unsafe_get large (base + !i + 1)) ;
              incr i
            done ;
            Float.Array.unsafe_set large (base + !i) v
          end
        end
      done
    done ;
    let count = Float.of_int take in
    let out = (((o * nb) + b) * n) + t0 in
    for t = 0 to w - 1 do
      let base = t * take in
      let sv = ref 0. and sp = ref 0. in
      for i = 0 to take - 1 do
        sv := !sv +. Float.Array.unsafe_get small (base + i) ;
        sp := !sp +. Float.Array.unsafe_get large (base + i)
      done ;
      Array1.unsafe_set valley (out + t) (!sv /. count) ;
      Array1.unsafe_set peak (out + t) (!sp /. count)
    done
  done

let extremes (type a) ~bands (x : (float, a) Nx.t) =
  let shape = Nx.shape x in
  let nd = Array.length shape in
  let k = shape.(nd - 2) and n = shape.(nd - 1) in
  let lead = Array.sub shape 0 (nd - 2) in
  let outer = Array.fold_left ( * ) 1 lead in
  let nb = Array.length bands / 3 in
  let most = ref 1 in
  for b = 0 to nb - 1 do
    most := Int.max !most bands.((3 * b) + 2)
  done ;
  let out_shape = Array.append lead [|nb; n|] in
  let tiles = (n + tile - 1) / tile in
  let grain = Int.max 1 (32_768 / Int.max 1 (k * tile)) in
  let units run =
    Par.ranges ~n:(outer * tiles) ~grain (fun lo hi ->
        let small = Float.Array.create (tile * !most)
        and large = Float.Array.create (tile * !most) in
        for u = lo to hi - 1 do
          let o = u / tiles and t0 = u mod tiles * tile in
          run o t0 (Int.min n (t0 + tile) - t0) small large
        done )
  in
  Store.make64_pair out_shape (fun valley peak ->
      match Nx.dtype x with
      | Nx.Float32 ->
          Store.read32 x (fun src ->
              units (extremes32 src valley peak ~k ~n ~bands) )
      | Nx.Float64 ->
          Store.read64 x (fun src ->
              units (extremes64 src valley peak ~k ~n ~bands) )
      | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
          Store.read64 (Nx.cast Nx.float64 x) (fun src ->
              units (extremes64 src valley peak ~k ~n ~bands) ) )

(* {1 Decibels}

   A parallel unit is a block of [db_block] consecutive elements of one channel.
   The first pass writes every value and records each block's greatest (NaN when
   the block holds one); the clamp, when asked for, reads each channel's
   greatest from its blocks and floors the channel in a second pass. The scale's
   product is bound to a name before the offset is taken from it, which keeps
   the multiply and the subtraction apart on every platform: two roundings, as
   the whole-tensor formulation rounds. *)

let db_block = 8192

let db32 (src : f32) (dst : f32) ~scale ~offset ~amin ~magnitude peaks b lo hi =
  let top = ref Float.neg_infinity and nan = ref false in
  for i = lo to hi - 1 do
    let v = Array1.unsafe_get src i in
    let v = if magnitude then Float.abs v else v in
    let v = if v < amin then amin else v in
    let p = Float.log v *. scale in
    let d = p -. offset in
    Array1.unsafe_set dst i d ;
    if d > !top then top := d else if d <> d then nan := true
  done ;
  Float.Array.unsafe_set peaks b (if !nan then Float.nan else !top)

let db64 (src : f64) (dst : f64) ~scale ~offset ~amin ~magnitude peaks b lo hi =
  let top = ref Float.neg_infinity and nan = ref false in
  for i = lo to hi - 1 do
    let v = Array1.unsafe_get src i in
    let v = if magnitude then Float.abs v else v in
    let v = if v < amin then amin else v in
    let p = Float.log v *. scale in
    let d = p -. offset in
    Array1.unsafe_set dst i d ;
    if d > !top then top := d else if d <> d then nan := true
  done ;
  Float.Array.unsafe_set peaks b (if !nan then Float.nan else !top)

let db32_64 (src : f32) (dst : f64) ~scale ~offset ~amin ~magnitude peaks b lo
    hi =
  let top = ref Float.neg_infinity and nan = ref false in
  for i = lo to hi - 1 do
    let v = Array1.unsafe_get src i in
    let v = if magnitude then Float.abs v else v in
    let v = if v < amin then amin else v in
    let p = Float.log v *. scale in
    let d = p -. offset in
    Array1.unsafe_set dst i d ;
    if d > !top then top := d else if d <> d then nan := true
  done ;
  Float.Array.unsafe_set peaks b (if !nan then Float.nan else !top)

(* [floor32 dst ~threshold lo hi] and [floor64] raise every element below
   [threshold] to it, and set every element to NaN when [threshold] is NaN. *)
let floor32 (dst : f32) ~threshold lo hi =
  if Float.is_nan threshold then
    for i = lo to hi - 1 do
      Array1.unsafe_set dst i Float.nan
    done
  else
    for i = lo to hi - 1 do
      if Array1.unsafe_get dst i < threshold then
        Array1.unsafe_set dst i threshold
    done

let floor64 (dst : f64) ~threshold lo hi =
  if Float.is_nan threshold then
    for i = lo to hi - 1 do
      Array1.unsafe_set dst i Float.nan
    done
  else
    for i = lo to hi - 1 do
      if Array1.unsafe_get dst i < threshold then
        Array1.unsafe_set dst i threshold
    done

(* [db_run ~shape ~range ~round fill floor] runs [fill peaks b lo hi] over every
   block, then, when [range] is given, [floor ~threshold lo hi] over every block
   with its channel's threshold: [round] of the channel's greatest value less
   [range], where [round] rounds into the result's dtype. A channel is the whole
   tensor up to rank two, one index of the leading axes beyond. *)
let db_run ~shape ~range ~round fill floor =
  let nd = Array.length shape in
  let total = Array.fold_left ( * ) 1 shape in
  let len = if nd <= 2 then total else shape.(nd - 2) * shape.(nd - 1) in
  let channels = if len = 0 then 0 else total / len in
  let per = (len + db_block - 1) / db_block in
  let blocks = channels * per in
  let span b =
    let c = b / per and q = b mod per in
    ((c * len) + (q * db_block), (c * len) + Int.min len ((q + 1) * db_block))
  in
  let peaks = Float.Array.make blocks Float.neg_infinity in
  Par.ranges ~n:blocks ~grain:1 (fun b0 b1 ->
      for b = b0 to b1 - 1 do
        let lo, hi = span b in
        fill peaks b lo hi
      done ) ;
  match range with
  | None ->
      ()
  | Some range ->
      let thresholds =
        Float.Array.init channels (fun c ->
            let top = ref Float.neg_infinity and nan = ref false in
            for q = 0 to per - 1 do
              let p = Float.Array.get peaks ((c * per) + q) in
              if p > !top then top := p else if Float.is_nan p then nan := true
            done ;
            if !nan then Float.nan else round (round !top -. range) )
      in
      Par.ranges ~n:blocks ~grain:1 (fun b0 b1 ->
          for b = b0 to b1 - 1 do
            let lo, hi = span b in
            floor ~threshold:(Float.Array.get thresholds (b / per)) lo hi
          done )

let round32 x =
  let a = Array1.create float32 c_layout 1 in
  Array1.unsafe_set a 0 x ; Array1.unsafe_get a 0

let decibels (type a) ~scale ~offset ~amin ~magnitude ~range
    (x : (float, a) Nx.t) : (float, a) Nx.t option =
  let shape = Nx.shape x in
  match Nx.dtype x with
  | Nx.Float32 ->
      Some
        (Store.make32 shape (fun dst ->
             Store.read32 x (fun src ->
                 db_run ~shape ~range ~round:round32
                   (db32 src dst ~scale ~offset ~amin ~magnitude)
                   (floor32 dst) ) ) )
  | Nx.Float64 ->
      Some
        (Store.make64 shape (fun dst ->
             Store.read64 x (fun src ->
                 db_run ~shape ~range ~round:Fun.id
                   (db64 src dst ~scale ~offset ~amin ~magnitude)
                   (floor64 dst) ) ) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      None

let decibels64 (type a) ~scale ~offset ~amin ~magnitude ~range
    (x : (float, a) Nx.t) : (float, Nx.float64_elt) Nx.t option =
  let shape = Nx.shape x in
  match Nx.dtype x with
  | Nx.Float32 ->
      Some
        (Store.make64 shape (fun dst ->
             Store.read32 x (fun src ->
                 db_run ~shape ~range ~round:Fun.id
                   (db32_64 src dst ~scale ~offset ~amin ~magnitude)
                   (floor64 dst) ) ) )
  | Nx.Float64 ->
      decibels ~scale ~offset ~amin ~magnitude ~range x
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      None

(* {1 Onset flux}

   A parallel unit is one tile of [tile] output frames of one plane: the bin
   rows are read one at a time, each frame's positive difference added to its
   accumulator, so every frame takes its bins in ascending order from [0.] — the
   association of {!Ordered.sum}. *)

let flux32 (src : f32) (dst : f64) ~k ~n ~lag ~first ~count o t0 w acc =
  Float.Array.fill acc 0 w 0. ;
  for j = 0 to k - 1 do
    let r = (((o * k) + j) * n) + first + t0 in
    for t = 0 to w - 1 do
      let d =
        Array1.unsafe_get src (r + t) -. Array1.unsafe_get src (r + t - lag)
      in
      let v = if d < 0. then 0. else d in
      Float.Array.unsafe_set acc t (Float.Array.unsafe_get acc t +. v)
    done
  done ;
  let bins = Float.of_int k in
  for t = 0 to w - 1 do
    Array1.unsafe_set dst
      ((o * count) + t0 + t)
      (Float.Array.unsafe_get acc t /. bins)
  done

let flux64 (src : f64) (dst : f64) ~k ~n ~lag ~first ~count o t0 w acc =
  Float.Array.fill acc 0 w 0. ;
  for j = 0 to k - 1 do
    let r = (((o * k) + j) * n) + first + t0 in
    for t = 0 to w - 1 do
      let d =
        Array1.unsafe_get src (r + t) -. Array1.unsafe_get src (r + t - lag)
      in
      let v = if d < 0. then 0. else d in
      Float.Array.unsafe_set acc t (Float.Array.unsafe_get acc t +. v)
    done
  done ;
  let bins = Float.of_int k in
  for t = 0 to w - 1 do
    Array1.unsafe_set dst
      ((o * count) + t0 + t)
      (Float.Array.unsafe_get acc t /. bins)
  done

let flux (type a) ~lag ~first ~count (x : (float, a) Nx.t) :
    (float, Nx.float64_elt) Nx.t option =
  let shape = Nx.shape x in
  let nd = Array.length shape in
  let k = shape.(nd - 2) and n = shape.(nd - 1) in
  let lead = Array.sub shape 0 (nd - 2) in
  let outer = Array.fold_left ( * ) 1 lead in
  let tiles = (count + tile - 1) / tile in
  let grain = Int.max 1 (32_768 / Int.max 1 (k * tile)) in
  let run kernel dst =
    Par.ranges ~n:(outer * tiles) ~grain (fun lo hi ->
        let acc = Float.Array.create tile in
        for u = lo to hi - 1 do
          let o = u / tiles and t0 = u mod tiles * tile in
          kernel dst o t0 (Int.min count (t0 + tile) - t0) acc
        done )
  in
  let out = Array.append lead [|count|] in
  match Nx.dtype x with
  | Nx.Float32 ->
      Some
        (Store.make64 out (fun dst ->
             Store.read32 x (fun src ->
                 run (fun dst -> flux32 src dst ~k ~n ~lag ~first ~count) dst ) )
        )
  | Nx.Float64 ->
      Some
        (Store.make64 out (fun dst ->
             Store.read64 x (fun src ->
                 run (fun dst -> flux64 src dst ~k ~n ~lag ~first ~count) dst ) )
        )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      None
