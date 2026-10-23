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

(* Linear recursive filters, as scipy.signal runs them: the transposed direct
   form II of lfilter (_lfilter.cc) and the cascade of the same structure that
   sosfilt runs per second-order section (_sosfilt.pyx), the steady states of
   lfilter_zi and sosfilt_zi, the forward-backward filtfilt and sosfiltfilt over
   scipy's padded extensions, the Butterworth design of butter through its
   zeros, poles and gain, and librosa's pre- and de-emphasis on top.

   Every recurrence is spelled in the reference's order of operations, and every
   multiply-add the reference's arm64 build contracts is written with let-bound
   operands, which is how OCaml contracts it too (fmadd, fmsub, fnmsub); the
   products the reference rounds on their own are let-bound products, which
   OCaml leaves alone. x86_64 contracts neither side. The state is double
   precision at both dtypes and the output is rounded once, so a stream carrying
   the state across chunks is the offline function bit for bit, and the offline
   function is the stream's one chunk.

   The design side reproduces numpy's own complex arithmetic: Smith's division
   with a reciprocal scale, the multiply and the modulus as numpy's loops round
   them, numpy's pairwise sum for the steady states. The one primitive left to
   the platform by the reference, the complex square root of the band
   transforms, is Friedland's algorithm here (see the interface). *)

type coefficients = (float, Nx.float64_elt) Nx.t

type state = (float, Nx.float64_elt) Nx.t

type band =
  [ `Lowpass of float
  | `Highpass of float
  | `Bandpass of float * float
  | `Bandstop of float * float ]

(* {1 Shapes and checks} *)

let numel shape = Array.fold_left ( * ) 1 shape

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let check_signal (type a) fn (x : (float, a) Nx.t) =
  if Nx.ndim x < 1 then
    invalid_arg
      (fn ^ ": cannot filter a rank-zero tensor (the time axis must exist)") ;
  match Nx.dtype x with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      invalid_arg (fn ^ ": cannot filter this dtype (float32 and float64 only)")

let continue_error fn l lead =
  invalid_arg
    (Printf.sprintf
       "%s: cannot continue a stream of leading shape [%s] with a chunk of \
        leading shape [%s]"
       fn (shape_string l) (shape_string lead) )

(* [vector fn name c] is the rank-one coefficient tensor [c] as a float
   array. *)
let vector fn name (c : coefficients) =
  if Nx.ndim c <> 1 then
    invalid_arg
      (Printf.sprintf "%s: cannot use %s of shape [%s] (%s must have rank 1)" fn
         name
         (shape_string (Nx.shape c))
         name ) ;
  let v : float array = Nx.to_array c in
  if Array.length v = 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use an empty %s (%s needs at least one coefficient)" fn
         name name ) ;
  if not (Array.for_all Float.is_finite v) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use the non-finite coefficients of %s (every coefficient \
          must be finite)"
         fn name ) ;
  v

(* {1 Coefficients}

   A transfer function holds [b] and [a] divided by [a.(0)] and zero-filled to
   [n = max nb na] taps, as scipy's kernel prepares them once per call, and the
   lengths given, over which the steady state sums. *)

type tf = {b: float array; a: float array; n: int; nb: int; na: int}

let tf_of fn (b : float array) (a : float array) =
  let a0 = a.(0) in
  if a0 = 0. then
    invalid_arg
      (Printf.sprintf
         "%s: cannot normalise by a.(0) = 0 (the leading denominator \
          coefficient must be nonzero)"
         fn ) ;
  let nb = Array.length b and na = Array.length a in
  let n = Int.max nb na in
  let norm v len =
    Array.init n (fun k -> (if k < len then v.(k) else 0.) /. a0)
  in
  {b= norm b nb; a= norm a na; n; nb; na}

let tf fn b a = tf_of fn (vector fn "b" b) (vector fn "a" a)

type sos = {c: float array; ns: int}

let sos_of fn (s : coefficients) =
  let shape = Nx.shape s in
  if Array.length shape <> 2 || shape.(1) <> 6 || shape.(0) < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use sections of shape [%s] (sos must be [n_sections; 6] \
          with at least one section)"
         fn (shape_string shape) ) ;
  let c : float array = Nx.to_array s in
  if not (Array.for_all Float.is_finite c) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use non-finite section coefficients (every coefficient \
          must be finite)"
         fn ) ;
  let ns = shape.(0) in
  for i = 0 to ns - 1 do
    if c.((6 * i) + 3) <> 1. then
      invalid_arg
        (Printf.sprintf
           "%s: cannot use section %d with a0 = %g (every section must have a0 \
            = 1)"
           fn i
           c.((6 * i) + 3) )
  done ;
  {c; ns}

(* {1 The kernels}

   Each runs [len] samples of one row from [src] at [so] into [dst] at [d],
   starting from the row's state at [z.(zo)] and leaving the state after the
   last sample there. *)

(* The transposed direct form II, _lfilter.cc's per-sample loop. *)
let tf_run c (z : float array) zo (src : Store.f64) so (dst : Store.f64) d len =
  let n = c.n and b = c.b and a = c.a in
  if n = 1 then begin
    let b0 = Array.unsafe_get b 0 in
    for t = 0 to len - 1 do
      let x = Bigarray.Array1.unsafe_get src (so + t) in
      Bigarray.Array1.unsafe_set dst (d + t) (x *. b0)
    done
  end
  else
    for t = 0 to len - 1 do
      let x = Bigarray.Array1.unsafe_get src (so + t) in
      let b0 = Array.unsafe_get b 0 and z0 = Array.unsafe_get z zo in
      let y = z0 +. (b0 *. x) in
      for k = 0 to n - 3 do
        let z1 = Array.unsafe_get z (zo + k + 1)
        and bk = Array.unsafe_get b (k + 1)
        and ak = Array.unsafe_get a (k + 1) in
        let u = z1 +. (x *. bk) in
        Array.unsafe_set z (zo + k) (u -. (y *. ak))
      done ;
      let bl = Array.unsafe_get b (n - 1) and al = Array.unsafe_get a (n - 1) in
      let p = y *. al in
      Array.unsafe_set z (zo + n - 2) ((x *. bl) -. p) ;
      Bigarray.Array1.unsafe_set dst (d + t) y
    done

(* The cascade, _sosfilt.pyx's loop: every section within each sample, the
   intermediate kept double between sections. *)
let sos_run s (z : float array) zo (src : Store.f64) so (dst : Store.f64) d len
    =
  let ns = s.ns and c = s.c in
  for t = 0 to len - 1 do
    let x = Bigarray.Array1.unsafe_get src (so + t) in
    let cur = ref x in
    for i = 0 to ns - 1 do
      let xc = !cur in
      let o = 6 * i and q = zo + (2 * i) in
      let b0 = Array.unsafe_get c o
      and b1 = Array.unsafe_get c (o + 1)
      and b2 = Array.unsafe_get c (o + 2)
      and a1 = Array.unsafe_get c (o + 4)
      and a2 = Array.unsafe_get c (o + 5) in
      let z0 = Array.unsafe_get z q and z1 = Array.unsafe_get z (q + 1) in
      let xn = z0 +. (b0 *. xc) in
      let p1 = a1 *. xn in
      let u = (b1 *. xc) -. p1 in
      Array.unsafe_set z q (u +. z1) ;
      let p2 = a2 *. xn in
      Array.unsafe_set z (q + 1) ((b2 *. xc) -. p2) ;
      cur := xn
    done ;
    Bigarray.Array1.unsafe_set dst (d + t) !cur
  done

(* librosa's de-emphasis without a state: the recursion of {!tf_run} at two taps
   from rest, minus [k coef ^ t] at the sample's index [t] in the signal,
   subtracted afterwards as librosa subtracts it. Returns the last output. *)
let deemph_run c ~coef ~k ~t0 (z : float array) zo (src : Store.f64) so
    (dst : Store.f64) d len =
  let b0 = c.b.(0) and bl = c.b.(1) and al = c.a.(1) in
  let last = ref 0. in
  for t = 0 to len - 1 do
    let x = Bigarray.Array1.unsafe_get src (so + t) in
    let z0 = Array.unsafe_get z zo in
    let y = z0 +. (b0 *. x) in
    let p = y *. al in
    Array.unsafe_set z zo ((x *. bl) -. p) ;
    let w = Float.pow coef (Float.of_int (t0 + t)) in
    let corr = k *. w in
    let out = y -. corr in
    Bigarray.Array1.unsafe_set dst (d + t) out ;
    last := out
  done ;
  !last

(* [filter_rows x ~cost run] is the tensor of [x]'s shape and dtype whose row
   [r] (along the last axis) is what [run r t src so dst d len] writes for the
   consecutive spans [[t, t + len)] of the row, in order. Rows run across
   domains; each carries its own state, so the split changes no bit. A float32
   row is widened and narrowed a block at a time around [run], which carries the
   state from block to block, so the blocking changes no bit either. *)
let block = 4096

let filter_rows (type a) (x : (float, a) Nx.t) ~cost
    (run : int -> int -> Store.f64 -> int -> Store.f64 -> int -> int -> unit) :
    (float, a) Nx.t =
  let shape = Nx.shape x in
  let m = last_dim x in
  let rows = numel (leading_shape x) in
  let grain = Int.max 1 (32_768 / Int.max 1 (m * cost)) in
  match Nx.dtype x with
  | Nx.Float64 ->
      Store.make64 shape (fun dst ->
          Store.read64 x (fun src ->
              Par.ranges ~n:rows ~grain (fun lo hi ->
                  for r = lo to hi - 1 do
                    run r 0 src (r * m) dst (r * m) m
                  done ) ) )
  | Nx.Float32 ->
      Store.make32 shape (fun dst ->
          Store.read32 x (fun src ->
              Par.ranges ~n:rows ~grain (fun lo hi ->
                  let w = Int.min block m in
                  let wide =
                    Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout w
                  and out =
                    Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout w
                  in
                  for r = lo to hi - 1 do
                    let t = ref 0 in
                    while !t < m do
                      let len = Int.min w (m - !t) and base = (r * m) + !t in
                      for i = 0 to len - 1 do
                        Bigarray.Array1.unsafe_set wide i
                          (Bigarray.Array1.unsafe_get src (base + i))
                      done ;
                      run r !t wide 0 out 0 len ;
                      for i = 0 to len - 1 do
                        Bigarray.Array1.unsafe_set dst (base + i)
                          (Bigarray.Array1.unsafe_get out i)
                      done ;
                      t := !t + len
                    done
                  done ) ) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      assert false

let run_tf c z x =
  let w = c.n - 1 in
  filter_rows x ~cost:(2 * c.n) (fun r _ src so dst d len ->
      tf_run c z (r * w) src so dst d len )

let run_sos s z x =
  filter_rows x ~cost:(5 * s.ns) (fun r _ src so dst d len ->
      sos_run s z (r * 2 * s.ns) src so dst d len )

let run_deemph c ~coef ~(k : float array) ~t0 z x =
  let last = Array.make (numel (leading_shape x)) 0. in
  let y =
    filter_rows x ~cost:8 (fun r t src so dst d len ->
        last.(r) <-
          deemph_run c ~coef ~k:k.(r) ~t0:(t0 + t) z r src so dst d len )
  in
  (y, last)

(* {1 States} *)

(* [check_tail fn ~tail zi] checks that the trailing axes of [zi] are [tail],
   the part of its shape the signal does not decide. *)
let check_tail fn ~tail zi =
  let s = Nx.shape zi in
  let r = Array.length s and k = Array.length tail in
  if r < k || Array.sub s (r - k) k <> tail then
    invalid_arg
      (Printf.sprintf
         "%s: cannot start from a state of shape [%s] (zi must be [...; %s])" fn
         (shape_string s) (shape_string tail) )

(* [initial fn ~tail zi lead] is the state of every row of a signal of leading
   shape [lead], [tail] values per row: [zi] broadcast against [lead @ tail], or
   rest. *)
let initial fn ~tail zi lead =
  let rows = numel lead and width = numel tail in
  match zi with
  | None ->
      Array.make (rows * width) 0.
  | Some zi ->
      check_tail fn ~tail zi ;
      let s = Nx.shape zi and target = Array.append lead tail in
      let extra = Array.length s - Array.length tail
      and l = Array.length lead in
      let fits = ref (extra <= l) in
      for i = 0 to extra - 1 do
        if !fits then begin
          let d = s.(i) and e = lead.(l - extra + i) in
          if d <> 1 && d <> e then fits := false
        end
      done ;
      if not !fits then
        invalid_arg
          (Printf.sprintf
             "%s: cannot start from a state of shape [%s] (zi must broadcast \
              to [%s])"
             fn (shape_string s) (shape_string target) ) ;
      if rows * width = 0 then [||]
      else
        Store.read64 (Nx.broadcast_to target zi) (fun v ->
            Array.init (rows * width) (fun i -> Bigarray.Array1.unsafe_get v i) )

let state_of lead tail (z : float array) =
  Store.make64 (Array.append lead tail) (fun d ->
      Array.iteri (fun i v -> Bigarray.Array1.unsafe_set d i v) z )

(* {1 Steady states} *)

(* numpy's pairwise sum of [v.(lo) .. v.(lo + n - 1)] (loops_utils.h.src):
   sequential below eight terms from [-0.], eight interleaved accumulators up to
   128, and halves at a multiple of eight above. *)
let rec pairwise (v : float array) lo n =
  if n < 8 then begin
    let res = ref (-0.) in
    for i = lo to lo + n - 1 do
      res := !res +. v.(i)
    done ;
    !res
  end
  else if n <= 128 then begin
    let r = Array.sub v lo 8 in
    let i = ref 8 in
    while !i < n - (n mod 8) do
      for j = 0 to 7 do
        r.(j) <- r.(j) +. v.(lo + !i + j)
      done ;
      i := !i + 8
    done ;
    let res =
      ref
        ( r.(0) +. r.(1)
        +. (r.(2) +. r.(3))
        +. (r.(4) +. r.(5) +. (r.(6) +. r.(7))) )
    in
    while !i < n do
      res := !res +. v.(lo + !i) ;
      incr i
    done ;
    !res
  end
  else
    let n2 = n / 2 in
    let n2 = n2 - (n2 mod 8) in
    pairwise v lo n2 +. pairwise v (lo + n2) (n - n2)

(* numpy's [sum] of the first [n] terms of [v]: the reduction adds the pairwise
   sum to its identity [+0.], so a sum of zeros is [+0.] whatever their
   signs. *)
let sum (v : float array) n = 0. +. pairwise v 0 n

(* scipy's lfilter_zi: [d = b - y_inf a] with [y_inf = sum b / sum a] the step
   response's limit, and the state its running sums from the end. *)
let zi_of fn c =
  let sa = sum c.a c.na in
  if sa = 0. then
    invalid_arg
      (Printf.sprintf
         "%s: cannot settle a filter whose denominator sums to 0 (a pole at z \
          = 1 has no steady state)"
         fn ) ;
  let y_inf = sum c.b c.nb /. sa in
  let n = c.n in
  let d =
    Array.init n (fun k ->
        let bk = if k < c.nb then c.b.(k) else 0.
        and ak = if k < c.na then c.a.(k) else 0. in
        let p = y_inf *. ak in
        bk -. p )
  in
  let zi = Array.make (n - 1) 0. in
  if n > 1 then begin
    let acc = ref d.(n - 1) in
    zi.(n - 2) <- !acc ;
    for j = n - 2 downto 1 do
      acc := !acc +. d.(j) ;
      zi.(j - 1) <- !acc
    done
  end ;
  zi

(* scipy's sosfilt_zi: each section's steady state, scaled by the DC gain of the
   sections before it. *)
let sos_zi_of fn s =
  let zi = Array.make (2 * s.ns) 0. in
  let scale = ref 1. in
  for i = 0 to s.ns - 1 do
    let b = Array.sub s.c (6 * i) 3 and a = Array.sub s.c ((6 * i) + 3) 3 in
    let z = zi_of fn (tf_of fn b a) in
    zi.(2 * i) <- !scale *. z.(0) ;
    zi.((2 * i) + 1) <- !scale *. z.(1) ;
    scale := !scale *. (sum b 3 /. sum a 3)
  done ;
  zi

(* {1 Pipeline stages} *)

let concat_chunks fn witness = function
  | [] -> (
    match !witness with
    | Some (dtype, lead) ->
        Nx.zeros dtype (Array.append lead [|0|])
    | None ->
        invalid_arg
          ( fn
          ^ ": cannot concatenate zero chunks before any chunk fixed the \
             element dtype" ) )
  | parts ->
      Nx.concatenate ~axis:(-1) parts

type carry = {mutable z: float array option; mutable lead: int array option}

(* The stage of a filter whose every input sample yields its output at once: the
   per-row state, made by [init] from the first chunk's leading shape and
   advanced by [run] over each chunk. No latency, nothing to flush. *)
let carry_stage fn ~init ~run =
  (* The element dtype is only witnessed by incoming chunks; [step] records it
     so [concat []] (reached only through [run], where a step always precedes)
     can build the empty output. *)
  let witness = ref None in
  let step s chunk =
    check_signal fn chunk ;
    let lead = leading_shape chunk in
    let z =
      match (s.z, s.lead) with
      | Some z, Some l when l = lead ->
          z
      | _, Some l ->
          continue_error fn l lead
      | _ ->
          let z = init lead in
          s.z <- Some z ;
          s.lead <- Some lead ;
          z
    in
    witness := Some (Nx.dtype chunk, lead) ;
    run z chunk
  in
  Pipeline.kernel
    ~reset:(fun s ->
      s.z <- None ;
      s.lead <- None )
    ~concat:(concat_chunks fn witness)
    ~prepare:(fun _format -> {z= None; lead= None})
    ~step:(fun s chunk -> Some (step s chunk))
    ()

(* {1 Transfer functions} *)

let lfilter_full fn ?zi b a x =
  let c = tf fn b a in
  check_signal fn x ;
  let lead = leading_shape x in
  let z = initial fn ~tail:[|c.n - 1|] zi lead in
  let y = run_tf c z x in
  (y, lead, c, z)

let lfilter ?zi b a x =
  let y, _, _, _ = lfilter_full "lfilter" ?zi b a x in
  y

let lfilter_zf ?zi b a x =
  let y, lead, c, z = lfilter_full "lfilter_zf" ?zi b a x in
  (y, state_of lead [|c.n - 1|] z)

let lfilter_zi b a =
  let fn = "lfilter_zi" in
  let c = tf fn b a in
  let zi = zi_of fn c in
  state_of [||] [|c.n - 1|] zi

let lfilter_stage ?zi b a =
  let fn = "lfilter_stage" in
  let c = tf fn b a in
  let tail = [|c.n - 1|] in
  Option.iter (check_tail fn ~tail) zi ;
  carry_stage fn ~init:(initial fn ~tail zi) ~run:(run_tf c)

(* {1 Zero-phase filtering} *)

let edge_of fn padlen ~default =
  match padlen with
  | None ->
      default
  | Some p ->
      if p < 0 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot pad by %d samples (padlen must be non-negative)" fn p ) ;
      p

(* [zero_phase fn ~pad ~edge ~zi run x] filters each row of [x] forward and then
   backward over its extension by [edge] samples at each end, from the steady
   state [zi] scaled by the first sample of each pass, and keeps the samples
   under the signal: scipy's filtfilt with method "pad". Rows run across
   domains, in double precision throughout; a float32 signal is widened exactly
   and its result rounded once. *)
let zero_phase (type a) fn ~pad ~edge ~(zi : float array) ~cost run
    (x : (float, a) Nx.t) : (float, a) Nx.t =
  let n = last_dim x in
  if n <= edge then
    invalid_arg
      (Printf.sprintf
         "%s: cannot extend a signal of %d samples by %d at each end (the \
          signal must be longer than padlen)"
         fn n edge ) ;
  let shape = Nx.shape x in
  let rows = numel (leading_shape x) in
  let len = n + (2 * edge) and width = Array.length zi in
  let wide : (float, Nx.float64_elt) Nx.t = Nx.cast Nx.float64 x in
  let grain = Int.max 1 (32_768 / Int.max 1 (len * cost)) in
  let y =
    Store.make64 shape (fun dst ->
        Store.read64 wide (fun src ->
            Par.ranges ~n:rows ~grain (fun lo hi ->
                let e =
                  Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout len
                and f =
                  Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout len
                in
                let z = Array.make width 0. in
                for r = lo to hi - 1 do
                  let base = r * n in
                  for t = 0 to n - 1 do
                    Bigarray.Array1.unsafe_set e (edge + t)
                      (Bigarray.Array1.unsafe_get src (base + t))
                  done ;
                  let x0 = Bigarray.Array1.unsafe_get src base
                  and xl = Bigarray.Array1.unsafe_get src (base + n - 1) in
                  for i = 0 to edge - 1 do
                    let lk = Bigarray.Array1.unsafe_get src (base + edge - i)
                    and rk =
                      Bigarray.Array1.unsafe_get src (base + n - 2 - i)
                    in
                    let left, right =
                      match pad with
                      | `Odd ->
                          let tl = 2. *. x0 and tr = 2. *. xl in
                          (tl -. lk, tr -. rk)
                      | `Reflect ->
                          (lk, rk)
                      | `Edge ->
                          (x0, xl)
                    in
                    Bigarray.Array1.unsafe_set e i left ;
                    Bigarray.Array1.unsafe_set e (edge + n + i) right
                  done ;
                  let e0 = Bigarray.Array1.unsafe_get e 0 in
                  for k = 0 to width - 1 do
                    z.(k) <- zi.(k) *. e0
                  done ;
                  run z e f len ;
                  for i = 0 to len - 1 do
                    Bigarray.Array1.unsafe_set e i
                      (Bigarray.Array1.unsafe_get f (len - 1 - i))
                  done ;
                  let y0 = Bigarray.Array1.unsafe_get e 0 in
                  for k = 0 to width - 1 do
                    z.(k) <- zi.(k) *. y0
                  done ;
                  run z e f len ;
                  for t = 0 to n - 1 do
                    Bigarray.Array1.unsafe_set dst (base + t)
                      (Bigarray.Array1.unsafe_get f (len - 1 - edge - t))
                  done
                done ) ) )
  in
  match Nx.dtype x with
  | Nx.Float64 ->
      y
  | Nx.Float32 ->
      Nx.cast Nx.float32 y
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      assert false

let filtfilt ?(pad = `Odd) ?padlen b a x =
  let fn = "filtfilt" in
  let c = tf fn b a in
  check_signal fn x ;
  let edge = edge_of fn padlen ~default:(3 * c.n) in
  let zi = zi_of fn c in
  zero_phase fn ~pad ~edge ~zi ~cost:(2 * c.n)
    (fun z e f len -> tf_run c z 0 e 0 f 0 len)
    x

(* {1 Second-order sections} *)

let sosfilt_full fn ?zi sos x =
  let s = sos_of fn sos in
  check_signal fn x ;
  let lead = leading_shape x in
  let z = initial fn ~tail:[|s.ns; 2|] zi lead in
  let y = run_sos s z x in
  (y, lead, s, z)

let sosfilt ?zi sos x =
  let y, _, _, _ = sosfilt_full "sosfilt" ?zi sos x in
  y

let sosfilt_zf ?zi sos x =
  let y, lead, s, z = sosfilt_full "sosfilt_zf" ?zi sos x in
  (y, state_of lead [|s.ns; 2|] z)

let sosfilt_zi sos =
  let fn = "sosfilt_zi" in
  let s = sos_of fn sos in
  state_of [||] [|s.ns; 2|] (sos_zi_of fn s)

let sosfilt_stage ?zi sos =
  let fn = "sosfilt_stage" in
  let s = sos_of fn sos in
  let tail = [|s.ns; 2|] in
  Option.iter (check_tail fn ~tail) zi ;
  carry_stage fn ~init:(initial fn ~tail zi) ~run:(run_sos s)

let sosfiltfilt ?(pad = `Odd) ?padlen sos x =
  let fn = "sosfiltfilt" in
  let s = sos_of fn sos in
  check_signal fn x ;
  let zeros col =
    let k = ref 0 in
    for i = 0 to s.ns - 1 do
      if s.c.((6 * i) + col) = 0. then incr k
    done ;
    !k
  in
  let ntaps = (2 * s.ns) + 1 - Int.min (zeros 2) (zeros 5) in
  let edge = edge_of fn padlen ~default:(3 * ntaps) in
  let zi = sos_zi_of fn s in
  zero_phase fn ~pad ~edge ~zi ~cost:(5 * s.ns)
    (fun z e f len -> sos_run s z 0 e 0 f 0 len)
    x

(* {1 Butterworth design}

   scipy's iirfilter path for butter: the analog prototype's poles, the
   frequency transformation to the band, the bilinear map, and the conversion to
   a transfer function or to sections — each value computed with the operations
   numpy computes it with. *)

let cx re im = {Complex.re; im}

(* numpy's complex multiply as its loop rounds it: the real part contracts the
   first product into the difference, the imaginary part the second into the
   sum. *)
let cmul (x : Complex.t) (y : Complex.t) =
  let ar = x.re and ai = x.im and br = y.re and bi = y.im in
  let p = ai *. bi and q = ai *. br in
  cx ((ar *. br) -. p) (q +. (ar *. bi))

(* numpy's complex division: Smith's algorithm with a reciprocal scale. *)
let cdiv (x : Complex.t) (y : Complex.t) =
  let ar = x.re and ai = x.im and br = y.re and bi = y.im in
  let abs_br = Float.abs br and abs_bi = Float.abs bi in
  if abs_br >= abs_bi then
    if abs_br = 0. && abs_bi = 0. then cx (ar /. abs_br) (ai /. abs_br)
    else
      let rat = bi /. br in
      let den = br +. (bi *. rat) in
      let scl = 1. /. den in
      let nr = ar +. (ai *. rat) and ni = ai -. (ar *. rat) in
      cx (nr *. scl) (ni *. scl)
  else
    let rat = br /. bi in
    let den = bi +. (br *. rat) in
    let scl = 1. /. den in
    let nr = ai +. (ar *. rat) and ni = (ai *. rat) -. ar in
    cx (nr *. scl) (ni *. scl)

(* The complex square root by Friedland's algorithm (CACM 10(10), 1967), on the
   correctly rounded modulus. *)
let csqrt (z : Complex.t) =
  let a = z.re and b = z.im in
  if a = 0. && b = 0. then cx 0. b
  else if a >= 0. then
    let t = Float.sqrt ((a +. Float.hypot a b) *. 0.5) in
    cx t (b /. (2. *. t))
  else
    let t = Float.sqrt ((-.a +. Float.hypot a b) *. 0.5) in
    cx (Float.abs b /. (2. *. t)) (Float.copy_sign t b)

(* numpy's modulus as its vector loop computes it. *)
let cabs (z : Complex.t) =
  let x = Float.abs z.re and y = Float.abs z.im in
  let l = if x >= y then x else y and s = if x >= y then y else x in
  if l = 0. then 0.
  else
    let q = s /. l in
    l *. Float.sqrt ((q *. q) +. 1.)

let cprod = Array.fold_left cmul (cx 1. 0.)

let cneg (z : Complex.t) = cx (-.z.re) (-.z.im)

let conj (z : Complex.t) = cx z.re (-.z.im)

(* [poly roots] is the monic polynomial with [roots], highest power first, by
   repeated convolution with [[1; -r]] — numpy's complex dot of two terms, the
   older coefficient's product first. *)
let poly (roots : Complex.t array) =
  Array.fold_left
    (fun (a : Complex.t array) (r : Complex.t) ->
      let nzr = -.r.re and nzi = -.r.im in
      let n = Array.length a in
      let out = Array.make (n + 1) a.(0) in
      for j = 1 to n - 1 do
        let ar1 = a.(j - 1).re and ai1 = a.(j - 1).im in
        let ar2 = a.(j).re and ai2 = a.(j).im in
        let p_rr = ar1 *. nzr
        and p_ii = ai1 *. nzi
        and p_ri = ar1 *. nzi
        and p_ir = ai1 *. nzr in
        let q_rr = ar2 *. 1.
        and q_ii = ai2 *. 0.
        and q_ri = ar2 *. 0.
        and q_ir = ai2 *. 1. in
        let rr = 0. +. p_rr +. q_rr
        and ii = 0. +. p_ii +. q_ii
        and ri = 0. +. p_ri +. q_ri
        and ir = 0. +. p_ir +. q_ir in
        out.(j) <- cx (rr -. ii) (ri +. ir)
      done ;
      let ar1 = a.(n - 1).re and ai1 = a.(n - 1).im in
      let p_rr = ar1 *. nzr
      and p_ii = ai1 *. nzi
      and p_ri = ar1 *. nzi
      and p_ir = ai1 *. nzr in
      out.(n) <- cx (0. +. p_rr -. (0. +. p_ii)) (0. +. p_ri +. (0. +. p_ir)) ;
      out )
    [|cx 1. 0.|]
    roots

let real_parts = Array.map (fun (c : Complex.t) -> c.re)

(* [butter_zpk fn ~order ~sample_rate band] is the digital Butterworth filter's
   zeros, poles and gain, after scipy's butter. *)
let butter_zpk fn ~order ~sample_rate band =
  if order < 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot design a filter of order %d (order must be non-negative)"
         fn order ) ;
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot design at %d Hz (sample_rate must be at least 1)" fn
         sample_rate ) ;
  let nyquist = Float.of_int sample_rate /. 2. in
  let cutoffs =
    match band with
    | `Lowpass f | `Highpass f ->
        [|f|]
    | `Bandpass (lo, hi) | `Bandstop (lo, hi) ->
        [|lo; hi|]
  in
  let wn = Array.map (fun f -> f /. nyquist) cutoffs in
  Array.iteri
    (fun i w ->
      if not (w > 0. && w < 1.) then
        invalid_arg
          (Printf.sprintf
             "%s: cannot place a cutoff at %g Hz (cutoffs must lie strictly \
              between 0 and sample_rate / 2 = %g Hz)"
             fn cutoffs.(i) nyquist ) )
    wn ;
  if Array.length wn = 2 && not (wn.(0) < wn.(1)) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use the band (%g, %g) Hz (its lower edge must lie below \
          its upper edge)"
         fn cutoffs.(0) cutoffs.(1) ) ;
  (* frequencies pre-warped onto the analog axis of the bilinear map at fs =
     2 *)
  let warped = Array.map (fun w -> 4. *. Float.tan (Float.pi *. w /. 2.)) wn in
  let n = order in
  (* buttap: [-exp (1j pi m / (2 n))], numpy evaluating the angle as [(1j pi) m]
     divided by [2 n] *)
  let proto =
    Array.init n (fun i ->
        let m = Float.of_int (-n + 1 + (2 * i)) in
        let num = cmul (cx 0. Float.pi) (cx m 0.) in
        let th = (cdiv num (cx (Float.of_int (2 * n)) 0.)).im in
        cx (-.Float.cos th) (-.Float.sin th) )
  in
  let inverse_gain () = (cdiv (cx 1. 0.) (cprod (Array.map cneg proto))).re in
  let shift wo2 q =
    let sq = cmul q q in
    csqrt (cx (sq.re -. wo2) (sq.im -. 0.))
  in
  let split wo qs =
    let wo2 = Float.pow wo 2. in
    let s = Array.map (shift wo2) qs in
    Array.append
      (Array.map2
         (fun (q : Complex.t) (r : Complex.t) ->
           cx (q.re +. r.re) (q.im +. r.im) )
         qs s )
      (Array.map2
         (fun (q : Complex.t) (r : Complex.t) ->
           cx (q.re -. r.re) (q.im -. r.im) )
         qs s )
  in
  let z, p, k =
    match band with
    | `Lowpass _ ->
        let wo = warped.(0) in
        ([||], Array.map (cmul (cx wo 0.)) proto, Float.pow wo (Float.of_int n))
    | `Highpass _ ->
        let wo = warped.(0) in
        ( Array.make n (cx 0. 0.)
        , Array.map (fun q -> cdiv (cx wo 0.) q) proto
        , inverse_gain () )
    | `Bandpass _ ->
        let bw = warped.(1) -. warped.(0)
        and wo = Float.sqrt (warped.(0) *. warped.(1)) in
        (* scipy's [p * bw / 2]: a complex multiply, then a complex division,
           which turns the [-0.] imaginary part of the real prototype pole into
           [+0.] and so picks the square root's branch below *)
        let lp =
          Array.map
            (fun (q : Complex.t) -> cdiv (cmul q (cx bw 0.)) (cx 2. 0.))
            proto
        in
        (Array.make n (cx 0. 0.), split wo lp, Float.pow bw (Float.of_int n))
    | `Bandstop _ ->
        let bw = warped.(1) -. warped.(0)
        and wo = Float.sqrt (warped.(0) *. warped.(1)) in
        let hp = Array.map (fun q -> cdiv (cx (bw /. 2.) 0.) q) proto in
        ( Array.append (Array.make n (cx 0. wo)) (Array.make n (cx (-0.) (-.wo)))
        , split wo hp
        , inverse_gain () )
  in
  (* the bilinear map at fs = 2, its zeros at infinity moved to -1 *)
  let bilinear q =
    cdiv (cx (4. +. q.Complex.re) (0. +. q.im)) (cx (4. -. q.re) (0. -. q.im))
  in
  let complement q = cx (4. -. q.Complex.re) (0. -. q.im) in
  let degree = Array.length p - Array.length z in
  let zd =
    Array.append (Array.map bilinear z) (Array.make degree (cx (-1.) 0.))
  in
  let pd = Array.map bilinear p in
  let kd =
    k
    *. (cdiv (cprod (Array.map complement z)) (cprod (Array.map complement p)))
         .re
  in
  (zd, pd, kd)

(* {2 Second-order sections: scipy's zpk2sos, pairing "nearest", digital} *)

let tol = 100. *. epsilon_float

let isreal (q : Complex.t) = q.im = 0.

(* [cplxreal z] is scipy's _cplxreal: one of each conjugate pair, the two
   averaged, sorted by real part and then by imaginary part; and the real
   values, sorted. *)
let cplxreal (z : Complex.t array) =
  if Array.length z = 0 then ([||], [||])
  else
    let sorted =
      List.stable_sort
        (fun (u : Complex.t) (v : Complex.t) ->
          let c = Float.compare u.re v.re in
          if c <> 0 then c else Float.compare (Float.abs u.im) (Float.abs v.im) )
        (Array.to_list z)
    in
    let real (q : Complex.t) = Float.abs q.im <= tol *. cabs q in
    let zr =
      Array.of_list
        (List.filter_map
           (fun (q : Complex.t) -> if real q then Some q.re else None)
           sorted )
    in
    let zc = List.filter (fun q -> not (real q)) sorted in
    let zp = Array.of_list (List.filter (fun (q : Complex.t) -> q.im > 0.) zc)
    and zn =
      Array.of_list (List.filter (fun (q : Complex.t) -> q.im < 0.) zc)
    in
    let n = Array.length zp in
    assert (Array.length zn = n) ;
    (* runs of (nearly) the same real part, each re-sorted by its imaginary
       parts *)
    let same j = zp.(j + 1).re -. zp.(j).re <= tol *. cabs zp.(j) in
    let i = ref 0 in
    while !i < n do
      let j = ref !i in
      while !j + 1 < n && same !j do
        incr j
      done ;
      if !j > !i then
        List.iter
          (fun (arr : Complex.t array) ->
            let seg = Array.to_list (Array.sub arr !i (!j - !i + 1)) in
            let seg =
              List.stable_sort
                (fun (u : Complex.t) (v : Complex.t) ->
                  Float.compare (Float.abs u.im) (Float.abs v.im) )
                seg
            in
            List.iteri (fun k q -> arr.(!i + k) <- q) seg )
          [zp; zn] ;
      i := !j + 1
    done ;
    let zc =
      Array.init n (fun k ->
          let s = cx (zp.(k).re +. zn.(k).re) (zp.(k).im +. -.zn.(k).im) in
          cx (s.re *. 0.5) (s.im *. 0.5) )
    in
    (zc, zr)

let remove (a : Complex.t array) i =
  Array.init (Array.length a - 1) (fun j -> if j < i then a.(j) else a.(j + 1))

(* the pole closest to the unit circle, the first of equals *)
let idx_worst (p : Complex.t array) =
  let best = ref 0 and value = ref Float.infinity in
  Array.iteri
    (fun i q ->
      let v = Float.abs (1. -. cabs q) in
      if v < !value then begin
        value := v ;
        best := i
      end )
    p ;
  !best

(* the index of the value of [fro] nearest [target] among those [which] selects,
   the first of equals *)
let nearest (fro : Complex.t array) (target : Complex.t) which =
  let d =
    Array.map
      (fun (q : Complex.t) -> cabs (cx (q.re -. target.re) (q.im -. target.im)))
      fro
  in
  let order =
    List.stable_sort
      (fun i j -> Float.compare d.(i) d.(j))
      (List.init (Array.length fro) Fun.id)
  in
  let ok i =
    match which with
    | `Any ->
        true
    | `Real ->
        isreal fro.(i)
    | `Complex ->
        not (isreal fro.(i))
  in
  List.find ok order

let single (z : Complex.t array) (p : Complex.t array) =
  let b = real_parts (poly z) and a = real_parts (poly p) in
  let row = Array.make 6 0. in
  Array.iteri (fun i v -> row.(3 - Array.length b + i) <- v) b ;
  Array.iteri (fun i v -> row.(6 - Array.length a + i) <- v) a ;
  row

let zpk2sos (z : Complex.t array) (p : Complex.t array) k =
  if Array.length z = 0 && Array.length p = 0 then [|[|k; 0.; 0.; 1.; 0.; 0.|]|]
  else begin
    let lz = Array.length z and lp = Array.length p in
    let p = Array.append p (Array.make (Int.max (lz - lp) 0) (cx 0. 0.))
    and z = Array.append z (Array.make (Int.max (lp - lz) 0) (cx 0. 0.)) in
    let nsec = (Int.max (Array.length p) (Array.length z) + 1) / 2 in
    let p, z =
      if Array.length p mod 2 = 1 then
        (Array.append p [|cx 0. 0.|], Array.append z [|cx 0. 0.|])
      else (p, z)
    in
    let flat (c, r) = Array.append c (Array.map (fun v -> cx v 0.) r) in
    let z = ref (flat (cplxreal z)) and p = ref (flat (cplxreal p)) in
    let nreal a =
      Array.fold_left (fun n q -> if isreal q then n + 1 else n) 0 a
    in
    let take r i =
      let q = !r.(i) in
      r := remove !r i ;
      q
    in
    let sos = Array.make_matrix nsec 6 0. in
    for si = nsec - 1 downto 0 do
      let p1 = take p (idx_worst !p) in
      if isreal p1 && nreal !p = 0 then begin
        (* the last real pole *)
        let z1 = take z (nearest !z p1 `Real) in
        sos.(si) <- single [|z1; cx 0. 0.|] [|p1; cx 0. 0.|]
      end
      else if
        Array.length !p + 1 = Array.length !z
        && (not (isreal p1))
        && nreal !p = 1
        && nreal !z = 1
      then begin
        (* one real pole and one real zero left: a complex zero must go here *)
        let z1 = take z (nearest !z p1 `Complex) in
        sos.(si) <- single [|z1; conj z1|] [|p1; conj p1|]
      end
      else begin
        let p2 =
          if isreal p1 then begin
            let reals =
              Array.of_list
                (List.filter
                   (fun i -> isreal !p.(i))
                   (List.init (Array.length !p) Fun.id) )
            in
            take p reals.(idx_worst (Array.map (fun i -> !p.(i)) reals))
          end
          else conj p1
        in
        if Array.length !z > 0 then begin
          let z1 = take z (nearest !z p1 `Any) in
          if not (isreal z1) then sos.(si) <- single [|z1; conj z1|] [|p1; p2|]
          else if Array.length !z > 0 then begin
            let z2 = take z (nearest !z p1 `Real) in
            sos.(si) <- single [|z1; z2|] [|p1; p2|]
          end
          else sos.(si) <- single [|z1|] [|p1; p2|]
        end
        else sos.(si) <- single [||] [|p1; p2|]
      end
    done ;
    for j = 0 to 2 do
      sos.(0).(j) <- sos.(0).(j) *. k
    done ;
    sos
  end

let butter ~order ~sample_rate band =
  let z, p, k = butter_zpk "butter" ~order ~sample_rate band in
  let b = Array.map (fun (c : Complex.t) -> k *. c.re) (poly z) in
  let a = real_parts (poly p) in
  ( Nx.create Nx.float64 [|Array.length b|] b
  , Nx.create Nx.float64 [|Array.length a|] a )

let butter_sos ~order ~sample_rate band =
  let z, p, k = butter_zpk "butter_sos" ~order ~sample_rate band in
  let sos = zpk2sos z p k in
  Nx.create Nx.float64
    [|Array.length sos; 6|]
    (Array.concat (Array.to_list sos))

(* {1 Pre- and de-emphasis} *)

let check_coef fn coef =
  if not (Float.is_finite coef) then
    invalid_arg
      (Printf.sprintf "%s: cannot use coef = %g (coef must be finite)" fn coef)

let pre_tf coef = tf_of "preemphasis" [|1.; -.coef|] [|1.|]

let de_tf coef = tf_of "deemphasis" [|1.|] [|1.; -.coef|]

let too_short fn =
  invalid_arg
    ( fn
    ^ ": cannot extrapolate a starting state from a single sample (without ~zi \
       the signal needs at least two)" )

(* [firsts x] is the first two samples of every row of [x]. *)
let firsts (type a) (x : (float, a) Nx.t) =
  let nd = Nx.ndim x in
  let head =
    Nx.slice
      (List.init nd (fun i -> if i = nd - 1 then Nx.R (0, 2) else Nx.A))
      x
  in
  let rows = numel (leading_shape x) in
  let y0 = Array.make rows 0. and y1 = Array.make rows 0. in
  let fill get =
    for r = 0 to rows - 1 do
      y0.(r) <- get (2 * r) ;
      y1.(r) <- get ((2 * r) + 1)
    done
  in
  ( match Nx.dtype x with
  | Nx.Float32 ->
      Store.read32 head (fun v -> fill (Bigarray.Array1.get v))
  | Nx.Float64 ->
      Store.read64 head (fun v -> fill (Bigarray.Array1.get v))
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      assert false ) ;
  (y0, y1)

(* librosa's default pre-emphasis state, [2 y0 - y1] *)
let extrapolate y0 y1 =
  let t = 2. *. y0 in
  t -. y1

(* librosa's de-emphasis correction constant, [((2 - coef) y0 - y1) / (3 -
   coef)] *)
let correction ~coef y0 y1 =
  let p = (2. -. coef) *. y0 in
  (p -. y1) /. (3. -. coef)

let preemphasis_full fn ~coef ~zi x =
  check_coef fn coef ;
  check_signal fn x ;
  let c = pre_tf coef in
  let lead = leading_shape x in
  let z =
    match zi with
    | Some _ ->
        initial fn ~tail:[|1|] zi lead
    | None ->
        let m = last_dim x in
        if m = 1 then too_short fn ;
        if m = 0 then Array.make (numel lead) 0.
        else
          let y0, y1 = firsts x in
          Array.map2 extrapolate y0 y1
  in
  let y = run_tf c z x in
  (y, state_of lead [|1|] z)

let preemphasis ?(coef = 0.97) ?zi x =
  fst (preemphasis_full "preemphasis" ~coef ~zi x)

let preemphasis_zf ?(coef = 0.97) ?zi x =
  preemphasis_full "preemphasis_zf" ~coef ~zi x

let deemphasis_full fn ~coef ~zi x =
  check_coef fn coef ;
  check_signal fn x ;
  let c = de_tf coef in
  let lead = leading_shape x in
  let rows = numel lead in
  match zi with
  | Some _ ->
      let z = initial fn ~tail:[|1|] zi lead in
      let y = run_tf c z x in
      (y, state_of lead [|1|] z)
  | None ->
      let m = last_dim x in
      if m = 1 then too_short fn ;
      if m = 0 then (run_tf c [||] x, state_of lead [|1|] (Array.make rows 0.))
      else
        let y0, y1 = firsts x in
        let k = Array.map2 (correction ~coef) y0 y1 in
        let y, last = run_deemph c ~coef ~k ~t0:0 (Array.make rows 0.) x in
        (* the state that continues the returned signal, not librosa's *)
        (y, state_of lead [|1|] (Array.map (fun v -> coef *. v) last))

let deemphasis ?(coef = 0.97) ?zi x =
  fst (deemphasis_full "deemphasis" ~coef ~zi x)

let deemphasis_zf ?(coef = 0.97) ?zi x =
  deemphasis_full "deemphasis_zf" ~coef ~zi x

(* {2 The emphasis stages}

   With a state they are the filter's {!carry_stage}. Without one, librosa's
   default reads the second sample before the first output, so the stage holds
   back the first sample until the second arrives — one sample of latency,
   declared — and from then on emits every sample at once: [first] holds that
   sample while it is alone, [k] the de-emphasis constants and [t] the samples
   emitted. *)

type 'a hold =
  { mutable hz: float array option
  ; mutable hlead: int array option
  ; mutable first: (float, 'a) Nx.t option
  ; mutable k: float array
  ; mutable t: int }

let emphasis_stage fn ~pre ~coef ~zi =
  check_coef fn coef ;
  let c = if pre then pre_tf coef else de_tf coef in
  match zi with
  | Some zi ->
      check_tail fn ~tail:[|1|] zi ;
      carry_stage fn ~init:(initial fn ~tail:[|1|] (Some zi)) ~run:(run_tf c)
  | None ->
      let witness = ref None in
      let emit s z x =
        if pre then run_tf c z x
        else begin
          let y, _ = run_deemph c ~coef ~k:s.k ~t0:s.t z x in
          s.t <- s.t + last_dim x ;
          y
        end
      in
      let step s chunk =
        check_signal fn chunk ;
        let lead = leading_shape chunk in
        ( match s.hlead with
        | None ->
            s.hlead <- Some lead
        | Some l when l = lead ->
            ()
        | Some l ->
            continue_error fn l lead ) ;
        witness := Some (Nx.dtype chunk, lead) ;
        match s.hz with
        | Some z ->
            emit s z chunk
        | None ->
            let ext =
              match s.first with
              | None ->
                  chunk
              | Some f ->
                  Nx.concatenate ~axis:(-1) [f; chunk]
            in
            let m = last_dim ext in
            if m < 2 then begin
              if m = 1 then s.first <- Some (Nx.copy ext) ;
              Nx.zeros (Nx.dtype chunk) (Array.append lead [|0|])
            end
            else begin
              s.first <- None ;
              let y0, y1 = firsts ext in
              let z =
                if pre then Array.map2 extrapolate y0 y1
                else begin
                  s.k <- Array.map2 (correction ~coef) y0 y1 ;
                  Array.make (Array.length y0) 0.
                end
              in
              s.hz <- Some z ;
              emit s z ext
            end
      in
      let flush s = if Option.is_some s.first then too_short fn else [] in
      Pipeline.kernel ~latency:1 ~flush
        ~reset:(fun s ->
          s.hz <- None ;
          s.hlead <- None ;
          s.first <- None ;
          s.k <- [||] ;
          s.t <- 0 )
        ~concat:(concat_chunks fn witness)
        ~prepare:(fun _format ->
          {hz= None; hlead= None; first= None; k= [||]; t= 0} )
        ~step:(fun s chunk -> Some (step s chunk))
        ()

let preemphasis_stage ?(coef = 0.97) ?zi () =
  emphasis_stage "preemphasis_stage" ~pre:true ~coef ~zi

let deemphasis_stage ?(coef = 0.97) ?zi () =
  emphasis_stage "deemphasis_stage" ~pre:false ~coef ~zi
