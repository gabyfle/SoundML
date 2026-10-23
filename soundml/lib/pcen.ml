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

(* Per-channel energy normalization, as librosa.pcen computes it: each band's
   magnitude is smoothed over time by the one-pole low-pass [y t = b x t + (1 -
   b) y (t - 1)], started as if every earlier frame had magnitude 1 (scipy's
   [lfilter_zi]), and the band is then divided by a power of its smoothed self
   and compressed. [max_size > 1] smooths, instead of the magnitude, its maximum
   over [max_size] neighbouring bands (scipy's [maximum_filter1d], reflected at
   the edges), which needs nothing outside the frame.

   The recurrence is spelled as scipy's [lfilter] runs it — the product [b x],
   then the sum with the carried state, then the state [(1 - b) y] — with the
   product let-bound so that no platform fuses it, and every value is double
   precision, rounded to the dtype once. The state is one value per band, so a
   stream carries it across chunks and the offline function is the stream's one
   chunk. *)

type params =
  { b: float
  ; gain: float
  ; bias: float
  ; power: float
  ; eps: float
  ; max_size: int
  ; zi: float (* the state before the first frame *) }

let check_finite fn name ~ok v =
  if not (Float.is_finite v && ok v) then
    invalid_arg
      (Printf.sprintf "%s: cannot use %s = %g (%s must be finite and %s)" fn
         name v name
         ( match name with
         | "eps" | "time_constant" ->
             "positive"
         | "b" ->
             "in (0, 1]"
         | _ ->
             "non-negative" ) )

let params fn ~hop ~gain ~bias ~power ~time_constant ~eps ~b ~max_size
    ~sample_rate =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot smooth at %d Hz (sample_rate must be at least 1)" fn
         sample_rate ) ;
  if hop < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot smooth at a hop of %d (hop must be at least 1)" fn hop ) ;
  let nonneg v = v >= 0. and pos v = v > 0. in
  check_finite fn "gain" ~ok:nonneg gain ;
  check_finite fn "bias" ~ok:nonneg bias ;
  check_finite fn "power" ~ok:nonneg power ;
  check_finite fn "eps" ~ok:pos eps ;
  check_finite fn "time_constant" ~ok:pos time_constant ;
  if max_size < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot take the maximum over %d bands (max_size must be at least \
          1)"
         fn max_size ) ;
  let b =
    match b with
    | Some b ->
        b
    | None ->
        (* the coefficient whose squared response falls to half at
           [time_constant] seconds: the root of [b^2 + (1 - b) / t - 2 = 0] *)
        let t = time_constant *. Float.of_int sample_rate /. Float.of_int hop in
        let tt = t *. t in
        let four = 4. *. tt in
        (Float.sqrt (1. +. four) -. 1.) /. (2. *. tt)
  in
  check_finite fn "b" ~ok:(fun b -> b > 0. && b <= 1.) b ;
  (* scipy's lfilter_zi for [[b]; [1; b - 1]]: the state a constant input of 1
     settles to *)
  let a1 = b -. 1. in
  let zi = -.(a1 *. b) /. (1. +. a1) in
  {b; gain; bias; power; eps; max_size; zi}

(* {1 The kernel} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

(* [reflect j n] is the band [maximum_filter1d]'s "reflect" mode reads at index
   [j] of [n] bands: [... b a | a b c ... | c b ...], periodic over [2 n]. *)
let reflect j n =
  let p = 2 * n in
  let j = ((j mod p) + p) mod p in
  if j < n then j else p - 1 - j

(* [ROW p ~log_eps ~bias_power src dst z ~bands ~m r] runs band row [r] ([bands]
   bands per block, [m] frames per row) from state [z.(r)] and leaves the state
   after its last frame there; one copy per element type, so the loads and
   stores stay unboxed. *)
let row32 p ~log_eps ~bias_power (src : Store.f32) (dst : Store.f32)
    (z : float array) ~bands ~m r =
  let base = r * m in
  let block = r / bands * bands and band = r mod bands in
  let c = 1. -. p.b in
  let state = ref z.(r) in
  for t = 0 to m - 1 do
    let x = Bigarray.Array1.unsafe_get src (base + t) in
    let reference =
      if p.max_size = 1 then x
      else begin
        let lo = band - (p.max_size / 2) in
        let v = ref Float.neg_infinity in
        for j = lo to lo + p.max_size - 1 do
          let u =
            Bigarray.Array1.unsafe_get src (((block + reflect j bands) * m) + t)
          in
          if u > !v then v := u
        done ;
        !v
      end
    in
    let bx = p.b *. reference in
    let y = bx +. !state in
    state := c *. y ;
    let smooth =
      Float.exp (-.p.gain *. (log_eps +. Float.log1p (y /. p.eps)))
    in
    let out =
      if p.power = 0. then Float.log1p (x *. smooth)
      else if p.bias = 0. then
        Float.exp (p.power *. (Float.log x +. Float.log smooth))
      else
        bias_power
        *. Float.expm1 (p.power *. Float.log1p (x *. smooth /. p.bias))
    in
    Bigarray.Array1.unsafe_set dst (base + t) out
  done ;
  z.(r) <- !state

let row64 p ~log_eps ~bias_power (src : Store.f64) (dst : Store.f64)
    (z : float array) ~bands ~m r =
  let base = r * m in
  let block = r / bands * bands and band = r mod bands in
  let c = 1. -. p.b in
  let state = ref z.(r) in
  for t = 0 to m - 1 do
    let x = Bigarray.Array1.unsafe_get src (base + t) in
    let reference =
      if p.max_size = 1 then x
      else begin
        let lo = band - (p.max_size / 2) in
        let v = ref Float.neg_infinity in
        for j = lo to lo + p.max_size - 1 do
          let u =
            Bigarray.Array1.unsafe_get src (((block + reflect j bands) * m) + t)
          in
          if u > !v then v := u
        done ;
        !v
      end
    in
    let bx = p.b *. reference in
    let y = bx +. !state in
    state := c *. y ;
    let smooth =
      Float.exp (-.p.gain *. (log_eps +. Float.log1p (y /. p.eps)))
    in
    let out =
      if p.power = 0. then Float.log1p (x *. smooth)
      else if p.bias = 0. then
        Float.exp (p.power *. (Float.log x +. Float.log smooth))
      else
        bias_power
        *. Float.expm1 (p.power *. Float.log1p (x *. smooth /. p.bias))
    in
    Bigarray.Array1.unsafe_set dst (base + t) out
  done ;
  z.(r) <- !state

(* [normalize p z x] is the normalization of the chunk [x] ([[...; bands;
   frames]], or [[frames]] alone) from the per-band states [z], which it
   advances past the chunk. Band rows run across domains; each carries only its
   own state, so the split changes no bit. *)
let normalize (type a) p (z : float array) (x : (float, a) Nx.t) :
    (float, a) Nx.t =
  let shape = Nx.shape x in
  let m = last_dim x in
  let rows = Array.length z in
  let bands = if Nx.ndim x >= 2 then shape.(Nx.ndim x - 2) else 1 in
  let log_eps = Float.log p.eps and bias_power = Float.pow p.bias p.power in
  let grain = Int.max 1 (32_768 / Int.max 1 (m * p.max_size)) in
  let run kernel =
    Par.ranges ~n:rows ~grain (fun lo hi ->
        for r = lo to hi - 1 do
          kernel r
        done )
  in
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.make32 shape (fun dst ->
          Store.read32 x (fun src ->
              run (row32 p ~log_eps ~bias_power src dst z ~bands ~m) ) )
  | Nx.Float64 ->
      Store.make64 shape (fun dst ->
          Store.read64 x (fun src ->
              run (row64 p ~log_eps ~bias_power src dst z ~bands ~m) ) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      assert false

let check_input (type a) fn p (x : (float, a) Nx.t) =
  let nd = Nx.ndim x in
  if nd < 1 then
    invalid_arg
      (fn ^ ": cannot normalize a rank-zero tensor (the frame axis must exist)") ;
  if p.max_size > 1 && nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot take the maximum over bands of a rank-1 tensor (max_size \
          %d needs [...; bands; frames])"
         fn p.max_size ) ;
  match Nx.dtype x with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      invalid_arg
        (fn ^ ": cannot normalize this dtype (float32 and float64 only)")

let fresh p lead = Array.make (Array.fold_left ( * ) 1 lead) p.zi

(* {1 The flat function} *)

let pcen ?(hop = 512) ?(gain = 0.98) ?(bias = 2.) ?(power = 0.5)
    ?(time_constant = 0.4) ?(eps = 1e-6) ?b ?(max_size = 1) ~sample_rate x =
  let fn = "pcen" in
  let p =
    params fn ~hop ~gain ~bias ~power ~time_constant ~eps ~b ~max_size
      ~sample_rate
  in
  check_input fn p x ;
  normalize p (fresh p (leading_shape x)) x

(* {1 Pipeline stage}

   The state is the band states, made from the first chunk's leading shape and
   advanced by each chunk; every input frame yields its output frame at once, so
   there is no latency and nothing to flush. *)

type state = {mutable z: float array option; mutable lead: int array option}

let stage ?(hop = 512) ?(gain = 0.98) ?(bias = 2.) ?(power = 0.5)
    ?(time_constant = 0.4) ?(eps = 1e-6) ?b ?(max_size = 1) ~sample_rate () =
  let fn = "pcen_stage" in
  let p =
    params fn ~hop ~gain ~bias ~power ~time_constant ~eps ~b ~max_size
      ~sample_rate
  in
  let step s chunk =
    check_input fn p chunk ;
    let lead = leading_shape chunk in
    let z =
      match (s.z, s.lead) with
      | Some z, Some l when l = lead ->
          z
      | Some _, Some l ->
          invalid_arg
            (Printf.sprintf
               "%s: cannot continue a stream of leading shape [%s] with a \
                chunk of leading shape [%s]"
               fn
               (String.concat "; " (Array.to_list (Array.map string_of_int l)))
               (String.concat "; "
                  (Array.to_list (Array.map string_of_int lead)) ) )
      | _ ->
          let z = fresh p lead in
          s.z <- Some z ;
          s.lead <- Some lead ;
          z
    in
    normalize p z chunk
  in
  (* The element dtype is only witnessed by incoming chunks; [step] records it
     so [concat []] (reached only through [run], where a step always precedes)
     can build the empty output. *)
  let witness = ref None in
  Pipeline.kernel
    ~reset:(fun s ->
      s.z <- None ;
      s.lead <- None )
    ~concat:(function
      | [] -> (
        match !witness with
        | Some dtype ->
            Nx.zeros dtype [|0|]
        | None ->
            invalid_arg
              "pcen_stage: cannot concatenate zero chunks before any chunk \
               fixed the element dtype" )
      | parts ->
          Nx.concatenate ~axis:(-1) parts )
    ~prepare:(fun _format -> {z= None; lead= None})
    ~step:(fun s chunk ->
      witness := Some (Nx.dtype chunk) ;
      Some (step s chunk) )
    ()
