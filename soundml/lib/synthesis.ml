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

(* Synthetic signals and mu-law companding, as librosa.tone, librosa.chirp (over
   scipy.signal.chirp), librosa.clicks, librosa.mu_compress and
   librosa.mu_expand compute them.

   Every value is computed in double precision, in the reference's order of
   operations. A tone or chirp sample is a function of its own integer index,
   never of an accumulated phase, and a companded value one of its own input
   element, so those loops split across domains without changing a bit. A click
   track adds its clicks in the order the positions are given, rounding to the
   track's dtype after every addition as numpy's in-place float32 sum does; its
   samples split across domains, each range adding, in that order, every click
   that covers it, so every sample sees the same additions in the same order.

   The spelling of the arithmetic is part of the parity: on arm64 the compiler
   fuses [a +. b *. c] into one multiply-add when the product's operands are
   let-bound values, which rounds once where the reference rounds twice. Every
   product that feeds an addition below is therefore let-bound itself, which
   keeps it a separate, rounded operation on every platform. *)

open Bigarray

type positions =
  [`Frames of int array | `Samples of int array | `Times of float array]

(* {1 Storage} *)

(* A result being written, at float32 or float64; float32 storage rounds each
   double once on store, the rounding [Nx.cast] performs. *)
type dst = To32 of Store.f32 | To64 of Store.f64

let[@inline] put dst i v =
  match dst with
  | To32 a ->
      Array1.unsafe_set a i v
  | To64 a ->
      Array1.unsafe_set a i v

(* An operand being read, widened exactly to double. *)
type src = From32 of Store.f32 | From64 of Store.f64

let[@inline] get src i =
  match src with
  | From32 a ->
      Array1.unsafe_get a i
  | From64 a ->
      Array1.unsafe_get a i

(* Elements per parallel range of a loop calling one or two libm functions per
   element: some tens of microseconds of work. *)
let grain = 4096

(* [generate dtype n fill] is the rank-one tensor of [n] elements of [dtype]
   whose elements [fill dst lo hi] writes over [\[lo, hi)], the ranges split
   across domains. The narrow dtypes are generated at float64 and cast once. *)
let generate : type a.
    (float, a) Nx.dtype -> int -> (dst -> int -> int -> unit) -> (float, a) Nx.t
    =
 fun dtype n fill ->
  let run dst = Par.ranges ~n ~grain (fill dst) in
  match dtype with
  | Nx.Float32 ->
      Store.make32 [|n|] (fun a -> run (To32 a))
  | Nx.Float64 ->
      Store.make64 [|n|] (fun a -> run (To64 a))
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast dtype (Store.make64 [|n|] (fun a -> run (To64 a)))

(* {1 Validation} *)

(* Sample counts derived from seconds must convert exactly to [int]. *)
let max_samples = 0x1p62

let check_sample_rate fn sample_rate =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot synthesize at %d Hz (sample_rate must be at least 1)" fn
         sample_rate )

let check_finite fn name v =
  if not (Float.is_finite v) then
    invalid_arg
      (Printf.sprintf "%s: cannot use %s = %g (%s must be finite)" fn name v
         name )

let check_positive fn name v =
  if not (Float.is_finite v && v > 0.) then
    invalid_arg
      (Printf.sprintf "%s: cannot use %s = %g (%s must be finite and positive)"
         fn name v name )

(* [samples fn ~sample_rate ?length ?duration ()] is the length of the signal in
   samples: [length] when given, otherwise [duration] seconds truncated toward
   zero, librosa's [int(duration * sr)]. Both are validated when given. *)
let samples fn ~sample_rate ?length ?duration () =
  Option.iter
    (fun n ->
      if n < 0 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot synthesize %d samples (length must be non-negative)" fn
             n ) )
    length ;
  let of_duration =
    Option.map
      (fun d ->
        if not (Float.is_finite d && d >= 0.) then
          invalid_arg
            (Printf.sprintf
               "%s: cannot use duration = %g (duration must be finite and \
                non-negative)"
               fn d ) ;
        let s = d *. Float.of_int sample_rate in
        if not (s < max_samples) then
          invalid_arg
            (Printf.sprintf
               "%s: cannot synthesize %g seconds at %d Hz (the signal would \
                hold 2^62 samples or more)"
               fn d sample_rate ) ;
        Float.to_int s )
      duration
  in
  match (length, of_duration) with
  | Some n, _ ->
      n
  | None, Some n ->
      n
  | None, None ->
      invalid_arg
        (fn ^ ": cannot size the signal (a length or a duration must be given)")

(* [two_pi] is numpy's [2 * np.pi]: doubling is exact. *)
let two_pi = 2. *. Float.pi

(* {1 Tones} *)

let tone dtype ?length ?duration ?(phase = -.Float.pi *. 0.5) ~sample_rate
    frequency =
  let fn = "tone" in
  check_sample_rate fn sample_rate ;
  check_finite fn "frequency" frequency ;
  check_finite fn "phase" phase ;
  let n = samples fn ~sample_rate ?length ?duration () in
  let s = Float.of_int sample_rate in
  let c = two_pi *. frequency in
  (* librosa: cos(2 * pi * frequency * arange(length) / sr + phi), evaluated
     left to right; the addition takes a quotient, which nothing fuses. *)
  generate dtype n (fun dst lo hi ->
      for k = lo to hi - 1 do
        put dst k (Float.cos ((c *. Float.of_int k /. s) +. phase))
      done )

(* {1 Chirps} *)

let chirp dtype ?length ?duration ?(linear = false) ?(phase = -.Float.pi *. 0.5)
    ~sample_rate ~fmin ~fmax () =
  let fn = "chirp" in
  check_sample_rate fn sample_rate ;
  check_finite fn "fmin" fmin ;
  check_finite fn "fmax" fmax ;
  check_finite fn "phase" phase ;
  (* scipy's test, product and all: it underflows for tiny frequencies *)
  if (not linear) && not (fmin *. fmax > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot sweep exponentially from %g Hz to %g Hz (fmin and fmax \
          must be nonzero and of the same sign)"
         fn fmin fmax ) ;
  let n = samples fn ~sample_rate ?length ?duration () in
  let s = Float.of_int sample_rate in
  (* librosa's sweep time: [(1 / sr) * length] when a length is given, the
     duration otherwise ([samples] has refused neither). A signal of no sample
     never reads it, so a zero sweep time divides nothing below. *)
  let t1 =
    match length with
    | Some l ->
        1. /. s *. Float.of_int l
    | None ->
        Option.value duration ~default:0.
  in
  (* scipy.signal.chirp takes the phase in degrees: librosa converts the radians
     there, [phi / pi * 180], and scipy back, [deg2rad], which is a
     multiplication by [pi / 180]; the round trip moves some phases by an
     ulp. *)
  let off =
    let deg = phase /. Float.pi *. 180. in
    deg *. (Float.pi /. 180.)
  in
  let fill =
    if linear then
      (* scipy: 2 * pi * (f0 * t + 0.5 * beta * t * t), beta = (f1 - f0) / t1 *)
      let hb = 0.5 *. ((fmax -. fmin) /. t1) in
      fun dst lo hi ->
        for k = lo to hi - 1 do
          let t = Float.of_int k /. s in
          let a = fmin *. t in
          let b = hb *. t *. t in
          let sum = a +. b in
          let ph = two_pi *. sum in
          put dst k (Float.cos (ph +. off))
        done
    else if Float.equal fmin fmax then
      (* scipy's constant-frequency branch: 2 * pi * f0 * t *)
      let c = two_pi *. fmin in
      fun dst lo hi ->
        for k = lo to hi - 1 do
          let t = Float.of_int k /. s in
          let ph = c *. t in
          put dst k (Float.cos (ph +. off))
        done
    else
      (* scipy: 2 * pi * beta * f0 * (pow(f1 / f0, t / t1) - 1.0), beta = t1 /
         log(f1 / f0) *)
      let r = fmax /. fmin in
      let c = two_pi *. (t1 /. Float.log r) *. fmin in
      fun dst lo hi ->
        for k = lo to hi - 1 do
          let t = Float.of_int k /. s in
          let g = Float.pow r (t /. t1) -. 1. in
          let ph = c *. g in
          put dst k (Float.cos (ph +. off))
        done
  in
  generate dtype n fill

(* {1 Click tracks} *)

(* [default_click ~sample_rate ~frequency ~duration] is librosa's default click,
   a sine at [frequency] under an envelope decaying exponentially from 1 to
   2^-10 over [int(sr*duration)] samples. The envelope is numpy's
   [logspace(0,-10,base=2)]: two to the power of each point of a [linspace],
   point [k] being [k*step+0] and the last exactly [-10] (a one-point linspace
   is [0]). *)
let default_click ~sample_rate ~frequency ~duration =
  let sr = Float.of_int sample_rate in
  let m = Float.to_int (sr *. duration) in
  let w = two_pi *. frequency /. sr in
  let step = -10. /. Float.of_int (m - 1) in
  Array.init m (fun k ->
      let y =
        if m = 1 then 0.
        else if k = m - 1 then -10.
        else
          let p = Float.of_int k *. step in
          p +. 0.
      in
      Float.pow 2. y *. Float.sin (w *. Float.of_int k) )

(* [starts fn ~sample_rate ~hop positions] is the sample index of every
   position, in the order given: librosa's [frames * hop] and [int(times * sr)],
   truncated toward zero, so a time within a sample before zero lands on sample
   zero. A position past [max_int] saturates there, past any length. *)
let starts fn ~sample_rate ~hop (positions : positions) =
  let check_sample p =
    if p < 0 then
      invalid_arg
        (Printf.sprintf
           "%s: cannot place a click at sample %d (positions must not fall \
            before the first sample)"
           fn p )
  in
  match positions with
  | `Samples s ->
      Array.iter check_sample s ; Array.copy s
  | `Frames f ->
      Array.map
        (fun p ->
          if p < 0 then
            invalid_arg
              (Printf.sprintf
                 "%s: cannot place a click at frame %d (frames must be \
                  non-negative)"
                 fn p ) ;
          if p > Int.max_int / hop then Int.max_int else p * hop )
        f
  | `Times t ->
      let sr = Float.of_int sample_rate in
      Array.map
        (fun v ->
          let x = v *. sr in
          if not (Float.is_finite v && x > -1.) then
            invalid_arg
              (Printf.sprintf
                 "%s: cannot place a click at %g s (times must be finite and \
                  not fall before the first sample)"
                 fn v ) ;
          if x < max_samples then Float.to_int x else Int.max_int )
        t

(* [place32 a ~rows ~total ~m ~wave kept lo hi] writes samples [\[lo, hi)] of
   every row of the float32 track [a]: zero, then each click of [kept] that
   covers them added in order, rounded to float32 after every addition. *)
let place32 (a : Store.f32) ~rows ~total ~m ~wave kept lo hi =
  for r = 0 to rows - 1 do
    let base = r * total and src = r * m in
    for i = base + lo to base + hi - 1 do
      Array1.unsafe_set a i 0.
    done ;
    for q = 0 to Array.length kept - 1 do
      let p = Array.unsafe_get kept q in
      let stop = if m < total - p then p + m else total in
      for i = Int.max p lo to Int.min stop hi - 1 do
        let j = base + i in
        Array1.unsafe_set a j
          (Array1.unsafe_get a j +. Array.unsafe_get wave (src + i - p))
      done
    done
  done

(* [place64] is {!place32} on a float64 track. *)
let place64 (a : Store.f64) ~rows ~total ~m ~wave kept lo hi =
  for r = 0 to rows - 1 do
    let base = r * total and src = r * m in
    for i = base + lo to base + hi - 1 do
      Array1.unsafe_set a i 0.
    done ;
    for q = 0 to Array.length kept - 1 do
      let p = Array.unsafe_get kept q in
      let stop = if m < total - p then p + m else total in
      for i = Int.max p lo to Int.min stop hi - 1 do
        let j = base + i in
        Array1.unsafe_set a j
          (Array1.unsafe_get a j +. Array.unsafe_get wave (src + i - p))
      done
    done
  done

(* Track samples per parallel range: zeroing and a few additions each. *)
let track_grain = 32_768

let clicks (type a) (dtype : (float, a) Nx.dtype) ?(hop = 512)
    ?(click : (float, a) Nx.t option) ?(click_frequency = 1000.)
    ?(click_duration = 0.1) ?length ~sample_rate (positions : positions) :
    (float, a) Nx.t =
  let fn = "clicks" in
  check_sample_rate fn sample_rate ;
  if hop < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot place frames at a hop of %d (hop must be at least 1)" fn
         hop ) ;
  check_positive fn "click_frequency" click_frequency ;
  check_positive fn "click_duration" click_duration ;
  if not (click_duration *. Float.of_int sample_rate < max_samples) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot synthesize a click of %g seconds at %d Hz (the click \
          would hold 2^62 samples or more)"
         fn click_duration sample_rate ) ;
  Option.iter
    (fun l ->
      if l < 1 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot synthesize %d samples (length must be at least 1)" fn l ) )
    length ;
  let starts = starts fn ~sample_rate ~hop positions in
  (* The click: [lead] rows of [m] samples in C order, in double. *)
  let lead, m, wave =
    match click with
    | None ->
        let w =
          default_click ~sample_rate ~frequency:click_frequency
            ~duration:click_duration
        in
        ([||], Array.length w, w)
    | Some c ->
        let nd = Nx.ndim c in
        if nd < 1 then
          invalid_arg
            (fn ^ ": cannot place a rank-zero click (the time axis must exist)") ;
        let w = Nx.to_array c in
        if not (Array.for_all Float.is_finite w) then
          invalid_arg
            ( fn
            ^ ": cannot place a click holding a non-finite sample (every \
               sample must be finite)" ) ;
        (Array.sub (Nx.shape c) 0 (nd - 1), Nx.dim (nd - 1) c, w)
  in
  let total, kept =
    match length with
    | Some l ->
        (* librosa drops the positions at or past the length *)
        (l, Array.of_list (List.filter (fun p -> p < l) (Array.to_list starts)))
    | None ->
        if Array.length starts = 0 then
          invalid_arg
            ( fn
            ^ ": cannot size a click track without clicks (a length or at \
               least one position must be given)" ) ;
        let last = Array.fold_left Int.max 0 starts in
        if last > Int.max_int - m then
          invalid_arg
            (Printf.sprintf
               "%s: cannot synthesize a click track ending past sample %d" fn
               last ) ;
        (last + m, starts)
  in
  let rows = Array.fold_left ( * ) 1 lead in
  let shape = Array.append lead [|total|] in
  let run place = Par.ranges ~n:total ~grain:track_grain place in
  match dtype with
  | Nx.Float32 ->
      Store.make32 shape (fun a -> run (place32 a ~rows ~total ~m ~wave kept))
  | Nx.Float64 ->
      Store.make64 shape (fun a -> run (place64 a ~rows ~total ~m ~wave kept))
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast dtype
        (Store.make64 shape (fun a ->
             run (place64 a ~rows ~total ~m ~wave kept) ) )

(* {1 Mu-law companding} *)

(* Quantizing needs [int(1 + mu)] levels of at least two, which [int] and
   [float] both hold exactly. *)
let max_quantized_mu = 0x1p53

let check_mu fn ~quantize mu =
  check_positive fn "mu" mu ;
  if quantize && not (mu >= 1. && mu < max_quantized_mu) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot quantize with mu = %g (quantize needs mu in [1, 2^53))" fn
         mu )

(* numpy's sign: [-0.] and [0.] both map to [0.]. *)
let[@inline] sign v = if v > 0. then 1. else if v < 0. then -1. else 0.

(* [compressed ~mu ~d x] is librosa's [sign(x) * log1p(mu * abs(x)) /
   log1p(mu)], with [d = log1p mu]. *)
let[@inline] compressed ~mu ~d x =
  let l = Float.log1p (mu *. Float.abs x) in
  sign x *. l /. d

(* [edge ~n ~step j] is point [j] of numpy's [linspace(-1, 1, n)]: [j * step +
   -1], the last exactly [1]. *)
let[@inline] edge ~n ~step j =
  if j = n - 1 then 1.
  else
    let p = Float.of_int j *. step in
    p -. 1.

(* [level ~n ~step c] is librosa's [digitize(c, linspace(-1, 1, n),
   right=True)]: the number of points strictly below [c]. The points do not
   decrease, so the count is estimated from the grid and corrected against the
   points themselves, with no table. *)
let level ~n ~step c =
  let g = Float.to_int (Float.floor ((c +. 1.) /. step)) in
  let g = ref (Int.max 0 (Int.min n g)) in
  while !g > 0 && edge ~n ~step (!g - 1) >= c do
    decr g
  done ;
  while !g < n && edge ~n ~step !g < c do
    incr g
  done ;
  !g

let compress_fill ~mu ~quantize src dst lo hi =
  let d = Float.log1p mu in
  if quantize then
    let n = Float.to_int (1. +. mu) in
    let half = Float.to_int (mu +. 1.) / 2
    and step = 2. /. Float.of_int (n - 1) in
    for i = lo to hi - 1 do
      let c = compressed ~mu ~d (get src i) in
      put dst i (Float.of_int (level ~n ~step c - half))
    done
  else
    for i = lo to hi - 1 do
      put dst i (compressed ~mu ~d (get src i))
    done

(* librosa: sign(x) / mu * (power(1 + mu, abs(x)) - 1), with a code first scaled
   to [x * 2.0 / (1 + mu)]. *)
let expand_fill ~mu ~quantize src dst lo hi =
  let base = 1. +. mu in
  for i = lo to hi - 1 do
    let x = get src i in
    let v = if quantize then x *. 2. /. base else x in
    let s = sign v /. mu in
    let p = Float.pow base (Float.abs v) -. 1. in
    put dst i (s *. p)
  done

(* [outside src n ~scale ~mu] is the first of the [n] elements of [src] that,
   multiplied by [2 / (1 + mu)] when [scale], does not lie in [\[-1, 1\]] — nan
   included — or [-1]. *)
let outside src n ~scale ~mu =
  let base = 1. +. mu in
  let rec go i =
    if i = n then -1
    else
      let x = get src i in
      let v = if scale then x *. 2. /. base else x in
      if v >= -1. && v <= 1. then go (i + 1) else i
  in
  go 0

(* [map x ~check fill] is [fill] over the elements of [x], in its shape and
   dtype, once [check] accepted them. The narrow dtypes go through float64: the
   widening is exact, and the result is rounded once. *)
let rec map : type a.
       (float, a) Nx.t
    -> check:(src -> int -> unit)
    -> (src -> dst -> int -> int -> unit)
    -> (float, a) Nx.t =
 fun x ~check fill ->
  let shape = Nx.shape x and n = Nx.numel x in
  let run src dst = Par.ranges ~n ~grain (fill src dst) in
  match Nx.dtype x with
  | Nx.Float32 ->
      Store.read32 x (fun a ->
          check (From32 a) n ;
          Store.make32 shape (fun d -> run (From32 a) (To32 d)) )
  | Nx.Float64 ->
      Store.read64 x (fun a ->
          check (From64 a) n ;
          Store.make64 shape (fun d -> run (From64 a) (To64 d)) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Nx.cast (Nx.dtype x) (map (Nx.cast Nx.float64 x) ~check fill)

let compress fn ~mu ~quantize x =
  map x
    ~check:(fun src n ->
      let i = outside src n ~scale:false ~mu in
      if i >= 0 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot compress %g (every sample must lie in [-1, 1])" fn
             (get src i) ) )
    (compress_fill ~mu ~quantize)

let expand fn ~mu ~quantize x =
  map x
    ~check:(fun src n ->
      let i = outside src n ~scale:quantize ~mu in
      if i >= 0 then
        invalid_arg
          ( if quantize then
              Printf.sprintf
                "%s: cannot expand the code %g (codes must lie in [-(1 + mu) / \
                 2, (1 + mu) / 2])"
                fn (get src i)
            else
              Printf.sprintf
                "%s: cannot expand %g (every value must lie in [-1, 1])" fn
                (get src i) ) )
    (expand_fill ~mu ~quantize)

let mu_compress ?(mu = 255.) ?(quantize = true) x =
  let fn = "mu_compress" in
  check_mu fn ~quantize mu ;
  compress fn ~mu ~quantize x

let mu_expand ?(mu = 255.) ?(quantize = true) x =
  let fn = "mu_expand" in
  check_mu fn ~quantize mu ; expand fn ~mu ~quantize x

let mu_compress_stage ?(mu = 255.) ?(quantize = true) () =
  let fn = "mu_compress_stage" in
  check_mu fn ~quantize mu ;
  Pipeline.stateless (fun x -> compress fn ~mu ~quantize x)

let mu_expand_stage ?(mu = 255.) ?(quantize = true) () =
  let fn = "mu_expand_stage" in
  check_mu fn ~quantize mu ;
  Pipeline.stateless (fun x -> expand fn ~mu ~quantize x)
