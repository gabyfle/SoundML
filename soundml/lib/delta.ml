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

(* Delta features: the Savitzky-Golay derivative along the frame axis, as
   librosa.feature.delta computes it — a least-squares polynomial of degree
   [order] fitted to each window of [width] frames, differentiated [order] times
   at the window's centre.

   With the polynomial's degree equal to the derivative's order, the derivative
   is constant over the window, so each output is one fixed weighted sum of a
   window's frames ([weights]). librosa's default [`Interp] border evaluates the
   fits of the first and last full windows at the frames they cover, which is
   those windows' own values; the [`Nearest] border extends the signal by copies
   of its edge frames. Every output frame is that sum, accumulated in double
   precision in window order and rounded once, and the flat function and the
   stage both read it through {!emit}: the stream reproduces the offline bits by
   construction. *)

type mode = [`Interp | `Nearest]

let check fn ~width ~order =
  if width < 3 || width mod 2 = 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot differentiate over a window of %d frames (width must be \
          an odd integer of at least 3)"
         fn width ) ;
  if order < 1 || order >= width then
    invalid_arg
      (Printf.sprintf
         "%s: cannot take a derivative of order %d over %d frames (order must \
          lie in [1, width - 1])"
         fn order width )

let check_input (type a) fn (x : (float, a) Nx.t) =
  if Nx.ndim x < 1 then
    invalid_arg
      ( fn
      ^ ": cannot differentiate a rank-zero tensor (the frame axis must exist)"
      ) ;
  match Nx.dtype x with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      invalid_arg
        (fn ^ ": cannot differentiate this dtype (float32 and float64 only)")

let check_frames fn ~width ~mode n =
  match mode with
  | `Interp when n < width ->
      invalid_arg
        (Printf.sprintf
           "%s: cannot fit a window of %d frames to %d frames (mode `Interp \
            needs at least width frames)"
           fn width n )
  | `Interp | `Nearest ->
      ()

(* {1 The derivative weights} *)

(* [weights ~width ~order] is the weight of each frame of a window in its
   derivative, oldest first: [order!] times row [order] of the least-squares
   solution of the degree-[order] polynomial over the offsets [-h .. h], with [h
   = width / 2]. The system is solved at offsets scaled by [h], which keeps its
   normal matrix well conditioned, and the weights are rescaled by [h ^
   -order]. *)
let weights ~width ~order =
  let h = width / 2 and n = order + 1 in
  let x k = Float.of_int k /. Float.of_int h in
  let power v e = if e = 0 then 1. else Float.pow v (Float.of_int e) in
  let g =
    Array.init n (fun i ->
        Array.init n (fun j ->
            let s = ref 0. in
            for k = -h to h do
              s := !s +. power (x k) (i + j)
            done ;
            !s ) )
  in
  let z = Array.init n (fun i -> if i = order then 1. else 0.) in
  (* Gaussian elimination with partial pivoting, then back substitution. *)
  for c = 0 to n - 1 do
    let p = ref c in
    for r = c + 1 to n - 1 do
      if Float.abs g.(r).(c) > Float.abs g.(!p).(c) then p := r
    done ;
    let swap a =
      let t = a.(c) in
      a.(c) <- a.(!p) ;
      a.(!p) <- t
    in
    swap g ;
    swap z ;
    for r = c + 1 to n - 1 do
      let f = g.(r).(c) /. g.(c).(c) in
      for j = c to n - 1 do
        g.(r).(j) <- g.(r).(j) -. (f *. g.(c).(j))
      done ;
      z.(r) <- z.(r) -. (f *. z.(c))
    done
  done ;
  for c = n - 1 downto 0 do
    let s = ref z.(c) in
    for j = c + 1 to n - 1 do
      s := !s -. (g.(c).(j) *. z.(j))
    done ;
    z.(c) <- !s /. g.(c).(c)
  done ;
  let factorial = ref 1. in
  for i = 2 to order do
    factorial := !factorial *. Float.of_int i
  done ;
  let scale = !factorial /. power (Float.of_int h) order in
  Float.Array.init width (fun i ->
      let s = ref 0. in
      for j = 0 to order do
        s := !s +. (power (x (i - h)) j *. z.(j))
      done ;
      scale *. !s )

(* {1 The kernel} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 1 then Nx.R (start, stop) else Nx.A))
    t

let concat_last parts = Nx.concatenate ~axis:(-1) parts

(* [centre ~h ~mode ~total t] is the frame the window of output frame [t] of a
   signal of [total] frames is centred on: [t] itself, or under [`Interp] the
   nearest full window's centre. The kernels read frames outside [0, total) as
   the nearest edge frame, which only [`Nearest] reaches. *)
let centre ~h ~mode ~total t =
  match mode with
  | `Interp ->
      Int.max h (Int.min (total - 1 - h) t)
  | `Nearest ->
      t

let fill32 (src : Store.f32) (dst : Store.f32) ~w ~h ~mode ~n ~base ~total ~lo
    ~hi r0 r1 =
  let count = hi - lo in
  for r = r0 to r1 - 1 do
    let row = (r * n) - base and out = r * count in
    for t = lo to hi - 1 do
      let c = centre ~h ~mode ~total t in
      let acc = ref 0. in
      for k = 0 to 2 * h do
        let f = Int.max 0 (Int.min (total - 1) (c - h + k)) in
        let wk = Float.Array.unsafe_get w k in
        let v = Bigarray.Array1.unsafe_get src (row + f) in
        acc := !acc +. (wk *. v)
      done ;
      Bigarray.Array1.unsafe_set dst (out + t - lo) !acc
    done
  done

let fill64 (src : Store.f64) (dst : Store.f64) ~w ~h ~mode ~n ~base ~total ~lo
    ~hi r0 r1 =
  let count = hi - lo in
  for r = r0 to r1 - 1 do
    let row = (r * n) - base and out = r * count in
    for t = lo to hi - 1 do
      let c = centre ~h ~mode ~total t in
      let acc = ref 0. in
      for k = 0 to 2 * h do
        let f = Int.max 0 (Int.min (total - 1) (c - h + k)) in
        let wk = Float.Array.unsafe_get w k in
        let v = Bigarray.Array1.unsafe_get src (row + f) in
        acc := !acc +. (wk *. v)
      done ;
      Bigarray.Array1.unsafe_set dst (out + t - lo) !acc
    done
  done

(* [emit ~w ~mode ext ~base ~total ~lo ~hi] is output frames [lo, hi) of a
   signal of [total] frames, of which [ext] holds frames [base, base + last_dim
   ext) — every frame those outputs read. Rows run across domains; each output
   is computed alone, so the split changes no bit. *)
let emit (type a) ~w ~mode (ext : (float, a) Nx.t) ~base ~total ~lo ~hi :
    (float, a) Nx.t =
  let h = Float.Array.length w / 2 in
  let lead = leading_shape ext in
  let rows = Array.fold_left ( * ) 1 lead and n = last_dim ext in
  let count = hi - lo in
  let shape = Array.append lead [|count|] in
  let grain = Int.max 1 (32_768 / Int.max 1 (count * Float.Array.length w)) in
  let run fill = Par.ranges ~n:rows ~grain fill in
  match Nx.dtype ext with
  | Nx.Float32 ->
      Store.make32 shape (fun dst ->
          Store.read32 ext (fun src ->
              run (fill32 src dst ~w ~h ~mode ~n ~base ~total ~lo ~hi) ) )
  | Nx.Float64 ->
      Store.make64 shape (fun dst ->
          Store.read64 ext (fun src ->
              run (fill64 src dst ~w ~h ~mode ~n ~base ~total ~lo ~hi) ) )
  | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      assert false

(* {1 The flat function} *)

let delta ?(width = 9) ?(order = 1) ?(mode = `Interp) x =
  let fn = "delta" in
  check fn ~width ~order ;
  check_input fn x ;
  let n = last_dim x in
  check_frames fn ~width ~mode n ;
  emit ~w:(weights ~width ~order) ~mode x ~base:0 ~total:n ~lo:0 ~hi:n

(* {1 Pipeline stage}

   State of one stream: [hist] holds the last [width] frames seen, a
   kernel-owned copy (incoming chunks are borrowed); [seen] and [emitted] count
   frames in and out. An output frame is emitted once every frame its window
   reads has arrived: frame [t] once [t + h] has, and under [`Interp] the first
   [h + 1] outputs once the first full window has, so the stage withholds [h]
   frames, [2 h] before its first window. The flush emits the last [h] outputs,
   whose windows the end of the stream bounds. *)

type 'a state =
  { mutable hist: (float, 'a) Nx.t option
  ; mutable seen: int
  ; mutable emitted: int
  ; mutable lead: int array option }

let state_create () = {hist= None; seen= 0; emitted= 0; lead= None}

let state_reset s =
  s.hist <- None ;
  s.seen <- 0 ;
  s.emitted <- 0 ;
  s.lead <- None

let stage ?(width = 9) ?(order = 1) ?(mode = `Interp) () =
  let fn = "delta_stage" in
  check fn ~width ~order ;
  let w = weights ~width ~order and h = width / 2 in
  let empty lead dtype = Nx.zeros dtype (Array.append lead [|0|]) in
  (* [run ... ~upto] emits every output below [upto] not yet emitted, reading
     [ext] (frames [base ...]) as the stream's frames so far. *)
  let run s ext ~base ~upto =
    let lo = s.emitted in
    s.emitted <- Int.max lo upto ;
    if upto > lo then Some (emit ~w ~mode ext ~base ~total:s.seen ~lo ~hi:upto)
    else None
  in
  let step s chunk =
    check_input fn chunk ;
    let lead = leading_shape chunk in
    ( match s.lead with
    | None ->
        s.lead <- Some lead
    | Some l when l = lead ->
        ()
    | Some l ->
        invalid_arg
          (Printf.sprintf
             "%s: cannot continue a stream of leading shape [%s] with a chunk \
              of leading shape [%s]"
             fn
             (String.concat "; " (Array.to_list (Array.map string_of_int l)))
             (String.concat "; " (Array.to_list (Array.map string_of_int lead))) )
    ) ;
    let m = last_dim chunk in
    let ext, base =
      match s.hist with
      | None ->
          (chunk, s.seen)
      | Some hist ->
          (concat_last [hist; chunk], s.seen - last_dim hist)
    in
    s.seen <- s.seen + m ;
    let ready =
      match mode with
      | `Interp ->
          if s.seen < width then 0 else s.seen - h
      | `Nearest ->
          Int.max 0 (s.seen - h)
    in
    let out = run s ext ~base ~upto:ready in
    let l = last_dim ext in
    if l > 0 then
      s.hist <- Some (Nx.copy (shrink_last ext (Int.max 0 (l - width)) l)) ;
    match out with Some y -> y | None -> empty lead (Nx.dtype chunk)
  in
  let flush s =
    check_frames fn ~width ~mode s.seen ;
    match s.hist with
    | None ->
        []
    | Some hist ->
        Option.to_list (run s hist ~base:(s.seen - last_dim hist) ~upto:s.seen)
  in
  (* The element dtype is only witnessed by incoming chunks; [step] records it
     so [concat []] (reached only through [run], where a step always precedes)
     can build the empty output. *)
  let witness = ref None in
  Pipeline.kernel
    ~latency:(match mode with `Interp -> 2 * h | `Nearest -> h)
    ~reset:state_reset ~flush
    ~concat:(function
      | [] -> (
        match !witness with
        | Some dtype ->
            Nx.zeros dtype [|0|]
        | None ->
            invalid_arg
              "delta_stage: cannot concatenate zero chunks before any chunk \
               fixed the element dtype" )
      | parts ->
          concat_last parts )
    ~prepare:(fun _format -> state_create ())
    ~step:(fun s chunk ->
      witness := Some (Nx.dtype chunk) ;
      Some (step s chunk) )
    ()
