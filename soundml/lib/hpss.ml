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

(* Harmonic/percussive separation by median filtering (Fitzgerald, DAFx 2010)
   with the soft-mask generalisation of Driedger, Müller and Disch (ISMIR 2014).

   A harmonic partial is a horizontal ridge of the magnitude spectrogram [S] and
   a percussive transient a vertical one, so

   harm[b, t] = median over the k_h frames centred on t of S[b, .] perc[b, t] =
   median over the k_p bins centred on b of S[., t]

   enhance one structure and suppress the other. The two enhanced spectrograms
   drive a pair of masks — Wiener-like for finite [p], hard for [p = infinity] —
   whose margins bias the decision:

   mask_h = f (harm, m_h * perc) mask_p = f (perc, m_p * harm)

   with f (x, r) = x^p / (x^p + r^p) at finite [p] and [x > r] at [p =
   infinity], both computed on inputs rescaled by their pointwise maximum. At
   [m_h = m_p = 1] the two finite-[p] masks sum to one and the components
   partition [S]; beyond that the masks only exclude each other and the
   difference [S - (H + P)] is the residual.

   {1 The median kernels}

   Both filters are one-dimensional, so each is a sliding-window median over a
   line: the window of index [i] is [[i - k/2, i + k - 1 - k/2]] (left-biased
   for even [k]) and the value is rank [k/2] of the ascending window — the upper
   middle, never an average of two. Out-of-range indices reflect
   half-sample-symmetrically with period [2n]: index [i] maps to [refl i n], for
   every [k], however far the window overhangs.

   The window is maintained sorted. Advancing by one drops one value and admits
   one, so a binary search locates the departing value and a single shift in one
   direction reinserts the arriving one: O(k) worst case per step, a few
   contiguous moves in practice, and no allocation inside the line.

   The time filter runs along the contiguous axis and is the one with memory:
   each lane keeps a ring of its last [k] frames beside its sorted window, and
   completes each frame as soon as its window is known — streamed or offline,
   the same sweep (see the time median as a stream, below). The frequency filter
   runs along the strided axis, and rather than transpose it keeps the windows
   of a range of frames live at once ([k] floats per frame) and streams that
   range of the source rows in storage order, so every read is contiguous. Each
   domain sweeps lanes of its own: rows for the time filter, frame ranges for
   the frequency filter. *)

(* {1 Line geometry} *)

(* [refl i n] is the half-sample-symmetric reflection of [i] into [[0, n)]:
   period [2n], with [-1] mapping to [0] and [n] to [n - 1]. Total for every
   integer, so windows may overhang a line by any amount. *)
let refl i n =
  let p = 2 * n in
  let j = ((i mod p) + p) mod p in
  if j < n then j else p - 1 - j

(* [sorted_insert win base k] sorts the entries [win[base, base + k)] ascending
   by insertion — the one full sort per line, at the initial window. *)
let sorted_insert (win : float array) base k =
  for i = base + 1 to base + k - 1 do
    let v = Array.unsafe_get win i in
    let j = ref (i - 1) in
    while !j >= base && Array.unsafe_get win !j > v do
      Array.unsafe_set win (!j + 1) (Array.unsafe_get win !j) ;
      decr j
    done ;
    Array.unsafe_set win (!j + 1) v
  done

(* [replace win base k old nw] turns the ascending run [win[base, base + k)]
   holding [old] into the ascending run holding [nw] instead: binary-search the
   leftmost slot of [old], then shift the entries between it and the slot of
   [nw] one place towards it. *)
let[@inline] replace (win : float array) base k old nw =
  let lo = ref base and hi = ref (base + k - 1) in
  while !lo < !hi do
    let m = (!lo + !hi) / 2 in
    if Array.unsafe_get win m < old then lo := m + 1 else hi := m
  done ;
  let i = ref !lo in
  if nw >= old then begin
    let stop = base + k - 1 in
    while !i < stop && Array.unsafe_get win (!i + 1) < nw do
      Array.unsafe_set win !i (Array.unsafe_get win (!i + 1)) ;
      incr i
    done ;
    Array.unsafe_set win !i nw
  end
  else begin
    while !i > base && Array.unsafe_get win (!i - 1) > nw do
      Array.unsafe_set win !i (Array.unsafe_get win (!i - 1)) ;
      decr i
    done ;
    Array.unsafe_set win !i nw
  end

(* {1 The kernels}

   One pair per element width: the buffers are monomorphic so the element
   accesses compile to direct loads and stores. *)

(* [median_cols src dst ~planes ~rows ~cols ~k] writes the length-[k] running
   median down each column of [planes] consecutive [rows * cols] matrices. All
   [cols] windows advance together, one source row at a time. *)

let median_cols_f64
    (src : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t)
    (dst : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t)
    ~rows ~cols ~k plane j0 j1 =
  let pl = k / 2 in
  let mid = k / 2 in
  let wins = Array.make ((j1 - j0) * k) 0. in
  let scratch = Array.make k 0. in
  begin
    let origin = plane * rows * cols in
    for j = j0 to j1 - 1 do
      for t = 0 to k - 1 do
        Array.unsafe_set scratch t
          (Bigarray.Array1.unsafe_get src
             (origin + (refl (t - pl) rows * cols) + j) )
      done ;
      sorted_insert scratch 0 k ;
      Array.blit scratch 0 wins ((j - j0) * k) k ;
      Bigarray.Array1.unsafe_set dst (origin + j) (Array.unsafe_get scratch mid)
    done ;
    for r = 1 to rows - 1 do
      let out_row = origin + (refl (r - pl - 1) rows * cols) in
      let in_row = origin + (refl (r + k - 1 - pl) rows * cols) in
      let d = origin + (r * cols) in
      for j = j0 to j1 - 1 do
        let base = (j - j0) * k in
        replace wins base k
          (Bigarray.Array1.unsafe_get src (out_row + j))
          (Bigarray.Array1.unsafe_get src (in_row + j)) ;
        Bigarray.Array1.unsafe_set dst (d + j)
          (Array.unsafe_get wins (base + mid))
      done
    done
  end

let median_cols_f32
    (src : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t)
    (dst : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t)
    ~rows ~cols ~k plane j0 j1 =
  let pl = k / 2 in
  let mid = k / 2 in
  let wins = Array.make ((j1 - j0) * k) 0. in
  let scratch = Array.make k 0. in
  begin
    let origin = plane * rows * cols in
    for j = j0 to j1 - 1 do
      for t = 0 to k - 1 do
        Array.unsafe_set scratch t
          (Bigarray.Array1.unsafe_get src
             (origin + (refl (t - pl) rows * cols) + j) )
      done ;
      sorted_insert scratch 0 k ;
      Array.blit scratch 0 wins ((j - j0) * k) k ;
      Bigarray.Array1.unsafe_set dst (origin + j) (Array.unsafe_get scratch mid)
    done ;
    for r = 1 to rows - 1 do
      let out_row = origin + (refl (r - pl - 1) rows * cols) in
      let in_row = origin + (refl (r + k - 1 - pl) rows * cols) in
      let d = origin + (r * cols) in
      for j = j0 to j1 - 1 do
        let base = (j - j0) * k in
        replace wins base k
          (Bigarray.Array1.unsafe_get src (out_row + j))
          (Bigarray.Array1.unsafe_get src (in_row + j)) ;
        Bigarray.Array1.unsafe_set dst (d + j)
          (Array.unsafe_get wins (base + mid))
      done
    done
  end

(* {1 The time median as a stream}

   The time filter is the one with memory. The window of frame [i] reaches [r =
   k - 1 - k/2] frames ahead, so frame [i] is final once frame [i + r] has
   arrived — unless its window crosses the end of the line, where the reflection
   depends on the line's length and the frame waits for the end.

   Each lane keeps a ring of its last [k] frames, frame [f] in slot [f mod k],
   and its sorted window. Admitting frame [a] first reads the slot it overwrites
   — frame [a - k], the value leaving the window of frame [a - r] — and then
   completes frame [a - r]: its first window is gathered from the ring and
   sorted, every later one replaces the leaving value by [a]. Indices before the
   start reflect into frames already seen, [-1] reading [0], and the reflection
   cannot wrap there: a window completed before the end reflects at most [k/2]
   indices, onto frames below [k/2 <= r + 1 <= a + 1]. At the end of the line
   the last [min n r] frames are completed with every index reflected over the
   whole line; each reads frames among the last [k], all in the ring.

   That is value for value the sweep of the lane extended by reflection — the
   same first window, the same replacements in the same order — so the windows,
   and the medians, are the same bits whatever the line's division into chunks,
   ties between signed zeros and NaNs included. The offline faces are its one
   chunk instance: every frame admitted in one call, which then ends the line.

   One routine per element width, as for the frequency filter: the buffers are
   monomorphic so the element accesses compile to direct loads and stores. *)

(* [first_window ring win rb k n] gathers the window of frame [0] of the lane at
   [rb] — indices reflected into a line of [n] frames, every one of them in the
   ring — and sorts it. *)
let first_window (ring : float array) (win : float array) rb k n =
  let pl = k / 2 in
  for j = 0 to k - 1 do
    Array.unsafe_set win (rb + j)
      (Array.unsafe_get ring (rb + (refl (j - pl) n mod k)))
  done ;
  sorted_insert win rb k

(* [advance ~k ~ring ~win ~seen ~m ~lo ~hi src harm delay] admits the [m] frames
   of [src] into lanes [q0, q1) of a line that has seen [seen] frames and
   completed [lo], and writes the medians of frames [lo, hi) to [harm] — and the
   frames themselves to [delay], when given. [hi] is [max lo (seen + m - r)], or
   the line's length [seen + m] when the call ends the line. *)

let advance_f64 ~k ~ring ~win ~seen ~m ~lo ~hi (src : Store.f64)
    (harm : Store.f64) (delay : Store.f64 option) q0 q1 =
  let pl = k / 2 in
  let r = k - 1 - pl in
  let n = seen + m and count = hi - lo in
  for q = q0 to q1 - 1 do
    let rb = q * k and sb = q * m and db = (q * count) - lo in
    for t = 0 to m - 1 do
      let a = seen + t in
      let slot = rb + (a mod k) in
      let leaving = Array.unsafe_get ring slot in
      let v = Bigarray.Array1.unsafe_get src (sb + t) in
      Array.unsafe_set ring slot v ;
      let i = a - r in
      if i >= 0 then begin
        ( if i = 0 then first_window ring win rb k (a + 1)
          else
            let o = i - 1 - pl in
            (* before the start, index [o] reflects to frame [-o - 1] *)
            replace win rb k
              (if o >= 0 then leaving else Array.unsafe_get ring (rb - o - 1))
              v ) ;
        Bigarray.Array1.unsafe_set harm (db + i)
          (Array.unsafe_get win (rb + pl)) ;
        match delay with
        | Some d ->
            Bigarray.Array1.unsafe_set d (db + i)
              (Array.unsafe_get ring (rb + (i mod k)))
        | None ->
            ()
      end
    done ;
    for i = Int.max 0 (n - r) to hi - 1 do
      if i = 0 then first_window ring win rb k n
      else
        replace win rb k
          (Array.unsafe_get ring (rb + (refl (i - 1 - pl) n mod k)))
          (Array.unsafe_get ring (rb + (refl (i + r) n mod k))) ;
      Bigarray.Array1.unsafe_set harm (db + i) (Array.unsafe_get win (rb + pl)) ;
      match delay with
      | Some d ->
          Bigarray.Array1.unsafe_set d (db + i)
            (Array.unsafe_get ring (rb + (i mod k)))
      | None ->
          ()
    done
  done

let advance_f32 ~k ~ring ~win ~seen ~m ~lo ~hi (src : Store.f32)
    (harm : Store.f32) (delay : Store.f32 option) q0 q1 =
  let pl = k / 2 in
  let r = k - 1 - pl in
  let n = seen + m and count = hi - lo in
  for q = q0 to q1 - 1 do
    let rb = q * k and sb = q * m and db = (q * count) - lo in
    for t = 0 to m - 1 do
      let a = seen + t in
      let slot = rb + (a mod k) in
      let leaving = Array.unsafe_get ring slot in
      let v = Bigarray.Array1.unsafe_get src (sb + t) in
      Array.unsafe_set ring slot v ;
      let i = a - r in
      if i >= 0 then begin
        ( if i = 0 then first_window ring win rb k (a + 1)
          else
            let o = i - 1 - pl in
            (* before the start, index [o] reflects to frame [-o - 1] *)
            replace win rb k
              (if o >= 0 then leaving else Array.unsafe_get ring (rb - o - 1))
              v ) ;
        Bigarray.Array1.unsafe_set harm (db + i)
          (Array.unsafe_get win (rb + pl)) ;
        match delay with
        | Some d ->
            Bigarray.Array1.unsafe_set d (db + i)
              (Array.unsafe_get ring (rb + (i mod k)))
        | None ->
            ()
      end
    done ;
    for i = Int.max 0 (n - r) to hi - 1 do
      if i = 0 then first_window ring win rb k n
      else
        replace win rb k
          (Array.unsafe_get ring (rb + (refl (i - 1 - pl) n mod k)))
          (Array.unsafe_get ring (rb + (refl (i + r) n mod k))) ;
      Bigarray.Array1.unsafe_set harm (db + i) (Array.unsafe_get win (rb + pl)) ;
      match delay with
      | Some d ->
          Bigarray.Array1.unsafe_set d (db + i)
            (Array.unsafe_get ring (rb + (i mod k)))
      | None ->
          ()
    done
  done

(* [delay ~k ~ring ~seen ~m ~lo ~hi src dst] is the same frame bookkeeping for a
   payload that only travels with the line: it writes frames [lo, hi) of lanes
   [q0, q1) to [dst] — those already seen from [ring], the rest from the [m]
   frames of [src] — and then keeps the last [k] frames in [ring]. *)

let delay_f64 ~k ~ring ~seen ~m ~lo ~hi (src : Store.f64) (dst : Store.f64) q0
    q1 =
  let count = hi - lo in
  for q = q0 to q1 - 1 do
    let rb = q * k and sb = (q * m) - seen and db = (q * count) - lo in
    for i = lo to hi - 1 do
      Bigarray.Array1.unsafe_set dst (db + i)
        ( if i < seen then Array.unsafe_get ring (rb + (i mod k))
          else Bigarray.Array1.unsafe_get src (sb + i) )
    done ;
    for a = Int.max seen (seen + m - k) to seen + m - 1 do
      Array.unsafe_set ring
        (rb + (a mod k))
        (Bigarray.Array1.unsafe_get src (sb + a))
    done
  done

let delay_f32 ~k ~ring ~seen ~m ~lo ~hi (src : Store.f32) (dst : Store.f32) q0
    q1 =
  let count = hi - lo in
  for q = q0 to q1 - 1 do
    let rb = q * k and sb = (q * m) - seen and db = (q * count) - lo in
    for i = lo to hi - 1 do
      Bigarray.Array1.unsafe_set dst (db + i)
        ( if i < seen then Array.unsafe_get ring (rb + (i mod k))
          else Bigarray.Array1.unsafe_get src (sb + i) )
    done ;
    for a = Int.max seen (seen + m - k) to seen + m - 1 do
      Array.unsafe_set ring
        (rb + (a mod k))
        (Bigarray.Array1.unsafe_get src (sb + a))
    done
  done

(* [line] is the state of the time median over every lane of a stream: the rings
   and the sorted windows, [k] floats each per lane, and the frames seen and
   completed. *)
type line =
  { k: int
  ; mutable rows: int
  ; mutable ring: float array
  ; mutable win: float array
  ; mutable seen: int
  ; mutable emitted: int }

let line k = {k; rows= 0; ring= [||]; win= [||]; seen= 0; emitted= 0}

let line_open l rows =
  l.rows <- rows ;
  l.ring <- Array.make (rows * l.k) 0. ;
  l.win <- Array.make (rows * l.k) 0.

let line_reset l =
  l.rows <- 0 ;
  l.ring <- [||] ;
  l.win <- [||] ;
  l.seen <- 0 ;
  l.emitted <- 0

(* [lookahead k] is the frames a window of [k] reaches past its centre: the
   frames the time median withholds. *)
let lookahead k = k - 1 - (k / 2)

(* [span l ~m ~finish] is the frames [lo, hi) completed once [m] more frames
   have arrived, and the line has ended if [finish]. *)
let span l ~m ~finish =
  let n = l.seen + m in
  (l.emitted, if finish then n else Int.max l.emitted (n - lookahead l.k))

(* {1 Tensor faces of the kernels}

   The last two axes are the bin and frame axes; leading axes are independent
   planes. The faces read the input's C-order storage in place (a contiguous
   copy only when it has none) and fill fresh tensors. Lanes are independent —
   rows for the time median, columns for the frequency median — so they are
   split across domains ({!Par}), each range with its own windows: the medians
   are those one domain computes. *)

(* About 64 Ksteps of window update per chunk. *)
let grain_lanes ~length = Int.max 1 (65_536 / Int.max 1 length)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

(* Stand-ins for the source of a call that admits no frame: never read. *)
let no_frames32 : Store.f32 =
  Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 0

let no_frames64 : Store.f64 =
  Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout 0

(* [time_median l dtype ~lead chunk ~finish ~keep] admits the frames of [chunk]
   ([None]: no frame) into the line [l] of the lanes [lead], ending it if
   [finish], and is the medians of the frames this completes, shaped [lead] by
   those frames — with the frames themselves when [keep]. The line's lanes must
   be open. *)
let time_median : type a.
       line
    -> (float, a) Nx.dtype
    -> lead:int array
    -> (float, a) Nx.t option
    -> finish:bool
    -> keep:bool
    -> (float, a) Nx.t * (float, a) Nx.t option =
 fun l dtype ~lead chunk ~finish ~keep ->
  let m = match chunk with Some c -> last_dim c | None -> 0 in
  let lo, hi = span l ~m ~finish in
  let shape = Array.append lead [|hi - lo|] in
  let k = l.k and ring = l.ring and win = l.win and seen = l.seen in
  let lanes run =
    Par.ranges ~n:l.rows
      ~grain:(grain_lanes ~length:(m + if finish then k else 0))
      run
  in
  let empty () =
    (Nx.zeros dtype shape, if keep then Some (Nx.zeros dtype shape) else None)
  in
  let out =
    if l.rows = 0 || (m = 0 && hi = lo) then empty ()
    else
      match dtype with
      | Nx.Float64 ->
          let sweep harm delay =
            let run src =
              lanes (advance_f64 ~k ~ring ~win ~seen ~m ~lo ~hi src harm delay)
            in
            match chunk with
            | Some c ->
                Store.read64 c run
            | None ->
                run no_frames64
          in
          if hi = lo then begin
            sweep no_frames64 None ; empty ()
          end
          else if keep then begin
            let kept = ref None in
            let harm =
              Store.make64 shape (fun harm ->
                  kept :=
                    Some
                      (Store.make64 shape (fun delay -> sweep harm (Some delay))) )
            in
            (harm, !kept)
          end
          else (Store.make64 shape (fun harm -> sweep harm None), None)
      | Nx.Float32 ->
          let sweep harm delay =
            let run src =
              lanes (advance_f32 ~k ~ring ~win ~seen ~m ~lo ~hi src harm delay)
            in
            match chunk with
            | Some c ->
                Store.read32 c run
            | None ->
                run no_frames32
          in
          if hi = lo then begin
            sweep no_frames32 None ; empty ()
          end
          else if keep then begin
            let kept = ref None in
            let harm =
              Store.make32 shape (fun harm ->
                  kept :=
                    Some
                      (Store.make32 shape (fun delay -> sweep harm (Some delay))) )
            in
            (harm, !kept)
          end
          else (Store.make32 shape (fun harm -> sweep harm None), None)
      | _ ->
          assert false
  in
  l.seen <- seen + m ;
  l.emitted <- hi ;
  out

(* [delayed l ring dtype ~lead chunk ~finish] is the frames of a payload that
   [time_median l] completes on the same call, kept in [ring]: call it before,
   while [l] still counts the frames before [chunk]. *)
let delayed : type a.
       line
    -> float array
    -> (float, a) Nx.dtype
    -> lead:int array
    -> (float, a) Nx.t option
    -> finish:bool
    -> (float, a) Nx.t =
 fun l ring dtype ~lead chunk ~finish ->
  let m = match chunk with Some c -> last_dim c | None -> 0 in
  let lo, hi = span l ~m ~finish in
  let shape = Array.append lead [|hi - lo|] in
  let k = l.k and seen = l.seen in
  let lanes run =
    Par.ranges ~n:l.rows
      ~grain:(grain_lanes ~length:(hi - lo + Int.min m k))
      run
  in
  if l.rows = 0 || (m = 0 && hi = lo) then Nx.zeros dtype shape
  else
    match dtype with
    | Nx.Float64 ->
        let sweep dst =
          let run src = lanes (delay_f64 ~k ~ring ~seen ~m ~lo ~hi src dst) in
          match chunk with
          | Some c ->
              Store.read64 c run
          | None ->
              run no_frames64
        in
        if hi = lo then begin
          sweep no_frames64 ; Nx.zeros dtype shape
        end
        else Store.make64 shape sweep
    | Nx.Float32 ->
        let sweep dst =
          let run src = lanes (delay_f32 ~k ~ring ~seen ~m ~lo ~hi src dst) in
          match chunk with
          | Some c ->
              Store.read32 c run
          | None ->
              run no_frames32
        in
        if hi = lo then begin
          sweep no_frames32 ; Nx.zeros dtype shape
        end
        else Store.make32 shape sweep
    | _ ->
        assert false

(* [median_time s ~k] is the time median of the whole spectrogram [s]: the one
   chunk instance of the line, admitting every frame and ending it. *)
let median_time s ~k =
  let lead = leading_shape s in
  let l = line k in
  line_open l (Array.fold_left ( * ) 1 lead) ;
  fst (time_median l (Nx.dtype s) ~lead (Some s) ~finish:true ~keep:false)

let median_freq : type a. (float, a) Nx.t -> k:int -> (float, a) Nx.t =
 fun s ~k ->
  let dtype = Nx.dtype s in
  let shape = Nx.shape s in
  let nd = Array.length shape in
  let cols = shape.(nd - 1) and rows = shape.(nd - 2) in
  let planes = if rows = 0 || cols = 0 then 0 else Nx.numel s / (rows * cols) in
  if planes = 0 then Nx.zeros dtype shape
  else
    (* a unit is one plane's column range; ranges never straddle planes *)
    let lanes run =
      Par.ranges ~n:(planes * cols) ~grain:(grain_lanes ~length:rows)
        (fun lo hi ->
          let u = ref lo in
          while !u < hi do
            let plane = !u / cols in
            let stop = Int.min hi ((plane + 1) * cols) in
            run plane (!u - (plane * cols)) (stop - (plane * cols)) ;
            u := stop
          done )
    in
    match dtype with
    | Nx.Float64 ->
        Store.make64 shape (fun dst ->
            Store.read64 s (fun src ->
                lanes (median_cols_f64 src dst ~rows ~cols ~k) ) )
    | Nx.Float32 ->
        Store.make32 shape (fun dst ->
            Store.read32 s (fun src ->
                lanes (median_cols_f32 src dst ~rows ~cols ~k) ) )
    | _ ->
        assert false

let tiny : type a. (float, a) Nx.dtype -> float = function
  | Nx.Float64 ->
      2.2250738585072014e-308
  | Nx.Float32 ->
      1.1754943508222875e-38
  | _ ->
      assert false

(* [powered x p] is [x ^ p]. Unit and square exponents are the identity and one
   multiply — the exact values a general power function need not return. *)
let powered x p =
  if p = 1. then x else if p = 2. then Nx.mul x x else Nx.pow_s x p

(* [softmask x xref ~power ~split_zeros] is the share of [x] against the
   reference [xref], both non-negative and of one shape.

   At finite [power] the pair is first rescaled by its pointwise maximum [z],
   which keeps [x^p] and [xref^p] representable for any [p] and leaves the
   quotient unchanged; the share is then [x^p / (x^p + xref^p)]. Where [z] falls
   below the smallest positive normal the quotient is undefined and the mask
   takes [0.5] when the decision is a partition ([split_zeros]) and [0.]
   otherwise.

   At infinite [power] the mask is the strict comparison [x > xref] as [0.] and
   [1.]: the limit of the finite masks away from equality, and [0.] on both
   sides of it. *)
let softmask x xref ~power ~split_zeros =
  let dtype = Nx.dtype x in
  if not (Float.is_finite power) then
    Nx.where (Nx.greater x xref) (Nx.ones_like x) (Nx.zeros_like x)
  else begin
    let z = Nx.maximum x xref in
    let bad = Nx.less z (Nx.full_like z (tiny dtype)) in
    let z = Nx.where bad (Nx.ones_like z) z in
    let m = powered (Nx.div x z) power in
    let r = powered (Nx.div xref z) power in
    Nx.where bad
      (Nx.full_like z (if split_zeros then 0.5 else 0.))
      (Nx.div m (Nx.add m r))
  end

(* [decide ~power ~margin ~harm ~perc] is the harmonic and percussive mask pair
   the enhanced spectrograms [harm] and [perc] decide. *)
let decide ~power ~margin:(m_h, m_p) ~harm ~perc =
  let split_zeros = m_h = 1. && m_p = 1. in
  ( softmask harm (Nx.mul_s perc m_h) ~power ~split_zeros
  , softmask perc (Nx.mul_s harm m_p) ~power ~split_zeros )

(* [masks kernel_size power margin s] is the harmonic and percussive mask pair
   of the magnitude spectrogram [s]. *)
let masks ~kernel_size:(k_h, k_p) ~power ~margin s =
  decide ~power ~margin ~harm:(median_time s ~k:k_h)
    ~perc:(median_freq s ~k:k_p)

(* {1 Validation} *)

let check_dtype : type a. string -> (float, a) Nx.dtype -> unit =
 fun fn dt ->
  match dt with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | dt ->
      invalid_arg
        (Stdlib.Format.asprintf
           "%s: cannot separate %a spectra (the median kernel carries float32 \
            and float64)"
           fn Nx.pp_dtype dt )

let check_rank fn s =
  if Nx.ndim s < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot separate a rank-%d tensor (a spectrogram carries a bin \
          axis and a frame axis)"
         fn (Nx.ndim s) )

let check_kernel fn (k_h, k_p) =
  if k_h < 1 || k_p < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot median-filter with a kernel of (%d, %d) (both kernel \
          sizes must be at least 1)"
         fn k_h k_p )

let check_power fn power =
  if Float.is_nan power || power <= 0. then
    invalid_arg
      (Printf.sprintf
         "%s: cannot raise the mask to the power %g (power must be strictly \
          positive, or infinite for a hard mask)"
         fn power )

let check_margin fn (m_h, m_p) =
  if not (Float.is_finite m_h && Float.is_finite m_p && m_h >= 1. && m_p >= 1.)
  then
    invalid_arg
      (Printf.sprintf
         "%s: cannot bias the decision by a margin of (%g, %g) (both margins \
          must be finite and at least 1)"
         fn m_h m_p )

let validate fn ~kernel_size ~power ~margin s =
  check_kernel fn kernel_size ;
  check_power fn power ;
  check_margin fn margin ;
  check_rank fn s ;
  check_dtype fn (Nx.dtype s)

(* {1 Spectrogram domain} *)

let default_kernel = (31, 31)

let default_margin = (1., 1.)

let hpss_masks ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) s =
  validate "hpss_masks" ~kernel_size ~power ~margin s ;
  masks ~kernel_size ~power ~margin s

let hpss_of_spectrogram ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) s =
  validate "hpss_of_spectrogram" ~kernel_size ~power ~margin s ;
  let mask_h, mask_p = masks ~kernel_size ~power ~margin s in
  (Nx.mul s mask_h, Nx.mul s mask_p)

(* {1 Complex domain}

   [component_witness cdtype] is the real storage whose width matches the
   components of [cdtype]: what the magnitude and the two phase planes carry. *)

type packed_rdtype = Rdtype : (float, 'a) Nx.dtype -> packed_rdtype

let component_witness : type c. (Complex.t, c) Nx.dtype -> packed_rdtype =
  function
  | Nx.Complex64 ->
      Rdtype Nx.float32
  | Nx.Complex128 ->
      Rdtype Nx.float64

(* [unit_phase mag re im] is the unit phase of the spectrum of magnitude [mag]
   and parts [re] and [im], component by component: a zero magnitude is offset
   to one so the quotient is defined, and its real part carries the offset back,
   so the phase there is [1 + 0i]. Complex division is avoided — its
   intermediate products underflow on denormal components. *)
let unit_phase mag re im =
  let zeros_to_ones =
    Nx.where
      (Nx.equal mag (Nx.zeros_like mag))
      (Nx.ones_like mag) (Nx.zeros_like mag)
  in
  let denominator = Nx.add mag zeros_to_ones in
  (Nx.add (Nx.div re denominator) zeros_to_ones, Nx.div im denominator)

(* [components cdtype mag phase masks] is the pair of [cdtype] spectra the masks
   keep of the magnitude [mag], each carrying the unit [phase]. *)
let components cdtype mag (phase_re, phase_im) (mask_h, mask_p) =
  let apply mask =
    let t = Nx.mul mag mask in
    Nx.complex cdtype ~re:(Nx.mul t phase_re) ~im:(Nx.mul t phase_im)
  in
  (apply mask_h, apply mask_p)

let hpss_of_stft ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) z =
  let cdtype = Nx.dtype z in
  let (Rdtype rdtype) = component_witness cdtype in
  let mag = Nx.magnitude rdtype z in
  validate "hpss_of_stft" ~kernel_size ~power ~margin mag ;
  components cdtype mag
    (unit_phase mag (Nx.real rdtype z) (Nx.imag rdtype z))
    (masks ~kernel_size ~power ~margin mag)

(* {1 Signal domain}

   [spectrum_witness dtype] is the complex storage whose component width matches
   [dtype]: what the signal-domain faces analyse into. *)

type packed_cdtype = Cdtype : (Complex.t, 'c) Nx.dtype -> packed_cdtype

let spectrum_witness : type b. (float, b) Nx.dtype -> packed_cdtype = function
  | Nx.Float64 ->
      Cdtype Nx.complex128
  | Nx.Float32 | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Cdtype Nx.complex64

(* [separate fn c ~kernel_size ~power ~margin x] is the analysis-separation-
   synthesis round trip: the components of the STFT of [x] on the geometry [c],
   each inverted back to [x]'s own length and dtype. *)
let separate fn c ~kernel_size ~power ~margin x =
  let dtype = Nx.dtype x in
  check_kernel fn kernel_size ;
  check_power fn power ;
  check_margin fn margin ;
  if Nx.ndim x < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot separate a rank-zero tensor (the time axis must exist)" fn ) ;
  check_dtype fn dtype ;
  let (Cdtype cdtype) = spectrum_witness dtype in
  let length = Nx.dim (Nx.ndim x - 1) x in
  let z_h, z_p =
    hpss_of_stft ~kernel_size ~power ~margin (Stft.transform cdtype c x)
  in
  (Stft.invert dtype c ~length z_h, Stft.invert dtype c ~length z_p)

let hpss c ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) x =
  separate "hpss" c ~kernel_size ~power ~margin x

let harmonic c ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) x =
  fst (separate "harmonic" c ~kernel_size ~power ~margin x)

let percussive c ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) x =
  snd (separate "percussive" c ~kernel_size ~power ~margin x)

(* {1 Pipeline stages}

   A stage is the line of the time median carried across chunks, with the
   frequency median, the masks and the products computed on the frames each call
   completes — operations on one frame at a time, and element-wise ones, so a
   frame's outputs are the same bits whichever call completes it. The stage
   withholds the [lookahead k_h] frames its windows reach past their centre,
   completes a frame on the push that delivers the last of them, and leaves the
   frames whose windows cross the end of the stream to the flush: a push emits
   no chunk when it completes no frame, and the flush at most one.

   [payloads] are the rings of the complex stage's real and imaginary parts,
   delayed alongside the magnitude; the spectrogram stages carry none. *)

type 'd stream =
  { line: line
  ; mutable lead: int array option
  ; mutable dtype: 'd option
  ; mutable payloads: float array * float array
  ; mutable drained: bool }

let shape_string a =
  String.concat "; " (Array.to_list (Array.map string_of_int a))

let stage_of fn ~kernel_size ~power ~margin ~payloads ~check ~emit =
  check_kernel fn kernel_size ;
  check_power fn power ;
  check_margin fn margin ;
  let k_h = fst kernel_size in
  let create () =
    { line= line k_h
    ; lead= None
    ; dtype= None
    ; payloads= ([||], [||])
    ; drained= false }
  in
  let reset s =
    line_reset s.line ;
    s.lead <- None ;
    s.dtype <- None ;
    s.payloads <- ([||], [||]) ;
    s.drained <- false
  in
  (* The element dtype and the leading shape are only witnessed by incoming
     chunks; [step] records them so [concat []] (reached only through [run],
     where a step always precedes) can build the empty pair. *)
  let witness = ref None in
  let step s chunk =
    if s.drained then
      invalid_arg
        ( fn
        ^ ": cannot feed a drained stream (flush consumed the tail; reset \
           before reusing)" ) ;
    check_rank fn chunk ;
    let dtype = Nx.dtype chunk in
    check dtype ;
    let lead = leading_shape chunk in
    ( match s.lead with
    | Some l when l = lead ->
        ()
    | Some l ->
        invalid_arg
          (Printf.sprintf
             "%s: cannot continue a stream of leading shape [%s] with a chunk \
              of leading shape [%s]"
             fn (shape_string l) (shape_string lead) )
    | None ->
        let rows = Array.fold_left ( * ) 1 lead in
        line_open s.line rows ;
        if payloads then
          s.payloads <- (Array.make (rows * k_h) 0., Array.make (rows * k_h) 0.) ;
        s.lead <- Some lead ;
        s.dtype <- Some dtype ) ;
    witness := Some (dtype, lead) ;
    emit s dtype ~lead (Some chunk) ~finish:false
  in
  let flush s =
    if s.drained then []
    else begin
      s.drained <- true ;
      match (s.dtype, s.lead) with
      | Some dtype, Some lead ->
          Option.to_list (emit s dtype ~lead None ~finish:true)
      | _ ->
          []
    end
  in
  Pipeline.kernel ~latency:(lookahead k_h) ~flush ~reset
    ~concat:(function
      | [] -> (
        match !witness with
        | Some (dtype, lead) ->
            let empty () = Nx.zeros dtype (Array.append lead [|0|]) in
            (empty (), empty ())
        | None ->
            invalid_arg
              ( fn
              ^ ": cannot concatenate zero chunks before any chunk fixed the \
                 element dtype" ) )
      | parts ->
          ( Nx.concatenate ~axis:(-1) (List.map fst parts)
          , Nx.concatenate ~axis:(-1) (List.map snd parts) ) )
    ~prepare:(fun _format -> create ())
    ~step ()

(* [spectrogram_stage fn ... ~apply] completes frames of a real spectrogram into
   [apply d masks], [d] the completed frames themselves. *)
let spectrogram_stage fn ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) ~apply () =
  let k_p = snd kernel_size in
  stage_of fn ~kernel_size ~power ~margin ~payloads:false
    ~check:(check_dtype fn) ~emit:(fun s dtype ~lead chunk ~finish ->
      let harm, d = time_median s.line dtype ~lead chunk ~finish ~keep:true in
      let d = Option.get d in
      if last_dim harm = 0 then None
      else
        Some
          (apply d (decide ~power ~margin ~harm ~perc:(median_freq d ~k:k_p))) )

let hpss_of_spectrogram_stage ?kernel_size ?power ?margin () =
  spectrogram_stage "hpss_of_spectrogram_stage" ?kernel_size ?power ?margin
    ~apply:(fun d (mask_h, mask_p) -> (Nx.mul d mask_h, Nx.mul d mask_p))
    ()

let hpss_masks_stage ?kernel_size ?power ?margin () =
  spectrogram_stage "hpss_masks_stage" ?kernel_size ?power ?margin
    ~apply:(fun _ masks -> masks)
    ()

let hpss_of_stft_stage ?(kernel_size = default_kernel) ?(power = 2.)
    ?(margin = default_margin) () =
  let k_p = snd kernel_size in
  stage_of "hpss_of_stft_stage" ~kernel_size ~power ~margin ~payloads:true
    ~check:ignore ~emit:(fun s cdtype ~lead chunk ~finish ->
      let (Rdtype rdtype) = component_witness cdtype in
      let part f = Option.map (f rdtype) chunk in
      let re_ring, im_ring = s.payloads in
      (* the payloads first: [time_median] advances the line's counts *)
      let re = delayed s.line re_ring rdtype ~lead (part Nx.real) ~finish in
      let im = delayed s.line im_ring rdtype ~lead (part Nx.imag) ~finish in
      let harm, mag =
        time_median s.line rdtype ~lead (part Nx.magnitude) ~finish ~keep:true
      in
      let mag = Option.get mag in
      if last_dim harm = 0 then None
      else
        Some
          (components cdtype mag (unit_phase mag re im)
             (decide ~power ~margin ~harm ~perc:(median_freq mag ~k:k_p)) ) )
