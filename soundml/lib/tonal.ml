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

(* Tonal features over a chromagram, as librosa.feature.chroma_cens and
   librosa.feature.tonnetz compute them.

   CENS (Müller, Kurth & Clausen 2005; Müller & Ewert 2011) divides each frame
   by its L1 length, quantises every share of the frame's energy on the
   logarithmic ladder 0.05, 0.1, 0.2, 0.4, smooths the quantised sequence along
   the frames with a window normalised to unit sum, and normalises each frame
   again. The tonal centroid (Harte, Sandler & Gasser 2006) projects each
   L1-normalised frame onto the circles of fifths, minor thirds and major
   thirds: six coordinates per frame.

   Both interiors run in double precision and round once into the dtype of the
   chromagram. The per-frame steps reuse {!Chroma}'s normalisation, its
   ascending sums and its underflow guard; the smoothing is the one step that
   reads across frames, a centred FIR whose every output is one fixed sum read
   through {!emit}, so the flat function and the stage compute the same bits. *)

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

let shape_string shape =
  String.concat "; " (Array.to_list (Array.map string_of_int shape))

(* {1 Checks} *)

let check_rank fn ~verb ~noun x =
  let nd = Nx.ndim x in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot %s a rank-%d %s (the chromagram needs [...; n_chroma; \
          frames])"
         fn verb nd noun )

(* [check_entries fn ~signed c] rejects the non-finite entries librosa's
   normalisation refuses — a NaN share would otherwise quantise to silence, and
   an infinite one turn its whole frame into NaN — and, unless [signed], the
   negative ones: CENS quantises shares of a frame's energy, which librosa's
   constant-Q chromagram clips at zero before they reach it. *)
let check_entries fn ~signed (c : (float, Nx.float64_elt) Nx.t) =
  let n = Nx.numel c in
  Store.read64 c (fun a ->
      for i = 0 to n - 1 do
        let v = Bigarray.Array1.unsafe_get a i in
        if not (Float.is_finite v) then
          invalid_arg
            (Printf.sprintf
               "%s: cannot normalise a chromagram with infinite or NaN values \
                (every entry must be finite)"
               fn ) ;
        if (not signed) && v < 0. then
          invalid_arg
            (Printf.sprintf
               "%s: cannot quantise a chromagram with negative values (chroma \
                energy must be non-negative)"
               fn )
      done )

(* {1 Quantisation} *)

(* [level v] is the quantised share [v]: [0.25] for each of the thresholds 0.4,
   0.2, 0.1 and 0.05 that [v] strictly exceeds, added in that order — exact
   multiples of [0.25] in any case. *)
let level v =
  let q = if v > 0.4 then 0.25 else 0. in
  let q = if v > 0.2 then q +. 0.25 else q in
  let q = if v > 0.1 then q +. 0.25 else q in
  if v > 0.05 then q +. 0.25 else q

(* [quantise fn ~tiny c] is the chromagram [c] normalised to unit L1 length per
   frame, in double precision and under {!Chroma}'s guard at [tiny], then
   quantised. Every frame is quantised alone, so the result does not depend on
   how the frames were chunked. *)
let quantise fn ~tiny c =
  let c = Nx.cast Nx.float64 c in
  check_entries fn ~signed:false c ;
  let v = Chroma.normalise ~norm:(`P 1.) ~tiny c in
  let n = Nx.numel v in
  Store.make64 (Nx.shape v) (fun dst ->
      Store.read64 v (fun src ->
          for i = 0 to n - 1 do
            Bigarray.Array1.unsafe_set dst i
              (level (Bigarray.Array1.unsafe_get src i))
          done ) )

(* {1 Smoothing} *)

(* The smoothing taps: the window's weights normalised to unit sum, kept where
   they exceed the double-precision epsilon (scipy.ndimage's footprint rule,
   which drops the zero endpoints of Hann, Bartlett and Blackman), in ascending
   frame offset. Output frame [i] is [0.] plus [weights.(j) * q (i +
   offsets.(j))], added in ascending [j] over the offsets that land inside the
   sequence: frames beyond either end read as zero. [future] and [past] are the
   largest offsets ahead of and behind the output frame. *)
type taps = {offsets: int array; weights: Float.Array.t; future: int; past: int}

(* [taps fn ~window l] is the smoothing of librosa's [win_len_smooth = l]: the
   symmetric [l + 2]-point [window], normalised by its ascending sum. Its point
   [m] weighs the frame [n / 2 - m] frames ahead, the alignment
   scipy.ndimage.convolve gives a window of [n] points. *)
let taps fn ~window l =
  let n = l + 2 in
  let w =
    (* [Window.make] reports domain errors under its own entry point; relabel
       them with this one, which the caller actually called. *)
    try Nx.to_array (Window.make Nx.float64 ~periodic:false window n)
    with Invalid_argument message ->
      let message =
        match String.index_opt message ':' with
        | Some i ->
            fn ^ String.sub message i (String.length message - i)
        | None ->
            fn ^ ": " ^ message
      in
      invalid_arg message
  in
  let sum = Array.fold_left ( +. ) 0. w in
  if not (Float.is_finite sum && sum > 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot smooth with a %d-point %s window summing to %g (its \
          weights must have a positive sum)"
         fn n
         (Format.asprintf "%a" Window.pp window)
         sum ) ;
  let h = n / 2 in
  let kept =
    List.filter
      (fun m -> Float.abs (w.(m) /. sum) > Float.epsilon)
      (List.init n (fun i -> n - 1 - i))
  in
  let offsets = Array.of_list (List.map (fun m -> h - m) kept) in
  let weights = Float.Array.of_list (List.map (fun m -> w.(m) /. sum) kept) in
  (* The weights sum to one, so some weight is at least [1 / n] and [kept] is
     never empty; the window is symmetric, so both extents are non-negative. *)
  { offsets
  ; weights
  ; future= offsets.(Array.length offsets - 1)
  ; past= -offsets.(0) }

(* Each term is one multiply-add of let-bound operands, which arm64 contracts
   into a fused instruction — as scipy's own arm64 build contracts the loop of
   ndimage.convolve — and x86_64 does not. A term outside the sequence would add
   [w * 0.], which changes no sum started from [0.], so it is skipped. *)
let fill (src : Store.f64) (dst : Store.f64) ~taps ~n ~base ~total ~lo ~hi r0 r1
    =
  let count = hi - lo and k = Array.length taps.offsets in
  for r = r0 to r1 - 1 do
    let row = (r * n) - base and out = r * count in
    for i = lo to hi - 1 do
      let acc = ref 0. in
      for j = 0 to k - 1 do
        let f = i + Array.unsafe_get taps.offsets j in
        if f >= 0 && f < total then begin
          let w = Float.Array.unsafe_get taps.weights j
          and v = Bigarray.Array1.unsafe_get src (row + f) in
          acc := !acc +. (w *. v)
        end
      done ;
      Bigarray.Array1.unsafe_set dst (out + i - lo) !acc
    done
  done

(* [emit ~taps ext ~base ~total ~lo ~hi] is smoothed frames [lo, hi) of a
   quantised sequence of [total] frames, of which [ext] holds frames [base, base
   + last_dim ext) — every frame inside the sequence those outputs read. Rows
   run across domains; each output is computed alone, so the split changes no
   bit. *)
let emit ~taps ext ~base ~total ~lo ~hi =
  let lead = leading_shape ext in
  let rows = Array.fold_left ( * ) 1 lead and n = last_dim ext in
  let count = hi - lo in
  let grain =
    Int.max 1 (32_768 / Int.max 1 (count * Array.length taps.offsets))
  in
  Store.make64 (Array.append lead [|count|]) (fun dst ->
      Store.read64 ext (fun src ->
          Par.ranges ~n:rows ~grain (fill src dst ~taps ~n ~base ~total ~lo ~hi) ) )

(* [finish ~norm ~tiny dtype y] is the final per-frame normalisation, rounded
   once into [dtype]. *)
let finish ~norm ~tiny dtype y = Nx.cast dtype (Chroma.normalise ~norm ~tiny y)

(* {1 CENS} *)

let settings fn ?(norm = `P 2.) ?(smoothing = Some 41)
    ?(smoothing_window = Window.Hann) () =
  Chroma.check_norm fn norm ;
  match smoothing with
  | None ->
      (norm, None)
  | Some l ->
      if l < 1 then
        invalid_arg
          (Printf.sprintf
             "%s: cannot smooth over %d frames (smoothing must be at least 1)"
             fn l ) ;
      (norm, Some (taps fn ~window:smoothing_window l))

let cens fn ~noun ~norm ~taps c =
  check_rank fn ~verb:"quantise" ~noun c ;
  let dtype = Nx.dtype c in
  if Nx.numel c = 0 then Nx.zeros dtype (Nx.shape c)
  else
    let tiny = Chroma.smallest_normal dtype in
    let q = quantise fn ~tiny c in
    let y =
      match taps with
      | None ->
          q
      | Some taps ->
          let t = last_dim q in
          emit ~taps q ~base:0 ~total:t ~lo:0 ~hi:t
    in
    finish ~norm ~tiny dtype y

let chroma_cens_of_chroma ?norm ?smoothing ?smoothing_window c =
  let fn = "chroma_cens_of_chroma" in
  let norm, taps = settings fn ?norm ?smoothing ?smoothing_window () in
  cens fn ~noun:"tensor" ~norm ~taps c

(* The raw chromagram is [Soundml.chroma_cqt cqt ~norm:`None x], rounded into
   the dtype of [x] exactly as that function rounds it. *)
let chroma_cens cqt ?n_chroma ?norm ?smoothing ?smoothing_window x =
  let fn = "chroma_cens" in
  let norm, taps = settings fn ?norm ?smoothing ?smoothing_window () in
  let raw =
    Chroma.of_cqt ?n_chroma ~norm:`None cqt (Cqt.power_spectrum ~power:1. cqt x)
  in
  cens fn ~noun:"tensor" ~norm ~taps raw

(* {2 Pipeline stage}

   State of one stream: [hist] holds the last [past + future] quantised frames
   seen, a kernel-owned copy; [seen] and [emitted] count frames in and out. An
   output frame is emitted once every frame its window reads has arrived — frame
   [i] once frame [i + future] has — so the stage withholds [future] frames, and
   the flush emits them against the end of the stream. Each chunk is quantised
   alone and every output is {!emit}'s fixed sum, so every partitioning yields
   the offline bits. *)

type 'a state =
  { mutable hist: (float, Nx.float64_elt) Nx.t option
  ; mutable seen: int
  ; mutable emitted: int
  ; mutable lead: int array option
  ; mutable dtype: (float, 'a) Nx.dtype option }

let state_create () = {hist= None; seen= 0; emitted= 0; lead= None; dtype= None}

let state_reset s =
  s.hist <- None ;
  s.seen <- 0 ;
  s.emitted <- 0 ;
  s.lead <- None ;
  s.dtype <- None

let smoothing_stage fn ~norm ~taps =
  let keep = taps.past + taps.future in
  let empty lead dtype = Nx.zeros dtype (Array.append lead [|0|]) in
  (* [run s ext ~base ~upto] emits every output below [upto] not yet emitted,
     reading [ext] (frames [base ...]) as the stream's frames so far. *)
  let run s ext ~base ~upto =
    let lo = s.emitted in
    s.emitted <- Int.max lo upto ;
    if upto > lo then Some (emit ~taps ext ~base ~total:s.seen ~lo ~hi:upto)
    else None
  in
  let step s chunk =
    check_rank fn ~verb:"quantise" ~noun:"chunk" chunk ;
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
             fn (shape_string l) (shape_string lead) ) ) ;
    let dtype = Nx.dtype chunk in
    s.dtype <- Some dtype ;
    let m = last_dim chunk in
    if m = 0 then empty lead dtype
    else begin
      let tiny = Chroma.smallest_normal dtype in
      let q = quantise fn ~tiny chunk in
      let ext, base =
        match s.hist with
        | None ->
            (q, s.seen)
        | Some hist ->
            (concat_last [hist; q], s.seen - last_dim hist)
      in
      s.seen <- s.seen + m ;
      let out = run s ext ~base ~upto:(Int.max 0 (s.seen - taps.future)) in
      let l = last_dim ext in
      s.hist <-
        ( if keep = 0 then None
          else Some (Nx.copy (shrink_last ext (Int.max 0 (l - keep)) l)) ) ;
      match out with
      | Some y ->
          finish ~norm ~tiny dtype y
      | None ->
          empty lead dtype
    end
  in
  let flush s =
    match (s.hist, s.dtype) with
    | Some hist, Some dtype ->
        let tiny = Chroma.smallest_normal dtype in
        Option.to_list
          (Option.map (finish ~norm ~tiny dtype)
             (run s hist ~base:(s.seen - last_dim hist) ~upto:s.seen) )
    | _ ->
        []
  in
  (* The element dtype is only witnessed by incoming chunks; [step] records it
     so [concat []] (reached only through [run], where a step always precedes)
     can build the empty output. *)
  let witness = ref None in
  Pipeline.kernel ~latency:taps.future ~reset:state_reset ~flush
    ~concat:(function
      | [] -> (
        match !witness with
        | Some dtype ->
            Nx.zeros dtype [|0|]
        | None ->
            invalid_arg
              ( fn
              ^ ": cannot concatenate zero chunks before any chunk fixed the \
                 element dtype" ) )
      | parts ->
          concat_last parts )
    ~prepare:(fun _format -> state_create ())
    ~step:(fun s chunk ->
      witness := Some (Nx.dtype chunk) ;
      Some (step s chunk) )
    ()

let chroma_cens_stage ?norm ?smoothing ?smoothing_window () =
  let fn = "chroma_cens_stage" in
  let norm, taps = settings fn ?norm ?smoothing ?smoothing_window () in
  match taps with
  | None ->
      Pipeline.stateless (cens fn ~noun:"chunk" ~norm ~taps:None)
  | Some taps ->
      smoothing_stage fn ~norm ~taps

(* {1 Tonal centroid} *)

(* The three circles, two coordinates each: the rate at which a coordinate's
   angle advances along the chroma axis, in half turns per semitone — a fifth
   moves the first circle by a twelfth of a turn, a minor third the second by a
   quarter and a major third the third by a third — and the circle's radius. *)
let rates = [|7. /. 6.; 7. /. 6.; 3. /. 2.; 3. /. 2.; 2. /. 3.; 2. /. 3.|]

let radii = [|1.; 1.; 1.; 1.; 0.5; 0.5|]

(* [centroid_matrix k] is the [[6; k]] projection over [k] chroma bands, in
   librosa's evaluation order: band [c] sits at [c * (12 / k)] semitones, its
   phase is the rate times that position — less a quarter turn on the even rows,
   which turns their cosine into a sine — and its weight the radius times the
   cosine of [pi] times the phase. The product is let-bound, so the subtraction
   never fuses with it and the weights are librosa's bits. *)
let centroid_matrix k =
  let step = 12. /. Float.of_int k in
  let w = Array.make (6 * k) 0. in
  for p = 0 to 5 do
    for c = 0 to k - 1 do
      let d = Float.of_int c *. step in
      let phase = rates.(p) *. d in
      let phase = if p mod 2 = 0 then phase -. 0.5 else phase in
      w.((p * k) + c) <- radii.(p) *. Float.cos (Float.pi *. phase)
    done
  done ;
  Ordered.matrix (Nx.create Nx.float64 [|6; k|] w)

(* [centroid fn ~matrix c] is the tonal centroid of the chromagram [c], whose
   [k] bands [matrix k] projects. Each frame is L1-normalised and projected
   alone ({!Ordered.project}), so its centroid does not depend on its batch. *)
let centroid fn ~matrix c =
  let dtype = Nx.dtype c in
  let shape = Nx.shape c in
  let nd = Array.length shape in
  if Array.exists (fun d -> d = 0) shape then begin
    let out = Array.copy shape in
    out.(nd - 2) <- 6 ;
    Nx.zeros dtype out
  end
  else
    let tiny = Chroma.smallest_normal dtype in
    let c = Nx.cast Nx.float64 c in
    check_entries fn ~signed:true c ;
    let v = Chroma.normalise ~norm:(`P 1.) ~tiny c in
    Nx.cast dtype (Ordered.project (matrix shape.(nd - 2)) v)

let tonnetz_of_chroma c =
  let fn = "tonnetz_of_chroma" in
  check_rank fn ~verb:"project" ~noun:"tensor" c ;
  centroid fn ~matrix:centroid_matrix c

let tonnetz cqt ?n_chroma x =
  let fn = "tonnetz" in
  let chroma =
    Chroma.of_cqt ?n_chroma cqt (Cqt.power_spectrum ~power:1. cqt x)
  in
  centroid fn ~matrix:centroid_matrix chroma

(* The projection is memoryless; the twelve-band matrix, the usual one, is
   prepared once with the stage, and any other band count per chunk. *)
let tonnetz_stage () =
  let fn = "tonnetz_stage" in
  let twelve = centroid_matrix 12 in
  let matrix k = if k = 12 then twelve else centroid_matrix k in
  Pipeline.stateless (fun c ->
      check_rank fn ~verb:"project" ~noun:"chunk" c ;
      centroid fn ~matrix c )
