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

(* Frame-rate energy features over framed audio: root-mean-square energy and
   zero-crossing rate, each as an offline batched computation and as the
   streaming carry behind its pipeline stage.

   The framing geometry is librosa 0.11's feature framing: the signal is
   centered by [frame_length / 2] positions of boundary extension on each side —
   constant zeros for [rms] ([librosa.feature.rms]'s [pad_mode] default), edge
   copies for [zero_crossing_rate] (hard-wired there) — and frames advance by
   [hop] over the padded stream. Framing goes through [Nx.extract_patches] in
   bounded blocks, exactly as [Stft] (see the header note there): the framed
   signal is never materialized whole, and per-frame work never routes through
   per-frame Nx calls. The interior is double precision with one cast at the
   boundary.

   The streaming state below is the [Stft] framing carry re-instantiated for the
   boundary modes these features need. Both are computable from one sample — a
   constant border needs none, an edge border copies the first or last raw
   sample — so the carry has no prelude phase and its tail is a single sample;
   the pending/skip bookkeeping is otherwise the same machine. *)

(* {1 Shape helpers (file-local, as in stft.ml)} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

(* [zero_leading t] is [true] iff a leading axis of [t] has size zero: the
   tensor holds no signals at all, and slicing machinery below cannot express
   zero-size ranges. Requires rank >= 1. *)
let zero_leading t = Array.exists (fun d -> d = 0) (leading_shape t)

let shrink_last t start stop =
  let nd = Nx.ndim t in
  Nx.shrink
    (Array.init nd (fun i ->
         if i = nd - 1 then (start, stop) else (0, Nx.dim i t) ) )
    t

let shrink_frame_axis t start stop =
  let nd = Nx.ndim t in
  Nx.shrink
    (Array.init nd (fun i ->
         if i = nd - 2 then (start, stop) else (0, Nx.dim i t) ) )
    t

let concat_last parts = Nx.concatenate ~axis:(-1) parts

let take_last t indices =
  let indices =
    Nx.create Nx.int32 [|Array.length indices|] (Array.map Int32.of_int indices)
  in
  Nx.take ~axis:(-1) ~indices t

(* {1 Preconditions} *)

let check_rank op t =
  if Nx.ndim t < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-zero tensor (the time axis must exist)" op )

let check_frame_length op frame_length =
  if frame_length < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse frames of %d samples (frame_length must be at \
          least 1)"
         op frame_length )

let check_hop op hop =
  if hop < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot advance frames by %d samples (hop must be at least 1)" op
         hop )

let check_threshold op threshold =
  if not (Float.is_finite threshold && threshold >= 0.) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot clamp signs with a threshold of %g (threshold must be \
          finite and non-negative)"
         op threshold )

(* {1 The frame grid}

   Frame [p] of a length-[n] signal covers padded positions [p * hop, p * hop +
   frame_length), where the padded stream is the signal extended by
   [frame_length / 2] positions on each side. An empty signal produces no frames
   — the [Stft] grid convention; librosa pads an all-border frame for even frame
   lengths and rejects empty input otherwise. *)

let count_frames ~frame_length ~hop n =
  if n = 0 then 0
  else
    let padded = n + (2 * (frame_length / 2)) in
    if padded < frame_length then 0 else 1 + ((padded - frame_length) / hop)

(* {1 The per-frame reductions}

   Each reduction maps a framed block [[...; frame_length; count]], already cast
   to double, to its per-frame values [[...; 1; count]] — one batched Nx
   expression, no frame loop. *)

(* [reduce_rms patches] is the root of the mean square over the frame axis:
   librosa's [mean(abs2(frames)); sqrt]. *)
let reduce_rms patches =
  Nx.sqrt (Nx.mean ~axes:[-2] ~keepdims:true (Nx.square patches))

(* [reduce_zcr ~frame_length ~threshold patches] is the fraction of
   consecutive-sample sign changes per frame. librosa clamps values in
   [-threshold, threshold] to positive zero and compares [signbit]s, so a sample
   is negative iff it lies strictly below [-threshold]; the first frame position
   carries no crossing ([pad=False] in [librosa.feature.zero_crossing_rate]),
   and the count is averaged over [frame_length]. *)
let reduce_zcr ~frame_length ~threshold patches =
  if frame_length = 1 then Nx.zeros Nx.float64 (Nx.shape patches)
  else
    let negative = Nx.less_s patches (-.threshold) in
    let crossings =
      Nx.cast Nx.float64
        (Nx.not_equal
           (shrink_frame_axis negative 1 frame_length)
           (shrink_frame_axis negative 0 (frame_length - 1)) )
    in
    Nx.div_s
      (Nx.sum ~axes:[-2] ~keepdims:true crossings)
      (Float.of_int frame_length)

(* {1 The batched frame computation} *)

(* Upper bound on the framed scratch materialized per reduction call, in
   scalars; framing is blocked to this budget because each block is gathered
   rather than viewed, so the budget caps the transient. *)
let block_elements = 65536

(* [analyse ~frame_length ~hop ~reduce samples count] is frames [0, count) of
   the padded-stream segment [samples], shaped [[...; 1; count]] in the dtype of
   [samples]; [samples] must span at least [(count - 1) * hop + frame_length]
   positions. Each block is one gather ([extract_patches]), one cast to double,
   one batched reduction and one cast back; frames depend only on their own
   samples, so blocking is exact. *)
let analyse ~frame_length ~hop ~reduce samples count =
  let block = Stdlib.max 1 (block_elements / frame_length) in
  let rec go p acc =
    if p >= count then List.rev acc
    else
      let nb = Stdlib.min block (count - p) in
      let seg =
        shrink_last samples (p * hop) (((p + nb - 1) * hop) + frame_length)
      in
      let patches =
        Nx.extract_patches ~kernel_size:[|frame_length|] ~stride:[|hop|]
          ~dilation:[|1|]
          ~padding:[|(0, 0)|]
          seg
      in
      let values =
        Nx.cast (Nx.dtype samples) (reduce (Nx.cast Nx.float64 patches))
      in
      go (p + nb) (values :: acc)
  in
  match go 0 [] with [one] -> one | many -> concat_last many

(* {1 The streaming carry} *)

(* State of one streaming analysis. The padded stream is the signal with its
   boundary extension; [pending_rev] holds the unconsumed suffix of that stream
   (in reverse), [skip] counts padded positions still to drop when the hop
   outruns the data. The left border installs with the first non-empty chunk —
   both border modes need at most the first raw sample — and [tail] keeps the
   last raw sample so the right border is computable at drain time. Retained
   tensors are always kernel-owned copies — incoming chunks are borrowed. *)
type 'a state =
  { frame_length: int
  ; hop: int
  ; border: int (* [frame_length / 2] positions on each side *)
  ; pad: [`Zeros | `Edge]
  ; reduce: (float, Nx.float64_elt) Nx.t -> (float, Nx.float64_elt) Nx.t
  ; mutable started: bool
  ; mutable drained: bool
  ; mutable pending_rev: (float, 'a) Nx.t list
  ; mutable pending_len: int
  ; mutable tail: (float, 'a) Nx.t option
  ; mutable skip: int }

let state_create ~frame_length ~hop pad reduce =
  { frame_length
  ; hop
  ; border= frame_length / 2
  ; pad
  ; reduce
  ; started= false
  ; drained= false
  ; pending_rev= []
  ; pending_len= 0
  ; tail= None
  ; skip= 0 }

let state_reset s =
  s.started <- false ;
  s.drained <- false ;
  s.pending_rev <- [] ;
  s.pending_len <- 0 ;
  s.tail <- None ;
  s.skip <- 0

(* [process s extra extra_len] appends [extra] (stream order, possibly borrowed)
   to the pending padded stream and emits every frame that became complete. The
   leftover suffix is copied out, so nothing borrowed is retained and the
   materialized concatenation dies with the call. *)
let process s extra extra_len =
  let fl = s.frame_length and hop = s.hop in
  let total = s.pending_len + extra_len in
  let count = if total < fl then 0 else 1 + ((total - fl) / hop) in
  if count = 0 then begin
    List.iter
      (fun t ->
        if last_dim t > 0 then s.pending_rev <- Nx.copy t :: s.pending_rev )
      extra ;
    s.pending_len <- total ;
    None
  end
  else begin
    let parts = List.rev_append s.pending_rev extra in
    let samples = match parts with [one] -> one | many -> concat_last many in
    let out = analyse ~frame_length:fl ~hop ~reduce:s.reduce samples count in
    let next_start = count * hop in
    if next_start >= total then begin
      s.skip <- s.skip + next_start - total ;
      s.pending_rev <- [] ;
      s.pending_len <- 0
    end
    else begin
      s.pending_rev <- [Nx.copy (shrink_last samples next_start total)] ;
      s.pending_len <- total - next_start
    end ;
    Some out
  end

(* [update_tail s chunk] keeps [tail] equal to the last raw sample; [chunk] must
   be non-empty on its last axis. *)
let update_tail s chunk =
  let m = last_dim chunk in
  s.tail <- Some (Nx.copy (shrink_last chunk (m - 1) m))

(* [left_pad s chunk] is the [border]-position left extension, computed from the
   head of the first non-empty chunk. *)
let left_pad s chunk =
  if s.border = 0 then None
  else
    Some
      ( match s.pad with
      | `Zeros ->
          Nx.zeros (Nx.dtype chunk)
            (Array.append (leading_shape chunk) [|s.border|])
      | `Edge ->
          take_last chunk (Array.make s.border 0) )

(* [right_pad s] is the [border]-position right extension, computed from the raw
   tail; defined only when [started] and [border > 0]. *)
let right_pad s =
  let tail = Option.get s.tail in
  match s.pad with
  | `Zeros ->
      Nx.zeros (Nx.dtype tail) (Array.append (leading_shape tail) [|s.border|])
  | `Edge ->
      take_last tail (Array.make s.border (last_dim tail - 1))

let state_step s chunk =
  if s.drained then
    invalid_arg
      "step: cannot feed a drained kernel (flush consumed the tail; reset \
       before reusing)" ;
  if zero_leading chunk then
    invalid_arg
      "step: cannot analyse a chunk with a zero-size leading axis (channels \
       must be at least 1)" ;
  let m = last_dim chunk in
  if m = 0 then None
  else begin
    update_tail s chunk ;
    if not s.started then begin
      s.started <- true ;
      match left_pad s chunk with
      | None ->
          process s [chunk] m
      | Some lp ->
          process s [lp; chunk] (s.border + m)
    end
    else if s.skip >= m then begin
      s.skip <- s.skip - m ;
      None
    end
    else begin
      let dropped = s.skip in
      s.skip <- 0 ;
      let chunk = if dropped = 0 then chunk else shrink_last chunk dropped m in
      process s [chunk] (m - dropped)
    end
  end

let state_flush s =
  if s.drained then None
  else begin
    s.drained <- true ;
    let out =
      if (not s.started) || s.border = 0 then None
      else begin
        let rp = right_pad s in
        let r = last_dim rp in
        if s.skip >= r then begin
          s.skip <- s.skip - r ;
          None
        end
        else begin
          let dropped = s.skip in
          s.skip <- 0 ;
          let rp = if dropped = 0 then rp else shrink_last rp dropped r in
          process s [rp] (r - dropped)
        end
      end
    in
    s.pending_rev <- [] ;
    s.pending_len <- 0 ;
    out
  end

(* {1 Offline entry points} *)

(* [frameless x count] is the [count]-frame all-zero feature of [x]'s leading
   shape: the result for the cases that evaluate no frame — an empty signal, or
   a zero-size leading axis (no signals at all). *)
let frameless x count =
  Nx.zeros (Nx.dtype x) (Array.append (leading_shape x) [|1; count|])

(* [run_offline op ~frame_length ~hop pad reduce x] is the one-chunk instance of
   the streaming carry: step on the whole signal, then flush — the flat function
   and the stage cannot disagree. *)
let run_offline op ~frame_length ~hop pad reduce x =
  check_frame_length op frame_length ;
  check_hop op hop ;
  check_rank op x ;
  let n = last_dim x in
  if zero_leading x || n = 0 then
    frameless x (count_frames ~frame_length ~hop n)
  else
    let s = state_create ~frame_length ~hop pad reduce in
    (* [@] evaluates its right operand first: sequence step before flush *)
    let stepped = state_step s x in
    let drained = state_flush s in
    match Option.to_list stepped @ Option.to_list drained with
    | [] ->
        frameless x 0
    | [one] ->
        one
    | many ->
        concat_last many

let rms ?(frame_length = 2048) ?(hop = 512) x =
  run_offline "rms" ~frame_length ~hop `Zeros reduce_rms x

let zero_crossing_rate ?(frame_length = 2048) ?(hop = 512) ?(threshold = 1e-10)
    x =
  check_threshold "zero_crossing_rate" threshold ;
  run_offline "zero_crossing_rate" ~frame_length ~hop `Edge
    (reduce_zcr ~frame_length ~threshold)
    x

(* [rms_of_spectrogram] is librosa's [S=] path: the frame power recovered from a
   magnitude spectrogram by Parseval's identity for the one-sided layout — the
   DC bin (and, for even frame lengths, the Nyquist bin) is halved, the doubled
   bin sum divides by [frame_length ^ 2]. *)
let rms_of_spectrogram ?(frame_length = 2048) s =
  check_frame_length "rms_of_spectrogram" frame_length ;
  if Nx.ndim s < 2 then
    invalid_arg
      (Printf.sprintf
         "rms_of_spectrogram: cannot analyse a rank-%d tensor (the spectrogram \
          path needs [...; bins; frames])"
         (Nx.ndim s) ) ;
  let bins = Nx.dim (Nx.ndim s - 2) s in
  if bins <> (frame_length / 2) + 1 then
    invalid_arg
      (Printf.sprintf
         "rms_of_spectrogram: cannot read %d frequency bins as frames of %d \
          samples (a magnitude spectrogram holds frame_length / 2 + 1 bins)"
         bins frame_length ) ;
  let weights = Array.make bins 1. in
  weights.(0) <- 0.5 ;
  if frame_length mod 2 = 0 then weights.(bins - 1) <- 0.5 ;
  let weights = Nx.create Nx.float64 [|bins; 1|] weights in
  let power =
    Nx.sum ~axes:[-2] ~keepdims:true
      (Nx.mul (Nx.square (Nx.cast Nx.float64 s)) weights)
  in
  let power =
    Nx.div_s (Nx.mul_s power 2.)
      (Float.of_int frame_length *. Float.of_int frame_length)
  in
  Nx.cast (Nx.dtype s) (Nx.sqrt power)

(* {1 Pipeline stages} *)

let ceil_div a b = (a + b - 1) / b

(* [stage_latency ~frame_length] is the involuntary lookahead the pipeline stage
   declares: frame [p] is centered at padded position [p * hop + border], so its
   right reach past the center is [frame_length / 2] samples. Both border modes
   install from the first raw sample, so no extra border lookahead accrues
   (unlike [`Reflect] in [Stft]). *)
let stage_latency ~frame_length = frame_length / 2

(* [frame_bound ~frame_length ~hop b] is the most frames one [step] or drained
   tail can emit for chunks of at most [b] samples: the withheld [latency]
   samples plus the chunk, framed at [hop] — the same bound [out_format]
   declares, so every emitted chunk honors the threaded format; flush output is
   additionally split to it. *)
let frame_bound ~frame_length ~hop b =
  ceil_div (b + stage_latency ~frame_length) hop + 1

let threaded_bound ~frame_length ~hop fmt =
  Option.map (frame_bound ~frame_length ~hop) (Pipeline.Format.max_items fmt)

let stage_rate ~hop = {Pipeline.Rate.num= 1; den= hop}

let stage_out_format ~frame_length ~hop fmt =
  let ips =
    Pipeline.Rate.(Pipeline.Format.items_per_second fmt * stage_rate ~hop)
  in
  fmt
  |> Pipeline.Format.with_items_per_second ips
  |> Pipeline.Format.with_max_items (threaded_bound ~frame_length ~hop fmt)

let split_frames bound t =
  match bound with
  | None ->
      [t]
  | Some bound ->
      let total = last_dim t in
      if total <= bound then [t]
      else
        let rec go start acc =
          if start >= total then List.rev acc
          else
            let stop = Stdlib.min total (start + bound) in
            go stop (shrink_last t start stop :: acc)
        in
        go 0 []

type 'a stage_state = {state: 'a state; bound: int option}

(* [feature_stage op ~frame_length ~hop pad reduce] is the streaming carry as a
   pipeline stage. The element dtype and the leading (channel) shape are only
   witnessed by incoming chunks; [step] records them so [concat []] (reached
   only through [run], where a step always precedes) can build the empty chunk
   with the stream's own leading axes — offline [frameless] does the same, so
   the two faces agree on an all-empty stream. One stage value is monomorphic in
   its element type, so the shared cell is coherent across prepares. *)
let feature_stage op ~frame_length ~hop pad reduce =
  let witness = ref None in
  Pipeline.kernel
    ~latency:(stage_latency ~frame_length)
    ~rate:(stage_rate ~hop)
    ~out_format:(stage_out_format ~frame_length ~hop)
    ~flush:(fun s ->
      match state_flush s.state with
      | None ->
          []
      | Some out ->
          split_frames s.bound out )
    ~reset:(fun s -> state_reset s.state)
    ~concat:(function
      | [] -> (
        match !witness with
        | Some (dtype, leading) ->
            Nx.zeros dtype (Array.append leading [|1; 0|])
        | None ->
            invalid_arg
              (Printf.sprintf
                 "%s: cannot concatenate zero chunks before any chunk fixed \
                  the element dtype"
                 op ) )
      | parts ->
          concat_last parts )
    ~prepare:(fun fmt ->
      { state= state_create ~frame_length ~hop pad reduce
      ; bound= threaded_bound ~frame_length ~hop fmt } )
    ~step:(fun s chunk ->
      witness := Some (Nx.dtype chunk, leading_shape chunk) ;
      state_step s.state chunk )
    ()

let rms_stage ?(frame_length = 2048) ?(hop = 512) () =
  check_frame_length "rms_stage" frame_length ;
  check_hop "rms_stage" hop ;
  feature_stage "rms_stage" ~frame_length ~hop `Zeros reduce_rms

let zero_crossing_rate_stage ?(frame_length = 2048) ?(hop = 512)
    ?(threshold = 1e-10) () =
  check_frame_length "zero_crossing_rate_stage" frame_length ;
  check_hop "zero_crossing_rate_stage" hop ;
  check_threshold "zero_crossing_rate_stage" threshold ;
  feature_stage "zero_crossing_rate_stage" ~frame_length ~hop `Edge
    (reduce_zcr ~frame_length ~threshold)
