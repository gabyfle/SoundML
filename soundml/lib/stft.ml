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

(* Internals:

   - Framing and transform are one [Nx.stft]: the padded stream is windowed
   through a strided frame view and carried by a single batched [rfft], so no
   framed scratch is ever gathered.

   - The windowed product is computed in double — the arithmetic the module's
   parity contract fixes (see the interface): the frames are multiplied by the
   float64 window (promoting float32 audio) and transformed in double, rounding
   once into the complex storage. The double promotion is applied to the stream,
   not the frames — widening is exact, so the product is unchanged, and it costs
   one cast per sample instead of one per framed position. [Nx.rfft] takes the
   output dtype first, so the caller's witness supplies that storage natively —
   no boundary cast of the spectrum.

   - Complex magnitudes are [Nx.magnitude], dtype-first, landing directly in the
   caller's float dtype. The real-valued conveniences pick the complex storage
   whose component width matches the input's dtype ([spectrum_witness]). *)

module Config = struct
  type t =
    { window: Window.t
    ; win_length: int
    ; hop: int
    ; alignment: [`Centered | `Left | `Right]
    ; pad: [`Reflect | `Constant of float | `Edge]
    ; scale: [`None | `Magnitude | `Psd]
    ; fft_size: int
    ; analysis_window: (float, Nx.float64_elt) Nx.t
          (* [win_length] points of [window], centered in [fft_size] samples and
             normalised per [scale]; precomputed once, in double. *) }

  let create ?(window = Window.Hann) ?win_length ?hop ?(alignment = `Centered)
      ?(pad = `Reflect) ?(scale = `None) ~fft_size () =
    if fft_size < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot use an FFT of size %d (fft_size must be at least 1)"
           fft_size ) ;
    let win_length = Option.value win_length ~default:fft_size in
    if win_length < 1 || win_length > fft_size then
      invalid_arg
        (Printf.sprintf
           "create: cannot use a %d-point window with an FFT of size %d \
            (win_length must lie in [1, fft_size])"
           win_length fft_size ) ;
    let hop = Option.value hop ~default:(Stdlib.max 1 (fft_size / 4)) in
    if hop < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot advance frames by %d samples (hop must be at least \
            1)"
           hop ) ;
    let analysis_window =
      let coefficients =
        (* [Window.make] reports domain errors under its own entry point;
           relabel them with this one, which the caller actually called. *)
        try Window.make Nx.float64 ~periodic:true window win_length
        with Invalid_argument message ->
          let message =
            match String.index_opt message ':' with
            | Some i ->
                "create" ^ String.sub message i (String.length message - i)
            | None ->
                "create: " ^ message
          in
          invalid_arg message
      in
      let padded =
        if win_length = fft_size then coefficients
        else
          let left = (fft_size - win_length) / 2 in
          Nx.pad [|(left, fft_size - win_length - left)|] 0. coefficients
      in
      match scale with
      | `None ->
          padded
      | `Magnitude ->
          Nx.div_s padded (Nx.item [] (Nx.sum padded))
      | `Psd ->
          Nx.div_s padded (Float.sqrt (Nx.item [] (Nx.sum (Nx.square padded))))
    in
    {window; win_length; hop; alignment; pad; scale; fft_size; analysis_window}

  let fft_size t = t.fft_size

  let hop t = t.hop

  let win_length t = t.win_length

  let window t = t.window

  let alignment t = t.alignment

  let pad t = t.pad

  let scale t = t.scale

  let bins t = (t.fft_size / 2) + 1

  let latency t =
    match t.alignment with `Centered -> t.fft_size / 2 | `Left | `Right -> 0

  let pp fmt t =
    let alignment =
      match t.alignment with
      | `Centered ->
          "centered"
      | `Left ->
          "left"
      | `Right ->
          "right"
    in
    let pad fmt = function
      | `Reflect ->
          Format.pp_print_string fmt "reflect"
      | `Constant v ->
          Format.fprintf fmt "constant(%g)" v
      | `Edge ->
          Format.pp_print_string fmt "edge"
    in
    let scale =
      match t.scale with
      | `None ->
          "none"
      | `Magnitude ->
          "magnitude"
      | `Psd ->
          "psd"
    in
    Format.fprintf fmt
      "stft(fft_size=%d, hop=%d, window=%a, win_length=%d, alignment=%s, \
       pad=%a, scale=%s)"
      t.fft_size t.hop Window.pp t.window t.win_length alignment pad t.pad scale

  let equal a b =
    Window.equal a.window b.window
    && a.win_length = b.win_length
    && a.hop = b.hop && a.alignment = b.alignment
    && ( match (a.pad, b.pad) with
      | `Reflect, `Reflect | `Edge, `Edge ->
          true
      | `Constant x, `Constant y ->
          Float.equal x y
      | _ ->
          false )
    && a.scale = b.scale && a.fft_size = b.fft_size
end

(* {1 The frame grid} *)

(* Widths of the boundary extension: frame [p] covers padded positions [p * hop,
   p * hop + fft_size), where padded position [q] is source sample [q -
   left_width]. *)
let left_width (c : Config.t) =
  match c.alignment with
  | `Centered ->
      c.fft_size / 2
  | `Left ->
      0
  | `Right ->
      c.fft_size - 1

let right_width (c : Config.t) =
  match c.alignment with `Centered -> c.fft_size / 2 | `Left | `Right -> 0

let check_length op n =
  if n < 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a signal of length %d (length must be \
          non-negative)"
         op n )

let frames c ~n =
  check_length "frames" n ;
  if n = 0 then 0
  else
    let padded = n + left_width c + right_width c in
    if padded < Config.fft_size c then 0
    else 1 + ((padded - Config.fft_size c) / Config.hop c)

let first_complete c =
  let l = left_width c in
  (l + Config.hop c - 1) / Config.hop c

let last_complete c ~n =
  check_length "last_complete" n ;
  let total = frames c ~n in
  if n = 0 then 0
  else
    let reach = n + left_width c - Config.fft_size c in
    if reach < 0 then 0 else Stdlib.min total ((reach / Config.hop c) + 1)

let check_sample_rate op sample_rate =
  if sample_rate < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot use a sample rate of %d Hz (sample_rate must be at least \
          1)"
         op sample_rate )

let times dtype c ~sample_rate ~n =
  check_sample_rate "times" sample_rate ;
  check_length "times" n ;
  let count = frames c ~n in
  (* [p * hop] is exact in double; the single rounding is the division — the
     reference arithmetic the parity contract fixes for the frame-time grid. *)
  Nx.arange_f Nx.float64 0. (Float.of_int count) 1.
  |> Fun.flip Nx.mul_s (Float.of_int (Config.hop c))
  |> Fun.flip Nx.div_s (Float.of_int sample_rate)
  |> Nx.cast dtype

let frequencies dtype c ~sample_rate =
  check_sample_rate "frequencies" sample_rate ;
  Nx.arange_f Nx.float64 0. (Float.of_int (Config.bins c)) 1.
  |> Fun.flip Nx.mul_s
       (Float.of_int sample_rate /. Float.of_int (Config.fft_size c))
  |> Nx.cast dtype

(* {1 Shape helpers} *)

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

let concat_last parts = Nx.concatenate ~axis:(-1) parts

let full_widths t left right =
  let nd = Nx.ndim t in
  Array.init nd (fun i -> if i = nd - 1 then (left, right) else (0, 0))

let check_rank op t =
  if Nx.ndim t < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-zero tensor (the time axis must exist)" op )

(* {1 Boundary extension} *)

(* [reflect_index n q] is the source index reflect padding reads for padded
   position [q] over a length-[n] signal: mirrored around the first and last
   samples without repeating them, with period [2 * (n - 1)]. *)
let reflect_index n q =
  if n = 1 then 0
  else
    let period = 2 * (n - 1) in
    let m = ((q mod period) + period) mod period in
    if m < n then m else period - m

let take_last t indices =
  let indices =
    Nx.create Nx.int32 [|Array.length indices|] (Array.map Int32.of_int indices)
  in
  Nx.take ~axis:(-1) ~indices t

(* [pad_signal c x] is [x] extended by the configured boundary mode: [left_width
   c] positions before the signal and [right_width c] after. Only the borders
   are gathered — index arrays never exceed the border widths, and the signal
   body passes through as-is — with the general (multi-reflection) index
   formula, valid for any signal length >= 1. *)
let pad_signal (c : Config.t) x =
  let left = left_width c and right = right_width c in
  if left = 0 && right = 0 then x
  else
    match c.pad with
    | `Constant v ->
        Nx.pad (full_widths x left right) v x
    | (`Reflect | `Edge) as mode ->
        let n = last_dim x in
        let index q =
          match mode with
          | `Reflect ->
              reflect_index n q
          | `Edge ->
              Stdlib.min (n - 1) (Stdlib.max 0 q)
        in
        let border width offset =
          if width = 0 then []
          else [take_last x (Array.init width (fun j -> index (offset + j)))]
        in
        concat_last (border left (-left) @ (x :: border right n))

(* {1 The batched frame computation} *)

(* [to_double t] is [t] as float64, sharing [t] when it already is: the one
   framing gather of [analyse], fused with the promotion to the double
   interior. *)
let to_double : type b. (float, b) Nx.t -> (float, Nx.float64_elt) Nx.t =
 fun t -> match Nx.dtype t with Nx.Float64 -> t | _ -> Nx.cast Nx.float64 t

(* [analyse c cdtype samples count] is frames [0, count) of the padded-stream
   segment [samples], shaped [[...; bins; count]]; [samples] must span at least
   [(count - 1) * hop + fft_size] positions. Shrinking to exactly that span is
   what fixes the frame count: [Nx.stft] frames every whole window the axis
   holds, so a longer segment would analyse past [count]. The analysis is one
   double-precision window multiply (see the header note) and one batched [rfft]
   straight into the caller's witness dtype; per-frame work never routes through
   per-frame Nx calls. *)
let analyse (c : Config.t) cdtype samples count =
  let fft = c.fft_size and hop = c.hop in
  let span = ((count - 1) * hop) + fft in
  let samples =
    if last_dim samples = span then samples else shrink_last samples 0 span
  in
  Nx.swapaxes (-1) (-2)
    (Nx.stft cdtype ~window:fft ~step:hop ~win:c.analysis_window
       (to_double samples) )

(* {1 The streaming kernel} *)

(* State of one streaming analysis. The padded stream is the signal with its
   boundary extension; [pending_rev] holds the unconsumed suffix of that stream
   (in reverse), [skip] counts padded positions still to drop when the hop
   outruns the data. Before the left extension is computable ([started = false])
   raw chunks accumulate in [prelude_rev]; [tail] keeps the last [right + 1] raw
   samples so the right extension is computable at drain time. Retained tensors
   are always kernel-owned copies — incoming chunks are borrowed. *)
type 'a state =
  { cfg: Config.t
  ; left: int
  ; right: int
  ; mutable started: bool
  ; mutable drained: bool
  ; mutable received: int
  ; mutable prelude_rev: (float, 'a) Nx.t list
  ; mutable pending_rev: (float, 'a) Nx.t list
  ; mutable pending_len: int
  ; mutable tail: (float, 'a) Nx.t option
  ; mutable skip: int }

let state_create (cfg : Config.t) =
  { cfg
  ; left= left_width cfg
  ; right= right_width cfg
  ; started= false
  ; drained= false
  ; received= 0
  ; prelude_rev= []
  ; pending_rev= []
  ; pending_len= 0
  ; tail= None
  ; skip= 0 }

let state_reset s =
  s.started <- false ;
  s.drained <- false ;
  s.received <- 0 ;
  s.prelude_rev <- [] ;
  s.pending_rev <- [] ;
  s.pending_len <- 0 ;
  s.tail <- None ;
  s.skip <- 0

(* [process s cdtype extra extra_len] appends [extra] (stream order, possibly
   borrowed) to the pending padded stream and emits every frame that became
   complete. The leftover suffix is copied out, so nothing borrowed is retained
   and the materialized concatenation dies with the call. *)
let process s cdtype extra extra_len =
  let fft = s.cfg.Config.fft_size and hop = s.cfg.Config.hop in
  let total = s.pending_len + extra_len in
  let count = if total < fft then 0 else 1 + ((total - fft) / hop) in
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
    let out = analyse s.cfg cdtype samples count in
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

(* [install_threshold cfg] is the number of raw samples the kernel buffers
   before the left extension is computable: reflection reads the first [left +
   1] samples, while a constant or edge extension only needs the first. *)
let install_threshold (cfg : Config.t) =
  match cfg.Config.pad with
  | `Reflect ->
      left_width cfg + 1
  | `Constant _ | `Edge ->
      1

(* [left_pad s x] is the [left]-position left extension computed from the signal
   head [x]; single reflection suffices because installation waits for [left +
   1] samples in the reflect case. *)
let left_pad s x =
  if s.left = 0 then None
  else
    match s.cfg.Config.pad with
    | `Constant v ->
        Some
          (Nx.full (Nx.dtype x)
             (Array.append
                (Array.sub (Nx.shape x) 0 (Nx.ndim x - 1))
                [|s.left|] )
             v )
    | `Reflect ->
        Some (take_last x (Array.init s.left (fun j -> s.left - j)))
    | `Edge ->
        Some (take_last x (Array.make s.left 0))

(* [install s cdtype x] switches to the started phase: [x] is every raw sample
   so far (at least [left + 1] of them), the left extension is computed, the raw
   tail snapshotted, and the frames that already fit are emitted. *)
let install s cdtype x =
  let n = last_dim x in
  if s.right > 0 then begin
    let keep = Stdlib.min (s.right + 1) n in
    s.tail <- Some (Nx.copy (shrink_last x (n - keep) n))
  end ;
  s.started <- true ;
  s.prelude_rev <- [] ;
  match left_pad s x with
  | None ->
      process s cdtype [x] n
  | Some lp ->
      process s cdtype [lp; x] (s.left + n)

(* [update_tail s chunk] keeps [tail] equal to the last [right + 1] raw samples
   across pushes; only small slices are copied. *)
let update_tail s chunk =
  let keep = s.right + 1 in
  let m = last_dim chunk in
  if m >= keep then s.tail <- Some (Nx.copy (shrink_last chunk (m - keep) m))
  else
    let combined =
      match s.tail with None -> chunk | Some t -> concat_last [t; chunk]
    in
    let cm = last_dim combined in
    let start = Stdlib.max 0 (cm - keep) in
    s.tail <- Some (Nx.copy (shrink_last combined start cm))

(* [right_pad s] is the [right]-position right extension computed from the raw
   tail; defined only when [started] and [right > 0]. *)
let right_pad s =
  let tail = Option.get s.tail in
  let tl = last_dim tail in
  match s.cfg.Config.pad with
  | `Constant v ->
      Nx.full (Nx.dtype tail)
        (Array.append
           (Array.sub (Nx.shape tail) 0 (Nx.ndim tail - 1))
           [|s.right|] )
        v
  | `Reflect ->
      take_last tail (Array.init s.right (fun i -> tl - 2 - i))
  | `Edge ->
      take_last tail (Array.make s.right (tl - 1))

let state_step s cdtype chunk =
  if s.drained then
    invalid_arg
      "step: cannot feed a drained kernel (flush consumed the tail; reset \
       before reusing)" ;
  let m = last_dim chunk in
  if zero_leading chunk then
    invalid_arg
      "step: cannot analyse a chunk with a zero-size leading axis (channels \
       must be at least 1)" ;
  if m = 0 then None
  else begin
    s.received <- s.received + m ;
    if not s.started then
      if s.received >= install_threshold s.cfg then begin
        let parts = List.rev (chunk :: s.prelude_rev) in
        let x = match parts with [one] -> one | many -> concat_last many in
        install s cdtype x
      end
      else begin
        s.prelude_rev <- Nx.copy chunk :: s.prelude_rev ;
        None
      end
    else begin
      if s.right > 0 then update_tail s chunk ;
      if s.skip >= m then begin
        s.skip <- s.skip - m ;
        None
      end
      else begin
        let dropped = s.skip in
        s.skip <- 0 ;
        let chunk =
          if dropped = 0 then chunk else shrink_last chunk dropped m
        in
        process s cdtype [chunk] (m - dropped)
      end
    end
  end

let state_flush s cdtype =
  if s.drained then None
  else begin
    s.drained <- true ;
    let out =
      if not s.started then
        if s.received = 0 then None
        else begin
          let parts = List.rev s.prelude_rev in
          let x = match parts with [one] -> one | many -> concat_last many in
          let padded = pad_signal s.cfg x in
          s.started <- true ;
          s.prelude_rev <- [] ;
          process s cdtype [padded] (last_dim padded)
        end
      else if s.right > 0 then begin
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
          process s cdtype [rp] (r - dropped)
        end
      end
      else None
    in
    s.pending_rev <- [] ;
    s.pending_len <- 0 ;
    out
  end

module Kernel = struct
  type ('a, 'c) t =
    { state: 'a state
    ; cdtype: (Complex.t, 'c) Nx.dtype
    ; dtype: (float, 'a) Nx.dtype }

  let prepare cdtype cfg dtype ~channels ~max_block =
    if channels < 1 then
      invalid_arg
        (Printf.sprintf
           "prepare: cannot analyse %d channels (channels must be at least 1)"
           channels ) ;
    if max_block < 1 then
      invalid_arg
        (Printf.sprintf
           "prepare: cannot accept blocks of %d samples (max_block must be at \
            least 1)"
           max_block ) ;
    {state= state_create cfg; cdtype; dtype}

  let step k chunk = state_step k.state k.cdtype chunk

  let flush k = state_flush k.state k.cdtype

  let reset k = state_reset k.state
end

(* {1 Offline entry points} *)

(* [frameless_spectrum cdtype c x count] is the [count]-frame all-zero spectrum
   of [x]'s leading shape: the result for the cases that evaluate no frame — an
   empty frame range, or a zero-size leading axis (no signals at all). *)
let frameless_spectrum cdtype (c : Config.t) x count =
  Nx.zeros cdtype (Array.append (leading_shape x) [|Config.bins c; count|])

let transform cdtype c x =
  check_rank "transform" x ;
  let n = last_dim x in
  if zero_leading x then frameless_spectrum cdtype c x (frames c ~n)
  else
    let channels = Array.fold_left (fun acc d -> acc * d) 1 (leading_shape x) in
    let k =
      Kernel.prepare cdtype c (Nx.dtype x) ~channels ~max_block:(Stdlib.max 1 n)
    in
    (* [@] evaluates its right operand first: sequence step before flush *)
    let stepped = Kernel.step k x in
    let drained = Kernel.flush k in
    match Option.to_list stepped @ Option.to_list drained with
    | [] ->
        frameless_spectrum cdtype c x 0
    | [one] ->
        one
    | many ->
        concat_last many

let transform_range cdtype c ~p0 ~p1 x =
  check_rank "transform_range" x ;
  let total = frames c ~n:(last_dim x) in
  if p0 < 0 || p0 > p1 || p1 > total then
    invalid_arg
      (Printf.sprintf
         "transform_range: cannot take frames [%d, %d) of a %d-frame transform \
          (the range must satisfy 0 <= p0 <= p1 <= frames)"
         p0 p1 total ) ;
  if p0 = p1 || zero_leading x then frameless_spectrum cdtype c x (p1 - p0)
  else
    let padded = pad_signal c x in
    let hop = Config.hop c and fft = Config.fft_size c in
    let seg = shrink_last padded (p0 * hop) (((p1 - 1) * hop) + fft) in
    analyse c cdtype seg (p1 - p0)

(* [magnitude_pow dtype power z] is [|z| ^ power] in [dtype]: one dtype-first
   [Nx.magnitude], one power. *)
let magnitude_pow dtype power z =
  let m = Nx.magnitude dtype z in
  if Float.equal power 2. then Nx.square m
  else if Float.equal power 1. then m
  else Nx.pow_s m power

(* [spectrum_witness dtype] is the complex storage whose component width matches
   [dtype]: what the real-valued conveniences analyse into when the caller never
   names a complex dtype. *)
type packed_cdtype = Cdtype : (Complex.t, 'c) Nx.dtype -> packed_cdtype

let spectrum_witness : type b. (float, b) Nx.dtype -> packed_cdtype = function
  | Nx.Float64 ->
      Cdtype Nx.complex128
  | Nx.Float32 | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      Cdtype Nx.complex64

let power_spectrum ?(power = 2.) c x =
  check_rank "power_spectrum" x ;
  let dtype = Nx.dtype x in
  let (Cdtype cdtype) = spectrum_witness dtype in
  magnitude_pow dtype power (transform cdtype c x)

(* {1 Pipeline stages} *)

let ceil_div a b = (a + b - 1) / b

(* [stage_latency c] is the involuntary lookahead the pipeline stage declares:
   the geometric analysis lookahead of the configuration, or the reflected left
   border's reach when that is larger ([`Right] alignment with [`Reflect]
   padding reads the first [fft_size - 1] samples before frame 0 can be
   produced). *)
let stage_latency (c : Config.t) =
  Stdlib.max (Config.latency c) (install_threshold c - 1)

(* [frame_bound c latency b] is the most frames one [step] or drained tail can
   emit for chunks of at most [b] samples: the withheld [latency] samples plus
   the chunk, framed at [hop]. It is also what [out_format] declares, so the
   threaded bound (this value, widened by the library to at least [ceil
   latency/hop]) equals it and every emitted chunk honors the threaded format;
   flush output is additionally split to this bound. *)
let frame_bound (c : Config.t) b =
  ceil_div (b + stage_latency c) (Config.hop c) + 1

let threaded_bound (c : Config.t) fmt =
  Option.map (frame_bound c) (Pipeline.Format.max_items fmt)

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

let stage_rate (c : Config.t) = {Pipeline.Rate.num= 1; den= Config.hop c}

let stage_out_format (c : Config.t) fmt =
  let ips =
    Pipeline.Rate.(Pipeline.Format.items_per_second fmt * stage_rate c)
  in
  fmt
  |> Pipeline.Format.with_items_per_second ips
  |> Pipeline.Format.with_max_items (threaded_bound c fmt)

let stage cdtype c =
  let bins = Config.bins c in
  Pipeline.kernel ~latency:(stage_latency c) ~rate:(stage_rate c)
    ~out_format:(stage_out_format c)
    ~flush:(fun s ->
      match state_flush s.state cdtype with
      | None ->
          []
      | Some out ->
          split_frames s.bound out )
    ~reset:(fun s -> state_reset s.state)
    ~concat:(function
      | [] -> Nx.zeros cdtype [|bins; 0|] | parts -> concat_last parts )
    ~prepare:(fun fmt -> {state= state_create c; bound= threaded_bound c fmt})
    ~step:(fun s chunk -> state_step s.state cdtype chunk)
    ()

let power_stage ?(power = 2.) c =
  let bins = Config.bins c in
  (* The element dtype is only witnessed by incoming chunks; [step] records it
     so [concat []] (reached only through [run], where a step always precedes)
     and [flush] can cast back. One stage value is monomorphic in its element
     type, so the shared cell is coherent across prepares. *)
  let witness = ref None in
  Pipeline.kernel ~latency:(stage_latency c) ~rate:(stage_rate c)
    ~out_format:(stage_out_format c)
    ~flush:(fun s ->
      match !witness with
      | Some dtype -> (
          let (Cdtype cdtype) = spectrum_witness dtype in
          match state_flush s.state cdtype with
          | None ->
              []
          | Some out ->
              split_frames s.bound (magnitude_pow dtype power out) )
      | None ->
          (* the kernel emits only after receiving samples, and every received
             chunk records the witness first; an unfed flush still drains the
             state machine but has nothing to emit *)
          ignore (state_flush s.state Nx.complex128) ;
          [] )
    ~reset:(fun s -> state_reset s.state)
    ~concat:(function
      | [] -> (
        match !witness with
        | Some dtype ->
            Nx.zeros dtype [|bins; 0|]
        | None ->
            invalid_arg
              "power_stage: cannot concatenate zero chunks before any chunk \
               fixed the element dtype" )
      | parts ->
          concat_last parts )
    ~prepare:(fun fmt -> {state= state_create c; bound= threaded_bound c fmt})
    ~step:(fun s chunk ->
      let dtype = Nx.dtype chunk in
      witness := Some dtype ;
      let (Cdtype cdtype) = spectrum_witness dtype in
      state_step s.state cdtype chunk |> Option.map (magnitude_pow dtype power) )
    ()
