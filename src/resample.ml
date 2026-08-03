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

(* The exact-rational polyphase resampler. The geometry, once and for all:

   [target / sample_rate] reduces to [L / M]. The prototype lowpass runs at the
   interpolated rate [L * sample_rate]; its length is rounded up to [2*K*L + 1]
   so the linear-phase group delay is exactly [K] input samples. Output [i] sits
   at input time [i * M / L]: with [t = i * M], its phase is [t mod L] and it
   reads the [2K + 1] input samples [t/L - K .. t/L + K] (delay already
   compensated). The bank stores one row per phase, phase-major, each row the
   prototype decimated by [L] and reversed so the executor reads input windows
   forward.

   The C executor (resample_stubs.c) computes each output as one dot product
   over [history ++ chunk] with a fixed summation order, so offline equals
   streaming bit for bit on every partitioning — the pipeline law is structural,
   not tested-in. OCaml orchestrates at chunk granularity only: exact integer
   phase state, no float accumulator anywhere. *)

type spec = {attenuation: float; passband: float}

type quality = [`Fast | `High | `Best | `Custom of spec]

(* {1 Small shape helpers (shared conventions with the other modules)} *)

let last_dim t = Nx.dim (Nx.ndim t - 1) t

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let check_rank op t =
  if Nx.ndim t < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot resample a rank-zero tensor (the time axis must exist)" op )

let check_dtype : type a. string -> (float, a) Nx.dtype -> unit =
 fun op dt ->
  match dt with
  | Nx.Float32 ->
      ()
  | Nx.Float64 ->
      ()
  | dt ->
      invalid_arg
        (Stdlib.Format.asprintf
           "%s: cannot resample %a audio (the executor carries float32 and \
            float64)"
           op Nx.pp_dtype dt )

(* [ceil_pos a b] is [ceil (a / b)] for [b > 0], clamped at zero for [a <=
   0]. *)
let ceil_pos a b = if a <= 0 then 0 else ((a - 1) / b) + 1

(* {1 Filter design}

   Kaiser-windowed sinc, designed by the standard empirical Kaiser relations
   (the [kaiserord] convention): the transition band runs from [passband * min
   (1/L, 1/M)] to [min (1/L, 1/M)] — in units of the interpolated-rate Nyquist —
   the cutoff sits mid-transition, and attenuation fixes both the window shape
   and the length. The prototype is designed in float64 once per configuration
   and cast to the kernel dtype at prepare. *)

let kaiser_beta att =
  if att > 50. then 0.1102 *. (att -. 8.7)
  else if att > 21. then
    (0.5842 *. ((att -. 21.) ** 0.4)) +. (0.07886 *. (att -. 21.))
  else 0.

(* [width] is the transition width as a fraction of the rate the filter runs at,
   Nyquist = 1. *)
let kaiser_numtaps att width =
  let n = Float.ceil (((att -. 7.95) /. 2.285 /. (Float.pi *. width)) +. 1.) in
  (* odd length: linear phase with an integral group delay *)
  if Float.rem n 2. = 0. then n +. 1. else n

(* [1 / k^2] for the [bessel_i0] series: the per-term division dominates the
   series cost (the window evaluates [I0] once per tap of a filter that can run
   tens of thousands of taps), so the reciprocals are precomputed once. The
   series at the betas admitted here converges within a few dozen terms; the
   table covers far more, and the fallback division keeps the function total. *)
let i0_inv_sq =
  Array.init 129 (fun k -> if k = 0 then 0. else 1. /. Float.of_int (k * k))

let bessel_i0 x =
  let hx2 = 0.25 *. x *. x in
  let rec go k term sum =
    let term =
      if k < 129 then term *. hx2 *. i0_inv_sq.(k)
      else term *. hx2 /. Float.of_int (k * k)
    in
    let sum' = sum +. term in
    if term <= Float.epsilon *. sum' || k > 1000 then sum'
    else go (k + 1) term sum'
  in
  go 1 1. 1.

(* [design_prototype ~l ~k ~fc ~beta] is the [2*K*L + 1]-tap prototype: a
   windowed sinc with cutoff [fc] (Nyquist units of the interpolated rate),
   normalised so the full filter sums to [L] — unit passband gain after the
   zero-stuffing of upsampling. Both factors are even in [z], so only the right
   half is evaluated and the left half mirrors it — the symmetry is exact, not a
   rounding shortcut. *)
let design_prototype ~l ~k ~fc ~beta =
  let mid = k * l in
  let n = (2 * mid) + 1 in
  let i0_beta = bessel_i0 beta in
  let h = Array.make n 0. in
  for i = mid to n - 1 do
    let z = Float.of_int (i - mid) in
    let s =
      if i = mid then fc else Float.sin (Float.pi *. fc *. z) /. (Float.pi *. z)
    in
    let r = z /. Float.of_int mid in
    let v = s *. (bessel_i0 (beta *. Float.sqrt (1. -. (r *. r))) /. i0_beta) in
    h.(i) <- v ;
    h.(n - 1 - i) <- v
  done ;
  let sum = Array.fold_left ( +. ) 0. h in
  let gain = Float.of_int l /. sum in
  Array.iteri (fun i v -> h.(i) <- v *. gain) h ;
  h

(* [bank_of_prototype ~l ~k h] lays the prototype out phase-major, [L] rows of
   [2K + 1] taps, each row reversed so the executor's dot product walks the
   input window forward: row [p], slot [s] holds [h.(p + (2K - s) * L)] (zero
   where that index passes the end). *)
let bank_of_prototype ~l ~k h =
  let taps = (2 * k) + 1 in
  let n = Array.length h in
  let b = Array.make (l * taps) 0. in
  for p = 0 to l - 1 do
    for s = 0 to taps - 1 do
      let idx = p + ((taps - 1 - s) * l) in
      if idx < n then b.((p * taps) + s) <- h.(idx)
    done
  done ;
  b

(* [gemm_bank ~l ~m ~k bank] is the [P; L] matrix [G] of the tensor formulation
   ([apply_gemm]), with [P = 2K + 1 + floor((L-1)*M/L)] and [G[q, r] =
   bank[(r*M) mod L, q - floor(r*M/L)]] where in range and zero elsewhere:
   column [r] is the phase-[(r*M) mod L] row of the polyphase bank, shifted down
   by the integer input advance [floor(r*M/L)] of output [r] within the block.
   [bank] is the phase-major array of [bank_of_prototype]. *)
let gemm_bank ~l ~m ~k bank =
  let taps = (2 * k) + 1 in
  let p_len = taps + ((l - 1) * m / l) in
  let g = Array.make (p_len * l) 0. in
  for r = 0 to l - 1 do
    let d = r * m / l and ph = r * m mod l in
    for s = 0 to taps - 1 do
      g.(((d + s) * l) + r) <- bank.((ph * taps) + s)
    done
  done ;
  Nx.create Nx.float64 [|p_len; l|] g

(* {1 Configuration} *)

let bank_budget_bytes = 8 * 1024 * 1024

type config =
  { sample_rate: int
  ; target: int
  ; quality: quality
  ; l: int (* output samples per L/M block, gcd-reduced *)
  ; m: int (* input samples per L/M block *)
  ; k: int (* group delay in input samples; 0 for the identity *)
  ; prototype: (float, Nx.float64_elt) Nx.t (* [2*K*L + 1] *)
  ; bank: (float, Nx.float64_elt) Nx.t (* [L; 2K + 1], rows reversed *)
  ; gemm: (float, Nx.float64_elt) Nx.t Lazy.t
        (* [P; L], built on the first [apply_gemm] and shared by every later
           call on this config; forcing is not domain-safe, like every other
           piece of state in this module *) }

let is_identity c = c.l = 1 && c.m = 1 && c.k = 0

let spec_of_quality = function
  | `Fast ->
      {attenuation= 100.; passband= 0.913}
  | `High ->
      {attenuation= 126.; passband= 0.913}
  | `Best ->
      {attenuation= 175.; passband= 0.913}
  | `Custom spec ->
      spec

let pp_bytes fmt bytes =
  let scaled v unit =
    if Float.is_integer v then Stdlib.Format.fprintf fmt "%.0f %s" v unit
    else Stdlib.Format.fprintf fmt "%.1f %s" v unit
  in
  if bytes >= 1024. *. 1024. *. 1024. then
    scaled (bytes /. (1024. *. 1024. *. 1024.)) "GB"
  else if bytes >= 1024. *. 1024. then scaled (bytes /. (1024. *. 1024.)) "MB"
  else scaled (bytes /. 1024.) "KB"

module Config = struct
  type t = config

  let rec gcd a b = if b = 0 then a else gcd b (a mod b)

  let create ?(quality = `High) ~sample_rate ~target () =
    if sample_rate < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot resample from %d Hz (sample_rate must be at least 1)"
           sample_rate ) ;
    if target < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot resample to %d Hz (target must be at least 1)" target ) ;
    let {attenuation; passband} = spec_of_quality quality in
    if
      not
        ( Float.is_finite attenuation
        && attenuation >= 40. && attenuation <= 200. )
    then
      invalid_arg
        (Printf.sprintf
           "create: cannot design a filter with %g dB of stop-band rejection \
            (attenuation must be finite, in [40, 200])"
           attenuation ) ;
    if not (Float.is_finite passband && passband >= 0.5 && passband <= 0.99)
    then
      invalid_arg
        (Printf.sprintf
           "create: cannot preserve %g of the band (passband must be finite, \
            in [0.5, 0.99])"
           passband ) ;
    let g = gcd sample_rate target in
    let l = target / g and m = sample_rate / g in
    if l = 1 && m = 1 then
      { sample_rate
      ; target
      ; quality
      ; l= 1
      ; m= 1
      ; k= 0
      ; prototype= Nx.create Nx.float64 [|1|] [|1.|]
      ; bank= Nx.create Nx.float64 [|1; 1|] [|1.|]
      ; gemm= lazy (Nx.create Nx.float64 [|1; 1|] [|1.|]) }
    else begin
      let width = (1. -. passband) /. Float.of_int (Stdlib.max l m) in
      let ntaps = kaiser_numtaps attenuation width in
      (* K in float first: the budget check must precede any conversion that
         could overflow or any array that could not be allocated *)
      let k_f = Float.ceil ((ntaps -. 1.) /. (2. *. Float.of_int l)) in
      let bank_bytes = Float.of_int l *. ((2. *. k_f) +. 1.) *. 8. in
      if bank_bytes > Float.of_int bank_budget_bytes then
        invalid_arg
          (Stdlib.Format.asprintf
             "create: cannot resample %d Hz to %d Hz (%d phases need a %a \
              bank; the budget is %a) hint: near-unity conversion is \
              clock-drift correction, which the fixed-ratio resampler does not \
              do"
             sample_rate target l pp_bytes bank_bytes pp_bytes
             (Float.of_int bank_budget_bytes) ) ;
      let k = Stdlib.max 1 (Float.to_int k_f) in
      let fc = (1. +. passband) /. (2. *. Float.of_int (Stdlib.max l m)) in
      let beta = kaiser_beta attenuation in
      let h = design_prototype ~l ~k ~fc ~beta in
      let taps = (2 * k) + 1 in
      let bank = Nx.create Nx.float64 [|l; taps|] (bank_of_prototype ~l ~k h) in
      { sample_rate
      ; target
      ; quality
      ; l
      ; m
      ; k
      ; prototype= Nx.create Nx.float64 [|Array.length h|] h
      ; bank
      ; gemm= lazy (gemm_bank ~l ~m ~k (Nx.to_array bank)) }
    end

  let sample_rate c = c.sample_rate

  let target c = c.target

  let quality c = c.quality

  let rate c = {Pipeline.Rate.num= c.l; den= c.m}

  let latency c = c.k

  let output_latency c =
    let num = c.k * c.l in
    if num = 0 then {Pipeline.Rate.num= 0; den= 1}
    else
      let g = gcd num c.m in
      {Pipeline.Rate.num= num / g; den= c.m / g}

  let output_frames c ~n =
    if n < 0 then
      invalid_arg
        (Printf.sprintf
           "output_frames: cannot resample a signal of length %d (length must \
            be non-negative)"
           n ) ;
    if n > 0 && n > Stdlib.max_int / c.l then
      invalid_arg
        (Printf.sprintf
           "output_frames: cannot resample a signal of length %d (n * %d \
            overflows)"
           n c.l ) ;
    ceil_pos (n * c.l) c.m

  let prototype dtype c = Nx.cast dtype (Nx.copy c.prototype)

  let pp fmt c =
    let quality fmt = function
      | `Fast ->
          Stdlib.Format.pp_print_string fmt "fast"
      | `High ->
          Stdlib.Format.pp_print_string fmt "high"
      | `Best ->
          Stdlib.Format.pp_print_string fmt "best"
      | `Custom {attenuation; passband} ->
          Stdlib.Format.fprintf fmt "custom(%g dB, %g)" attenuation passband
    in
    if is_identity c then
      Stdlib.Format.fprintf fmt "resample(%d Hz, identity)" c.sample_rate
    else
      Stdlib.Format.fprintf fmt
        "resample(%d -> %d Hz, quality=%a, L/M=%d/%d, taps=%dx%d, latency=%d)"
        c.sample_rate c.target quality c.quality c.l c.m c.l
        ((2 * c.k) + 1)
        c.k

  let equal a b =
    a.sample_rate = b.sample_rate
    && a.target = b.target
    &&
    match (a.quality, b.quality) with
    | `Fast, `Fast | `High, `High | `Best, `Best ->
        true
    | `Custom x, `Custom y ->
        Float.equal x.attenuation y.attenuation
        && Float.equal x.passband y.passband
    | _ ->
        false
end

(* {1 The C seam} *)

(* One call per chunk; the stub does all slicing internally, validates every
   extent against the arrays it was actually handed, and releases the runtime
   lock around the bulk work (hence no [@@noalloc]). Argument order: bank,
   history, scratch, input, output, n, n_out, channels, K, L, M, phase0, s0,
   is_flush. *)
external resample_step_c :
     (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> int
  -> bool
  -> unit = "soundml_resample_step_bc" "soundml_resample_step"

let array1_of t = Nx_buffer.to_bigarray1 (Nx.to_buffer t)

(* {1 Incremental kernel} *)

module Kernel = struct
  type 'a t =
    { cfg: config
    ; dtype: (float, 'a) Nx.dtype
    ; channels: int
    ; max_block: int
    ; bank: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
    ; hist: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* [channels * 2K], planar *)
    ; scratch: (float, 'a, Bigarray.c_layout) Bigarray.Array1.t
          (* [2K + max max_block K], one lane shared across channels *)
    ; mutable fed: int (* input samples consumed so far *)
    ; mutable emitted: int (* output samples emitted so far *)
    ; mutable drained: bool
    ; mutable leading: int array (* leading shape of the last chunk *) }

  (* [ready c fed] is the number of outputs computable once [fed] input samples
     have arrived: output [i] needs input [floor (i*M/L) + K], so every [i] with
     [floor (i*M/L) <= fed - 1 - K] — that is [ceil ((fed - K) * L / M)] of
     them. Feeding the [K] virtual zeros of [flush] makes this exactly [ceil
     (fed * L / M)]: the deterministic-length contract, with no truncation
     step. *)
  let ready c fed = ceil_pos ((fed - c.k) * c.l) c.m

  let prepare cfg dtype ~channels ~max_block =
    check_dtype "prepare" dtype ;
    if channels < 1 then
      invalid_arg
        (Printf.sprintf
           "prepare: cannot resample %d channels (channels must be at least 1)"
           channels ) ;
    if max_block < 1 then
      invalid_arg
        (Printf.sprintf
           "prepare: cannot accept blocks of %d samples (max_block must be at \
            least 1)"
           max_block ) ;
    let hist_len = channels * 2 * cfg.k in
    let scratch_len = (2 * cfg.k) + Stdlib.max max_block cfg.k in
    { cfg
    ; dtype
    ; channels
    ; max_block
    ; bank= array1_of (Nx.cast dtype cfg.bank)
    ; hist= array1_of (Nx.zeros dtype [|hist_len|])
    ; scratch= array1_of (Nx.zeros dtype [|scratch_len|])
    ; fed= 0
    ; emitted= 0
    ; drained= false
    ; leading= [|channels|] }

  (* [emit k ~n ~n_out ~is_flush x] runs the executor over [n] input samples
     ([x] when streaming, virtual zeros when draining) and is the [n_out]
     freshly computed output samples shaped [leading ++ [n_out]] — planar,
     exactly the C layout. The phase state of the first output [i = emitted] is
     exact integer arithmetic: [t = i * M], phase [t mod L], window start [t / L
     + K - fed] in scratch coordinates. *)
  let emit k ~n ~n_out ~is_flush x =
    let t = k.emitted * k.cfg.m in
    let phase0 = t mod k.cfg.l in
    let s0 = (t / k.cfg.l) + k.cfg.k - k.fed in
    if n_out = 0 then begin
      (* nothing computable yet: the call only threads the history through *)
      resample_step_c k.bank k.hist k.scratch x k.scratch n 0 k.channels k.cfg.k
        k.cfg.l k.cfg.m phase0 s0 is_flush ;
      None
    end
    else begin
      let out = Nx.empty k.dtype (Array.append k.leading [|n_out|]) in
      resample_step_c k.bank k.hist k.scratch x (array1_of out) n n_out
        k.channels k.cfg.k k.cfg.l k.cfg.m phase0 s0 is_flush ;
      k.emitted <- k.emitted + n_out ;
      Some out
    end

  let step k chunk =
    if k.drained then
      invalid_arg
        "step: cannot feed a drained kernel (flush consumed the tail; reset \
         before reusing)" ;
    check_rank "step" chunk ;
    let n = last_dim chunk in
    if n > k.max_block then
      invalid_arg
        (Printf.sprintf "step: cannot feed a %d-sample chunk (max_block is %d)"
           n k.max_block ) ;
    let lead = leading_shape chunk in
    let channels = Array.fold_left ( * ) 1 lead in
    if channels <> k.channels then
      invalid_arg
        (Printf.sprintf
           "step: cannot feed %d-channel chunks (the kernel was prepared for \
            %d channels)"
           channels k.channels ) ;
    k.leading <- lead ;
    if n = 0 then None
    else begin
      let n_out = ready k.cfg (k.fed + n) - k.emitted in
      let out = emit k ~n ~n_out ~is_flush:false (array1_of chunk) in
      k.fed <- k.fed + n ;
      out
    end

  let flush k =
    if k.drained then None
    else begin
      k.drained <- true ;
      let n_out = ready k.cfg (k.fed + k.cfg.k) - k.emitted in
      if n_out = 0 then None else emit k ~n:k.cfg.k ~n_out ~is_flush:true k.hist
    end

  let reset k =
    Bigarray.Array1.fill k.hist 0. ;
    k.fed <- 0 ;
    k.emitted <- 0 ;
    k.drained <- false ;
    k.leading <- [|k.channels|]
end

(* {1 Offline} *)

let apply c x =
  check_rank "apply" x ;
  check_dtype "apply" (Nx.dtype x) ;
  if is_identity c then x
  else
    let n = last_dim x in
    let lead = leading_shape x in
    let total = Config.output_frames c ~n in
    let channels = Array.fold_left ( * ) 1 lead in
    if channels = 0 || n = 0 then
      Nx.zeros (Nx.dtype x) (Array.append lead [|total|])
    else begin
      let k = Kernel.prepare c (Nx.dtype x) ~channels ~max_block:n in
      (* sequence step before flush *)
      let stepped = Kernel.step k x in
      let drained = Kernel.flush k in
      match Option.to_list stepped @ Option.to_list drained with
      | [] ->
          Nx.zeros (Nx.dtype x) (Array.append lead [|total|])
      | [one] ->
          one
      | many ->
          Nx.concatenate ~axis:(-1) many
    end

(* {1 Offline, tensor formulation}

   The GEMM surface: the same conversion as [apply], written as one dense tensor
   expression over the same config-owned filter. Outputs are grouped into blocks
   of [L] (one full phase cycle): block [b] holds outputs [b*L + r], [r] in
   [0..L-1], and output [b*L + r] reads the input window starting at [b*M +
   floor(r*M/L) - K]. One patch of [P = 2K + 1 + floor((L-1)*M/L)] samples per
   block covers all [L] windows, so the whole conversion is patches [n_frames;
   P] times a [P; L] arrangement of the bank — the strided formulation whose
   arithmetic redundancy is [P / taps]. The matrix product sums in whatever
   order the backend blocks it, which is exactly why this surface is documented
   as numerically distinct and carries no partitioning law. The [P; L] matrix
   itself is [gemm_bank] above, built once per config on the first call. *)

let apply_gemm c x =
  check_rank "apply_gemm" x ;
  check_dtype "apply_gemm" (Nx.dtype x) ;
  if is_identity c then x
  else
    let dtype = Nx.dtype x in
    let n = last_dim x in
    let lead = leading_shape x in
    let total = Config.output_frames c ~n in
    let channels = Array.fold_left ( * ) 1 lead in
    if channels = 0 || n = 0 then Nx.zeros dtype (Array.append lead [|total|])
    else begin
      let taps = (2 * c.k) + 1 in
      let p_len = taps + ((c.l - 1) * c.m / c.l) in
      let frames = ceil_pos total c.l in
      (* [K] zeros on the left (the delay-compensated window of output 0 starts
         at input [-K]); on the right, exactly enough for the last block's
         patch. The clamp only ever discards surplus signal, and the frame axis
         is cut back to [frames] below either way. *)
      let right = Stdlib.max 0 (((frames - 1) * c.m) + p_len - (c.k + n)) in
      let pad_spec =
        Array.init
          (Array.length lead + 1)
          (fun i -> if i = Array.length lead then (c.k, right) else (0, 0))
      in
      let padded = Nx.pad pad_spec 0. x in
      let patches =
        Nx.extract_patches ~kernel_size:[|p_len|] ~stride:[|c.m|] ~dilation:[|1|]
          ~padding:[|(0, 0)|]
          padded
      in
      (* [lead ++ [P; frames']] with [frames' >= frames]; keep [frames] *)
      let rank = Nx.ndim patches in
      let patches =
        Nx.shrink
          (Array.init rank (fun i ->
               if i = rank - 1 then (0, frames) else (0, Nx.dim i patches) ) )
          patches
      in
      let axes =
        List.init rank (fun i ->
            if i = rank - 2 then rank - 1
            else if i = rank - 1 then rank - 2
            else i )
      in
      let y =
        Nx.matmul
          (Nx.transpose ~axes patches)
          (Nx.cast dtype (Lazy.force c.gemm))
      in
      let y = Nx.reshape (Array.append lead [|frames * c.l|]) y in
      Nx.shrink
        (Array.init (Nx.ndim y) (fun i ->
             if i = Nx.ndim y - 1 then (0, total) else (0, Nx.dim i y) ) )
        y
    end

(* {1 Pipeline stage} *)

type 'a stage_state =
  { cfg: config
  ; channels: int
  ; bound: int option
  ; mutable kern: 'a Kernel.t option
        (* prepared on the first chunk: the chunk witnesses the element dtype,
           which the dynamic format only carries existentially *) }

let stage_kernel s chunk =
  match s.kern with
  | Some k ->
      k
  | None ->
      let max_block =
        match s.bound with
        | Some b ->
            b
        | None ->
            (* unbounded chunks: the offline driver pushes exactly once *)
            Stdlib.max 1 (last_dim chunk)
      in
      let k =
        Kernel.prepare s.cfg (Nx.dtype chunk) ~channels:s.channels ~max_block
      in
      s.kern <- Some k ;
      k

let stage cfg =
  let witness = ref None in
  let concat = function
    | [] -> (
      match !witness with
      | Some (dtype, leading) ->
          Nx.zeros dtype (Array.append leading [|0|])
      | None ->
          invalid_arg
            "stage: cannot concatenate zero chunks before any chunk fixed the \
             element dtype" )
    | parts ->
        Nx.concatenate ~axis:(-1) parts
  in
  let prepare fmt =
    let ips = Pipeline.Format.items_per_second fmt in
    if not (Pipeline.Rate.equal ips {num= cfg.sample_rate; den= 1}) then
      invalid_arg
        (Stdlib.Format.asprintf
           "prepare: cannot resample a stream at %a items/s (the configuration \
            converts from %d Hz)"
           Pipeline.Rate.pp ips cfg.sample_rate ) ;
    { cfg
    ; channels= Pipeline.Format.channels fmt
    ; bound= Pipeline.Format.max_items fmt
    ; kern= None }
  in
  let step s chunk =
    witness := Some (Nx.dtype chunk, leading_shape chunk) ;
    Kernel.step (stage_kernel s chunk) chunk
  in
  let flush s =
    match s.kern with None -> [] | Some k -> Option.to_list (Kernel.flush k)
  in
  let reset s = Option.iter Kernel.reset s.kern in
  if is_identity cfg then
    (* transparent to the accounting: latency 0, rate 1/1, format untouched —
       but the chunk-ownership contract still demands a fresh output, so the
       identity kernel copies through the same executor *)
    Pipeline.kernel ~flush ~reset ~concat ~prepare ~step ()
  else
    let rate = {Pipeline.Rate.num= cfg.l; den= cfg.m} in
    let out_format fmt =
      let ips = Pipeline.Rate.(Pipeline.Format.items_per_second fmt * rate) in
      let bound =
        Option.map
          (fun b -> ceil_pos (b * cfg.l) cfg.m + 1)
          (Pipeline.Format.max_items fmt)
      in
      fmt
      |> Pipeline.Format.with_items_per_second ips
      |> Pipeline.Format.with_max_items bound
    in
    Pipeline.kernel ~latency:cfg.k ~rate ~out_format ~flush ~reset ~concat
      ~prepare ~step ()
