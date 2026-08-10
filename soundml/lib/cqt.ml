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

(* {1 Window bandwidth}

   The equivalent noise bandwidth of a window, [n * sum w^2 / (sum w)^2] (Harris
   1978, eq. 11), in FFT bins. It sets how far a filter's main lobe reaches
   above its centre frequency, hence the Nyquist admissibility bound. The
   closed-form families are tabulated at their exact limit values; the
   parametric families are measured on 1000 periodic samples, the sampling the
   tabulated numbers themselves were taken at. *)

let enbw_measured window =
  let w = Window.make Nx.float64 ~periodic:true window 1000 in
  let sum = Nx.item [] (Nx.sum w) in
  1000. *. Nx.item [] (Nx.sum (Nx.square w)) /. (sum *. sum)

let enbw : Window.t -> float = function
  | Window.Hann ->
      1.50018310546875
  | Window.Hamming ->
      1.3629455320350348
  | Window.Blackman ->
      1.7269681554262326
  | Window.Blackman_harris ->
      2.0045975283585014
  | Window.Nuttall ->
      1.9763500280946082
  | Window.Bartlett ->
      1.3334961334912805
  | Window.Flat_top ->
      2.7762255046484143
  | Window.Rectangular ->
      1.0
  | (Window.Kaiser _ | Window.Gaussian _ | Window.Tukey _) as window ->
      enbw_measured window

(* {1 Small integer helpers} *)

let ceil_div a b = (a + b - 1) / b

(* [two_factors x] is the exponent of two in [x], zero for non-positive [x]. *)
let two_factors x =
  let rec go n x = if x > 0 && x mod 2 = 0 then go (n + 1) (x / 2) else n in
  go 0 x

(* [next_power_of_two x] is the least power of two at or above the positive real
   [x], [2 ^ ceil (log2 x)]. *)
let next_power_of_two x =
  let e = Float.of_int (Float.to_int (Float.ceil (Float.log2 x))) in
  Float.to_int (Float.pow 2. e)

module Config = struct
  type gamma = [`Constant_q | `Erb | `Fixed of float]

  (* One octave of the recursive plan: [count] filters starting at global bin
     [first], analysed at [hop] on a signal decimated [halvings] times, through
     an [n_fft]-point rectangular-window STFT, then projected by [kernel], the
     [[count; n_fft / 2 + 1]] half-spectrum filter matrix. *)
  type octave =
    { count: int
    ; first: int
    ; n_fft: int
    ; hop: int
    ; halvings: int
    ; stft: Stft.Config.t
    ; kernel: (Complex.t, Nx.complex64_elt) Nx.t }

  type t =
    { fmin: float
    ; bins_per_octave: int
    ; gamma: gamma
    ; tuning: float
    ; filter_scale: float
    ; norm: float
    ; window: Window.t
    ; scale: bool
    ; hop: int
    ; pad: [`Reflect | `Constant of float | `Edge]
    ; n_bins: int
    ; sample_rate: int
    ; n_octaves: int
    ; early: int  (** number of 2:1 decimations applied before octave zero *)
    ; half: Resample.Config.t  (** the one exact 1/2 conversion plan *)
    ; octaves: octave array  (** highest octave first *)
    ; freqs: float array
    ; alphas: float array
    ; gammas: float array
    ; lengths: float array  (** fractional filter lengths at [sample_rate] *)
    ; cutoff: float
    ; divisors: (Complex.t, Nx.complex64_elt) Nx.t option
          (** [[n_bins; 1]] square roots of the filter lengths at the rate the
              transform runs at, when [scale] is set *)
    }

  (* {2 Geometry}

     The centre frequencies form an equal-tempered ladder anchored at [fmin]
     shifted by [tuning] bins: bin [k] of octave [o], [k = o * bpo + j], sits at
     [fmin * 2^(tuning/bpo) * 2^o * 2^(j/bpo)]. The ladder is built octave by
     octave so that a bin and its octave transposition differ by exactly a
     factor of two. *)
  let centre_frequencies ~fmin ~tuning ~bins_per_octave ~n_bins =
    let anchor =
      fmin *. Float.pow 2. (tuning /. Float.of_int bins_per_octave)
    in
    let ratios =
      Array.init bins_per_octave (fun j ->
          Float.pow 2. (Float.of_int j /. Float.of_int bins_per_octave) )
    in
    Array.init n_bins (fun k ->
        let o = k / bins_per_octave and j = k mod bins_per_octave in
        Float.pow 2. (Float.of_int o) *. ratios.(j) *. anchor )

  (* [relative_bandwidths freqs] is the local relative bandwidth [alpha] of each
     centre frequency: with [b] the local octave resolution read off the
     centered difference of [log2 f] (one-sided at the ends), [alpha = (2^(2/b)
     - 1) / (2^(2/b) + 1)]. On a geometric ladder [alpha] is constant and
     satisfies [(1 + alpha) f_(k-1) = (1 - alpha) f_(k+1)], but the difference
     is taken numerically so that the value carries the ladder's own
     rounding. *)
  let relative_bandwidths freqs =
    let n = Array.length freqs in
    let logf = Array.map Float.log2 freqs in
    let bpo =
      Array.init n (fun k ->
          if k = 0 then 1. /. (logf.(1) -. logf.(0))
          else if k = n - 1 then 1. /. (logf.(n - 1) -. logf.(n - 2))
          else 2. /. (logf.(k + 1) -. logf.(k - 1)) )
    in
    Array.map
      (fun b ->
        let r = Float.pow 2. (2. /. b) in
        (r -. 1.) /. (r +. 1.) )
      bpo

  (* A single bin carries no local difference: its relative bandwidth is the
     equal-tempered one, [(r^2 - 1) / (r^2 + 1)] for [r = 2^(1/bpo)]. *)
  let equal_tempered_bandwidth bins_per_octave =
    let r = Float.pow 2. (1. /. Float.of_int bins_per_octave) in
    ((r *. r) -. 1.) /. ((r *. r) +. 1.)

  let gamma_values gamma alphas =
    match gamma with
    | `Constant_q ->
        Array.map (fun _ -> 0.) alphas
    | `Fixed g ->
        Array.map (fun _ -> g) alphas
    | `Erb ->
        Array.map (fun a -> a *. 24.7 /. 0.108) alphas

  (* [filter_lengths_at ~sample_rate] is the fractional support of each filter,
     [Q_k * sr / (f_k + gamma_k / alpha_k)] with [Q_k = filter_scale / alpha_k]:
     the window length that gives bin [k] a bandwidth of [alpha_k * f_k +
     gamma_k]. *)
  let filter_lengths_at ~filter_scale ~freqs ~alphas ~gammas ~sample_rate =
    Array.init (Array.length freqs) (fun k ->
        let q = filter_scale /. alphas.(k) in
        q *. sample_rate /. (freqs.(k) +. (gammas.(k) /. alphas.(k))) )

  (* [filter_cutoff] is the highest frequency any filter's main lobe reaches:
     the centre pushed up by half the window's noise bandwidth in units of [f_k
     / Q_k], plus half the bandwidth offset. All of it must stay below Nyquist
     for the octave recursion to be alias-free. *)
  let filter_cutoff ~enbw ~filter_scale ~freqs ~alphas ~gammas =
    let bound = ref Float.neg_infinity and limit = ref 0 in
    Array.iteri
      (fun k f ->
        let q = filter_scale /. alphas.(k) in
        let v = (f *. (1. +. (0.5 *. enbw /. q))) +. (0.5 *. gammas.(k)) in
        if v > !bound then (
          bound := v ;
          limit := k ) )
      freqs ;
    (!bound, !limit)

  (* {2 Filter bank}

     One octave's half-spectrum kernel. Filter [j] is a complex sinusoid at
     [f_j] windowed over [floor (l/2) + ceil (l/2)] samples centred on zero,
     normalised to unit [norm]-norm, rescaled by [l / n_fft], centre-padded to
     [n_fft] and transformed; only the non-redundant bins are kept, so the
     kernel multiplies a real-signal spectrum directly. *)
  let octave_kernel ~window ~norm ~filter_scale ~freqs ~alphas ~gammas
      ~sample_rate ~n_fft =
    let m = Array.length freqs in
    let unit_norm = Float.equal norm 1. in
    let lengths =
      filter_lengths_at ~filter_scale ~freqs ~alphas ~gammas ~sample_rate
    in
    let re = Array.make (m * n_fft) 0. and im = Array.make (m * n_fft) 0. in
    Array.iteri
      (fun j length ->
        let first = Float.to_int (Float.floor (-.length /. 2.))
        and last = Float.to_int (Float.floor (length /. 2.)) in
        let count = last - first in
        let coefficients =
          Nx.to_array (Window.make Nx.float64 ~periodic:true window count)
        in
        let real = Array.make count 0. and imag = Array.make count 0. in
        let mass = ref 0. in
        for i = 0 to count - 1 do
          let t = Float.of_int (first + i) in
          let angle = t *. 2. *. Float.pi *. freqs.(j) /. sample_rate in
          let a = Float.cos angle *. coefficients.(i)
          and b = Float.sin angle *. coefficients.(i) in
          real.(i) <- a ;
          imag.(i) <- b ;
          let modulus = Float.hypot a b in
          mass := !mass +. if unit_norm then modulus else Float.pow modulus norm
        done ;
        (* An all-but-vanishing filter is left unnormalised rather than
           amplified by a denormal. *)
        let mass = if unit_norm then !mass else Float.pow !mass (1. /. norm) in
        let mass = if mass < Float.min_float then 1. else mass in
        let rescale = length /. Float.of_int n_fft in
        let left = (n_fft - count) / 2 in
        for i = 0 to count - 1 do
          let base = (j * n_fft) + left + i in
          re.(base) <- real.(i) /. mass *. rescale ;
          im.(base) <- imag.(i) /. mass *. rescale
        done )
      lengths ;
    let basis =
      Nx.complex Nx.complex128
        ~re:(Nx.create Nx.float64 [|m; n_fft|] re)
        ~im:(Nx.create Nx.float64 [|m; n_fft|] im)
    in
    Nx.shrink [|(0, m); (0, (n_fft / 2) + 1)|] (Nx.fft ~axis:1 basis)

  (* [octave_fft_size] is the transform length the octave's filters are padded
     to: the least power of two at or above the longest of them. *)
  let octave_fft_size ~filter_scale ~freqs ~alphas ~gammas ~sample_rate =
    let lengths =
      filter_lengths_at ~filter_scale ~freqs ~alphas ~gammas ~sample_rate
    in
    next_power_of_two (Array.fold_left Float.max 0. lengths)

  let sub_array a first count = Array.sub a first count

  (* {2 Validation} *)

  let check_positive_int name value floor =
    if value < floor then
      invalid_arg
        (Printf.sprintf "create: cannot use %s = %d (%s must be at least %d)"
           name value name floor )

  let check_finite name value predicate description =
    if not (Float.is_finite value && predicate value) then
      invalid_arg
        (Printf.sprintf "create: cannot use %s = %g (%s must be %s)" name value
           name description )

  let relabel_window f =
    (* [Window.make] reports domain errors under its own entry point; relabel
       them with this one, which the caller actually called. *)
    try f ()
    with Invalid_argument message ->
      let message =
        match String.index_opt message ':' with
        | Some i ->
            "create" ^ String.sub message i (String.length message - i)
        | None ->
            "create: " ^ message
      in
      invalid_arg message

  let create ?(fmin = 32.70319566257483) ?(bins_per_octave = 12)
      ?(gamma = `Constant_q) ?(tuning = 0.) ?(filter_scale = 1.) ?(norm = 1.)
      ?(window = Window.Hann) ?(scale = true) ?(hop = 512) ?(pad = `Constant 0.)
      ~n_bins ~sample_rate () =
    check_positive_int "n_bins" n_bins 1 ;
    check_positive_int "bins_per_octave" bins_per_octave 1 ;
    check_positive_int "hop" hop 1 ;
    check_positive_int "sample_rate" sample_rate 1 ;
    check_finite "fmin" fmin (fun v -> v > 0.) "finite and positive" ;
    check_finite "tuning" tuning (fun _ -> true) "finite" ;
    check_finite "filter_scale" filter_scale
      (fun v -> v > 0.)
      "finite and positive" ;
    check_finite "norm" norm (fun v -> v > 0.) "finite and positive" ;
    ( match gamma with
    | `Fixed g ->
        check_finite "gamma" g (fun v -> v >= 0.) "finite and non-negative"
    | `Constant_q | `Erb ->
        () ) ;
    let enbw = relabel_window (fun () -> enbw window) in
    let freqs = centre_frequencies ~fmin ~tuning ~bins_per_octave ~n_bins in
    if not (Array.for_all Float.is_finite freqs) then
      invalid_arg
        (Printf.sprintf
           "create: cannot place %d bins from %g Hz at %d bins per octave (the \
            frequency ladder overflows)"
           n_bins fmin bins_per_octave ) ;
    let alphas =
      if n_bins = 1 then [|equal_tempered_bandwidth bins_per_octave|]
      else relative_bandwidths freqs
    in
    let gammas = gamma_values gamma alphas in
    let sr = Float.of_int sample_rate in
    let lengths =
      filter_lengths_at ~filter_scale ~freqs ~alphas ~gammas ~sample_rate:sr
    in
    let cutoff, limit =
      filter_cutoff ~enbw ~filter_scale ~freqs ~alphas ~gammas
    in
    let nyquist = sr /. 2. in
    if cutoff > nyquist then
      invalid_arg
        (Printf.sprintf
           "create: cannot place %d bins from %g Hz at a sample rate of %d Hz \
            (bin %d, centred at %g Hz, reaches %g Hz, above the Nyquist \
            frequency %g)"
           n_bins fmin sample_rate limit freqs.(limit) cutoff nyquist ) ;
    let n_octaves = ceil_div n_bins bins_per_octave in
    let n_filters = Stdlib.min bins_per_octave n_bins in
    (* Early decimation drops the rate to just above what the filters need,
       while the hop keeps a factor of two for every remaining octave. *)
    let early =
      let head =
        Stdlib.max 0
          (Float.to_int (Float.ceil (Float.log2 (nyquist /. cutoff))) - 2)
      in
      let tail = Stdlib.max 0 (two_factors hop - n_octaves + 1) in
      Stdlib.min head tail
    in
    let base_rate = sr /. Float.of_int (1 lsl early) in
    let base_hop = hop / (1 lsl early) in
    let rate = ref base_rate and octave_hop = ref base_hop and halved = ref 0 in
    let plan = ref [] in
    for i = 0 to n_octaves - 1 do
      let first = Stdlib.max 0 (n_bins - (n_filters * (i + 1))) in
      let count = n_bins - (n_filters * i) - first in
      let freqs = sub_array freqs first count
      and alphas = sub_array alphas first count
      and gammas = sub_array gammas first count in
      let n_fft =
        octave_fft_size ~filter_scale ~freqs ~alphas ~gammas ~sample_rate:!rate
      in
      if n_fft < 1 then
        invalid_arg
          (Printf.sprintf
             "create: cannot resolve %d bins from %g Hz with filter_scale = %g \
              (the filters of octave %d span less than one sample)"
             n_bins fmin filter_scale i ) ;
      let kernel =
        relabel_window (fun () ->
            octave_kernel ~window ~norm ~filter_scale ~freqs ~alphas ~gammas
              ~sample_rate:!rate ~n_fft )
      in
      (* Every octave analyses the same band of the decimated signal, so its
         response carries the energy of the whole rate reduction it sits
         under. *)
      let gain = Float.sqrt (base_rate /. !rate) in
      let kernel = Nx.mul_s kernel Complex.{re= gain; im= 0.} in
      let stft =
        Stft.Config.create ~window:Window.Rectangular ~hop:!octave_hop
          ~alignment:`Centered ~pad ~scale:`None ~fft_size:n_fft ()
      in
      plan :=
        {count; first; n_fft; hop= !octave_hop; halvings= !halved; stft; kernel}
        :: !plan ;
      if !octave_hop mod 2 = 0 then begin
        octave_hop := !octave_hop / 2 ;
        rate := !rate /. 2. ;
        incr halved
      end
    done ;
    let octaves = Array.of_list (List.rev !plan) in
    let divisors =
      if not scale then None
      else
        let lengths =
          filter_lengths_at ~filter_scale ~freqs ~alphas ~gammas
            ~sample_rate:base_rate
        in
        let roots =
          Nx.create Nx.float64 [|n_bins; 1|] (Array.map Float.sqrt lengths)
        in
        Some
          (Nx.complex Nx.complex128 ~re:roots
             ~im:(Nx.zeros Nx.float64 [|n_bins; 1|]) )
    in
    { fmin
    ; bins_per_octave
    ; gamma
    ; tuning
    ; filter_scale
    ; norm
    ; window
    ; scale
    ; hop
    ; pad
    ; n_bins
    ; sample_rate
    ; n_octaves
    ; early
    ; half= Resample.Config.create ~quality:`High ~sample_rate:2 ~target:1 ()
    ; octaves
    ; freqs
    ; alphas
    ; gammas
    ; lengths
    ; cutoff
    ; divisors }

  let n_bins t = t.n_bins

  let bins_per_octave t = t.bins_per_octave

  let sample_rate t = t.sample_rate

  let hop t = t.hop

  let fmin t = t.fmin

  let gamma t = t.gamma

  let tuning t = t.tuning

  let filter_scale t = t.filter_scale

  let norm t = t.norm

  let window t = t.window

  let scale t = t.scale

  let pad t = t.pad

  let n_octaves t = t.n_octaves

  let cutoff t = t.cutoff

  let pp_gamma fmt = function
    | `Constant_q ->
        Format.pp_print_string fmt "constant_q"
    | `Erb ->
        Format.pp_print_string fmt "erb"
    | `Fixed g ->
        Format.fprintf fmt "fixed(%g)" g

  let pp fmt t =
    let pad fmt = function
      | `Reflect ->
          Format.pp_print_string fmt "reflect"
      | `Constant v ->
          Format.fprintf fmt "constant(%g)" v
      | `Edge ->
          Format.pp_print_string fmt "edge"
    in
    Format.fprintf fmt
      "cqt(n_bins=%d, bins_per_octave=%d, sample_rate=%d, fmin=%g, hop=%d, \
       gamma=%a, tuning=%g, filter_scale=%g, norm=%g, window=%a, scale=%b, \
       pad=%a)"
      t.n_bins t.bins_per_octave t.sample_rate t.fmin t.hop pp_gamma t.gamma
      t.tuning t.filter_scale t.norm Window.pp t.window t.scale pad t.pad

  let equal a b =
    Float.equal a.fmin b.fmin
    && a.bins_per_octave = b.bins_per_octave
    && ( match (a.gamma, b.gamma) with
      | `Constant_q, `Constant_q | `Erb, `Erb ->
          true
      | `Fixed x, `Fixed y ->
          Float.equal x y
      | _ ->
          false )
    && Float.equal a.tuning b.tuning
    && Float.equal a.filter_scale b.filter_scale
    && Float.equal a.norm b.norm
    && Window.equal a.window b.window
    && a.scale = b.scale && a.hop = b.hop
    && ( match (a.pad, b.pad) with
      | `Reflect, `Reflect | `Edge, `Edge ->
          true
      | `Constant x, `Constant y ->
          Float.equal x y
      | _ ->
          false )
    && a.n_bins = b.n_bins
    && a.sample_rate = b.sample_rate
end

(* {1 The frequency ladder} *)

let frequencies dtype (c : Config.t) =
  Nx.cast dtype (Nx.create Nx.float64 [|c.Config.n_bins|] c.Config.freqs)

let filter_lengths dtype (c : Config.t) =
  Nx.cast dtype (Nx.create Nx.float64 [|c.Config.n_bins|] c.Config.lengths)

(* {1 The frame grid} *)

(* [octave_length c i ~n] is the number of samples octave [i] analyses of a
   length-[n] signal: every decimation halves the length, rounding up. *)
let octave_length (c : Config.t) (o : Config.octave) ~n =
  ceil_div n (1 lsl (c.Config.early + o.Config.halvings))

let frames (c : Config.t) ~n =
  if n < 0 then
    invalid_arg
      (Printf.sprintf
         "frames: cannot analyse a signal of length %d (length must be \
          non-negative)"
         n ) ;
  Array.fold_left
    (fun acc (o : Config.octave) ->
      Stdlib.min acc (Stft.frames o.Config.stft ~n:(octave_length c o ~n)) )
    max_int c.Config.octaves

(* {1 The transform} *)

let leading_shape t =
  let shape = Nx.shape t in
  Array.sub shape 0 (Array.length shape - 1)

let check_rank op t =
  if Nx.ndim t < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot analyse a rank-zero tensor (the time axis must exist)" op )

(* The decimation runs at the caller's sample type: the recursion is where the
   transform spends its samples, and keeping float32 audio in float32 there is
   the point of the plan. *)
let check_resampling_dtype : type a. string -> (float, a) Nx.dtype -> unit =
 fun op dtype ->
  match dtype with
  | Nx.Float32 | Nx.Float64 ->
      ()
  | _ ->
      invalid_arg
        (Format.asprintf
           "%s: cannot decimate a %a signal (the octave recursion resamples, \
            which needs float32 or float64 audio)"
           op Nx.pp_dtype dtype )

(* [decimate c y] is [y] at half its rate: the exact 1/2 conversion, run through
   the resampler's dense offline form. That form carries no partitioning law,
   which costs this module nothing — the transform is offline and evaluates the
   whole signal at once — and it is the same filter, the same compensated group
   delay and the same output length as the streaming executor. *)
let decimate (c : Config.t) y = Resample.apply_gemm c.Config.half y

(* [halve c y] is [decimate] with the energy convention of the octave recursion:
   rescaled so that a band common to both rates keeps its magnitude. *)
let halve (c : Config.t) y = Nx.div_s (decimate c y) (Float.sqrt 0.5)

(* [resamples c] is [true] iff the plan decimates anywhere: an odd hop stops the
   recursion from halving after its first octave, and a configuration whose
   filters already fill the band needs no early stage either. *)
let resamples (c : Config.t) =
  let octaves = c.Config.octaves in
  c.Config.early > 0
  || Array.exists
       (fun (o : Config.octave) -> o.Config.hop mod 2 = 0)
       (Array.sub octaves 0 (Array.length octaves - 1))

let shrink_last t stop =
  let nd = Nx.ndim t in
  Nx.shrink
    (Array.init nd (fun i -> if i = nd - 1 then (0, stop) else (0, Nx.dim i t)))
    t

let transform : type c a.
       (Complex.t, c) Nx.dtype
    -> Config.t
    -> (float, a) Nx.t
    -> (Complex.t, c) Nx.t =
 fun cdtype c x ->
  check_rank "transform" x ;
  let n = Nx.dim (Nx.ndim x - 1) x in
  let count = frames c ~n in
  if Array.exists (fun d -> d = 0) (leading_shape x) || count = 0 then
    Nx.zeros cdtype
      (Array.append (leading_shape x) [|c.Config.n_bins; Stdlib.max 0 count|])
  else begin
    if resamples c then check_resampling_dtype "transform" (Nx.dtype x) ;
    let y = ref x in
    if c.Config.early > 0 then begin
      for _ = 1 to c.Config.early do
        y := decimate c !y
      done ;
      let factor = Float.sqrt (Float.of_int (1 lsl c.Config.early)) in
      (* The early stage carries the same rescaling as one octave step per
         halving; without length normalisation downstream it carries it twice,
         once for the rate and once for the filter support. *)
      y := Nx.mul_s !y factor ;
      if not c.Config.scale then y := Nx.mul_s !y factor
    end ;
    let last = Array.length c.Config.octaves - 1 in
    (* Octave zero holds the highest bins and runs at the highest rate; the
       signal is decimated between octaves, so the responses must be produced in
       order. *)
    let parts = ref [] in
    for i = 0 to last do
      let o = c.Config.octaves.(i) in
      let d = Stft.transform Nx.complex128 o.Config.stft !y in
      parts := shrink_last (Nx.matmul o.Config.kernel d) count :: !parts ;
      if i < last && o.Config.hop mod 2 = 0 then y := halve c !y
    done ;
    (* Stacking from the lowest octave up rebuilds the ladder in ascending
       order, and a partial bottom octave contributes only the bins it
       carries. *)
    let parts = !parts in
    let stacked =
      match parts with [one] -> one | parts -> Nx.concatenate ~axis:(-2) parts
    in
    let stacked =
      match c.Config.divisors with
      | None ->
          stacked
      | Some divisors ->
          Nx.div stacked divisors
    in
    Nx.cast cdtype stacked
  end

(* [magnitude_pow dtype power z] is [|z| ^ power] in [dtype]: one dtype-first
   [Nx.magnitude], one power. *)
let magnitude_pow dtype power z =
  let m = Nx.magnitude dtype z in
  if Float.equal power 2. then Nx.square m
  else if Float.equal power 1. then m
  else Nx.pow_s m power

(* [spectrum_witness dtype] is the complex storage whose component width matches
   [dtype]: what the real-valued convenience analyses into when the caller never
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
