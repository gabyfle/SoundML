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

module Config = struct
  type t =
    { f_min: float
    ; f_max: float
    ; scale: [`Slaney | `Htk]
    ; norm: [`Slaney | `None]
    ; n_mels: int
    ; sample_rate: int
    ; fft_size: int
    ; weights: (float, Nx.float64_elt) Nx.t
          (* The [[n_mels; bins]] filter matrix, precomputed once in double;
             config-owned and immutable — accessors hand it out, cast to their
             dtype. *)
    ; projection: Ordered.matrix
          (* [weights] prepared for the fixed-order projection {!apply} runs,
             each filter reduced to its triangle's support. *) }

  (* [fft_frequencies sample_rate fft_size bins] is the center frequency of each
     bin in hertz, evaluated as one reciprocal and one multiply per bin — the
     reference arithmetic the parity contract fixes, so the bin frequencies
     match it bit for bit. *)
  let fft_frequencies sample_rate fft_size bins =
    let step =
      1.0 /. (Float.of_int fft_size *. (1.0 /. Float.of_int sample_rate))
    in
    Nx.mul_s (Nx.arange_f Nx.float64 0. (Float.of_int bins) 1.) step

  (* [breakpoints scale f_min f_max count] is [count] frequencies equally spaced
     on the mel scale between [f_min] and [f_max], in hertz. The linspace is one
     precomputed step [delta / div] scaled by the index, the endpoint pinned to
     the exact upper mel — the reference arithmetic the parity contract
     fixes. *)
  let breakpoints scale f_min f_max count =
    let bounds =
      Convert.hz_to_mel ~scale (Nx.create Nx.float64 [|2|] [|f_min; f_max|])
    in
    let mel_min = Nx.item [0] bounds and mel_max = Nx.item [1] bounds in
    let step = (mel_max -. mel_min) /. Float.of_int (count - 1) in
    Convert.mel_to_hz ~scale
      (Nx.create Nx.float64 [|count|]
         (Array.init count (fun i ->
              if i = count - 1 then mel_max
              else (Float.of_int i *. step) +. mel_min ) ) )

  let has_true t = Nx.item [] (Nx.any t)

  (* [weights_of c] is the [[n_mels; bins]] triangular filter matrix, in double
     precision: each filter ramps up from breakpoint [m] to [m + 1] and down to
     [m + 2], sampled at the FFT bin frequencies and clamped at zero. *)
  let weights_of ~f_min ~f_max ~scale ~norm ~n_mels ~sample_rate ~fft_size =
    let bins = (fft_size / 2) + 1 in
    let count = n_mels + 2 in
    let points = breakpoints scale f_min f_max count in
    let steps =
      Nx.sub
        (Nx.slice [Nx.R (1, count)] points)
        (Nx.slice [Nx.R (0, count - 1)] points)
    in
    if has_true (Nx.less_equal steps (Nx.scalar Nx.float64 0.)) then
      invalid_arg
        (Printf.sprintf
           "create: cannot resolve %d mel bands between %g and %g Hz (adjacent \
            breakpoints collapse in double precision)"
           n_mels f_min f_max ) ;
    let ramps =
      Nx.sub
        (Nx.reshape [|count; 1|] points)
        (Nx.reshape [|1; bins|] (fft_frequencies sample_rate fft_size bins))
    in
    let lower =
      Nx.div
        (Nx.neg (Nx.slice [Nx.R (0, n_mels); Nx.R (0, bins)] ramps))
        (Nx.reshape [|n_mels; 1|] (Nx.slice [Nx.R (0, n_mels)] steps))
    in
    let upper =
      Nx.div
        (Nx.slice [Nx.R (2, count); Nx.R (0, bins)] ramps)
        (Nx.reshape [|n_mels; 1|] (Nx.slice [Nx.R (1, n_mels + 1)] steps))
    in
    let weights =
      Nx.maximum (Nx.zeros Nx.float64 [|n_mels; bins|]) (Nx.minimum lower upper)
    in
    let support = Nx.max ~axes:[-1] weights in
    if has_true (Nx.less_equal support (Nx.scalar Nx.float64 0.)) then
      invalid_arg
        (Printf.sprintf
           "create: cannot support %d mel bands with an FFT of size %d (at \
            least one filter spans no FFT bin; raise fft_size or lower n_mels)"
           n_mels fft_size ) ;
    match norm with
    | `None ->
        weights
    | `Slaney ->
        let span =
          Nx.sub
            (Nx.slice [Nx.R (2, count)] points)
            (Nx.slice [Nx.R (0, n_mels)] points)
        in
        Nx.mul weights
          (Nx.reshape [|n_mels; 1|] (Nx.div (Nx.scalar Nx.float64 2.) span))

  let create ?(f_min = 0.) ?f_max ?(scale = `Slaney) ?(norm = `Slaney) ~n_mels
      ~sample_rate ~fft_size () =
    if n_mels < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot build %d mel bands (n_mels must be at least 1)"
           n_mels ) ;
    if sample_rate < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot use a sample rate of %d Hz (sample_rate must be at \
            least 1)"
           sample_rate ) ;
    if fft_size < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot use an FFT of size %d (fft_size must be at least 1)"
           fft_size ) ;
    if not (Float.is_finite f_min && f_min >= 0.) then
      invalid_arg
        (Printf.sprintf
           "create: cannot start the filterbank at %g Hz (f_min must be finite \
            and non-negative)"
           f_min ) ;
    let nyquist = Float.of_int sample_rate /. 2. in
    let f_max = Option.value f_max ~default:nyquist in
    if not (Float.is_finite f_max && f_max > f_min) then
      invalid_arg
        (Printf.sprintf
           "create: cannot span [%g, %g] Hz (f_max must be finite and greater \
            than f_min)"
           f_min f_max ) ;
    if f_max > nyquist then
      (* The offending value prints with full precision: with the usual six
         significant digits an f_max just above Nyquist would display as the
         bound itself, making the message assert an apparently false
         inequality. *)
      invalid_arg
        (Printf.sprintf
           "create: cannot extend the filterbank to %.17g Hz at a sample rate \
            of %d Hz (f_max must not exceed the Nyquist frequency %g)"
           f_max sample_rate nyquist ) ;
    let weights =
      weights_of ~f_min ~f_max ~scale ~norm ~n_mels ~sample_rate ~fft_size
    in
    let projection = Ordered.matrix weights in
    { f_min
    ; f_max
    ; scale
    ; norm
    ; n_mels
    ; sample_rate
    ; fft_size
    ; weights
    ; projection }

  let n_mels t = t.n_mels

  let sample_rate t = t.sample_rate

  let fft_size t = t.fft_size

  let bins t = (t.fft_size / 2) + 1

  let f_min t = t.f_min

  let f_max t = t.f_max

  let scale t = t.scale

  let norm t = t.norm

  let pp fmt t =
    let scale = match t.scale with `Slaney -> "slaney" | `Htk -> "htk" in
    let norm = match t.norm with `Slaney -> "slaney" | `None -> "none" in
    Format.fprintf fmt
      "mel(n_mels=%d, sample_rate=%d, fft_size=%d, f_min=%g, f_max=%g, \
       scale=%s, norm=%s)"
      t.n_mels t.sample_rate t.fft_size t.f_min t.f_max scale norm

  let equal a b =
    Float.equal a.f_min b.f_min
    && Float.equal a.f_max b.f_max
    && a.scale = b.scale && a.norm = b.norm && a.n_mels = b.n_mels
    && a.sample_rate = b.sample_rate
    && a.fft_size = b.fft_size
end

(* Tensors are immutable, so handing out the config-owned matrix — [Nx.cast]
   returns it itself at float64 — cannot let a caller corrupt the config. *)
let filterbank dtype (c : Config.t) = Nx.cast dtype c.Config.weights

let apply (c : Config.t) s =
  let nd = Nx.ndim s in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "apply: cannot project a rank-%d tensor (the mel projection needs \
          [...; bins; frames])"
         nd ) ;
  let bins = Config.bins c in
  let shape = Nx.shape s in
  if shape.(nd - 2) <> bins then
    invalid_arg
      (Printf.sprintf
         "apply: cannot project %d frequency bins through a filterbank built \
          for an FFT of size %d (%d bins)"
         shape.(nd - 2)
         c.Config.fft_size bins ) ;
  let dtype = Nx.dtype s in
  if Array.exists (fun d -> d = 0) shape then begin
    (* No frames, or no signals at all: the projection has nothing to reduce
       over, so produce the broadcast-consistent empty result directly. *)
    let out = Array.copy shape in
    out.(nd - 2) <- c.Config.n_mels ;
    Nx.zeros dtype out
  end
  else
    (* The projection runs in double precision in a fixed association — each
       band adds its bins in ascending order — so a frame projects to the same
       bits alone or among others, whatever the chunking; the cast rounds once,
       at the boundary. *)
    Ordered.project_cast c.Config.projection s

let stage c = Pipeline.stateless (apply c)

(* [power_spectrogram stft c x] is [apply c (Stft.power_spectrum stft x)] at
   float32 and float64, the power projected tile by tile as the transform leaves
   it ({!Ordered.project_power_cast}) instead of materialized whole; [None] for
   the other dtypes and for a rank-zero [x], which take the two-step path. *)
let power_spectrogram (type a) stft (c : Config.t) (x : (float, a) Nx.t) :
    (float, a) Nx.t option =
  if Nx.ndim x < 1 then None
  else
    let rows cdtype = Nx.swapaxes (-1) (-2) (Stft.transform cdtype stft x) in
    match Nx.dtype x with
    | Nx.Float32 ->
        Ordered.project_power_cast c.Config.projection Nx.float32
          (rows Nx.complex64)
    | Nx.Float64 ->
        Ordered.project_power_cast c.Config.projection Nx.float64
          (rows Nx.complex128)
    | Nx.Float16 | Nx.BFloat16 | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
        None
