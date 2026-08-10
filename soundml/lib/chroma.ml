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

type norm = [`Inf | `P of float | `None]

(* {1 Per-frame normalisation}

   Each frame is divided by its own length in the chosen norm. A frame whose
   length underflows the smallest normal of the caller's sample type is left
   alone rather than amplified: dividing a silent frame by its own noise would
   turn rounding into a pitch profile. *)

let check_norm op = function
  | `Inf | `None ->
      ()
  | `P p ->
      if not (Float.is_finite p && p > 0.) then
        invalid_arg
          (Printf.sprintf
             "%s: cannot normalise in the %g-norm (the exponent must be finite \
              and positive)"
             op p )

let smallest_normal : type b. (float, b) Nx.dtype -> float = function
  | Nx.Float64 ->
      Float.min_float
  | Nx.Float32 ->
      0x1p-126
  | Nx.Float16 ->
      0x1p-14
  | Nx.BFloat16 ->
      0x1p-126
  | Nx.Float8_e4m3 ->
      0x1p-6
  | Nx.Float8_e5m2 ->
      0x1p-14

let frame_lengths norm x =
  let magnitude = Nx.abs x in
  match norm with
  | `Inf ->
      Nx.max ~axes:[-2] ~keepdims:true magnitude
  | `P p when Float.equal p 1. ->
      Nx.sum ~axes:[-2] ~keepdims:true magnitude
  | `P p when Float.equal p 2. ->
      Nx.sqrt (Nx.sum ~axes:[-2] ~keepdims:true (Nx.square magnitude))
  | `P p ->
      Nx.pow_s
        (Nx.sum ~axes:[-2] ~keepdims:true (Nx.pow_s magnitude p))
        (1. /. p)

let normalise ~norm ~tiny x =
  match norm with
  | `None ->
      x
  | (`Inf | `P _) as norm ->
      let lengths = frame_lengths norm x in
      let shape = Nx.shape lengths in
      let lengths =
        Nx.where
          (Nx.less lengths (Nx.scalar Nx.float64 tiny))
          (Nx.ones Nx.float64 shape) lengths
      in
      Nx.div x lengths

(* [round_half_even x] rounds to the nearest integer and ties to the even
   neighbour — the rule the reference geometry below is stated in, and not the
   ties-away-from-zero rule of [Float.round]. *)
let round_half_even x =
  let below = Float.floor x in
  let fraction = x -. below in
  if fraction > 0.5 then below +. 1.
  else if fraction < 0.5 then below
  else if Float.equal (Float.rem below 2.) 0. then below
  else below +. 1.

module Config = struct
  type t =
    { n_chroma: int
    ; tuning: float
    ; ctroct: float
    ; octwidth: float option
    ; base_c: bool
    ; sample_rate: int
    ; fft_size: int
    ; weights: (float, Nx.float64_elt) Nx.t
          (* The [[n_chroma; bins]] projection matrix, precomputed once in
             double; config-owned — accessors copy, kernels read without
             copying. *) }

  (* [pitch_positions] is the position of each FFT bin on the chroma scale:
     [n_chroma * log2 (f / (A440 / 16))] with [A440] shifted by [tuning] bins.
     Bin zero carries no frequency, so it takes the position of bin one lowered
     by an octave and a half — far enough below the band that its bump
     contributes nothing. *)
  let pitch_positions ~n_chroma ~tuning ~sample_rate ~fft_size =
    let a440 = 440.0 *. Float.pow 2. (tuning /. Float.of_int n_chroma) /. 16. in
    let step = Float.of_int sample_rate /. Float.of_int fft_size in
    let position j =
      Float.of_int n_chroma *. Float.log2 (Float.of_int j *. step /. a440)
    in
    Array.init fft_size (fun j ->
        if j = 0 then position 1 -. (1.5 *. Float.of_int n_chroma)
        else position j )

  (* [bump_widths positions] is the width of each bin's Gaussian bump in chroma
     units: the local spacing of the pitch positions, floored at one so that the
     widely spaced low bins do not collapse to a spike. The last bin has no
     successor and takes the floor. *)
  let bump_widths positions =
    let n = Array.length positions in
    Array.init n (fun j ->
        if j = n - 1 then 1.
        else Float.max (positions.(j + 1) -. positions.(j)) 1. )

  (* [weights_of] is the [[n_chroma; bins]] projection matrix: bin [j]
     contributes to chroma [c] through a Gaussian in the wrapped distance
     between its pitch position and [c], columns normalised to unit euclidean
     length, optionally damped by an octave-dominance envelope centred on
     [ctroct], and rolled so that row zero is C rather than A. *)
  let weights_of ~n_chroma ~tuning ~ctroct ~octwidth ~base_c ~sample_rate
      ~fft_size =
    let bins = (fft_size / 2) + 1 in
    let positions = pitch_positions ~n_chroma ~tuning ~sample_rate ~fft_size in
    let widths = bump_widths positions in
    let half = round_half_even (Float.of_int n_chroma /. 2.) in
    let chroma = Float.of_int n_chroma in
    let weights = Array.make (n_chroma * bins) 0. in
    for c = 0 to n_chroma - 1 do
      for j = 0 to bins - 1 do
        let d = positions.(j) -. Float.of_int c in
        let wrapped =
          Float.rem (d +. half +. (10. *. chroma)) chroma
          |> fun v -> (if v < 0. then v +. chroma else v) -. half
        in
        let spread = 2. *. wrapped /. widths.(j) in
        weights.((c * bins) + j) <- Float.exp (-0.5 *. spread *. spread)
      done
    done ;
    (* Column normalisation, in the euclidean norm the feature path fixes. *)
    for j = 0 to bins - 1 do
      let sum = ref 0. in
      for c = 0 to n_chroma - 1 do
        let v = weights.((c * bins) + j) in
        sum := !sum +. (v *. v)
      done ;
      let length = Float.sqrt !sum in
      let length = if length < Float.min_float then 1. else length in
      for c = 0 to n_chroma - 1 do
        weights.((c * bins) + j) <- weights.((c * bins) + j) /. length
      done
    done ;
    Option.iter
      (fun octwidth ->
        for j = 0 to bins - 1 do
          let offset = ((positions.(j) /. chroma) -. ctroct) /. octwidth in
          let envelope = Float.exp (-0.5 *. offset *. offset) in
          for c = 0 to n_chroma - 1 do
            weights.((c * bins) + j) <- weights.((c * bins) + j) *. envelope
          done
        done )
      octwidth ;
    let rolled =
      if not base_c then weights
      else
        let shift = 3 * (n_chroma / 12) in
        Array.init (n_chroma * bins) (fun i ->
            let c = i / bins and j = i mod bins in
            weights.(((c + shift) mod n_chroma * bins) + j) )
    in
    Nx.create Nx.float64 [|n_chroma; bins|] rolled

  let create ?(n_chroma = 12) ?(tuning = 0.) ?(ctroct = 5.)
      ?(octwidth = Some 2.) ?(base_c = true) ~sample_rate ~fft_size () =
    if n_chroma < 1 then
      invalid_arg
        (Printf.sprintf
           "create: cannot build %d chroma bands (n_chroma must be at least 1)"
           n_chroma ) ;
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
    if not (Float.is_finite tuning) then
      invalid_arg
        (Printf.sprintf
           "create: cannot shift the scale by %g bins (tuning must be finite)"
           tuning ) ;
    if not (Float.is_finite ctroct) then
      invalid_arg
        (Printf.sprintf
           "create: cannot centre the octave envelope at %g (ctroct must be \
            finite)"
           ctroct ) ;
    Option.iter
      (fun w ->
        if not (Float.is_finite w && w > 0.) then
          invalid_arg
            (Printf.sprintf
               "create: cannot use an octave envelope of half-width %g \
                (octwidth must be finite and positive)"
               w ) )
      octwidth ;
    let weights =
      weights_of ~n_chroma ~tuning ~ctroct ~octwidth ~base_c ~sample_rate
        ~fft_size
    in
    {n_chroma; tuning; ctroct; octwidth; base_c; sample_rate; fft_size; weights}

  let n_chroma t = t.n_chroma

  let tuning t = t.tuning

  let ctroct t = t.ctroct

  let octwidth t = t.octwidth

  let base_c t = t.base_c

  let sample_rate t = t.sample_rate

  let fft_size t = t.fft_size

  let bins t = (t.fft_size / 2) + 1

  let pp fmt t =
    let octwidth fmt = function
      | None ->
          Format.pp_print_string fmt "none"
      | Some w ->
          Format.fprintf fmt "%g" w
    in
    Format.fprintf fmt
      "chroma(n_chroma=%d, sample_rate=%d, fft_size=%d, tuning=%g, ctroct=%g, \
       octwidth=%a, base_c=%b)"
      t.n_chroma t.sample_rate t.fft_size t.tuning t.ctroct octwidth t.octwidth
      t.base_c

  let equal a b =
    a.n_chroma = b.n_chroma
    && Float.equal a.tuning b.tuning
    && Float.equal a.ctroct b.ctroct
    && ( match (a.octwidth, b.octwidth) with
      | None, None ->
          true
      | Some x, Some y ->
          Float.equal x y
      | _ ->
          false )
    && a.base_c = b.base_c
    && a.sample_rate = b.sample_rate
    && a.fft_size = b.fft_size
end

(* [Nx.cast] copies even onto the same dtype, so the config-owned matrix never
   escapes: mutating a returned filterbank cannot corrupt the config. *)
let filterbank dtype (c : Config.t) = Nx.cast dtype c.Config.weights

(* [project op ~mismatch weights bands s] is the batched product of a [[bands;
   bins]] projection against a [[...; bins; frames]] spectrum, in double
   precision. It is [Empty shape] when some axis of [s] has size zero: the
   product has nothing to reduce over, and [shape] is the broadcast-consistent
   result. *)
type product = Empty of int array | Product of (float, Nx.float64_elt) Nx.t

let project op ~mismatch weights bands s =
  let nd = Nx.ndim s in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot project a rank-%d tensor (the projection needs [...; \
          bins; frames])"
         op nd ) ;
  let shape = Nx.shape s in
  let bins = Nx.dim 1 weights in
  if shape.(nd - 2) <> bins then invalid_arg (mismatch shape.(nd - 2) bins) ;
  if Array.exists (fun d -> d = 0) shape then begin
    let out = Array.copy shape in
    out.(nd - 2) <- bands ;
    Empty out
  end
  else Product (Nx.matmul weights (Nx.cast Nx.float64 s))

let apply ?(norm = `Inf) (c : Config.t) s =
  check_norm "apply" norm ;
  let dtype = Nx.dtype s in
  let mismatch got bins =
    Printf.sprintf
      "apply: cannot project %d frequency bins through a matrix built for an \
       FFT of size %d (%d bins)"
      got c.Config.fft_size bins
  in
  match project "apply" ~mismatch c.Config.weights c.Config.n_chroma s with
  | Empty shape ->
      Nx.zeros dtype shape
  | Product raw ->
      Nx.cast dtype (normalise ~norm ~tiny:(smallest_normal dtype) raw)

(* {1 Constant-Q projection}

   Folding a constant-Q ladder onto pitch classes is a 0/1 assignment: every bin
   belongs to exactly one class, the [bins_per_octave / n_chroma] bins of a step
   are merged, the merge window is centred on the step rather than starting at
   it, and the whole assignment is rotated so that the ladder's lowest bin lands
   on its own pitch class. *)

let projection_matrix ~n_chroma ~bins_per_octave ~n_bins ~fmin =
  let merge = bins_per_octave / n_chroma in
  (* MIDI numbers count twelve semitones to the octave whatever the chroma
     resolution, so the rotation is converted at the end. *)
  let midi = (12. *. (Float.log2 fmin -. Float.log2 440.0)) +. 69. in
  let midi = Float.rem (Float.rem midi 12. +. 12.) 12. in
  let shift =
    Float.to_int (round_half_even (midi *. (Float.of_int n_chroma /. 12.)))
  in
  Array.init (n_chroma * n_bins) (fun i ->
      let c = i / n_bins and k = i mod n_bins in
      let source = (((c - shift) mod n_chroma) + n_chroma) mod n_chroma in
      let step = ((k mod bins_per_octave) + (merge / 2)) mod bins_per_octave in
      if step / merge = source then 1.0 else 0.0 )

let check_resolution op ~n_chroma ~bins_per_octave =
  if n_chroma < 1 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot build %d chroma bands (n_chroma must be at least 1)" op
         n_chroma ) ;
  if bins_per_octave mod n_chroma <> 0 then
    invalid_arg
      (Printf.sprintf
         "%s: cannot fold %d bins per octave onto %d chroma bands \
          (bins_per_octave must be an integer multiple of n_chroma)"
         op bins_per_octave n_chroma )

let cq_matrix op ?(n_chroma = 12) (c : Cqt.Config.t) =
  let bins_per_octave = Cqt.Config.bins_per_octave c in
  check_resolution op ~n_chroma ~bins_per_octave ;
  Nx.create Nx.float64
    [|n_chroma; Cqt.Config.n_bins c|]
    (projection_matrix ~n_chroma ~bins_per_octave ~n_bins:(Cqt.Config.n_bins c)
       ~fmin:(Cqt.Config.fmin c) )

let cqt_projection dtype ?n_chroma c =
  Nx.cast dtype (cq_matrix "cqt_projection" ?n_chroma c)

let of_cqt ?(n_chroma = 12) ?(norm = `Inf) c s =
  check_norm "of_cqt" norm ;
  let dtype = Nx.dtype s in
  let weights = cq_matrix "of_cqt" ~n_chroma c in
  let mismatch got bins =
    Printf.sprintf
      "of_cqt: cannot project %d constant-Q bins through a configuration of %d \
       bins"
      got bins
  in
  match project "of_cqt" ~mismatch weights n_chroma s with
  | Empty shape ->
      Nx.zeros dtype shape
  | Product raw ->
      Nx.cast dtype (normalise ~norm ~tiny:(smallest_normal dtype) raw)
