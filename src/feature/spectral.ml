(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023                                                       *)
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

type mode =
  | PSD (* Power Spectral Density *)
  | Angle (* Phase angle *)
  | Phase (* Alias for Angle *)
  | Magnitude (* Magnitude spectrum *)
  | Complex (* Complex spectrum *)
  | Default (* Default to PSD *)

type side =
  | OneSided (* Single-sided spectrum *)
  | TwoSided (* Double-sided spectrum *)

module Window = struct
  type t =
    | Hann
    | Hamming
    | Blackman
    | Rectangle
    | Custom of (int -> (float, Bigarray.float64_elt) Audio.G.t)

  let get_window = function
    | Hann ->
        Owl.Signal.hann
    | Hamming ->
        Owl.Signal.hamming
    | Blackman ->
        Owl.Signal.blackman
    | Rectangle ->
        fun n -> Audio.G.ones Bigarray.float64 [|n|]
    | Custom f ->
        f

  let default = Hann
end

module Detrend = struct
  let none = Fun.id

  let constant x =
    let mean = Audio.G.mean' x in
    let trend =
      Audio.G.scalar_mul mean
        (Audio.G.ones Bigarray.complex32 (Audio.G.shape x))
    in
    Audio.G.sub x trend

  let linear x =
    let dim = (Audio.G.shape x |> Array.get) 0 in
    assert (dim <= 1) ;
    (* only works for <=1 dimensional arrays *)
    match dim with
    | 0 ->
        Audio.G.zeros Bigarray.complex32 [|0|]
    | _ ->
        let y = Audio.G.linspace Bigarray.float32 0. (float_of_int dim) dim in
        let y = Audio.G.cast_s2c y in
        let cov = Utils.cov ?b:(Some y) ~a:x in
        let b =
          Complex.div (Audio.G.get cov [|0; 1|]) (Audio.G.get cov [|0; 0|])
        in
        let a =
          Complex.sub (Audio.G.mean' x) (Complex.mul b (Audio.G.mean' y))
        in
        Audio.G.sub x (Audio.G.add_scalar (Audio.G.mul_scalar y b) a)
end

(* Ported and adapted from the spectral helper from matplotlib.mlab All credits
   to the original matplotlib.mlab authors and mainteners *)
let spectral_helper ?(nfft : int = 256) ?(fs : int = 2) ?(window = Window.Hann)
    ?(detrend : 'a -> 'a = Detrend.none) ?(noverlap : int = 0)
    ?(side = OneSided) ?(mode = Default) ?(pad_to = None)
    ?(scale_by_freq = None) ?(y : Audio.audio option = None) (x : Audio.audio) =
  let ttime = Unix.gettimeofday () in
  let window = Window.get_window window in
  let same_data =
    match y with Some y -> Audio.data x = Audio.data y | None -> true
  in
  let pad_to = match pad_to with Some x -> x | None -> nfft in
  assert (pad_to >= nfft) ;
  assert (noverlap < nfft) ;
  let mode = match mode with Default -> PSD | _ -> mode in
  if (not same_data) && mode = PSD then assert false ;
  (* we're making copies of the data from x and y to then use in place padding
     and operations *)
  let x = Audio.G.copy (Audio.data x) in
  let y =
    match y with
    | Some y ->
        Audio.G.copy (Audio.data y)
    | None ->
        Audio.G.copy x
  in
  (* We're making sure the arrays are at least of size nfft *)
  let xshp = Audio.G.shape x in
  ( if Array.get xshp 0 < nfft then
      let delta = nfft - Array.get xshp 0 in
      (* we're doing this in place in hope to gain a little bit of speed *)
      Audio.G.pad_ ~out:x ~v:0. [[0; delta - 1]; [0; 0]] x ) ;
  (* only padding the y array if it has been provided *)
  ( if not same_data then
      let yshp = Audio.G.shape y in
      if Array.get yshp 0 < nfft then
        let delta = nfft - Array.get yshp 0 in
        Audio.G.pad_ ~out:y ~v:0. [[0; delta - 1]; [0; 0]] y ) ;
  let scale_by_freq =
    match mode with
    | PSD -> (
      match scale_by_freq with Some x -> x | None -> true )
    | _ -> (
      match scale_by_freq with Some x -> x | None -> false )
  in
  let num_freqs, scaling_factor, freq_center =
    match (side, pad_to mod 2) with
    | OneSided, 1 ->
        ((pad_to / 2) + 1, 2., 0)
    | OneSided, _ ->
        ((pad_to + 1) / 2, 2., 0)
    | TwoSided, 1 ->
        (pad_to, 1., pad_to / 2)
    | TwoSided, _ ->
        (pad_to, 1., (pad_to - 1) / 2)
  in
  let window = window nfft |> Audio.G.cast_d2s in
  let window =
    Audio.G.reshape Audio.G.(window * ones Bigarray.float32 [|nfft|]) [|-1; 1|]
  in
  let res =
    Audio.G.slide ~window:nfft ~step:(nfft - noverlap) x |> Audio.G.transpose
  in
  let res = detrend res in
  let res = Audio.G.(res * window) in
  let res = Owl.Fft.S.rfft res ~axis:0 in
  let freqs = Utils.fftfreq pad_to (1. /. float_of_int fs) in
  ( if not same_data then (
      let res_y = Audio.G.slide ~window:nfft ~step:(nfft - noverlap) y in
      Audio.G.transpose_ ~out:res_y res_y ;
      let res_y = detrend res_y in
      Audio.G.mul_ ~out:res_y res_y window ;
      let res_y = Owl.Fft.S.rfft res_y ~axis:0 in
      let len = Array.get (Audio.G.shape res_y) 0 in
      Audio.G.pad_ ~out:res_y ~v:Complex.zero [[0; pad_to - len]; [0; 0]] res_y ;
      Audio.G.get_slice_ ~out:res_y [[0; num_freqs - 1]; []] res_y ;
      Audio.G.conj_ ~out:res res ;
      Audio.G.mul_ ~out:res res res_y )
    else
      match mode with
      | PSD | Default ->
          let conj = Audio.G.conj res in
          Audio.G.mul_ ~out:res conj res
      | Magnitude ->
          Audio.G.abs_ ~out:res res ;
          Audio.G.scalar_div_ ~out:res
            Complex.{re= Audio.G.sum' window; im= 0.}
            res
      | Phase | Angle ->
          let angle = Audio.G.angle res in
          Audio.G.set_slice_ ~out:res [[0; num_freqs]; []] angle res
      | Complex ->
          Audio.G.scalar_div_ ~out:res
            Complex.{re= Audio.G.sum' window; im= 0.}
            res ) ;
  if mode = PSD then (
    let slice = if nfft mod 2 = 0 then [[1; -1]; []] else [[1]; []] in
    let gslice = Audio.G.get_slice slice res in
    Audio.G.mul_scalar_ ~out:gslice gslice Complex.{re= scaling_factor; im= 0.} ;
    Audio.G.set_slice slice res gslice ;
    if scale_by_freq then (
      let window = Audio.G.abs window in
      Audio.G.div_scalar_ ~out:res res Complex.{re= float_of_int fs; im= 0.} ;
      let n = Audio.G.sum' (Audio.G.pow_scalar window (float_of_int 2)) in
      Audio.G.div_scalar_ ~out:res res Complex.{re= n; im= 0.} )
    else
      let window = Audio.G.abs window in
      let n = Float.pow (Audio.G.sum' window) 2. in
      Audio.G.div_scalar_ ~out:res res Complex.{re= n; im= 0.} ) ;
  let res, freqs =
    match side with
    | TwoSided ->
        (* this will center the freqs range around 0 *)
        (Utils.roll res (-freq_center), Utils.roll freqs (-freq_center))
    | OneSided ->
        (res, freqs)
  in
  Owl.Log.debug "Elapsed time for specgram compute: %f"
    (Unix.gettimeofday () -. ttime) ;
  (res, freqs)

let specgram ?(nfft : int = 256) ?(window : Window.t = Window.default)
    ?(fs : int = 2) ?(noverlap : int = 128) ?(detrend : 'a -> 'a = Detrend.none)
    (x : Audio.audio) =
  let res, freqs = spectral_helper ~nfft ~fs ~window ~noverlap ~detrend x in
  let res = Owl.Dense.Ndarray.Generic.re_c2s res in
  (res, freqs)

let complex_specgram ?(nfft : int = 256) ?(window : Window.t = Window.default)
    ?(fs : int = 2) ?(noverlap : int = 128) ?(detrend : 'a -> 'a = Detrend.none)
    (x : Audio.audio) =
  let res, freqs = spectral_helper ~nfft ~fs ~window ~noverlap ~detrend x in
  (res, freqs)

let phase_specgram ?(nfft : int = 256) ?(window : Window.t = Window.default)
    ?(fs : int = 2) ?(noverlap : int = 128) (x : Audio.audio) =
  let res, freqs = spectral_helper ~nfft ~fs ~window ~noverlap x ~mode:Phase in
  let res = Owl.Dense.Ndarray.Generic.re_c2s res in
  (Utils.unwrap ~axis:0 res, freqs)

let magnitude_specgram ?(nfft : int = 256) ?(window : Window.t = Window.default)
    ?(fs : int = 2) ?(noverlap : int = 128) (x : Audio.audio) =
  let res, freqs =
    spectral_helper ~nfft ~fs ~window ~noverlap x ~mode:Magnitude
  in
  let res = Owl.Dense.Ndarray.Generic.re_c2s res in
  (res, freqs)
