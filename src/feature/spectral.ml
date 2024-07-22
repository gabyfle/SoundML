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

let fft (a : Audio.audio) : (Complex.t, Bigarray.complex32_elt) Audio.G.t =
  Owl.Fft.S.rfft (Audio.data a)

let ifft (ft : (Complex.t, Bigarray.complex32_elt) Audio.G.t) :
    (float, Bigarray.float32_elt) Audio.G.t =
  Owl.Fft.S.irfft ft

module Detrend = struct
  let none = Fun.id
end

type mode = PSD | Angle | Phase | Magnitude | Complex | Default

type side = OneSided | TwoSided

(* Ported and adapted from the spectral helper from matplotlib.mlab All credits
   to the original matplotlib.mlab authors and mainteners *)
let spectral_helper ?(nfft : int = 256) ?(fs : int = 2)
    ?(window = Owl.Signal.hann) ?(detrend : 'a -> 'a = Detrend.none)
    ?(noverlap : int = 0) ?(side = OneSided) ?(mode = Default) ?(pad_to = None)
    ?(scale_by_freq = None) ?(y : Audio.audio option = None) (x : Audio.audio) =
  let same_data =
    match y with Some y -> Audio.data x = Audio.data y | None -> true
  in
  let pad_to = match pad_to with Some x -> x | None -> nfft in
  assert (pad_to >= nfft) ;
  (* TODO: make a nice exception system for the whole library *)
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
      let res_y = Audio.G.transpose res_y in
      let res_y = detrend res_y in
      let res_y = Audio.G.(res_y * window) in
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
          Audio.G.abs_ res ;
          Audio.G.scalar_mul_ Complex.{re= Audio.G.sum' window; im= 0.} res
      | Phase | Angle ->
          let angle = Audio.G.angle res in
          Audio.G.set_slice_ ~out:res [[0; num_freqs - 1]; []] angle res
      | Complex ->
          Audio.G.scalar_div_ Complex.{re= Audio.G.sum' window; im= 0.} res ) ;
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
  (* TODO: implement the unwrap function *)
  (res, freqs)

let specgram ?(nfft : int = 256) ?(fs : int = 2) ?(noverlap : int = 128)
    ?(detrend : 'a -> 'a = Detrend.none) (x : Audio.audio) =
  let res, freqs = spectral_helper ~nfft ~fs ~noverlap ~detrend x in
  (res, freqs)
