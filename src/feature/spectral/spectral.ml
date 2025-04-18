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

module Filterbank = Filterbank

module Config = struct
  type t =
    { nfft: int
    ; window: Window.t
    ; fs: int
    ; noverlap: int
    ; side: side
    ; mode: mode
    ; pad_to: int option
    ; scale_by_freq: bool option }

  let default : t =
    { nfft= 2048
    ; window= Window.default
    ; fs= 2
    ; noverlap= 0
    ; side= OneSided
    ; mode= Default
    ; pad_to= None
    ; scale_by_freq= None }
end

(* Ported and adapted from the spectral helper from matplotlib.mlab All credits
   to the original matplotlib.mlab authors and mainteners *)
let spectral_helper ?(config : Config.t = Config.default)
    ?(detrend : 'a -> 'a = Detrend.none)
    ?(y : (float, Bigarray.float32_elt) Audio.G.t option = None)
    (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let window = Window.get_window config.window in
  let same_data = match y with Some y -> x = y | None -> true in
  let pad_to = match config.pad_to with Some x -> x | None -> config.nfft in
  assert (pad_to >= config.nfft) ;
  assert (config.noverlap < config.nfft) ;
  let mode = match config.mode with Default -> PSD | _ -> config.mode in
  if (not same_data) && mode = PSD then assert false ;
  (* we're making copies of the data from x and y to then use in place padding
     and operations *)
  let x = Audio.G.copy x in
  let y = match y with Some y -> Audio.G.copy y | None -> Audio.G.copy x in
  (* We're making sure the arrays are at least of size nfft *)
  let xshp = Audio.G.shape x in
  ( if Array.get xshp 0 < config.nfft then
      let delta = config.nfft - Array.get xshp 0 in
      (* we're doing this in place in hope to gain a little bit of speed *)
      Audio.G.pad_ ~out:x ~v:0. [[0; delta - 1]; [0; 0]] x ) ;
  (* only padding the y array if it has been provided *)
  ( if not same_data then
      let yshp = Audio.G.shape y in
      if Array.get yshp 0 < config.nfft then
        let delta = config.nfft - Array.get yshp 0 in
        Audio.G.pad_ ~out:y ~v:0. [[0; delta - 1]; [0; 0]] y ) ;
  let scale_by_freq =
    match mode with
    | PSD -> (
      match config.scale_by_freq with Some x -> x | None -> true )
    | _ -> (
      match config.scale_by_freq with Some x -> x | None -> false )
  in
  let num_freqs, scaling_factor, freq_center =
    match (config.side, pad_to mod 2) with
    | OneSided, 1 ->
        ((pad_to / 2) + 1, 2., 0)
    | OneSided, _ ->
        ((pad_to + 1) / 2, 2., 0)
    | TwoSided, 1 ->
        (pad_to, 1., pad_to / 2)
    | TwoSided, _ ->
        (pad_to, 1., (pad_to - 1) / 2)
  in
  let window = window config.nfft |> Audio.G.cast_d2s in
  let window =
    Audio.G.reshape
      Audio.G.(window * ones Bigarray.float32 [|config.nfft|])
      [|-1; 1|]
  in
  let res =
    Audio.G.slide ~window:config.nfft ~step:(config.nfft - config.noverlap) x
    |> Audio.G.transpose
  in
  let res = detrend res in
  let res = Audio.G.(res * window) in
  let res = Owl.Fft.S.rfft res ~axis:0 in
  ( if not same_data then (
      let res_y =
        Audio.G.slide ~window:config.nfft
          ~step:(config.nfft - config.noverlap)
          y
      in
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
    let slice = if config.nfft mod 2 = 0 then [[1; -1]; []] else [[1]; []] in
    let gslice = Audio.G.get_slice slice res in
    Audio.G.mul_scalar_ ~out:gslice gslice Complex.{re= scaling_factor; im= 0.} ;
    Audio.G.set_slice slice res gslice ;
    if scale_by_freq then (
      let window = Audio.G.abs window in
      Audio.G.div_scalar_ ~out:res res
        Complex.{re= float_of_int config.fs; im= 0.} ;
      let n = Audio.G.sum' (Audio.G.pow_scalar window (float_of_int 2)) in
      Audio.G.div_scalar_ ~out:res res Complex.{re= n; im= 0.} )
    else
      let window = Audio.G.abs window in
      let n = Float.pow (Audio.G.sum' window) 2. in
      Audio.G.div_scalar_ ~out:res res Complex.{re= n; im= 0.} ) ;
  let res =
    match config.side with
    | TwoSided ->
        (* this will center the freqs range around 0 *)
        Utils.roll res (-freq_center)
    | OneSided ->
        res
  in
  res

let specgram ?(config : Config.t = Config.default)
    ?(detrend : 'a -> 'a = Detrend.none)
    (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let res = spectral_helper ~config ~detrend x in
  let res = Audio.G.re_c2s res in
  res

let complex_specgram ?(config : Config.t = Config.default)
    ?(detrend : 'a -> 'a = Detrend.none)
    (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let res = spectral_helper ~config ~detrend x in
  res

let phase_specgram ?(config : Config.t = {Config.default with mode= Phase})
    (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let res = spectral_helper ~config x in
  let res = Audio.G.re_c2s res in
  Utils.unwrap ~axis:0 res

let magnitude_specgram
    ?(config : Config.t = {Config.default with mode= Magnitude})
    (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let res = spectral_helper ~config x in
  let res = Audio.G.re_c2s res in
  res

let mel_specgram ?(config : Config.t = {Config.default with mode= Magnitude})
    ?(nmels : int = 128) ?(fmin : float = 0.) ?(fmax : float option = None)
    ?(htk : bool = false) ?(norm : Filterbank.norm option = None)
    ?(sample_rate : int = 44100) (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let res = spectral_helper ~config x in
  let res = Audio.G.re_c2s res in
  let weights =
    Filterbank.mel ~fmax ~htk ~sample_rate ~nfft:config.nfft ~nmels ~fmin ~norm
  in
  let res = Audio.G.dot weights res in
  res

let mfcc ?(config : Config.t = Config.default) ?(nmfcc : int = 20)
    ?(nmels : int = 128) ?(fmin : float = 0.) ?(fmax : float option = None)
    ?(htk : bool = false) ?norm
    ?(dct_type : Owl.Fft.Generic.ttrig_transform = II) ?(lifter : int = 0)
    ?(sample_rate : int = 44100) (x : (float, Bigarray.float32_elt) Audio.G.t) =
  assert (lifter >= 0) ;
  let x = mel_specgram ~config ~nmels ~fmin ~fmax ~htk ~norm ~sample_rate x in
  let x = Utils.Convert.power_to_db (RefFloat 1.0) x in
  let norm =
    match dct_type with
    | II | III ->
        Owl.Fft.Generic.Ortho
    | _ ->
        Owl.Fft.Generic.Backward
  in
  let m = Owl.Fft.Generic.dct ~axis:(-2) ~norm ~ttype:dct_type x in
  let ndims = Audio.G.num_dims m in
  let slices =
    let open Owl in
    List.init ndims (fun i ->
        (* Second-to-last dimension: take first n_mfcc values *)
        if i = ndims - 2 then R [0; nmfcc - 1]
        else R [] (* All other dimensions: take everything *) )
  in
  let m = Audio.G.get_fancy slices m in
  match lifter with
  | 0 ->
      m
  | n ->
      let ndims = Audio.G.num_dims m in
      let pi = Owl.Const.pi in
      let n = float_of_int n in
      let li =
        Audio.G.(
          let range = linspace (kind m) 1. (1. +. float_of_int nmfcc) nmfcc in
          sin (pi $* range /$ n) )
      in
      let li_expanded =
        Audio.G.(1. $+ (n /. 2. $* Audio.G.expand ~hi:true li ndims))
      in
      Audio.G.(m * li_expanded)

let rms ?(window : int = 2048) ?(step : int = 512)
    (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let x = Audio.G.slide ~step ~window x in
  let x = Audio.G.abs2 x in
  let x = Audio.G.mean x in
  x

let zero_crossings ?(threshold = 1e-10) ?(zero_pos = true)
    (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let op =
    match zero_pos with true -> Audio.G.( <.$ ) | false -> Audio.G.( <=.$ )
  in
  (* everything below this threshold will be zeroed *)
  let mask = Audio.G.(abs x >.$ threshold) in
  let x = Audio.G.(x * mask) in
  let x = op x 0. in
  (* we're dealing with 1D arrays *)
  let shape = Audio.G.shape x in
  let result = Audio.G.zeros (Audio.G.kind x) shape in
  (* problem using diff here, as we want to keep the same dim *)
  let x0 = Audio.G.get_slice [[]; [0; shape.(1) - 2]] x in
  let x1 = Audio.G.get_slice [[]; [1; shape.(1) - 1]] x in
  let crossing = Audio.G.(( !=. ) x0 x1) in
  Audio.G.set_slice [[]; [1; shape.(1) - 1]] result crossing ;
  result

let zero_crossing_rate ?(window = 2048) ?(hop_length = 512) ?(threshold = 1e-10)
    ?(zero_pos = false) (x : (float, Bigarray.float32_elt) Audio.G.t) =
  let frames = Audio.G.slide ~window ~step:(window - hop_length) x in
  let crossings = zero_crossings ~threshold ~zero_pos frames in
  (* Calculate mean and normalize by frame length - 1 *)
  let rate = Audio.G.(mean ~axis:1 ~keep_dims:true crossings |> transpose) in
  rate
