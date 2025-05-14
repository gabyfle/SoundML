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

open Types

(** {1 Window Functions}

    This module provides a few commonly used window functions.  *)

(** The type of window functions. *)
type window = [`Hanning | `Hamming | `Blackman | `Boxcar]

val get :
     window
  -> ('a, 'b) precision
  -> ?fftbins:bool
  -> int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(** 
    [get window precision n] generates a window of size [n] using the given window function type.

    {2 Parameters}

    @param window The type of window to generate. 
    @param precision The precision of the Bigarray elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val cosine_sum :
     ?fftbins:bool
  -> ('a, 'b) precision
  -> float array
  -> int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(**
    [cosine_sum precision coeffs n] generates a cosine-sum window of size [n] using the given coefficients.

    {2 Parameters}
    @param precision The precision of the Bigarray elements.
    @param coeffs The coefficients of the cosine-sum window. The length of the coefficients array must be greater than 0.
    @param n The size of the window to generate. The size of the window must be greater than 0.

    {2 Raises}
    @raise Invalid_argument if [n] is less than or equal to 0.
    @raise Invalid_argument if the length of [coeffs] is less than 1.

    {2 References}
    @see https://en.wikipedia.org/wiki/Window_function#Cosine-sum_windows
    @see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_cosine.html *)

val hanning :
     ?fftbins:bool
  -> ('a, 'b) precision
  -> int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(** 
    [hanning precision n] generates a Hanning window of size [n].

    {2 Parameters}

    @param precision The precision of the Bigarray elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val hamming :
     ?fftbins:bool
  -> ('a, 'b) precision
  -> int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(** 
    [hamming precision n] generates a Hamming window of size [n].

    {2 Parameters}

    @param precision The precision of the Bigarray elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val blackman :
     ?fftbins:bool
  -> ('a, 'b) precision
  -> int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(** 
    [blackman precision n] generates a Blackman window of size [n].

    {2 Parameters}

    @param precision The precision of the Bigarray elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val boxcar :
     ?fftbins:bool
  -> ('a, 'b) precision
  -> int
  -> (float, 'a) Owl_dense_ndarray.Generic.t
(** 
    [boxcar precision n] generates a Rectangular window of size [n].

    {2 Parameters}

    @param precision The precision of the Bigarray elements.
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)
