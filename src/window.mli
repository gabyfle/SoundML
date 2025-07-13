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

(** {1 Window Functions}

    This module provides a few commonly used window functions.  *)

(** The type of window functions. *)
type window = [`Hanning | `Hamming | `Blackman | `Boxcar]

val get :
  window -> (float, 'b) Nx.dtype -> ?fftbins:bool -> int -> (float, 'b) Nx.t
(** 
    [get window dtype n] generates a window of size [n] using the given window function type.

    {2 Parameters}

    @param window The type of window to generate. 
    @param dtype The datatype of the tensor elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val cosine_sum :
     ?fftbins:bool
  -> (float, 'b) Nx.dtype
  -> float array
  -> int
  -> (float, 'b) Nx.t
(**
    [cosine_sum datatype coeffs n] generates a cosine-sum window of size [n] using the given coefficients.

    {2 Parameters}
    @param dtype The datatype of the tensor elements.
    @param coeffs The coefficients of the cosine-sum window. The length of the coefficients array must be greater than 0.
    @param n The size of the window to generate. The size of the window must be greater than 0.

    {2 Raises}
    @raise Invalid_argument if [n] is less than or equal to 0.
    @raise Invalid_argument if the length of [coeffs] is less than 1.

    {2 References}
    @see https://en.wikipedia.org/wiki/Window_function#Cosine-sum_windows
    @see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.general_cosine.html *)

val hanning : ?fftbins:bool -> (float, 'b) Nx.dtype -> int -> (float, 'b) Nx.t
(** 
    [hanning dtype n] generates a Hanning window of size [n].

    {2 Parameters}

    @param dtype The datatype of the tensor elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val hamming : ?fftbins:bool -> (float, 'b) Nx.dtype -> int -> (float, 'b) Nx.t
(** 
    [hamming dtype n] generates a Hamming window of size [n].

    {2 Parameters}

    @param dtype The datatype of the tensor elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val blackman : ?fftbins:bool -> (float, 'b) Nx.dtype -> int -> (float, 'b) Nx.t
(** 
    [blackman dtype n] generates a Blackman window of size [n].

    {2 Parameters}

    @param dtype The datatype of the tensor elements. 
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)

val boxcar : ('a, 'b) Nx.dtype -> int -> ('a, 'b) Nx.t
(** 
    [boxcar dtype n] generates a Rectangular window of size [n].

    {2 Parameters}

    @param dtype The datatype of the tensor elements.
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0. *)
