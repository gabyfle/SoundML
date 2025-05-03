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
open Bigarray

(** {1 Window Functions}

    This module provides a few commonly used window functions.  *)

val hanning : (float, 'a) kind -> int -> (float, 'a) Audio.G.t
(** 
    [hanning kind n] generates a Hanning window of size [n].

    {2 Parameters}

    @param kind The kind of the Bigarray elements. It must be either [Bigarray.Float32] or [Bigarray.Float64] or it'll raise.
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0 or if the given kind is unsupported. *)

val hamming : (float, 'a) kind -> int -> (float, 'a) Audio.G.t
(** 
    [hamming kind n] generates a Hamming window of size [n].

    {2 Parameters}

    @param kind The kind of the Bigarray elements. It must be either [Bigarray.Float32] or [Bigarray.Float64] or it'll raise.
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0 or if the given kind is unsupported. *)

val blackman : (float, 'a) kind -> int -> (float, 'a) Audio.G.t
(** 
    [blackman kind n] generates a Blackman window of size [n].

    {2 Parameters}

    @param kind The kind of the Bigarray elements. It must be either [Bigarray.Float32] or [Bigarray.Float64] or it'll raise.
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0 or if the given kind is unsupported. *)

val rectangular : (float, 'a) kind -> int -> (float, 'a) Audio.G.t
(** 
    [rectangular kind n] generates a Rectangular window of size [n].

    {2 Parameters}

    @param kind The kind of the Bigarray elements. It must be either [Bigarray.Float32] or [Bigarray.Float64] or it'll raise.
    @param n The size of the window to generate. The size of the window must be greater than 0.
    
    @raise Invalid_argument if [n] is less than or equal to 0 or if the given kind is unsupported. *)
