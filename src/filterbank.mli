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

type norm = Slaney | PNorm of float

val mel :
     ?fmax:float option
  -> ?htk:bool
  -> ?norm:norm option
  -> (float, 'b) Bigarray.kind
  -> int
  -> int
  -> int
  -> float
  -> (float, 'b) Owl_dense_ndarray.Generic.t
(** 
   [mel ?fmax ?htk ?norm sample_rate nfft nmels fmin]
   
   Returns a matrix of shape [nmels, nfft/2+1] containing the mel filterbank. *)
