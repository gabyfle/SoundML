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

(**
    The {!Feature.Temporal} module focus on extracting time domain features
    of an audio data. *)

(**
    {1 Temporal}

    This module defines functions makes time-domain analysis of an audio signal.  *)

val rms :
     ?window:int
  -> ?step:int
  -> Audio.audio
  -> (float, Bigarray.float32_elt) Owl_dense_ndarray.Generic.t
(**
    [rms ~window ~step audio] computes the Root Mean Square (RMS) of the given audio data for each frame.

    [?window] is the window size to use for the RMS computation. Default is [2048].
    [?step] is the step size to use for the RMS computation. Default is [1024].

    Examples:

    {[
        let () =
            let src = read file.wav wav in
            let rms = rms src in
            (* ... *)
    ]} *)
