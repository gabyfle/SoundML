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

(** Tonal features over a chromagram: chroma energy normalised statistics
    (CENS) and the tonal centroid (tonnetz).

    Library-internal: [Soundml] re-exports these functions flat and this
    module is not part of the public interface — the reference documentation
    lives on the re-exports. Semantics follow librosa 1.0
    ([librosa.feature.chroma_cens] and [librosa.feature.tonnetz]); parity is
    enforced against committed golden vectors in the test suite. *)

val chroma_cens :
     Cqt.Config.t
  -> ?n_chroma:int
  -> ?norm:Chroma.norm
  -> ?smoothing:int option
  -> ?smoothing_window:Window.t
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [chroma_cens cqt x] is the CENS chromagram of the audio [x]. See
    [Soundml.chroma_cens]. *)

val chroma_cens_of_chroma :
     ?norm:Chroma.norm
  -> ?smoothing:int option
  -> ?smoothing_window:Window.t
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [chroma_cens_of_chroma c] is the CENS chromagram of the chromagram [c].
    See [Soundml.chroma_cens_of_chroma]. *)

val chroma_cens_stage :
     ?norm:Chroma.norm
  -> ?smoothing:int option
  -> ?smoothing_window:Window.t
  -> unit
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [chroma_cens_stage ()] is {!chroma_cens_of_chroma} as a pipeline stage.
    See [Soundml.chroma_cens_stage]. *)

val tonnetz :
  Cqt.Config.t -> ?n_chroma:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [tonnetz cqt x] is the tonal centroid of the audio [x]. See
    [Soundml.tonnetz]. *)

val tonnetz_of_chroma : (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [tonnetz_of_chroma c] is the tonal centroid of the chromagram [c]. See
    [Soundml.tonnetz_of_chroma]. *)

val tonnetz_stage : unit -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [tonnetz_stage ()] is {!tonnetz_of_chroma} as a pipeline stage. See
    [Soundml.tonnetz_stage]. *)
