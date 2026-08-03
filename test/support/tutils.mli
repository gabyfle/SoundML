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

(** Shared test utilities: the golden-file parity harness.

    Golden vectors are generated once by [test/generate_vectors.py] from
    pinned reference versions (librosa 0.11) and committed under
    [test/vectors/]; the tests replay them without Python. See
    [test/README.md]. *)

module Golden : sig
  (** The type for one golden case. *)
  type case =
    { name: string  (** Unique case identifier within the suite. *)
    ; params: (string * Yojson.Safe.t) list
          (** The generator parameters of the case. *)
    ; shape: int array  (** Shape of the expected tensor. *)
    ; values: float array  (** Expected float64 values, flattened in C order. *)
    }

  (** The type for one loaded golden file. *)
  type file =
    { path: string  (** The file the cases were loaded from. *)
    ; suite: string  (** The suite name recorded in the file. *)
    ; generator: (string * string) list
          (** Exact reference versions the file was generated with. *)
    ; cases: case list }

  val load : string -> file
  (** [load path] is the golden file at [path]. Raises [Failure] if the file
      does not follow the schema documented in [test/generate_vectors.py]. *)

  val load_dir : ?filter:(string -> bool) -> string -> file list
  (** [load_dir ?filter dir] is every [.json] golden file in [dir], in
      lexicographic order. [filter] defaults to accepting every file name. *)

  val param : case -> string -> Yojson.Safe.t
  (** [param case key] is the [key] parameter of [case]. Fails the running
      test case if [key] is absent. *)

  val bool_param : case -> string -> bool
  (** [bool_param case key] is {!param} as a boolean. *)

  val int_param : case -> string -> int
  (** [int_param case key] is {!param} as an integer. *)

  val float_param : case -> string -> float
  (** [float_param case key] is {!param} as a float. *)

  val string_param : case -> string -> string
  (** [string_param case key] is {!param} as a string. *)
end

val float64_rtol : float
(** [float64_rtol] is the relative tolerance for float64 parity, [1e-12]. *)

val float64_atol : float
(** [float64_atol] is the absolute tolerance for float64 parity, [1e-15]. *)

val float32_rtol : float
(** [float32_rtol] is the relative tolerance for float32 parity, [1e-6]. *)

val float32_atol : float
(** [float32_atol] is the absolute tolerance for float32 parity, [1e-7]. *)

val check_close :
     ?rtol:float
  -> ?atol:float
  -> ?shape:int array
  -> msg:string
  -> expected:float array
  -> (float, 'a) Nx.t
  -> unit
(** [check_close ?rtol ?atol ?shape ~msg ~expected actual] fails the running
    test case unless [actual] matches [expected] elementwise within
    [atol + rtol * |expected|], reporting the first offending index. [rtol]
    defaults to {!float64_rtol} and [atol] to {!float64_atol}; [shape], when
    given, must equal the shape of [actual]. [expected] is compared against
    [actual] flattened in C order. *)
