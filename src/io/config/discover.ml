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

module C = Configurator.V1

let () =
  C.main ~name:"sndfile-pkg-config" (fun c ->
      let default : C.Pkg_config.package_conf =
        {libs= ["-lsndfile"]; cflags= []}
      in
      let conf =
        match C.Pkg_config.get c with
        | None ->
            default
        | Some pc -> (
          match C.Pkg_config.query pc ~package:"sndfile" with
          | None ->
              default
          | Some deps ->
              deps )
      in
      C.Flags.write_sexp "c_flags.sexp" conf.cflags ;
      C.Flags.write_sexp "c_library_flags.sexp" conf.libs )
