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
        {libs= ["-lsndfile"; "-lsoxr"]; cflags= []}
      in
      let conf =
        match C.Pkg_config.get c with
        | None ->
            default
        | Some pc -> (
          match C.Pkg_config.query pc ~package:"sndfile soxr" with
          | None ->
              default
          | Some deps ->
              deps )
      in
      (* Add C++ standard library to the linking flags *)
      let cpp_libs =
        match Sys.os_type with
        | "Unix" ->
            (* On macOS and Linux, we need the C++ standard library *)
            if Sys.command "uname -s | grep -q Darwin" = 0 then ["-lc++"]
              (* macOS uses libc++ *)
            else ["-lstdc++"] (* Linux typically uses libstdc++ *)
        | _ ->
            ["-lstdc++"]
        (* Default fallback *)
      in
      let all_libs = conf.libs @ cpp_libs in
      C.Flags.write_sexp "c_flags.sexp" conf.cflags ;
      C.Flags.write_sexp "c_library_flags.sexp" all_libs )
