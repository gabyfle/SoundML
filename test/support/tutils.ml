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

module Golden = struct
  type case =
    { name: string
    ; params: (string * Yojson.Safe.t) list
    ; shape: int array
    ; values: float array }

  type file =
    { path: string
    ; suite: string
    ; generator: (string * string) list
    ; cases: case list }

  let case_of_json json =
    let open Yojson.Safe.Util in
    { name= json |> member "name" |> to_string
    ; params= json |> member "params" |> to_assoc
    ; shape=
        json |> member "shape" |> to_list |> List.map to_int |> Array.of_list
    ; values=
        json |> member "values" |> to_list |> List.map to_number
        |> Array.of_list }

  let load path =
    try
      let module U = Yojson.Safe.Util in
      let json = Yojson.Safe.from_file path in
      { path
      ; suite= json |> U.member "suite" |> U.to_string
      ; generator=
          json |> U.member "generator" |> U.to_assoc
          |> List.map (fun (key, value) -> (key, U.to_string value))
      ; cases= json |> U.member "cases" |> U.to_list |> List.map case_of_json }
    with Yojson.Safe.Util.Type_error (message, _) ->
      failwith (Printf.sprintf "golden file %s: %s" path message)

  let load_dir ?(filter = fun _ -> true) dir =
    Sys.readdir dir |> Array.to_list |> List.sort String.compare
    |> List.filter (fun name ->
        Filename.check_suffix name ".json" && filter name )
    |> List.map (fun name -> load (Filename.concat dir name))

  let param case key =
    match List.assoc_opt key case.params with
    | Some value ->
        value
    | None ->
        Alcotest.failf "golden case %s: missing parameter %s" case.name key

  let bool_param case key = Yojson.Safe.Util.to_bool (param case key)

  let int_param case key = Yojson.Safe.Util.to_int (param case key)

  let float_param case key = Yojson.Safe.Util.to_number (param case key)

  let string_param case key = Yojson.Safe.Util.to_string (param case key)
end

let float64_rtol = 1e-12

let float64_atol = 1e-15

let float32_rtol = 1e-6

let float32_atol = 1e-7

let pp_shape shape =
  shape |> Array.to_list |> List.map string_of_int |> String.concat "; "

let check_close ?(rtol = float64_rtol) ?(atol = float64_atol) ?shape ~msg
    ~expected actual =
  ( match shape with
  | Some shape when shape <> Nx.shape actual ->
      Alcotest.failf "%s: shape [%s], expected [%s]" msg
        (pp_shape (Nx.shape actual))
        (pp_shape shape)
  | _ ->
      () ) ;
  let got = Nx.to_array actual in
  if Array.length got <> Array.length expected then
    Alcotest.failf "%s: %d elements, expected %d" msg (Array.length got)
      (Array.length expected) ;
  Array.iteri
    (fun i e ->
      let a = got.(i) in
      let tolerance = atol +. (rtol *. Float.abs e) in
      let close =
        (Float.is_nan e && Float.is_nan a) || Float.abs (a -. e) <= tolerance
      in
      if not close then
        Alcotest.failf
          "%s: index %d: got %.17g, expected %.17g (delta %.3g, tolerance %.3g)"
          msg i a e
          (Float.abs (a -. e))
          tolerance )
    expected
