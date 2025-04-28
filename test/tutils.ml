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

let test_audio_dir = Sys.getcwd () ^ "/audio/"

let test_vectors_dir = Sys.getcwd () ^ "/vectors/"

let typ_to_readable = function "timeseries" -> "Io module" | _ -> "Unkown"

module StrMap = Map.Make (String)

module Parameters = struct
  open Yojson.Basic.Util

  type t = Yojson.Basic.t

  let create (path : string) = Yojson.Basic.from_file path

  let get_string (name : string) (params : t) =
    member name params |> to_string_option

  let get_int (name : string) (params : t) = member name params |> to_int_option

  let get_float (name : string) (params : t) =
    member name params |> to_float_option

  let get_bool (name : string) (params : t) =
    member name params |> to_bool_option
end

module Testdata = struct
  type t = (string * string * Parameters.t) list StrMap.t

  let get_test_type (basename : string) : string option =
    let split = Str.(split (regexp {|_|}) basename) in
    if List.length split >= 1 then Some (List.hd split) else None

  let get_test_filename (basename : string) : string option =
    let split = Str.(split (regexp {|_|}) basename) in
    if List.length split >= 1 then Some (List.nth split 1) else None

  let list_filter_filename (dir : string) (name : string) : string option =
    try
      let files = Sys.readdir dir in
      Option.map (fun elt -> Filename.concat dir elt)
      @@ Array.find_opt (fun elt -> Filename.remove_extension elt = name) files
    with Sys_error msg ->
      Printf.eprintf "Error reading directory '%s': %s\n" dir msg ;
      None

  let list_tests_files (dir : string) (ext : string) : string list =
    let has_extension path = Filename.check_suffix path ext in
    let process_entry base_dir entry =
      let full_path = Filename.concat base_dir entry in
      try
        if Sys.is_directory full_path then
          List.filter has_extension
          @@ List.map (Filename.concat full_path)
          @@ Array.to_list @@ Sys.readdir full_path
        else if has_extension full_path then [full_path]
        else []
      with Sys_error _ -> []
    in
    try
      Sys.readdir dir |> Array.to_list |> List.concat_map (process_entry dir)
      (* Appliquer process_entry et concaténer les résultats *)
    with Sys_error msg ->
      Printf.eprintf "Error reading directory '%s': %s\n" dir msg ;
      []

  let filter_test_type (typ : string) =
    let filter (typ : string) (full_path : string) : string option =
      let basename = Filename.basename full_path in
      let split = get_test_type basename in
      Option.bind split (fun x ->
          if x = typ then Some (Filename.remove_extension full_path) else None )
    in
    List.filter_map @@ filter typ

  let construct_parameters (audio_dir : string) (files : string list) =
    let aux (file : string) =
      let base = Filename.basename file |> Filename.remove_extension in
      let audio_filename =
        Option.value (get_test_filename base) ~default:base
      in
      let audio_file_opt = list_filter_filename audio_dir audio_filename in
      match audio_file_opt with
      | None ->
          Printf.eprintf "Warning: couldn't find audio file with name %s\n" base ;
          None
      | Some audio_file_path ->
          let npy_file = file ^ ".npy" in
          let json_file = file ^ ".json" in
          let params = Parameters.create json_file in
          Some (npy_file, audio_file_path, params)
    in
    List.filter_map aux files

  let create (vectors_dir : string) (audio_dir : string) (types : string list) :
      t =
    let vectors_files = list_tests_files vectors_dir ".json" in
    let fold map typ =
      let l = filter_test_type typ vectors_files in
      StrMap.add typ l map
    in
    let files_map = List.fold_left fold StrMap.empty types in
    StrMap.map (construct_parameters audio_dir) files_map

  let get (typ : string) (data : t) : (string * string * Parameters.t) list =
    StrMap.find typ data
end

module type Testable = sig
  val typ : string

  val create_test_set :
       (string * string * Parameters.t) list
    -> (string * [> `Slow] * (unit -> unit)) list
end

module Check = struct
  open Soundml

  let shape (x : ('a, 'b) Audio.G.t) (y : ('a, 'b) Audio.G.t) =
    let shape_x = Audio.G.shape x in
    let shape_y = Audio.G.shape y in
    if Array.length shape_x <> Array.length shape_y then false
    else Array.for_all2 (fun x y -> x = y) shape_x shape_y

  let rallclose ?(rtol = 1e-05) ?(atol = 1e-10) (x : ('a, 'b) Audio.G.t)
      (y : ('a, 'b) Audio.G.t) : bool =
    if not (shape x y) then false
    else
      let abs_diff = Audio.G.abs (Audio.G.sub x y) in
      let tolerance = Audio.G.(add_scalar (mul_scalar (abs y) rtol) atol) in
      let comparison_mask = Audio.G.elt_less_equal abs_diff tolerance in
      Audio.G.min' comparison_mask >= 1.0

  let callclose : type a.
         ?rtol:float
      -> ?atol:float
      -> (Complex.t, a) Audio.G.t
      -> (Complex.t, a) Audio.G.t
      -> bool =
   fun ?(rtol = 1e-05) ?(atol = 1e-08) (x : (Complex.t, a) Audio.G.t)
       (y : (Complex.t, a) Audio.G.t) ->
    if not (shape x y) then false
    else
      let x, y =
        match Audio.G.kind x with
        | Bigarray.Complex32 ->
            (Audio.G.cast_c2z x, Audio.G.cast_c2z y)
        | Bigarray.Complex64 ->
            (x, y)
        | _ ->
            .
      in
      let diff = Audio.G.sub x y in
      let abs_diff = Audio.G.abs2_z2d diff in
      let abs_y = Audio.G.abs2_z2d y in
      let tolerance = Audio.G.(add_scalar (mul_scalar abs_y rtol) atol) in
      let comparison_mask = Audio.G.elt_less_equal abs_diff tolerance in
      Audio.G.min' comparison_mask >= 1.0
end

(* This snippet has been gathered from the exact same code but for Matrix in
   Owl. See:
   https://github.com/tachukao/owl/blob/046f703a6890a5ed5ecf4a8c5750d4e392e4ec54/src/owl/dense/owl_dense_matrix_generic.ml#L606-L609
   Unfortunately, for the moment this is not yet available for Ndarrays. *)
let load_npy (path : string) (kind : ('a, 'b) Bigarray.kind) :
    ('a, 'b) Audio.G.t =
  let npy : ('a, 'b) Audio.G.t =
    match Npy.read_copy path |> Npy.to_bigarray Bigarray.c_layout kind with
    | Some x ->
        x
    | None ->
        failwith Printf.(sprintf "%s: incorrect format" path)
  in
  npy
