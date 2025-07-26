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

let typ_to_readable = function
  | "timeseries" ->
      "Io.Read"
  | "stft" ->
      "Spectral.stft"
  | _ ->
      "Unkown"

let string_to_resample_typ = function
  | "soxr_vhq" ->
      Io.SOXR_VHQ
  | "soxr_hq" ->
      Io.SOXR_HQ
  | "soxr_mq" ->
      Io.SOXR_MQ
  | "soxr_lq" ->
      Io.SOXR_LQ
  | _ ->
      Io.NONE

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
    if List.length split >= 1 then Some (String.concat "_" (List.tl split))
    else None

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
  type a

  type b

  val dtype : (a, b) Nx.dtype

  val typ : string

  val generate :
       (a, b) Nx.dtype
    -> string * string * Parameters.t
    -> (float, 'c) Nx.t
    -> (a, b) Nx.t
end

module Tests_cases (T : Testable) = struct
  include T

  let read_audio (type c) (audio_dtype : (float, c) Nx.dtype) (path : string)
      (res_typ : Io.resampling_t) (sample_rate : int) (mono : bool) =
    let audio, _ = Io.read ~res_typ ~sample_rate ~mono audio_dtype path in
    audio

  let read_npy : type a b. (a, b) Nx.dtype -> string -> (a, b) Nx.t =
   fun dtype path ->
    let packed = Nx_io.load_npy path in
    match dtype with
    | Nx.Float32 ->
        Nx_io.to_float32 packed
    | Nx.Float64 ->
        Nx_io.to_float64 packed
    | _ ->
        failwith "unsupported datatype"

  let create_tests (data : (string * string * Parameters.t) list) (rtol : float)
      (atol : float) : unit Alcotest.test_case list =
    List.concat_map
      (fun (case : string * string * Parameters.t) ->
        let vector_path, audio_path, params = case in
        let raw_basename =
          Filename.basename vector_path |> Filename.remove_extension
        in
        let basename =
          Option.value ~default:raw_basename
            (Testdata.get_test_filename raw_basename)
        in
        let sr =
          Option.value ~default:22050 @@ Parameters.get_int "sr" params
        in
        let mono =
          Option.value ~default:true @@ Parameters.get_bool "mono" params
        in
        let resampler =
          string_to_resample_typ
            ( Option.value ~default:"None"
            @@ Parameters.get_string "res_type" params )
        in
        let audio = read_audio Nx.float64 audio_path resampler sr mono in
        let generated = generate dtype case audio in
        let vector = read_npy dtype vector_path in
        let test_dense () =
          Alcotest.check
            (Tutils.tensor_testable dtype ~rtol ~atol)
            (typ ^ "_dense_" ^ basename)
            generated vector
        in
        let test_dense = ("DENSE:    " ^ basename, `Slow, test_dense) in
        [test_dense] )
      data

  let run (name : string) (tests : unit Alcotest.test_case list) =
    Alcotest.run name [(typ_to_readable typ, tests)]
end

let tests = ["timeseries"]

let data = Testdata.create test_vectors_dir test_audio_dir tests
