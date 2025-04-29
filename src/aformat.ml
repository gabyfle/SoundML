(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2023-2025                                                  *)
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

type ftype =
  | WAV
  | AIFF
  | AU
  | RAW
  | PAF
  | SVX
  | NIST
  | VOC
  | IRCAM
  | W64
  | MAT4
  | MAT5
  | PVF
  | XI
  | HTK
  | SDS
  | AVR
  | WAVEX
  | SD2
  | FLAC
  | CAF
  | WVE
  | OGG
  | MPC2K
  | RF64
  | MP3

type subtype =
  | PCM_S8
  | PCM_16
  | PCM_24
  | PCM_32
  | PCM_U8
  | FLOAT
  | DOUBLE
  | ULAW
  | ALAW
  | IMA_ADPCM
  | MS_ADPCM
  | GSM610
  | VOX_ADPCM
  | NMS_ADPCM_16
  | NMS_ADPCM_24
  | NMS_ADPCM_32
  | G721_32
  | G723_24
  | G723_40
  | DVW_12
  | DVW_16
  | DVW_24
  | DVW_N
  | DPCM_8
  | DPCM_16
  | VORBIS
  | OPUS
  | ALAC_16
  | ALAC_20
  | ALAC_24
  | ALAC_32
  | MPEG_LAYER_I
  | MPEG_LAYER_II
  | MPEG_LAYER_III

type endianness = FILE | LITTLE | BIG | CPU

type t = {ftype: ftype; sub: subtype; endian: endianness}

let show_ftype : ftype -> string = function
  | WAV ->
      "WAV"
  | AIFF ->
      "AIFF"
  | AU ->
      "AU"
  | RAW ->
      "RAW"
  | PAF ->
      "PAF"
  | SVX ->
      "SVX"
  | NIST ->
      "NIST"
  | VOC ->
      "VOC"
  | IRCAM ->
      "IRCAM"
  | W64 ->
      "W64"
  | MAT4 ->
      "MAT4"
  | MAT5 ->
      "MAT5"
  | PVF ->
      "PVF"
  | XI ->
      "XI"
  | HTK ->
      "HTK"
  | SDS ->
      "SDS"
  | AVR ->
      "AVR"
  | WAVEX ->
      "WAVEX"
  | SD2 ->
      "SD2"
  | FLAC ->
      "FLAC"
  | CAF ->
      "CAF"
  | WVE ->
      "WVE"
  | OGG ->
      "OGG"
  | MPC2K ->
      "MPC2K"
  | RF64 ->
      "RF64"
  | MP3 ->
      "MP3"

(* These assoc lists are directly extracted from
   https://github.com/libsndfile/libsndfile/blob/master/include/sndfile.h#L48-L129 *)
let ftype_assoc =
  [ (WAV, 0x010000)
  ; (AIFF, 0x020000)
  ; (AU, 0x030000)
  ; (RAW, 0x040000)
  ; (PAF, 0x050000)
  ; (SVX, 0x060000)
  ; (NIST, 0x070000)
  ; (VOC, 0x080000)
  ; (IRCAM, 0x0A0000)
  ; (W64, 0x0B0000)
  ; (MAT4, 0x0C0000)
  ; (MAT5, 0x0D0000)
  ; (PVF, 0x0E0000)
  ; (XI, 0x0F0000)
  ; (HTK, 0x100000)
  ; (SDS, 0x110000)
  ; (AVR, 0x120000)
  ; (WAVEX, 0x130000)
  ; (SD2, 0x160000)
  ; (FLAC, 0x170000)
  ; (CAF, 0x180000)
  ; (WVE, 0x190000)
  ; (OGG, 0x200000)
  ; (MPC2K, 0x210000)
  ; (RF64, 0x220000)
  ; (MP3, 0x230000) ]

let subtype_assoc =
  [ (PCM_S8, 0x0001)
  ; (PCM_16, 0x0002)
  ; (PCM_24, 0x0003)
  ; (PCM_32, 0x0004)
  ; (PCM_U8, 0x0005)
  ; (FLOAT, 0x0006)
  ; (DOUBLE, 0x0007)
  ; (ULAW, 0x0010)
  ; (ALAW, 0x0011)
  ; (IMA_ADPCM, 0x0012)
  ; (MS_ADPCM, 0x0013)
  ; (GSM610, 0x0020)
  ; (VOX_ADPCM, 0x0021)
  ; (NMS_ADPCM_16, 0x0022)
  ; (NMS_ADPCM_24, 0x0023)
  ; (NMS_ADPCM_32, 0x0024)
  ; (G721_32, 0x0030)
  ; (G723_24, 0x0031)
  ; (G723_40, 0x0032)
  ; (DVW_12, 0x0040)
  ; (DVW_16, 0x0041)
  ; (DVW_24, 0x0042)
  ; (DVW_N, 0x0043)
  ; (DPCM_8, 0x0050)
  ; (DPCM_16, 0x0051)
  ; (VORBIS, 0x0060)
  ; (OPUS, 0x0064)
  ; (ALAC_16, 0x0070)
  ; (ALAC_20, 0x0071)
  ; (ALAC_24, 0x0072)
  ; (ALAC_32, 0x0073)
  ; (MPEG_LAYER_I, 0x0080)
  ; (MPEG_LAYER_II, 0x0081)
  ; (MPEG_LAYER_III, 0x0082) ]

let endianness_assoc =
  [ (FILE, 0x00000000)
  ; (LITTLE, 0x10000000)
  ; (BIG, 0x20000000)
  ; (CPU, 0x30000000) ]

(* See
   https://github.com/libsndfile/libsndfile/blob/master/include/sndfile.h#L131-L133 *)
let format_submask = 0x0000FFFF

let format_typemask = 0x0FFF0000

let format_endmask = 0x30000000

(* For a maximum compatibility with librosa, this is the exact same default
   subtypes that is used here:
   https://python-soundfile.readthedocs.io/en/0.13.1/_modules/soundfile.html *)
let default_subtype_spec =
  [ (WAV, PCM_16)
  ; (AIFF, PCM_16)
  ; (AU, PCM_16)
  ; (PAF, PCM_16)
  ; (SVX, PCM_16)
  ; (NIST, PCM_16)
  ; (VOC, PCM_16)
  ; (IRCAM, PCM_16)
  ; (W64, PCM_16)
  ; (MAT4, DOUBLE)
  ; (MAT5, DOUBLE)
  ; (PVF, PCM_16)
  ; (XI, DPCM_16)
  ; (HTK, PCM_16)
  ; (SDS, PCM_16)
  ; (AVR, PCM_16)
  ; (WAVEX, PCM_16)
  ; (SD2, PCM_16)
  ; (FLAC, PCM_16)
  ; (CAF, PCM_16)
  ; (WVE, ALAW)
  ; (OGG, VORBIS)
  ; (MPC2K, PCM_16)
  ; (RF64, PCM_16)
  ; (MP3, MPEG_LAYER_III) ]

module StrMap = Map.Make (String)

module FtypeMap = Map.Make (struct
  type t = ftype

  let compare = compare
end)

module SubtypeMap = Map.Make (struct
  type t = subtype

  let compare = compare
end)

module EndiannessMap = Map.Make (struct
  type t = endianness

  let compare = compare
end)

let ftype_to_int_map =
  FtypeMap.of_list (List.map (fun (v, i) -> (v, i)) ftype_assoc)

let subtype_to_int_map =
  SubtypeMap.of_list (List.map (fun (v, i) -> (v, i)) subtype_assoc)

let endianness_to_int_map =
  EndiannessMap.of_list (List.map (fun (v, i) -> (v, i)) endianness_assoc)

let default_subtype_map =
  List.fold_left
    (fun map (ftype, sub) -> FtypeMap.add ftype sub map)
    FtypeMap.empty default_subtype_spec

let fmt_string_map =
  List.fold_left
    (fun map (ftype, _) -> StrMap.add ("." ^ show_ftype ftype) ftype map)
    StrMap.empty ftype_assoc

let int_of_ftype_format v = FtypeMap.find v ftype_to_int_map

let int_of_subtype v = SubtypeMap.find v subtype_to_int_map

let int_of_endianness v = EndiannessMap.find v endianness_to_int_map

let get_default_subtype (ftype_fmt : ftype) : subtype option =
  match ftype_fmt with
  | RAW ->
      None
  | _ ->
      Some (FtypeMap.find ftype_fmt default_subtype_map)

let ftype_of_string (ext : string) : ftype option =
  StrMap.find_opt ext fmt_string_map

let create ?subtype ?(endian = FILE) (ftype : ftype) : (t, string) Result.t =
  match subtype with
  | Some sub ->
      Ok {ftype; sub; endian}
  | None -> (
      let sub = get_default_subtype ftype in
      match sub with
      | None ->
          Error "Couldn't find a default subtype for the given file type."
      | Some sub ->
          Ok {ftype; sub; endian} )

let to_int (fmt : t) : int =
  let ftype_code = int_of_ftype_format fmt.ftype in
  let sub_code = int_of_subtype fmt.sub in
  let endian_code = int_of_endianness fmt.endian in
  ftype_code lor sub_code lor endian_code

let of_int (code : int) : (t, string) result =
  let ftype_code = code land format_typemask in
  let sub_code = code land format_submask in
  let endian_code = code land format_endmask in
  let reconstructed_code = ftype_code lor sub_code lor endian_code in
  if code <> reconstructed_code then
    Error
      (Printf.sprintf "Invalid or unknown bits set in format code: 0x%X" code)
  else
    let find_variant map code =
      List.find_map (fun (v, i) -> if i = code then Some v else None) map
    in
    match
      ( find_variant ftype_assoc ftype_code
      , find_variant subtype_assoc sub_code
      , find_variant endianness_assoc endian_code )
    with
    | Some ftype, Some sub, Some endian ->
        Ok {ftype; sub; endian}
    | None, _, _ ->
        Error (Printf.sprintf "Unknown ftype format code: 0x%X" ftype_code)
    | _, None, _ ->
        Error (Printf.sprintf "Unknown subtype code: 0x%X" sub_code)
    | _, _, None ->
        Error (Printf.sprintf "Unknown endianness code: 0x%X" endian_code)

let of_ext ?sub ?(endian = FILE) (ext : string) : (t, string) result =
  match ftype_of_string (String.uppercase_ascii ext) with
  | None ->
      Error
        (Printf.sprintf "Couldn't find any format matching extension: %s" ext)
  | Some ftype ->
      if ftype = RAW && sub = None then
        Error "The RAW format needs to have it's subtype specified"
      else
        let sub = Option.get (get_default_subtype ftype) in
        Ok {ftype; sub; endian}
