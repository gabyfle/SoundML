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

open Windtrap
open Soundml

let vectors_dir = "vectors"

(* [spec_of_params name params] is the Window.t described by the "window"
   parameter of a golden case, taking shape parameters from [params]. *)
let spec_of_params name params =
  let float_param key =
    match List.assoc_opt key params with
    | Some json ->
        Yojson.Safe.Util.to_number json
    | None ->
        failf "golden case %s: missing parameter %s" name key
  in
  match List.assoc_opt "window" params with
  | Some (`String "hann") ->
      Window.Hann
  | Some (`String "hamming") ->
      Window.Hamming
  | Some (`String "blackman") ->
      Window.Blackman
  | Some (`String "blackman_harris") ->
      Window.Blackman_harris
  | Some (`String "nuttall") ->
      Window.Nuttall
  | Some (`String "bartlett") ->
      Window.Bartlett
  | Some (`String "flat_top") ->
      Window.Flat_top
  | Some (`String "rectangular") ->
      Window.Rectangular
  | Some (`String "kaiser") ->
      Window.Kaiser (float_param "beta")
  | Some (`String "gaussian") ->
      Window.Gaussian (float_param "std")
  | Some (`String "tukey") ->
      Window.Tukey (float_param "taper")
  | _ ->
      failf "golden case %s: unknown window" name

let golden_files =
  Tutils.Golden.load_dir ~filter:(fun name -> name <> "cola.json") vectors_dir

(* {2 Parity against librosa 0.11 golden vectors} *)

let parity_case dtype ~rtol ~atol (case : Tutils.Golden.case) =
  test case.name (fun () ->
      let window = spec_of_params case.name case.params in
      let periodic = Tutils.Golden.bool_param case "periodic" in
      let n = Tutils.Golden.int_param case "n" in
      let actual = Window.make dtype ~periodic window n in
      Tutils.check_close ~rtol ~atol ~shape:case.shape ~msg:case.name
        ~expected:case.values actual )

let parity_tests dtype ~rtol ~atol =
  List.concat_map
    (fun (file : Tutils.Golden.file) ->
      List.map (parity_case dtype ~rtol ~atol) file.cases )
    golden_files

(* {2 Constant overlap-add} *)

let cola_golden_tests =
  let json = Yojson.Safe.from_file (Filename.concat vectors_dir "cola.json") in
  let open Yojson.Safe.Util in
  json |> member "cases" |> to_list
  |> List.map (fun case ->
      let name = case |> member "name" |> to_string in
      let params = case |> member "params" |> to_assoc in
      let expected = case |> member "expected" |> to_bool in
      test name (fun () ->
          let window = spec_of_params name params in
          let length =
            params |> List.assoc "length" |> Yojson.Safe.Util.to_int
          in
          let hop = params |> List.assoc "hop" |> Yojson.Safe.Util.to_int in
          equal ~msg:name bool expected (Window.cola window ~length ~hop) ) )

let cola_truths () =
  let check msg expected window ~length ~hop =
    equal ~msg bool expected (Window.cola window ~length ~hop)
  in
  check "hann at hop = n/2" true Window.Hann ~length:1024 ~hop:512 ;
  check "hann at hop = n/4" true Window.Hann ~length:1024 ~hop:256 ;
  check "hann at hop = 511" false Window.Hann ~length:1024 ~hop:511 ;
  check "hann at hop = n" false Window.Hann ~length:1024 ~hop:1024 ;
  check "blackman-harris at hop = n/2" false Window.Blackman_harris ~length:1024
    ~hop:512 ;
  check "blackman-harris at hop = n/4" true Window.Blackman_harris ~length:1024
    ~hop:256 ;
  check "rectangular at hop = n" true Window.Rectangular ~length:512 ~hop:512 ;
  check "one-point window at hop = 1" true Window.Hann ~length:1 ~hop:1

(* {2 Dtype and shape behavior} *)

let dtype_behavior () =
  let n = 33 in
  let f64 = Window.make Nx.float64 Window.Hamming n in
  let f32 = Window.make Nx.float32 Window.Hamming n in
  equal ~msg:"float64 shape" (array int) [|n|] (Nx.shape f64) ;
  equal ~msg:"float32 shape" (array int) [|n|] (Nx.shape f32) ;
  (* the float32 window is the float64 window rounded once at creation *)
  Tutils.check_close ~rtol:Tutils.float32_rtol ~atol:Tutils.float32_atol
    ~msg:"float32 tracks float64" ~expected:(Nx.to_array f64) f32

(* {2 Tiny-length edge cases} *)

let edge_lengths () =
  let values window ~periodic n =
    Nx.to_array (Window.make Nx.float64 ~periodic window n)
  in
  equal ~msg:"periodic hann, n = 1"
    (array (float 0.))
    [|1.|]
    (values Window.Hann ~periodic:true 1) ;
  equal ~msg:"symmetric hann, n = 1"
    (array (float 0.))
    [|1.|]
    (values Window.Hann ~periodic:false 1) ;
  equal ~msg:"periodic kaiser, n = 1"
    (array (float 0.))
    [|1.|]
    (values (Window.Kaiser 8.6) ~periodic:true 1) ;
  equal ~msg:"periodic hann, n = 2"
    (array (float 0.))
    [|0.; 1.|]
    (values Window.Hann ~periodic:true 2) ;
  equal ~msg:"symmetric hann, n = 2"
    (array (float 0.))
    [|0.; 0.|]
    (values Window.Hann ~periodic:false 2) ;
  equal ~msg:"symmetric hann, n = 3"
    (array (float 0.))
    [|0.; 1.; 0.|]
    (values Window.Hann ~periodic:false 3) ;
  equal ~msg:"periodic hann, n = 3"
    (array (float 1e-15))
    [|0.; 0.75; 0.75|]
    (values Window.Hann ~periodic:true 3) ;
  equal ~msg:"rectangular, n = 3"
    (array (float 0.))
    [|1.; 1.; 1.|]
    (values Window.Rectangular ~periodic:true 3)

(* {2 Precondition violations} *)

let errors () =
  let check msg message thunk =
    raises_invalid_arg ~msg message (fun () -> ignore (thunk ()))
  in
  check "make, n = 0"
    "make: cannot make a 0-point window (length must be at least 1)" (fun () ->
      Window.make Nx.float64 Window.Hann 0 ) ;
  check "make, n = -3"
    "make: cannot make a -3-point window (length must be at least 1)" (fun () ->
      Window.make Nx.float64 Window.Hann (-3) ) ;
  check "make, negative kaiser beta"
    "make: cannot use a kaiser window with beta -1 (beta must be finite and \
     non-negative)" (fun () -> Window.make Nx.float64 (Window.Kaiser (-1.)) 8 ) ;
  check "make, zero gaussian deviation"
    "make: cannot use a gaussian window with standard deviation 0 (standard \
     deviation must be finite and positive)" (fun () ->
      Window.make Nx.float64 (Window.Gaussian 0.) 8 ) ;
  check "make, tukey taper above one"
    "make: cannot use a tukey window with taper 1.5 (taper must lie in [0, 1])"
    (fun () -> Window.make Nx.float64 (Window.Tukey 1.5) 8 ) ;
  check "cola, length = 0"
    "cola: cannot check overlap-add of a 0-point window (length must be at \
     least 1)" (fun () -> Window.cola Window.Hann ~length:0 ~hop:1 ) ;
  check "cola, hop = 0"
    "cola: cannot check overlap-add at hop 0 (hop must lie in [1, 8])"
    (fun () -> Window.cola Window.Hann ~length:8 ~hop:0 ) ;
  check "cola, hop above length"
    "cola: cannot check overlap-add at hop 9 (hop must lie in [1, 8])"
    (fun () -> Window.cola Window.Hann ~length:8 ~hop:9 ) ;
  check "cola, tukey taper above one"
    "cola: cannot use a tukey window with taper 2 (taper must lie in [0, 1])"
    (fun () -> Window.cola (Window.Tukey 2.) ~length:8 ~hop:4 )

(* {2 Specification equality and printing} *)

let spec_operations () =
  is_true ~msg:"equal hann" (Window.equal Window.Hann Window.Hann) ;
  is_false ~msg:"hann is not hamming" (Window.equal Window.Hann Window.Hamming) ;
  is_true ~msg:"equal kaiser"
    (Window.equal (Window.Kaiser 8.6) (Window.Kaiser 8.6)) ;
  is_false ~msg:"kaiser betas differ"
    (Window.equal (Window.Kaiser 8.6) (Window.Kaiser 8.5)) ;
  is_false ~msg:"kaiser is not gaussian"
    (Window.equal (Window.Kaiser 8.6) (Window.Gaussian 8.6)) ;
  let print window = Format.asprintf "%a" Window.pp window in
  equal ~msg:"pp hann" string "hann" (print Window.Hann) ;
  equal ~msg:"pp blackman_harris" string "blackman_harris"
    (print Window.Blackman_harris) ;
  equal ~msg:"pp kaiser" string "kaiser(8.6)" (print (Window.Kaiser 8.6)) ;
  equal ~msg:"pp tukey" string "tukey(0.25)" (print (Window.Tukey 0.25))

let suite =
  [ group "parity-float64"
      (parity_tests Nx.float64 ~rtol:Tutils.float64_rtol
         ~atol:Tutils.float64_atol )
  ; group "parity-float32"
      (parity_tests Nx.float32 ~rtol:Tutils.float32_rtol
         ~atol:Tutils.float32_atol )
  ; group "cola-golden" cola_golden_tests
  ; group "behavior"
      [ test "cola truths" cola_truths
      ; test "dtype and shape" dtype_behavior
      ; test "tiny lengths" edge_lengths
      ; test "error messages" errors
      ; test "pp and equal" spec_operations ] ]

let () = run "Soundml.Window" suite
