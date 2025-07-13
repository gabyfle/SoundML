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

type data = (float, Bigarray.float32_elt) Nx.t

let data_testable : data Alcotest.testable =
  ( module struct
    type t = data

    let pp : t Fmt.t =
     fun fmt ndarray ->
      let shape_array = Nx.shape ndarray in
      let pp_shape = Fmt.brackets (Fmt.array ~sep:Fmt.semi Fmt.int) in
      Fmt.pf fmt "%a" pp_shape shape_array

    let equal : t -> t -> bool = Tutils.Check.rallclose
  end )

module Test_pad_center = struct
  let create_data (arr : float array) : data =
    Nx.create Float32 [|Array.length arr|] arr
  (* Create 1D Ndarray *)

  let test_no_padding () =
    let input_data = create_data [|1.; 2.; 3.|] in
    let target_size = 3 in
    let pad_value = 0. in
    let expected_output = create_data [|1.; 2.; 3.|] in
    let actual_output = Utils.pad_center input_data target_size pad_value in
    Alcotest.check data_testable "no_padding: Correct padding" expected_output
      actual_output

  let test_even_padding () =
    let input_data = create_data [|1.; 2.|] in
    let target_size = 6 in
    let pad_value = 0. in
    let expected_output = create_data [|0.; 0.; 1.; 2.; 0.; 0.|] in
    let actual_output = Utils.pad_center input_data target_size pad_value in
    Alcotest.check data_testable "even_padding: Correct padding" expected_output
      actual_output

  let test_odd_padding () =
    let input_data = create_data [|1.; 2.; 3.|] in
    let target_size = 6 in
    let pad_value = 0. in
    let expected_output = create_data [|0.; 1.; 2.; 3.; 0.; 0.|] in
    let actual_output = Utils.pad_center input_data target_size pad_value in
    Alcotest.check data_testable "odd_padding: Correct padding" expected_output
      actual_output

  let test_empty_input () =
    let input_data = create_data [||] in
    let target_size = 4 in
    let pad_value = 0. in
    let expected_output = create_data [|0.; 0.; 0.; 0.|] in
    let actual_output = Utils.pad_center input_data target_size pad_value in
    Alcotest.check data_testable "empty_input: Correct padding" expected_output
      actual_output

  let test_error_target_too_small () =
    let input_data = create_data [|1.; 2.; 3.; 4.|] in
    let target_size = 2 in
    let pad_value = 0. in
    let expected_exn =
      Invalid_argument
        "An error occured while trying to pad: current_size > target_size"
    in
    Alcotest.check_raises
      "error_target_too_small: raises Invalid_argument when target_size < \
       input_size"
      expected_exn (fun () ->
        ignore (Utils.pad_center input_data target_size pad_value) )

  let test_non_zero_padding () =
    let input_data = create_data [|5.; 6.|] in
    let target_size = 5 in
    let pad_value = -1.5 in
    let expected_output = create_data [|-1.5; 5.; 6.; -1.5; -1.5|] in
    let actual_output = Utils.pad_center input_data target_size pad_value in
    Alcotest.check data_testable "non_zero_padding: Correct padding"
      expected_output actual_output

  let test_zero_target_empty_input () =
    let input_data = create_data [||] in
    flush_all () ;
    let target_size = 0 in
    let pad_value = 0. in
    let expected_output = create_data [||] in
    let actual_output = Utils.pad_center input_data target_size pad_value in
    Alcotest.check data_testable "zero_target_empty_input: Correct padding"
      expected_output actual_output

  let test_zero_target_non_empty_input () =
    let input_data = create_data [|1.; 2.|] in
    let target_size = 0 in
    let pad_value = 0. in
    let expected_exn =
      Invalid_argument
        "An error occured while trying to pad: current_size > target_size"
    in
    Alcotest.check_raises
      "zero_target_non_empty_input: raises Invalid_argument when target_size < \
       input_size"
      expected_exn (fun () ->
        ignore (Utils.pad_center input_data target_size pad_value) )

  let suite =
    [ Alcotest.test_case "no_padding" `Quick test_no_padding
    ; Alcotest.test_case "even_padding" `Quick test_even_padding
    ; Alcotest.test_case "odd_padding" `Quick test_odd_padding
    ; Alcotest.test_case "empty_input" `Quick test_empty_input
    ; Alcotest.test_case "error_target_too_small" `Quick
        test_error_target_too_small
    ; Alcotest.test_case "non_zero_padding" `Quick test_non_zero_padding
    ; Alcotest.test_case "zero_target_empty_input" `Quick
        test_zero_target_empty_input
    ; Alcotest.test_case "zero_target_non_empty_input" `Quick
        test_zero_target_non_empty_input ]
end

let () =
  Alcotest.run "SoundML Utils Tests" [("Pad Center", Test_pad_center.suite)]
