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

open Soundml

type data = (float, Rune.float32_elt, [`c]) Rune.t

let data_testable : data Alcotest.testable =
  ( module struct
    type t = data

    let pp : t Fmt.t = Rune.pp

    let equal : t -> t -> bool = Tutils.Check.rallclose
  end )

module Test_pad_center = struct
  let create_data (arr : float array) : data =
    Rune.create Rune.c Rune.float32 [|Array.length arr|] arr
  (* Create 1D Rune tensor *)

  let test_no_padding () =
    let input_data = create_data [|1.; 2.; 3.|] in
    let target_size = 3 in
    let pad_value = 0. in
    let expected_output = create_data [|1.; 2.; 3.|] in
    let actual_output =
      Utils.pad_center input_data ~size:target_size ~pad_value
    in
    Alcotest.check data_testable "no_padding: Correct padding" expected_output
      actual_output

  let test_even_padding () =
    let input_data = create_data [|1.; 2.|] in
    let target_size = 6 in
    let pad_value = 0. in
    let expected_output = create_data [|0.; 0.; 1.; 2.; 0.; 0.|] in
    let actual_output =
      Utils.pad_center input_data ~size:target_size ~pad_value
    in
    Alcotest.check data_testable "even_padding: Correct padding" expected_output
      actual_output

  let test_odd_padding () =
    let input_data = create_data [|1.; 2.; 3.|] in
    let target_size = 6 in
    let pad_value = 0. in
    let expected_output = create_data [|0.; 1.; 2.; 3.; 0.; 0.|] in
    let actual_output =
      Utils.pad_center input_data ~size:target_size ~pad_value
    in
    Alcotest.check data_testable "odd_padding: Correct padding" expected_output
      actual_output

  let test_empty_input () =
    let input_data = create_data [||] in
    let target_size = 4 in
    let pad_value = 0. in
    let expected_output = create_data [|0.; 0.; 0.; 0.|] in
    let actual_output =
      Utils.pad_center input_data ~size:target_size ~pad_value
    in
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
        ignore (Utils.pad_center input_data ~size:target_size ~pad_value) )

  let test_non_zero_padding () =
    let input_data = create_data [|5.; 6.|] in
    let target_size = 5 in
    let pad_value = -1.5 in
    let expected_output = create_data [|-1.5; 5.; 6.; -1.5; -1.5|] in
    let actual_output =
      Utils.pad_center input_data ~size:target_size ~pad_value
    in
    Alcotest.check data_testable "non_zero_padding: Correct padding"
      expected_output actual_output

  let test_zero_target_empty_input () =
    let input_data = create_data [||] in
    flush_all () ;
    let target_size = 0 in
    let pad_value = 0. in
    let expected_output = create_data [||] in
    let actual_output =
      Utils.pad_center input_data ~size:target_size ~pad_value
    in
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
        ignore (Utils.pad_center input_data ~size:target_size ~pad_value) )

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

module Test_melfreq = struct
  let test_default () =
    let expected =
      Rune.create Rune.c Rune.float32 [|128|]
        [| 0.
         ; 26.199787
         ; 52.399574
         ; 78.59936
         ; 104.79915
         ; 130.99893
         ; 157.19872
         ; 183.39851
         ; 209.5983
         ; 235.79808
         ; 261.99786
         ; 288.19766
         ; 314.39743
         ; 340.59723
         ; 366.79703
         ; 392.9968
         ; 419.1966
         ; 445.3964
         ; 471.59616
         ; 497.79596
         ; 523.9957
         ; 550.19556
         ; 576.3953
         ; 602.5951
         ; 628.79486
         ; 654.9947
         ; 681.19446
         ; 707.3942
         ; 733.59406
         ; 759.7938
         ; 785.9936
         ; 812.1934
         ; 838.3932
         ; 864.59296
         ; 890.7928
         ; 916.99255
         ; 943.1923
         ; 969.39215
         ; 995.5919
         ; 1022.7277
         ; 1050.7377
         ; 1079.5149
         ; 1109.0801
         ; 1139.4551
         ; 1170.662
         ; 1202.7236
         ; 1235.6632
         ; 1269.505
         ; 1304.2737
         ; 1339.9945
         ; 1376.6937
         ; 1414.3981
         ; 1453.1349
         ; 1492.9327
         ; 1533.8206
         ; 1575.8281
         ; 1618.9862
         ; 1663.3263
         ; 1708.8807
         ; 1755.6829
         ; 1803.7667
         ; 1853.1675
         ; 1903.9211
         ; 1956.065
         ; 2009.6367
         ; 2064.6758
         ; 2121.2222
         ; 2179.3174
         ; 2239.0034
         ; 2300.3245
         ; 2363.3247
         ; 2428.0503
         ; 2494.5486
         ; 2562.8682
         ; 2633.059
         ; 2705.172
         ; 2779.26
         ; 2855.3772
         ; 2933.579
         ; 3013.9226
         ; 3096.4666
         ; 3181.2712
         ; 3268.3984
         ; 3357.9119
         ; 3449.877
         ; 3544.3606
         ; 3641.4321
         ; 3741.162
         ; 3843.6233
         ; 3948.8909
         ; 4057.0413
         ; 4168.154
         ; 4282.309
         ; 4399.5913
         ; 4520.0854
         ; 4643.8794
         ; 4771.064
         ; 4901.732
         ; 5035.978
         ; 5173.9014
         ; 5315.602
         ; 5461.183
         ; 5610.7515
         ; 5764.4165
         ; 5922.2896
         ; 6084.487
         ; 6251.126
         ; 6422.329
         ; 6598.221
         ; 6778.93
         ; 6964.5884
         ; 7155.3315
         ; 7351.299
         ; 7552.633
         ; 7759.481
         ; 7971.994
         ; 8190.3276
         ; 8414.641
         ; 8645.098
         ; 8881.865
         ; 9125.118
         ; 9375.032
         ; 9631.792
         ; 9895.583
         ; 10166.599
         ; 10445.037
         ; 10731.102
         ; 11025. |]
    in
    let actual = Utils.melfreqs Tutils.device Rune.float32 in
    Alcotest.check data_testable "melfreq_default" expected actual

  let test_custom () =
    let expected =
      Rune.create Rune.c Rune.float32 [|10|]
        [| 1000.
         ; 1203.3604
         ; 1431.0475
         ; 1685.9714
         ; 1971.3903
         ; 2290.952
         ; 2648.741
         ; 3049.3298
         ; 3497.8389
         ; 4000. |]
    in
    let actual =
      Utils.melfreqs ~n_mels:10 ~f_min:1000. ~f_max:4000. ~htk:true
        Tutils.device Rune.float32
    in
    Alcotest.check data_testable "melfreq_custom" expected actual

  let suite =
    [ Alcotest.test_case "default" `Quick test_default
    ; Alcotest.test_case "custom" `Quick test_custom ]
end

module Test_unwrap = struct
  let test_1d () =
    let p =
      Rune.create Rune.c Rune.float32 [|8|]
        [|0.; 0.1; 0.2; 5.0; 5.1; 5.2; -0.1; -0.2|]
    in
    let expected =
      Rune.create Rune.c Rune.float32 [|8|]
        [| 0.
         ; 0.1
         ; 0.2
         ; -1.283185
         ; -1.1831851
         ; -1.0831852
         ; -0.09999905
         ; -0.19999905 |]
    in
    let actual = Utils.unwrap p in
    Alcotest.check data_testable "unwrap_1d" expected actual

  let test_2d_axis0 () =
    let p =
      Rune.create Rune.c Rune.float32 [|2; 3|] [|0.; 0.1; 6.2; 0.; 0.1; 6.2|]
    in
    let expected =
      Rune.create Rune.c Rune.float32 [|2; 3|] [|0.; 0.1; 6.2; 0.; 0.1; 6.2|]
    in
    let actual = Utils.unwrap ~axis:0 p in
    Alcotest.check data_testable "unwrap_2d_axis0" expected actual

  let test_2d_axis1 () =
    let p =
      Rune.create Rune.c Rune.float32 [|2; 3|] [|0.; 0.1; 6.2; 0.; 0.1; 6.2|]
    in
    let expected =
      Rune.create Rune.c Rune.float32 [|2; 3|]
        [|0.; 0.1; -0.08318615; 0.; 0.1; -0.08318615|]
    in
    let actual = Utils.unwrap ~axis:1 p in
    Alcotest.check data_testable "unwrap_2d_axis1" expected actual

  let suite =
    [ Alcotest.test_case "1d" `Quick test_1d
    ; Alcotest.test_case "2d_axis0" `Quick test_2d_axis0
    ; Alcotest.test_case "2d_axis1" `Quick test_2d_axis1 ]
end

module Test_outer = struct
  let test_add () =
    let x = Rune.create Rune.c Rune.float32 [|3|] [|1.; 2.; 3.|] in
    let y = Rune.create Rune.c Rune.float32 [|4|] [|4.; 5.; 6.; 7.|] in
    let expected =
      Rune.create Rune.c Rune.float32 [|3; 4|]
        [|5.; 6.; 7.; 8.; 6.; 7.; 8.; 9.; 7.; 8.; 9.; 10.|]
    in
    let actual = Utils.outer Rune.add x y in
    Alcotest.check data_testable "outer_add" expected actual

  let test_mul () =
    let x = Rune.create Rune.c Rune.float32 [|3|] [|1.; 2.; 3.|] in
    let y = Rune.create Rune.c Rune.float32 [|4|] [|4.; 5.; 6.; 7.|] in
    let expected =
      Rune.create Rune.c Rune.float32 [|3; 4|]
        [|4.; 5.; 6.; 7.; 8.; 10.; 12.; 14.; 12.; 15.; 18.; 21.|]
    in
    let actual = Utils.outer Rune.mul x y in
    Alcotest.check data_testable "outer_mul" expected actual

  let suite =
    [ Alcotest.test_case "add" `Quick test_add
    ; Alcotest.test_case "mul" `Quick test_mul ]
end

module Test_convert = struct
  let test_hz_to_mel_htk () =
    let freqs =
      Rune.create Rune.c Rune.float32 [|10|]
        [|0.; 1225.; 2450.; 3675.; 4900.; 6125.; 7350.; 8575.; 9800.; 11025.|]
    in
    let expected =
      Rune.create Rune.c Rune.float32 [|10|]
        [| 0.
         ; 1140.0684
         ; 1695.0864
         ; 2065.3086
         ; 2343.5186
         ; 2566.467
         ; 2752.5107
         ; 2912.1501
         ; 3051.957
         ; 3176.3184 |]
    in
    let actual = Utils.Convert.hz_to_mel ~htk:true freqs in
    Alcotest.check data_testable "hz_to_mel_htk" expected actual

  let test_mel_to_hz_htk () =
    let mels =
      Rune.create Rune.c Rune.float32 [|10|]
        [| 0.
         ; 444.44446
         ; 888.8889
         ; 1333.3334
         ; 1777.7778
         ; 2222.2222
         ; 2666.6667
         ; 3111.111
         ; 3555.5557
         ; 4000. |]
    in
    let expected =
      Rune.create Rune.c Rune.float32 [|10|]
        [| 0.
         ; 338.40695
         ; 840.4128
         ; 1585.1075
         ; 2689.8167
         ; 4328.584
         ; 6759.595
         ; 10365.85
         ; 15715.508
         ; 23651.396 |]
    in
    let actual = Utils.Convert.mel_to_hz ~htk:true mels in
    Alcotest.check data_testable "mel_to_hz_htk" expected actual

  let test_power_to_db () =
    let s =
      Rune.create Rune.c Rune.float32 [|2; 4|]
        [|1.; 0.1; 0.01; 0.001; 1e-11; 1e-12; 1e-13; 1e-14|]
    in
    let expected_ref1_topdb80 =
      Rune.create Rune.c Rune.float32 [|2; 4|]
        [|0.; -10.; -20.; -30.; -80.; -80.; -80.; -80.|]
    in
    let actual_ref1_topdb80 =
      Utils.Convert.power_to_db ~top_db:80.0 (Utils.Convert.RefFloat 1.0) s
    in
    Alcotest.check data_testable "power_to_db ref=1.0 top_db=80.0"
      expected_ref1_topdb80 actual_ref1_topdb80 ;
    let expected_refmax_topdb80 =
      Rune.create Rune.c Rune.float32 [|2; 4|]
        [|0.; -10.; -20.; -30.; -80.; -80.; -80.; -80.|]
    in
    let actual_refmax_topdb80 =
      Utils.Convert.power_to_db ~top_db:80.0
        (Utils.Convert.RefFunction (fun x -> Rune.unsafe_get [] (Rune.max x)))
        s
    in
    Alcotest.check data_testable "power_to_db ref=max top_db=80.0"
      expected_refmax_topdb80 actual_refmax_topdb80 ;
    let expected_ref1_topdbNone =
      Rune.create Rune.c Rune.float32 [|2; 4|]
        [|0.; -10.; -20.; -30.; -110.; -120.; -130.; -140.|]
    in
    let actual_ref1_topdbNone =
      Utils.Convert.power_to_db ~amin:1e-10 ?top_db:None
        (Utils.Convert.RefFloat 1.0) s
    in
    Alcotest.check data_testable "power_to_db ref=1.0 top_db=None"
      expected_ref1_topdbNone actual_ref1_topdbNone

  let test_db_to_power () =
    let db_s =
      Rune.create Rune.c Rune.float32 [|2; 4|]
        [|0.; -10.; -20.; -30.; -80.; -80.; -80.; -80.|]
    in
    let expected =
      Rune.create Rune.c Rune.float32 [|2; 4|]
        [|1.; 0.1; 0.01; 0.001; 1e-8; 1e-8; 1e-8; 1e-8|]
    in
    let actual = Utils.Convert.db_to_power (Utils.Convert.RefFloat 1.0) db_s in
    Alcotest.check data_testable "db_to_power" expected actual

  let suite =
    [ Alcotest.test_case "hz_to_mel htk" `Quick test_hz_to_mel_htk
    ; Alcotest.test_case "mel_to_hz htk" `Quick test_mel_to_hz_htk
    ; Alcotest.test_case "power_to_db" `Quick test_power_to_db
    ; Alcotest.test_case "db_to_power" `Quick test_db_to_power ]
end

module Test_frame = struct
  (* Helper function to create random data for testing *)
  let create_random_data shape seed =
    Random.init seed ;
    let size = Array.fold_left ( * ) 1 shape in
    let data = Array.init size (fun _ -> Random.float 2.0 -. 1.0) in
    Rune.create Rune.c Rune.float32 shape data

  (* Test 1D framing with parametrized frame_length, hop_length, and axis *)
  let test_frame1d frame_length hop_length axis () =
    let y = create_random_data [|32|] 42 in
    let y_frame = Utils.frame y ~frame_length ~hop_length ~axis in
    let y_frame_adj = if axis = -1 then Rune.transpose y_frame else y_frame in
    let num_frames = (Rune.shape y_frame_adj).(0) in
    for i = 0 to num_frames - 1 do
      let frame_i = Rune.get [i] y_frame_adj in
      let start_idx = i * hop_length in
      let end_idx = min (start_idx + frame_length) 32 in
      let expected_slice = Rune.slice_ranges [start_idx] [end_idx] y in
      Alcotest.check data_testable
        (Printf.sprintf "frame1d_%d_%d_%d_frame_%d" frame_length hop_length axis
           i )
        expected_slice frame_i
    done

  (* Test 2D framing with parametrized frame_length, hop_length, axis, and array
     order *)
  let test_frame2d frame_length hop_length axis is_fortran_order () =
    let y_base = create_random_data [|16; 32|] 123 in
    let y =
      if is_fortran_order then
        (* Simulate Fortran order by transposing *)
        Rune.transpose y_base
      else y_base
    in
    let y_frame = Utils.frame y ~frame_length ~hop_length ~axis in
    let y_frame_adj, y_adj =
      if axis = -1 then (Rune.transpose y_frame, Rune.transpose y)
      else (y_frame, y)
    in
    let num_frames = (Rune.shape y_frame_adj).(0) in
    for i = 0 to num_frames - 1 do
      let frame_i = Rune.get [i] y_frame_adj in
      let start_idx = i * hop_length in
      let end_idx = min (start_idx + frame_length) (Rune.shape y_adj).(0) in
      let expected_slice = Rune.slice_ranges [start_idx] [end_idx] y_adj in
      Alcotest.check data_testable
        (Printf.sprintf "frame2d_%d_%d_%d_%b_frame_%d" frame_length hop_length
           axis is_fortran_order i )
        expected_slice frame_i
    done

  (* Test framing with 0-stride (padding) *)
  let test_frame_0stride () =
    let x = Rune.arange Rune.c Rune.float32 0 10 1 in
    let xpad = Rune.expand [|1; 10|] x in
    let xpad2 = Rune.reshape [|1; 10|] x in
    let xf = Utils.frame x ~frame_length:3 ~hop_length:1 in
    let xfpad = Utils.frame xpad ~frame_length:3 ~hop_length:1 in
    let xfpad2 = Utils.frame xpad2 ~frame_length:3 ~hop_length:1 in
    (* Check that shapes are correctly different due to extra dimensions *)
    let xf_shape = Rune.shape xf in
    let xfpad_shape = Rune.shape xfpad in
    let xfpad2_shape = Rune.shape xfpad2 in
    (* xf should be [3; 8] for axis=-1 *)
    Alcotest.check (Alcotest.array Alcotest.int) "xf_shape" [|3; 8|] xf_shape ;
    (* xfpad should be [1; 3; 8] - preserving the extra dimension *)
    Alcotest.check
      (Alcotest.array Alcotest.int)
      "xfpad_shape" [|1; 3; 8|] xfpad_shape ;
    (* xfpad2 should be same as xfpad *)
    Alcotest.check
      (Alcotest.array Alcotest.int)
      "xfpad2_shape" [|1; 3; 8|] xfpad2_shape ;
    (* Check that the core data is the same by comparing xf with xfpad[0] *)
    let xfpad_squeezed = Rune.get [0] xfpad in
    Alcotest.check data_testable "frame_0stride_xf_vs_xfpad_data" xf
      xfpad_squeezed ;
    let xfpad2_squeezed = Rune.get [0] xfpad2 in
    Alcotest.check data_testable "frame_0stride_xf_vs_xfpad2_data" xf
      xfpad2_squeezed

  (* Test high-dimensional framing *)
  let test_frame_highdim frame_length hop_length ndim () =
    let shape = Array.make ndim 20 in
    let x = create_random_data shape 456 in
    let xf = Utils.frame x ~frame_length ~hop_length in
    let first_dim_size = (Rune.shape x).(0) in
    for i = 0 to first_dim_size - 1 do
      let x_i = Rune.get [i] x in
      let xf0 = Utils.frame x_i ~frame_length ~hop_length in
      let xf_i = Rune.get [i] xf in
      Alcotest.check data_testable
        (Printf.sprintf "frame_highdim_%d_%d_%d_slice_%d" frame_length
           hop_length ndim i )
        xf0 xf_i
    done

  (* Test target axis framing *)
  let test_frame_targetaxis in_shape axis expected_out_shape () =
    let x = Rune.zeros Rune.c Rune.float32 in_shape in
    let xf = Utils.frame x ~frame_length:10 ~hop_length:2 ~axis in
    let actual_shape = Rune.shape xf in
    Alcotest.check
      (Alcotest.array Alcotest.int)
      "frame_targetaxis_shape" expected_out_shape actual_shape

  (* Test error cases *)
  let test_frame_too_short axis () =
    let x = Rune.arange Rune.c Rune.float32 0 16 1 in
    let expected_exn =
      Invalid_argument "Input is too short (n=16) for frame_length=17"
    in
    Alcotest.check_raises "frame_too_short" expected_exn (fun () ->
        ignore (Utils.frame x ~frame_length:17 ~hop_length:1 ~axis) )

  let test_frame_bad_hop () =
    let x = Rune.arange Rune.c Rune.float32 0 16 1 in
    let expected_exn = Invalid_argument "Invalid hop_length: 0" in
    Alcotest.check_raises "frame_bad_hop" expected_exn (fun () ->
        ignore (Utils.frame x ~frame_length:4 ~hop_length:0) )

  (* Generate parametrized test cases *)
  let frame1d_tests =
    let frame_lengths = [4; 8] in
    let hop_lengths = [2; 4] in
    let axes = [0; -1] in
    List.fold_left
      (fun acc frame_length ->
        List.fold_left
          (fun acc hop_length ->
            List.fold_left
              (fun acc axis ->
                let test_name =
                  Printf.sprintf "frame1d_fl%d_hl%d_ax%d" frame_length
                    hop_length axis
                in
                (test_name, `Quick, test_frame1d frame_length hop_length axis)
                :: acc )
              acc axes )
          acc hop_lengths )
      [] frame_lengths

  let frame2d_tests =
    let frame_lengths = [4; 8] in
    let hop_lengths = [2; 4] in
    let test_cases =
      [(-1, true); (* Fortran-like order *) (0, false) (* C order *)]
    in
    List.fold_left
      (fun acc frame_length ->
        List.fold_left
          (fun acc hop_length ->
            List.fold_left
              (fun acc (axis, fortran_order) ->
                let test_name =
                  Printf.sprintf "frame2d_fl%d_hl%d_ax%d_%s" frame_length
                    hop_length axis
                    (if fortran_order then "fortran" else "c")
                in
                ( test_name
                , `Quick
                , test_frame2d frame_length hop_length axis fortran_order )
                :: acc )
              acc test_cases )
          acc hop_lengths )
      [] frame_lengths

  let frame_highdim_tests =
    let frame_lengths = [5; 10] in
    let hop_lengths = [1; 2] in
    let ndims = [2; 3; 4; 5] in
    List.fold_left
      (fun acc frame_length ->
        List.fold_left
          (fun acc hop_length ->
            List.fold_left
              (fun acc ndim ->
                let test_name =
                  Printf.sprintf "frame_highdim_fl%d_hl%d_nd%d" frame_length
                    hop_length ndim
                in
                ( test_name
                , `Quick
                , test_frame_highdim frame_length hop_length ndim )
                :: acc )
              acc ndims )
          acc hop_lengths )
      [] frame_lengths

  let frame_targetaxis_tests =
    let test_cases =
      [ ([|20; 20; 20; 20|], 0, [|6; 10; 20; 20; 20|])
      ; ([|20; 20; 20; 20|], 1, [|20; 6; 10; 20; 20|])
      ; ([|20; 20; 20; 20|], 2, [|20; 20; 6; 10; 20|])
      ; ([|20; 20; 20; 20|], 3, [|20; 20; 20; 6; 10|])
      ; ([|20; 20; 20; 20|], -1, [|20; 20; 20; 10; 6|])
      ; ([|20; 20; 20; 20|], -2, [|20; 20; 10; 6; 20|])
      ; ([|20; 20; 20; 20|], -3, [|20; 10; 6; 20; 20|])
      ; ([|20; 20; 20; 20|], -4, [|10; 6; 20; 20; 20|]) ]
    in
    List.mapi
      (fun i (in_shape, axis, out_shape) ->
        let test_name = Printf.sprintf "frame_targetaxis_%d" i in
        (test_name, `Quick, test_frame_targetaxis in_shape axis out_shape) )
      test_cases

  let frame_error_tests =
    let axes = [0; -1] in
    List.mapi
      (fun _ axis ->
        let test_name = Printf.sprintf "frame_too_short_ax%d" axis in
        (test_name, `Quick, test_frame_too_short axis) )
      axes

  let suite =
    List.flatten
      [ frame1d_tests
      ; frame2d_tests
      ; [("frame_0stride", `Quick, test_frame_0stride)]
      ; frame_highdim_tests
      ; frame_targetaxis_tests
      ; [("frame_bad_hop", `Quick, test_frame_bad_hop)] @ frame_error_tests ]
end

let () =
  Alcotest.run "SoundML Utils Tests"
    [ ("Pad Center", Test_pad_center.suite)
    ; ("Mel Frequencies", Test_melfreq.suite)
    ; ("Unwrap", Test_unwrap.suite)
    ; ("Outer", Test_outer.suite)
    ; ("Conversions", Test_convert.suite)
    ; ("Frame", Test_frame.suite) ]
