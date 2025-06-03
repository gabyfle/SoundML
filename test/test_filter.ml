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

module F = Effects.Filter

let float_ndarray = Tutils.dense_testable ~atol:1e-7 Bigarray.Float64

let impulse =
  Audio.G.of_array Float64
    [| 1.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000
     ; 0.0000000 |]
    [|32|]

let test_lp_100hz () =
  let lp_filter = F.IIR.LowPass.create {cutoff= 100.0; sample_rate= 44100} in
  let output = F.IIR.LowPass.process lp_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.0070735
       ; 0.0140470
       ; 0.0138483
       ; 0.0136523
       ; 0.0134592
       ; 0.0132688
       ; 0.0130811
       ; 0.0128960
       ; 0.0127136
       ; 0.0125337
       ; 0.0123564
       ; 0.0121816
       ; 0.0120093
       ; 0.0118394
       ; 0.0116719
       ; 0.0115068
       ; 0.0113440
       ; 0.0111835
       ; 0.0110253
       ; 0.0108693
       ; 0.0107155
       ; 0.0105639
       ; 0.0104145
       ; 0.0102672
       ; 0.0101219
       ; 0.0099787
       ; 0.0098375
       ; 0.0096984
       ; 0.0095612
       ; 0.0094259
       ; 0.0092926
       ; 0.0091611 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "lp_100hz impulse response" expected_output output

let test_lp_1000hz () =
  let lp_filter = F.IIR.LowPass.create {cutoff= 1000.0; sample_rate= 44100} in
  let output = F.IIR.LowPass.process lp_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.0666058
       ; 0.1243389
       ; 0.1077755
       ; 0.0934186
       ; 0.0809741
       ; 0.0701875
       ; 0.0608377
       ; 0.0527334
       ; 0.0457087
       ; 0.0396198
       ; 0.0343420
       ; 0.0297672
       ; 0.0258019
       ; 0.0223648
       ; 0.0193855
       ; 0.0168031
       ; 0.0145648
       ; 0.0126246
       ; 0.0109428
       ; 0.0094851
       ; 0.0082216
       ; 0.0071264
       ; 0.0061771
       ; 0.0053542
       ; 0.0046410
       ; 0.0040227
       ; 0.0034869
       ; 0.0030224
       ; 0.0026198
       ; 0.0022708
       ; 0.0019683
       ; 0.0017061 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "lp_1000hz impulse response" expected_output output

let test_lp_5000hz () =
  let lp_filter = F.IIR.LowPass.create {cutoff= 5000.0; sample_rate= 44100} in
  let output = F.IIR.LowPass.process lp_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.2711683
       ; 0.3952721
       ; 0.1809016
       ; 0.0827920
       ; 0.0378909
       ; 0.0173413
       ; 0.0079365
       ; 0.0036322
       ; 0.0016623
       ; 0.0007608
       ; 0.0003482
       ; 0.0001594
       ; 0.0000729
       ; 0.0000334
       ; 0.0000153
       ; 0.0000070
       ; 0.0000032
       ; 0.0000015
       ; 0.0000007
       ; 0.0000003
       ; 0.0000001
       ; 0.0000001
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "lp_5000hz impulse response" expected_output output

let test_hp_100hz () =
  let hp_filter = F.IIR.HighPass.create {cutoff= 100.0; sample_rate= 44100} in
  let output = F.IIR.HighPass.process hp_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.9929265
       ; -0.0140470
       ; -0.0138483
       ; -0.0136523
       ; -0.0134592
       ; -0.0132688
       ; -0.0130811
       ; -0.0128960
       ; -0.0127136
       ; -0.0125337
       ; -0.0123564
       ; -0.0121816
       ; -0.0120093
       ; -0.0118394
       ; -0.0116719
       ; -0.0115068
       ; -0.0113440
       ; -0.0111835
       ; -0.0110253
       ; -0.0108693
       ; -0.0107155
       ; -0.0105639
       ; -0.0104145
       ; -0.0102672
       ; -0.0101219
       ; -0.0099787
       ; -0.0098375
       ; -0.0096984
       ; -0.0095612
       ; -0.0094259
       ; -0.0092926
       ; -0.0091611 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "hp_100hz impulse response" expected_output output

let test_hp_1000hz () =
  let hp_filter = F.IIR.HighPass.create {cutoff= 1000.0; sample_rate= 44100} in
  let output = F.IIR.HighPass.process hp_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.9333942
       ; -0.1243389
       ; -0.1077755
       ; -0.0934186
       ; -0.0809741
       ; -0.0701875
       ; -0.0608377
       ; -0.0527334
       ; -0.0457087
       ; -0.0396198
       ; -0.0343420
       ; -0.0297672
       ; -0.0258019
       ; -0.0223648
       ; -0.0193855
       ; -0.0168031
       ; -0.0145648
       ; -0.0126246
       ; -0.0109428
       ; -0.0094851
       ; -0.0082216
       ; -0.0071264
       ; -0.0061771
       ; -0.0053542
       ; -0.0046410
       ; -0.0040227
       ; -0.0034869
       ; -0.0030224
       ; -0.0026198
       ; -0.0022708
       ; -0.0019683
       ; -0.0017061 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "hp_1000hz impulse response" expected_output output

let test_hp_5000hz () =
  let hp_filter = F.IIR.HighPass.create {cutoff= 5000.0; sample_rate= 44100} in
  let output = F.IIR.HighPass.process hp_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.7288317
       ; -0.3952721
       ; -0.1809016
       ; -0.0827920
       ; -0.0378909
       ; -0.0173413
       ; -0.0079365
       ; -0.0036322
       ; -0.0016623
       ; -0.0007608
       ; -0.0003482
       ; -0.0001594
       ; -0.0000729
       ; -0.0000334
       ; -0.0000153
       ; -0.0000070
       ; -0.0000032
       ; -0.0000015
       ; -0.0000007
       ; -0.0000003
       ; -0.0000001
       ; -0.0000001
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000
       ; -0.0000000 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "hp_5000hz impulse response" expected_output output

let test_iir_generic_lp_manual () =
  let iir_filter =
    F.IIR.Generic.create
      {b= [|0.1000000; 0.1000000|]; a= [|1.0000000; -0.8000000|]}
  in
  let output = F.IIR.Generic.process iir_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.1000000
       ; 0.1800000
       ; 0.1440000
       ; 0.1152000
       ; 0.0921600
       ; 0.0737280
       ; 0.0589824
       ; 0.0471859
       ; 0.0377487
       ; 0.0301990
       ; 0.0241592
       ; 0.0193274
       ; 0.0154619
       ; 0.0123695
       ; 0.0098956
       ; 0.0079165
       ; 0.0063332
       ; 0.0050665
       ; 0.0040532
       ; 0.0032426
       ; 0.0025941
       ; 0.0020753
       ; 0.0016602
       ; 0.0013282
       ; 0.0010625
       ; 0.0008500
       ; 0.0006800
       ; 0.0005440
       ; 0.0004352
       ; 0.0003482
       ; 0.0002785
       ; 0.0002228 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "iir_generic_lp_manual impulse response" expected_output output

let test_iir_generic_hp_manual () =
  let iir_filter =
    F.IIR.Generic.create
      {b= [|0.5000000; -0.5000000|]; a= [|1.0000000; -0.8000000|]}
  in
  let output = F.IIR.Generic.process iir_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.5000000
       ; -0.1000000
       ; -0.0800000
       ; -0.0640000
       ; -0.0512000
       ; -0.0409600
       ; -0.0327680
       ; -0.0262144
       ; -0.0209715
       ; -0.0167772
       ; -0.0134218
       ; -0.0107374
       ; -0.0085899
       ; -0.0068719
       ; -0.0054976
       ; -0.0043980
       ; -0.0035184
       ; -0.0028147
       ; -0.0022518
       ; -0.0018014
       ; -0.0014412
       ; -0.0011529
       ; -0.0009223
       ; -0.0007379
       ; -0.0005903
       ; -0.0004722
       ; -0.0003778
       ; -0.0003022
       ; -0.0002418
       ; -0.0001934
       ; -0.0001547
       ; -0.0001238 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "iir_generic_hp_manual impulse response" expected_output output

let test_iir_generic_allpass () =
  let iir_filter =
    F.IIR.Generic.create
      {b= [|-0.5000000; 1.0000000|]; a= [|1.0000000; -0.5000000|]}
  in
  let output = F.IIR.Generic.process iir_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| -0.5000000
       ; 0.7500000
       ; 0.3750000
       ; 0.1875000
       ; 0.0937500
       ; 0.0468750
       ; 0.0234375
       ; 0.0117188
       ; 0.0058594
       ; 0.0029297
       ; 0.0014648
       ; 0.0007324
       ; 0.0003662
       ; 0.0001831
       ; 0.0000916
       ; 0.0000458
       ; 0.0000229
       ; 0.0000114
       ; 0.0000057
       ; 0.0000029
       ; 0.0000014
       ; 0.0000007
       ; 0.0000004
       ; 0.0000002
       ; 0.0000001
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "iir_generic_allpass impulse response" expected_output output

let test_fir_generic_delay () =
  let fir_filter =
    F.FIR.Generic.create {b= [|0.0000000; 0.0000000; 1.0000000|]}
  in
  let output = F.FIR.Generic.process fir_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.0000000
       ; 0.0000000
       ; 1.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "fir_generic_delay impulse response" expected_output output

let test_fir_generic_moving_average_3pt () =
  let fir_filter =
    F.FIR.Generic.create {b= [|0.3333333; 0.3333333; 0.3333333|]}
  in
  let output = F.FIR.Generic.process fir_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.3333333
       ; 0.3333333
       ; 0.3333333
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "fir_generic_moving_average_3pt impulse response" expected_output output

let test_fir_generic_simple_lp () =
  let fir_filter =
    F.FIR.Generic.create {b= [|0.2500000; 0.5000000; 0.2500000|]}
  in
  let output = F.FIR.Generic.process fir_filter impulse in
  let expected_output =
    Audio.G.of_array Float64
      [| 0.2500000
       ; 0.5000000
       ; 0.2500000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000
       ; 0.0000000 |]
      [|32|]
  in
  Alcotest.(check float_ndarray)
    "fir_generic_simple_lp impulse response" expected_output output

let suite =
  [ ( "IIR LowPass"
    , [ Alcotest.test_case "LowPass 100 Hz" `Quick test_lp_100hz
      ; Alcotest.test_case "LowPass 1000 Hz" `Quick test_lp_1000hz
      ; Alcotest.test_case "LowPass 5000 Hz" `Quick test_lp_5000hz ] )
  ; ( "IIR HighPass"
    , [ Alcotest.test_case "HighPass 100 Hz" `Quick test_hp_100hz
      ; Alcotest.test_case "HighPass 1000 Hz" `Quick test_hp_1000hz
      ; Alcotest.test_case "HighPass 5000 Hz" `Quick test_hp_5000hz ] )
  ; ( "IIR Generic"
    , [ Alcotest.test_case "iir_generic_lp_manual" `Quick
          test_iir_generic_lp_manual
      ; Alcotest.test_case "iir_generic_hp_manual" `Quick
          test_iir_generic_hp_manual
      ; Alcotest.test_case "iir_generic_allpass" `Quick test_iir_generic_allpass
      ] )
  ; ( "FIR Generic"
    , [ Alcotest.test_case "fir_generic_delay" `Quick test_fir_generic_delay
      ; Alcotest.test_case "fir_generic_moving_average_3pt" `Quick
          test_fir_generic_moving_average_3pt
      ; Alcotest.test_case "fir_generic_simple_lp" `Quick
          test_fir_generic_simple_lp ] ) ]

let () = Alcotest.run "Effects Filter Tests" suite
