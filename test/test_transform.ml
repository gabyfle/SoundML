open Soundml

let test_stft_defaults_window_length_to_n_fft () =
  let signal = Nx.arange_f Nx.float64 0. 64. 1. in
  let stft = Transform.stft ~n_fft:16 ~hop_length:8 signal in
  Alcotest.check
    (Alcotest.array Alcotest.int)
    "STFT shape" [|9; 9|] (Nx.shape stft)

let test_stft_accepts_float32 () =
  let signal = Nx.create Nx.float32 [|4|] [|0.; 1.; 2.; 3.|] in
  let actual =
    Transform.stft ~window:`Boxcar ~center:false ~n_fft:4 ~hop_length:4 signal
    |> Nx.to_array
  in
  let expected =
    [| {Complex.re= 6.; im= 0.}
     ; {Complex.re= -2.; im= 2.}
     ; {Complex.re= -2.; im= 0.} |]
  in
  Alcotest.check Alcotest.int "spectrum length" (Array.length expected)
    (Array.length actual) ;
  Array.iteri
    (fun i expected ->
      Alcotest.check (Alcotest.float 1e-5)
        (Printf.sprintf "bin %d real" i)
        expected.Complex.re actual.(i).Complex.re ;
      Alcotest.check (Alcotest.float 1e-5)
        (Printf.sprintf "bin %d imaginary" i)
        expected.Complex.im actual.(i).Complex.im )
    expected

let () =
  Alcotest.run "SoundML Transform"
    [ ( "STFT"
      , [ Alcotest.test_case "win_length defaults to n_fft" `Quick
            test_stft_defaults_window_length_to_n_fft
        ; Alcotest.test_case "accepts float32 input" `Quick
            test_stft_accepts_float32 ] ) ]
