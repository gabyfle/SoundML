(* A mel spectrogram from a synthesized signal: the shortest end-to-end path
   through SoundML's analysis stack. Swap the synthetic tensor for
   [Soundml_io.read] to analyse a file. *)

let () =
  let sample_rate = 22050 in
  (* Two seconds of a 440 Hz tone with a touch of noise. *)
  let n = 2 * sample_rate in
  let signal =
    Nx.init Nx.float32 [|n|] (fun i ->
        let t = Float.of_int i.(0) /. Float.of_int sample_rate in
        (0.6 *. Float.sin (2.0 *. Float.pi *. 440.0 *. t))
        +. (0.01 *. Float.sin (2.0 *. Float.pi *. 4419.0 *. t)) )
  in
  let stft = Soundml.Stft.Config.create ~fft_size:2048 ~hop:512 () in
  let mel =
    Soundml.Mel.Config.create ~n_mels:128 ~sample_rate ~fft_size:2048 ()
  in
  let spectrogram = Soundml.mel_spectrogram stft mel signal in
  let shape = Nx.shape spectrogram in
  Printf.printf "mel spectrogram: %d bands x %d frames\n" shape.(0) shape.(1) ;
  let peak = Nx.item [] (Nx.max spectrogram) in
  Printf.printf "peak mel power: %.6f\n" peak
