let () =
  Owl.Log.set_level Owl.Log.DEBUG ;
  Printexc.record_backtrace true ;
  let open Soundml in
  let audio = Io.read "test/noise.wav" "wav" in
  let rms = Feature.Temporal.rms ~window:2048 ~step:1024 audio in
  let spec, _ = Feature.Spectral.specgram audio in
  let mag, _ =
    Feature.Spectral.magnitude_specgram audio ~nfft:(Audio.rawsize audio)
      ~noverlap:0
  in
  Npy.write rms "rms.npy" ;
  Npy.write spec "spec.npy" ;
  Npy.write mag "mag.npy" ;
  Io.write audio "output.wav" "wav"
