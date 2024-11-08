let () =
  Owl.Log.set_level Owl.Log.DEBUG ;
  Printexc.record_backtrace true ;
  let open Soundml in
  let audio = Io.read "test/sin_2k.wav" "wav" in
  let mel, _ = Feature.Spectral.mel_specgram audio in
  Npy.write (Audio.data audio) "audio.npy" ;
  Npy.write mel "mel.npy"
