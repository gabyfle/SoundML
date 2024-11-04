let () =
  Owl.Log.set_level Owl.Log.DEBUG ;
  Printexc.record_backtrace true ;
  let open Soundml in
  let audio = Io.read "music_is_moving.mp3" "mp3" in
  let mel, _ = Feature.Spectral.mel_specgram audio in
  Npy.write (Audio.data audio) "audio.npy" ;
  Npy.write mel "mel.npy"
