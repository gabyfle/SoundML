let () =
  Owl.Log.set_level Owl.Log.DEBUG ;
  Printexc.record_backtrace true ;
  let open Soundml in
  Owl.Log.debug "Starting to read audio file" ;
  let start = Sys.time () in
  let audio = Io.read "music_is_moving.mp3" "mp3" in
  Owl.Log.debug "Done in %f;\n" (Sys.time () -. start) ;
  flush stdout ;
  Owl.Log.debug "Starting to compute spectrogram audio file" ;
  let start = Sys.time () in
  let spectrogram, _freqs = Feature.Spectral.specgram audio in
  Owl.Log.debug "Done in %f\n" (Sys.time () -. start) ;
  flush stdout ;
  let spectrogram = Audio.G.abs spectrogram in
  Npy.write (spectrogram |> Audio.G.re_c2s) "spectrogram.npy" ;
  Npy.write (Audio.data audio) "audio.npy"
