let () =
  Owl.Log.set_level Owl.Log.DEBUG ;
  Printexc.record_backtrace true ;
  let open Soundml in
  let audio = Io.read "music_is_moving.mp3" "mp3" in
  let time = Unix.gettimeofday () in
  let mag, _ = Feature.Spectral.magnitude_specgram audio in
  Printf.printf "Elapsed time for magnitude specgram: %f\n"
    (Unix.gettimeofday () -. time) ;
  Npy.write mag "mag.npy" ;
  Io.write audio "output.wav" "wav"
