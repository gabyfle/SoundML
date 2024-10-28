let () =
  Owl.Log.set_level Owl.Log.DEBUG ;
  Printexc.record_backtrace true ;
  let open Soundml in
  let audio = Io.read "music_is_moving.mp3" "mp3" in
  let time = Unix.gettimeofday () in
  let phase, _ = Feature.Spectral.phase_specgram audio in
  Printf.printf "Elapsed time for magnitude specgram: %f\n"
    (Unix.gettimeofday () -. time) ;
  Npy.write phase "phase.npy" ;
  Io.write audio "output.wav" "wav"
