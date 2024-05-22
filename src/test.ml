let () =
  Printexc.record_backtrace true ;
  let open Soundml in
  let audio = Io.read_audio "test/sin_1k.wav" "wav" in
  let audio = Audio.normalise audio in
  Printf.printf "Audio size: %d\n" (Audio.size audio) ;
  Io.write_audio audio "output.mp3" "libmp3lame"
