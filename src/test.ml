let () =
  Printexc.record_backtrace true ;
  let open Soundml in
  let audio = Io.read_audio "test/sin_1k.wav" "wav" in
  Printf.printf "Audio size: %d\n" (Audio.size audio) ;
  Io.write_audio audio "testing.flac" "flac"
