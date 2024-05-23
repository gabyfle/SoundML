let () =
  Printexc.record_backtrace true ;
  let open Soundml in
  let beg = Sys.time () in
  Printf.printf "Starting to read audio file\n" ;
  let start = Sys.time () in
  let audio = Io.read_audio "test/sin_1k.wav" "wav" in
  let meta = Audio.meta audio in
  Printf.printf "Rawsize: %d\n" (Audio.rawsize audio) ;
  Printf.printf "Sample rate %d\n" (Audio.Metadata.sample_rate meta) ;
  Printf.printf "Channels %d\n" (Audio.Metadata.channels meta) ;
  Printf.printf "Sample width %d\n" (Audio.Metadata.sample_width meta) ;
  Printf.printf "Done in %f; Length %d\n"
    (Sys.time () -. start)
    (Audio.length audio) ;
  (*Printf.printf "Starting to normalize audio file\n" ; let start = Sys.time ()
    in Audio.normalise audio ; Io.write_audio audio "output.mp3" "mp3" ;*)
  Printf.printf "Done in %f; Total time: %f\n"
    (Sys.time () -. start)
    (Sys.time () -. beg)
