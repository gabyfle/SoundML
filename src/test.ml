let () =
  Printexc.record_backtrace true ;
  let open Soundml in
  let beg = Sys.time () in
  Printf.printf "Starting to read audio file\n" ;
  let start = Sys.time () in
  let audio = Io.read_audio "testing.wav" "wav" in
  Audio.normalise audio ;
  Printf.printf "Done in %f; Size %d\n"
    (Sys.time () -. start)
    (Audio.size audio) ;
  (*Printf.printf "Starting to compute fft\n" ; let start = Sys.time () in let
    filtered = Analysis.fft audio in Printf.printf "Done in %f\n" (Sys.time ()
    -. start) ; Printf.printf "Starting to compute ifft\n" ; let start =
    Sys.time () in let reversed = Analysis.ifft filtered in Printf.printf "Done
    in %f\n" (Sys.time () -. start) ; let audio = Audio.set_data audio reversed
    in*)
  Printf.printf "Starting to write back audio\n" ;
  let start = Sys.time () in
  Io.write_audio audio "output.wav" "wav" ;
  Printf.printf "Done in %f; Total time: %f\n"
    (Sys.time () -. start)
    (Sys.time () -. beg)
