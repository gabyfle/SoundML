open! Core
open! Core_bench
open Soundml

let path = Sys_unix.getcwd () ^ "/bench/stft/wav_stereo_44100hz_1s.wav"

let audio, _ = Io.read ~res_typ:Io.NONE Nx.Float64 path

let main () =
  Command_unix.run
    (Bench.make_command
       [ Bench.Test.create ~name:"float64" (fun () ->
             ignore (Transform.stft audio) ) ] )

let () = main ()
