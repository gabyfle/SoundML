open! Core
open! Core_bench
open Soundml

let path = Sys_unix.getcwd () ^ "/bench/stft/wav_stereo_44100hz_1s.wav"

let f32audio = Audio.data @@ Io.read ~res_typ:Io.NONE Bigarray.Float32 path

let f64audio = Audio.data @@ Io.read ~res_typ:Io.NONE Bigarray.Float64 path

let main () =
  Command_unix.run
    (Bench.make_command
       [ Bench.Test.create ~name:"float32" (fun () ->
             ignore (Transform.stft Types.B32 f32audio) )
       ; Bench.Test.create ~name:"float64" (fun () ->
             ignore (Transform.stft Types.B64 f64audio) ) ] )

let () = main ()
