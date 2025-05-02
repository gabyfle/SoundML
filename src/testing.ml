open Soundml

let () =
  let a =
    Io.read Bigarray.Float32 ~sample_rate:22050 ~res_typ:Io.SOXR_VHQ ~mono:false
      "/home/gabyfle/Code/soundml/test/audio/freemusicarchiveorgadajonestaffy.mp3"
  in
  Printf.printf "Samples: %d; Channels: %d\n" (Audio.samples a)
    (Audio.channels a) ;
  Printf.printf "Shape of audio:\n" ;
  Array.iter (fun x -> Printf.printf "%d;" x) (Audio.G.shape (Audio.data a)) ;
  Printf.printf "\n"
