open Soundml

let () =
  let a =
    Io.read Bigarray.Float32 ~sample_rate:22050 ~res_typ:Io.SOXR_VHQ ~mono:false
      "/home/gabyfle/Code/soundml/test/audio/freesoundorgsadiquecatquaidelaglyoiseaux.flac"
  in
  Printf.printf "Samples: %d; Channels: %d\n" (Audio.samples a)
    (Audio.channels a)
