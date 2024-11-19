let () =
  let open Soundml in
  let a = Io.read "test/sin_2k.wav" "wav" in
  let mfcc = Feature.Spectral.mfcc a in
  Npy.write mfcc "mfcc.npy"
