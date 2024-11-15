let () =
  let open Soundml in
  let t = Io.read "test/sin_15k.wav" "wav" in
  let tt = Sys.time () in
  let mfcc = Feature.Spectral.mfcc t in
  Printf.printf "Time: %f\n" (Sys.time () -. tt) ;
  Npy.write mfcc "mfcc.npy"
