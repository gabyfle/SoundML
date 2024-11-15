let () =
  let open Soundml in
  let t = Io.read "test/sin_15k.wav" "wav" in
  let tt = Sys.time () in
  let mfcc = Feature.Spectral.mfcc t in
  Printf.printf "Time: %f\n" (Sys.time () -. tt) ;
  let tt =
    Audio.G.of_array Bigarray.Float32
      [|1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.|]
      [|10|]
  in
  let t = Utils.Convert.power_to_db (RefFloat 1.) tt in
  Audio.G.print t ; Npy.write mfcc "mfcc.npy"
