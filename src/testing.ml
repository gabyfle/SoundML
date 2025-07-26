open Soundml


let () =
  let a, sr = Io.read Nx.Float64 ~res_typ:Io.NONE "/Users/gabyfle/Documents/SoundML/databench/wav/audio_001.wav" in
  Printf.printf "Numels: %d\n" (Nx.numel a);
  Printf.printf "Ndim: %d\n" (Nx.ndim a);
  Printf.printf "Sample rate: %d\n" sr