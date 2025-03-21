let () =
  let open Soundml in
  let v = Io.read "mono.wav" "wav" in
  let stretched = Effects.Time.time_stretch v 0.5 in
  Io.write stretched "stretched.wav" "wav"
