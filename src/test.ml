let () =
  let open Soundml in
  let v = Io.read "mono.wav" "wav" in
  let stretched = Effects.Time.pitch_shift v (-6) in
  Io.write stretched "shifted.wav" "wav"
