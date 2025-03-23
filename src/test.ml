let () =
  let open Soundml in
  let v = Io.read "not_a_name.wav" "wav" in
  let stretched = Effects.Time.time_stretch v 4. in
  Io.write stretched "not_a_chipmunks.wav" "wav"
