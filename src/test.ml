(*let () = let mat = Owl.Dense.Ndarray.D.zeros [|6; 6|] in let vector =
  Owl.Dense.Ndarray.D.ones [|6|] in let vector = Owl.Dense.Ndarray.D.reshape
  vector [|6; 1|] in Owl.Dense.Ndarray.D.print mat ;
  Owl.Dense.Ndarray.D.set_slice [[]; [0]] mat vector ; Owl.Dense.Ndarray.D.print
  mat*)

let value_to_color value =
  let normalized_value = min 1.0 (max 0.0 value) in
  let g = int_of_float (255. *. (1. -. normalized_value)) in
  Graphics.rgb g g g

let _plot_spectrogram f t spectrogram =
  let spectrogram = Audio.G.abs spectrogram |> Audio.G.re_z2d in
  let width = 1280 in
  let height = 720 in
  let spec_width = Array.length t in
  let spec_height = Array.length f in
  Graphics.open_graph (Printf.sprintf " %dx%d" width height) ;
  let min_db = -120. in
  let max_db = 0. in
  for i = 0 to spec_width - 1 do
    for j = 0 to spec_height - 1 do
      let value =
        Audio.G.get spectrogram [|j; i|] |> log10 |> fun x -> 10. *. x
      in
      let normalized_value = (value -. min_db) /. (max_db -. min_db) in
      Graphics.set_color (value_to_color normalized_value) ;
      let x = i * width / spec_width in
      let y = j * height / spec_height in
      Graphics.fill_rect x y (width / spec_width) (height / spec_height)
    done
  done ;
  try
    let key_pressed = Graphics.wait_next_event [Graphics.Key_pressed] in
    if key_pressed.Graphics.keypressed then
      match key_pressed.Graphics.key with 'q' -> raise Exit | _ -> ()
  with Exit | Graphics.Graphic_failure _ -> Graphics.close_graph () ; exit 0

let () =
  Owl.Log.set_level Owl.Log.DEBUG ;
  Printexc.record_backtrace true ;
  let open Soundml in
  let beg = Sys.time () in
  Owl.Log.debug "Starting to read audio file" ;
  let start = Sys.time () in
  let audio = Io.read_audio "test.wav" "wav" in
  Owl.Log.debug "Done in %f;\n" (Sys.time () -. start) ;
  flush stdout ;
  Owl.Log.debug "Starting to compute spectrogram audio file" ;
  let start = Sys.time () in
  let spectrogram, _freqs = Specgram.specgram audio in
  Owl.Log.debug "Done in %f\n" (Sys.time () -. start) ;
  flush stdout ;
  (*Npy.write (Audio.data audio) "audio.npy" ;*)
  Npy.write spectrogram "spectrogram.npy" ;
  (*Npy.write freqs "freqs.npy" ;*)
  Owl.Log.debug "Starting to write audio file" ;
  let start = Sys.time () in
  Io.write_audio audio "output.mp3" "mp3" ;
  Owl.Log.debug "Done in %f\n" (Sys.time () -. start) ;
  Owl.Log.debug "Total time: %f\n" (Sys.time () -. beg)
