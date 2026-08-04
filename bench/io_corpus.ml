(* The deterministic corpus behind the io benchmark rows.

   Generated once into a stable directory (SOUNDML_IO_BENCH_CORPUS, or
   [<tmp>/soundml-io-bench-corpus-<version>]) by soundml-io's own writer, so the
   C ceiling harness (bench/bench_sndfile.c), the Python cross-reference
   (bench/bench_soundml_io.py) and the thumper rows all read identical bytes —
   the fair-fight precondition.

   Signal recipe, fixed for reproducibility: per channel [c], a chirp [0.45 *
   sin(2*pi*(f0 + 400 t) t)] with [f0 = 220 (c+1)], plus one-pole lowpassed LCG
   noise at 0.3 amplitude — audio-like content (FLAC compresses it to roughly
   half size; pure noise would be incompressible, a pure sine trivially so).
   Seeds are pinned per file below.

   Axes: the six write-surface format families x {1 s mono 22050 Hz (the
   small/ingest class), 30 s stereo 44100 Hz and 30 s mono 44100 Hz (the bulk
   class, with and without the planar transpose)}, plus the many-small-files
   workload: 1000 x 1 s mono 22050 Hz clips in wav-pcm16, wav-float32 and flac,
   with a [.list] file per class for the C harness. *)

let version = "v2"

let dir () =
  match Sys.getenv_opt "SOUNDML_IO_BENCH_CORPUS" with
  | Some d ->
      d
  | None ->
      Filename.concat
        (Filename.get_temp_dir_name ())
        ("soundml-io-bench-corpus-" ^ version)

let formats =
  [ ("wav-pcm16", Soundml_io.Format.create ~encoding:`Pcm_16 `Wav, ".wav")
  ; ("wav-pcm24", Soundml_io.Format.create ~encoding:`Pcm_24 `Wav, ".wav")
  ; ("wav-pcm32", Soundml_io.Format.create ~encoding:`Pcm_32 `Wav, ".wav")
  ; ("wav-float32", Soundml_io.Format.create ~encoding:`Float32 `Wav, ".wav")
  ; ("flac", Soundml_io.Format.create ~encoding:`Pcm_16 `Flac, ".flac")
  ; ("ogg", Soundml_io.Format.create `Ogg, ".ogg") ]

let many_classes = ["wav-pcm16"; "wav-float32"; "flac"]

let small_path ~root label ext =
  Filename.concat root (Printf.sprintf "%s_22050_mo_1s%s" label ext)

let bulk_path ~root label ext =
  Filename.concat root (Printf.sprintf "%s_44100_st_30s%s" label ext)

let bulk_mono_path ~root label ext =
  Filename.concat root (Printf.sprintf "%s_44100_mo_30s%s" label ext)

let small label =
  let _, _, ext = List.find (fun (l, _, _) -> l = label) formats in
  small_path ~root:(dir ()) label ext

let bulk label =
  let _, _, ext = List.find (fun (l, _, _) -> l = label) formats in
  bulk_path ~root:(dir ()) label ext

let bulk_mono label =
  let _, _, ext = List.find (fun (l, _, _) -> l = label) formats in
  bulk_mono_path ~root:(dir ()) label ext

let many_dir label = Filename.concat (Filename.concat (dir ()) "many") label

let many label =
  let _, _, ext = List.find (fun (l, _, _) -> l = label) formats in
  List.init 1000 (fun i ->
      Filename.concat (many_dir label) (Printf.sprintf "clip_%04d%s" i ext) )

let signal ~channels ~frames ~sample_rate seed =
  let data = Array.make (channels * frames) 0. in
  for c = 0 to channels - 1 do
    let state = ref (seed + (100 * c)) in
    let draw () =
      state := ((1103515245 * !state) + 12345) land 0x3FFFFFFF ;
      (2. *. (Float.of_int !state /. Float.of_int (1 lsl 30))) -. 1.
    in
    let lp = ref 0. in
    let f0 = 220. *. Float.of_int (c + 1) in
    for i = 0 to frames - 1 do
      let t = Float.of_int i /. Float.of_int sample_rate in
      lp := (0.85 *. !lp) +. (0.15 *. draw ()) ;
      data.((c * frames) + i) <-
        (0.45 *. sin (2. *. Float.pi *. (f0 +. (400. *. t)) *. t))
        +. (0.3 *. !lp)
    done
  done ;
  Nx.create Nx.float32 [|channels; frames|] data

let write_or_die path data sample_rate format =
  match Soundml_io.write ~format path {data; sample_rate} with
  | Ok () ->
      ()
  | Error e ->
      failwith (Stdlib.Format.asprintf "io corpus: %a" Soundml_io.pp_error e)

let generate root =
  let mkdir d = if not (Sys.file_exists d) then Sys.mkdir d 0o755 in
  mkdir root ;
  let one_second = signal ~channels:1 ~frames:22050 ~sample_rate:22050 101 in
  let thirty = signal ~channels:2 ~frames:(30 * 44100) ~sample_rate:44100 202 in
  let thirty_mono =
    signal ~channels:1 ~frames:(30 * 44100) ~sample_rate:44100 303
  in
  List.iter
    (fun (label, format, ext) ->
      write_or_die (small_path ~root label ext) one_second 22050 format ;
      write_or_die (bulk_path ~root label ext) thirty 44100 format ;
      write_or_die (bulk_mono_path ~root label ext) thirty_mono 44100 format )
    formats ;
  let many_root = Filename.concat root "many" in
  mkdir many_root ;
  List.iter
    (fun label ->
      let _, format, ext = List.find (fun (l, _, _) -> l = label) formats in
      let d = Filename.concat many_root label in
      mkdir d ;
      let paths =
        List.init 1000 (fun i ->
            let clip =
              signal ~channels:1 ~frames:22050 ~sample_rate:22050 (20000 + i)
            in
            let p = Filename.concat d (Printf.sprintf "clip_%04d%s" i ext) in
            write_or_die p clip 22050 format ;
            p )
      in
      Out_channel.with_open_text
        (Filename.concat many_root (label ^ ".list"))
        (fun oc ->
          List.iter (fun p -> Out_channel.output_string oc (p ^ "\n")) paths ) )
    many_classes

(* [ensure ()] is the corpus root, generated on first use (a marker commits a
   complete generation; a partial one is redone). Generation also serves as the
   page-cache warmup for the read rows. *)
let ensure () =
  let root = dir () in
  let marker = Filename.concat root ("COMPLETE-" ^ version) in
  if not (Sys.file_exists marker) then begin
    generate root ;
    Out_channel.with_open_text marker (fun oc ->
        Out_channel.output_string oc "complete\n" )
  end ;
  root
