(* The io gate runner: min-of-N wall time (median as sanity) for every cell the
   io performance gates compare against the C ceiling harness — the same
   statistic, the same corpus, one session. The thumper rows remain the
   regression ratchets; this runner is the measurement instrument behind the
   P1-P6 verdicts in bench/README.md.

   Run from the repo root: dune exec bench/profile/gate_io.exe Output:
   name,n,min_us,median_us rows (CSV). *)

module Io = Soundml_io

let ok ~what = function
  | Ok v ->
      v
  | Error e ->
      failwith (Stdlib.Format.asprintf "%s: %a" what Io.pp_error e)

let measure name n f =
  ignore (f ()) ;
  let times = Array.make n 0. in
  for i = 0 to n - 1 do
    let t0 = Unix.gettimeofday () in
    ignore (f ()) ;
    times.(i) <- Unix.gettimeofday () -. t0
  done ;
  Array.sort compare times ;
  Printf.printf "%s,%d,%.2f,%.2f\n%!" name n
    (times.(0) *. 1e6)
    (times.(n / 2) *. 1e6)

let families =
  ["wav-pcm16"; "wav-pcm24"; "wav-pcm32"; "wav-float32"; "flac"; "ogg"]

let () =
  let (_ : string) = Io_corpus.ensure () in
  (* reads: allocating and ?out, small / bulk stereo / bulk mono *)
  List.iter
    (fun fam ->
      let cells =
        [ ("1s", Io_corpus.small fam, 200)
        ; ("30s", Io_corpus.bulk fam, 30)
        ; ("30s-mo", Io_corpus.bulk_mono fam, 30) ]
      in
      List.iter
        (fun (tag, path, n) ->
          measure (Printf.sprintf "read %s %s" fam tag) n (fun () ->
              (ok ~what:fam (Io.read Nx.float32 path)).Io.data ) ;
          let i = ok ~what:fam (Io.info path) in
          let out =
            Nx.zeros Nx.float32 [|i.Io.Info.channels; i.Io.Info.frames|]
          in
          measure (Printf.sprintf "reader_out %s %s" fam tag) n (fun () ->
              let r = ok ~what:fam (Io.Reader.open_ Nx.float32 path) in
              let c =
                ok ~what:fam (Io.Reader.read ~out r ~frames:i.Io.Info.frames)
              in
              Io.Reader.close r ;
              match c with Some _ -> () | None -> failwith "empty" ) )
        cells ;
      measure (Printf.sprintf "info %s" fam) 300 (fun () ->
          ok ~what:fam (Io.info (Io_corpus.bulk fam)) ) )
    families ;
  (* the many-small-files ingest loops *)
  List.iter
    (fun cls ->
      let paths = Io_corpus.many cls in
      measure (Printf.sprintf "many %s" cls) 12 (fun () ->
          List.iter
            (fun p -> ignore (ok ~what:cls (Io.read Nx.float32 p) : _ Io.audio))
            paths ) ;
      let out = Nx.zeros Nx.float32 [|1; 22050|] in
      measure (Printf.sprintf "many_out %s" cls) 12 (fun () ->
          List.iter
            (fun p ->
              let r = ok ~what:cls (Io.Reader.open_ Nx.float32 p) in
              ( match Io.Reader.read ~out r ~frames:22050 with
              | Ok (Some _) ->
                  ()
              | _ ->
                  failwith "empty" ) ;
              Io.Reader.close r )
            paths ) )
    Io_corpus.many_classes ;
  (* writes: the 30 s stereo float32 payload, close-to-close *)
  let src =
    ok ~what:"write source" (Io.read Nx.float32 (Io_corpus.bulk "wav-float32"))
  in
  List.iter
    (fun (label, format, ext) ->
      let tmp =
        Filename.concat (Io_corpus.dir ()) ("gate_write_tmp_" ^ label ^ ext)
      in
      measure (Printf.sprintf "write %s" label) 12 (fun () ->
          ok ~what:label (Io.write ~format tmp src) ) ;
      Sys.remove tmp )
    [ ("pcm16", Io.Format.create ~encoding:`Pcm_16 `Wav, ".wav")
    ; ("pcm24", Io.Format.create ~encoding:`Pcm_24 `Wav, ".wav")
    ; ("float32", Io.Format.create ~encoding:`Float32 `Wav, ".wav")
    ; ("flac", Io.Format.create ~encoding:`Pcm_16 `Flac, ".flac")
    ; ("ogg", Io.Format.create `Ogg, ".ogg") ] ;
  (* the fused resampled read against its offline decomposition *)
  let bulk = Io_corpus.bulk "wav-float32" in
  List.iter
    (fun (pair, target) ->
      measure (Printf.sprintf "read_resample %s" pair) 12 (fun () ->
          (ok ~what:pair (Io.read ~sample_rate:target Nx.float32 bulk)).Io.data ) ;
      measure (Printf.sprintf "read_plus_apply %s" pair) 12 (fun () ->
          let a = ok ~what:pair (Io.read Nx.float32 bulk) in
          let cfg =
            Soundml.Resample.Config.create ~sample_rate:a.Io.sample_rate ~target
              ()
          in
          Soundml.Resample.apply cfg a.Io.data ) )
    [("44k1-16k", 16000); ("44k1-48k", 48000)]
