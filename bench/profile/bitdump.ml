(* Cross-build bit-identity probe: hex digests of resampled outputs on a
   deterministic corpus. Run at two commits; identical stdout means
   bit-identical outputs on every row (offline + streamed partitionings). *)
module R = Soundml.Resample

let pairs =
  [ ("44k1->48k", 44100, 48000)
  ; ("48k->44k1", 48000, 44100)
  ; ("44k1->16k", 44100, 16000)
  ; ("16k->44k1", 16000, 44100)
  ; ("8k->48k", 8000, 48000)
  ; ("48k->8k", 48000, 8000) ]

let tiers = [("fast", `Fast); ("high", `High); ("best", `Best)]

let digest t =
  Digest.to_hex
    (Digest.string (Marshal.to_string (Nx.to_array (Nx.cast Nx.float64 t)) []))

let stream (type a) c (dtype : (float, a) Nx.dtype) x chunk =
  let n = Nx.dim 0 x in
  let k = R.Kernel.prepare c dtype ~channels:1 ~max_block:chunk in
  let outs = ref [] in
  let i = ref 0 in
  while !i < n do
    let len = min chunk (n - !i) in
    ( match R.Kernel.step k (Nx.shrink [|(!i, !i + len)|] x) with
    | Some o ->
        outs := o :: !outs
    | None ->
        () ) ;
    i := !i + len
  done ;
  (match R.Kernel.flush k with Some o -> outs := o :: !outs | None -> ()) ;
  Nx.concatenate ~axis:0 (List.rev !outs)

let () =
  List.iter
    (fun (name, sr, tgt) ->
      List.iter
        (fun (tname, q) ->
          let c = R.Config.create ~quality:q ~sample_rate:sr ~target:tgt () in
          let n = sr in
          let x64 =
            Nx.create Nx.float64 [|n|]
              (Array.init n (fun i -> Float.sin (0.61 *. Float.of_int i)))
          in
          let x32 = Nx.cast Nx.float32 x64 in
          Printf.printf "%s\t%s\tf64\tapply\t%s\n" name tname
            (digest (R.apply c x64)) ;
          Printf.printf "%s\t%s\tf32\tapply\t%s\n" name tname
            (digest (R.apply c x32)) ;
          Printf.printf "%s\t%s\tf64\tstrm4096\t%s\n" name tname
            (digest (stream c Nx.float64 x64 4096)) ;
          Printf.printf "%s\t%s\tf32\tstrm160\t%s\n" name tname
            (digest (stream c Nx.float32 x32 160)) )
        tiers )
    pairs
