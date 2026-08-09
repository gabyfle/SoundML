(* Decomposition driver for the io read path: one row per layer, min-of-N over
   many calls, to locate per-file overhead against the C ceiling. Not a thumper
   suite — a profiling scratch tool. *)

let path = Sys.argv.(1)

let reps = try int_of_string Sys.argv.(2) with _ -> 2000

let time name f =
  ignore (f ()) ;
  let best = ref infinity in
  let t_all = Unix.gettimeofday () in
  for _ = 1 to reps do
    let t0 = Unix.gettimeofday () in
    ignore (f ()) ;
    let dt = Unix.gettimeofday () -. t0 in
    if dt < !best then best := dt
  done ;
  let total = Unix.gettimeofday () -. t_all in
  Printf.printf "%-32s min %8.2f us   mean %8.2f us\n%!" name (!best *. 1e6)
    (total /. Float.of_int reps *. 1e6)

let () =
  let module Io = Soundml_io in
  time "info (open+name+close)" (fun () ->
      match Io.info path with Ok i -> i.Io.Info.frames | Error _ -> 0 ) ;
  time "read f32 (allocating)" (fun () ->
      match Io.read Nx.float32 path with
      | Ok a ->
          (Nx.shape a.Io.data).(1)
      | Error _ ->
          0 ) ;
  let frames, channels =
    match Io.info path with
    | Ok i ->
        (i.Io.Info.frames, i.Io.Info.channels)
    | Error _ ->
        failwith "info"
  in
  let out = Nx.zeros Nx.float32 [|channels; frames|] in
  time "reader ?out (open+read+close)" (fun () ->
      match Io.Reader.open_ Nx.float32 path with
      | Error _ ->
          0
      | Ok r ->
          let n =
            match Io.Reader.read ~out r ~frames with
            | Ok (Some c) ->
                (Nx.shape c).(1)
            | _ ->
                0
          in
          Io.Reader.close r ; n ) ;
  time "reader open+close only" (fun () ->
      match Io.Reader.open_ Nx.float32 path with
      | Error _ ->
          0
      | Ok r ->
          Io.Reader.close r ; 1 ) ;
  time "buffer create+of_buffer" (fun () ->
      let buf = Nx_buffer.create Nx_buffer.float32 (channels * frames) in
      let t = Nx.of_buffer buf ~shape:[|channels; frames|] in
      (Nx.shape t).(1) )
