open Printf
open Unix

let mb_divisor = 1024. *. 1024.

let is_ext_file filename ext =
  String.lowercase_ascii (Filename.extension filename) = ext

let find_ext_files root_dir ext =
  let rec find acc dir =
    try
      let dh = opendir dir in
      try
        let rec loop acc =
          match readdir dh with
          | exception End_of_file ->
              closedir dh ; acc
          | "." | ".." ->
              loop acc
          | entry -> (
              let full_path = Filename.concat dir entry in
              try
                match (stat full_path).st_kind with
                | S_REG when is_ext_file full_path ext ->
                    loop (full_path :: acc)
                | S_DIR ->
                    loop (find acc full_path)
                | _ ->
                    loop acc
              with Unix_error (_, _, _) -> loop acc )
        in
        loop acc
      with ex ->
        closedir dh ;
        eprintf "\nError reading  directory '%s': %s\n%!" dir
          (Printexc.to_string ex) ;
        acc
    with Unix_error (e, _, p) ->
      eprintf "\nError  opening directory '%s': %s\n%!" p (error_message e) ;
      acc
  in
  find [] root_dir

let get_file_size filename =
  try
    let stats = stat filename in
    if stats.st_kind = S_REG then Ok (float_of_int stats.st_size /. mb_divisor)
    else Error (sprintf "Not a regular file: %s" filename)
  with
  | Unix_error (e, _, _) ->
      Error (sprintf "Cannot stat file '%s': %s" filename (error_message e))
  | Sys_error msg ->
      Error (sprintf "System error  statting '%s': %s" filename msg)

let benchmark_read kind filename sample_rate =
  match get_file_size filename with
  | Error msg ->
      Error (filename, msg)
  | Ok size_mb -> (
      if size_mb <= 0.0 then Error (filename, "Incorrect file size")
      else
        try
          let res_typ =
            match sample_rate with 0 -> Io.NONE | _ -> Io.SOXR_HQ
          in
          let start_time = Unix.gettimeofday () in
          let _audio =
            Soundml.Io.read ~res_typ ~sample_rate ~mono:false kind filename
          in
          let end_time = Unix.gettimeofday () in
          let duration = end_time -. start_time in
          Ok (duration, size_mb)
        with ex -> Error (filename, Printexc.to_string ex) )

let run_benchmark root sample_rate extension max_files =
  let kind = Bigarray.Float32 in
  let all_files = find_ext_files root extension in
  let all_files =
    List.filteri (fun i _ -> if i >= max_files then false else true) all_files
  in
  let total_files = List.length all_files in
  if total_files = 0 then exit 0 ;
  let warmup_count = min 5 total_files in
  ( if warmup_count > 0 then
      let warmup_files = List.filteri (fun i _ -> i < warmup_count) all_files in
      List.iter
        (fun f ->
          match benchmark_read kind f sample_rate with
          | Ok _ ->
              ()
          | Error _ ->
              () )
        warmup_files ) ;
  let total_time = ref 0.0 in
  let total_size = ref 0.0 in
  List.iter
    (fun filename ->
      match benchmark_read kind filename sample_rate with
      | Ok (duration, size_mb) ->
          total_time := !total_time +. duration ;
          total_size := !total_size +. size_mb
      | Error _ ->
          () )
    all_files ;
  if !total_time > 0.0 && !total_size > 0.0 then
    let avg_speed = !total_size /. !total_time in
    printf "%.5f\n" avg_speed

let () =
  if Array.length Sys.argv <> 5 then
    eprintf "Usage: %s  <root_directory> <sample_rate> <format> <max_files>\n"
      Sys.argv.(0)
  else
    let root_dir = Sys.argv.(1) in
    let sample_rate = int_of_string Sys.argv.(2) in
    let extension = Sys.argv.(3) in
    let max_files = int_of_string Sys.argv.(4) in
    if not (Sys.file_exists root_dir && Sys.is_directory root_dir) then
      eprintf "Can't read directory: %s.\n" root_dir
    else
      try run_benchmark root_dir sample_rate extension max_files
      with ex ->
        eprintf "An unexpected  error occurred: %s\n" (Printexc.to_string ex) ;
        exit 1
