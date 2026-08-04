(* Reader semantics: the handle contract around the laws of io_law.ml — EOF is
   [Ok None] forever, the short final chunk is exact, [?out] is written and
   viewed (never retained), [info] stays native while [format] describes the
   post-option output, truncation surfaces at the failing chunk through every
   reading face, and closed readers refuse typed misuse with
   [Invalid_argument]. *)

open Windtrap
open Soundml_io

let tmp_dir = fixture (fun () -> Filename.temp_dir "soundml_io_reader" "")

let ok ~msg = function Ok v -> v | Error e -> failf "%s: %a" msg pp_error e

let fixture_path =
  (* 5000 frames: reads of 4096 leave a 904-frame short final chunk *)
  fixture (fun () ->
      let path = Filename.concat (tmp_dir ()) "clip.wav" in
      let frames = 5000 in
      let data =
        Nx.create Nx.float32 [|2; frames|]
          (Array.init (2 * frames) (fun i ->
               sin (Float.of_int i /. 50.) *. 0.5 ) )
      in
      ( match
          write
            ~format:(Format.create ~encoding:`Float32 `Wav)
            path {data; sample_rate= 22050}
        with
      | Ok () ->
          ()
      | Error e ->
          failf "fixture: %a" pp_error e ) ;
      path )

(* {2 Chunk protocol} *)

let protocol_tests =
  [ test "the short final chunk is exact and EOF is None forever" (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 (fixture_path ())) in
        ( match Reader.read r ~frames:4096 with
        | Ok (Some c) ->
            equal ~msg:"full chunk" (array int) [|2; 4096|] (Nx.shape c)
        | _ ->
            fail "expected a full first chunk" ) ;
        ( match Reader.read r ~frames:4096 with
        | Ok (Some c) ->
            equal ~msg:"short final chunk" (array int) [|2; 904|] (Nx.shape c)
        | _ ->
            fail "expected the short final chunk" ) ;
        for _ = 1 to 3 do
          match Reader.read r ~frames:4096 with
          | Ok None ->
              ()
          | Ok (Some _) ->
              fail "EOF delivered data"
          | Error e ->
              failf "EOF errored: %a" pp_error e
        done ;
        Reader.close r )
  ; test "?out is written through and the full-chunk view is out itself"
      (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 (fixture_path ())) in
        let out = Nx.zeros Nx.float32 [|2; 1024|] in
        ( match Reader.read ~out r ~frames:1024 with
        | Ok (Some c) ->
            is_true ~msg:"the returned chunk is out" (c == out) ;
            let reference =
              ok ~msg:"read" (read Nx.float32 (fixture_path ()))
            in
            let e =
              Nx.to_array (Nx.shrink [|(0, 2); (0, 1024)|] reference.data)
            in
            let a = Nx.to_array out in
            Array.iteri
              (fun i ev ->
                if Float.compare ev a.(i) <> 0 then
                  failf "out sample %d is %.9g, the file holds %.9g" i a.(i) ev )
              e
        | _ ->
            fail "expected a chunk" ) ;
        Reader.close r )
  ; test "a mono downmix reader delivers [1; n] chunks" (fun () ->
        let r =
          ok ~msg:"open_" (Reader.open_ ~mono:true Nx.float64 (fixture_path ()))
        in
        ( match Reader.read r ~frames:100 with
        | Ok (Some c) ->
            equal ~msg:"shape" (array int) [|1; 100|] (Nx.shape c)
        | _ ->
            fail "expected a chunk" ) ;
        Reader.close r ) ]

(* {2 info and format} *)

let probe_tests =
  [ test "info is the native probe; format is the target description" (fun () ->
        let r =
          ok ~msg:"open_"
            (Reader.open_ ~sample_rate:16000 ~mono:true Nx.float32
               (fixture_path ()) )
        in
        let i = Reader.info r in
        equal ~msg:"native rate" int 22050 i.Info.sample_rate ;
        equal ~msg:"native channels" int 2 i.Info.channels ;
        equal ~msg:"native frames" int 5000 i.Info.frames ;
        let f = Reader.format r in
        let module PF = Soundml.Pipeline.Format in
        let module Rate = Soundml.Pipeline.Rate in
        is_true ~msg:"target items per second"
          (Rate.equal {Rate.num= 16000; den= 1} (PF.items_per_second f)) ;
        equal ~msg:"post-downmix channels" int 1 (PF.channels f) ;
        is_true ~msg:"zero upstream latency"
          (Rate.equal {Rate.num= 0; den= 1} (PF.upstream_latency f)) ;
        Reader.close r ) ]

(* {2 Truncation at the failing chunk, through every face} *)

let trunc = "corpus/malformed/trunc.flac"

let truncation_tests =
  [ test "Reader.read surfaces Truncated at the failing chunk, then EOF"
      (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 trunc) in
        (* the chunks that exist are served; the chunk that crosses the
           shortfall is the failing one *)
        let rec drain served =
          match Reader.read r ~frames:4096 with
          | Ok (Some c) ->
              drain (served + (Nx.shape c).(1))
          | Ok None ->
              failf "EOF without an error after %d served frames" served
          | Error (Truncated {expected_frames= 22050; read_frames; _}) ->
              is_true ~msg:"shortfall" (read_frames < 22050) ;
              is_true ~msg:"the failing chunk was not partially served"
                (served <= read_frames)
          | Error e ->
              failf "unexpected %a" pp_error e
        in
        drain 0 ;
        ( match Reader.read r ~frames:4096 with
        | Ok None ->
            ()
        | _ ->
            fail "after Truncated the reader is at EOF" ) ;
        Reader.close r )
  ; test "fold surfaces Truncated" (fun () ->
        match fold Nx.float64 trunc ~init:0 ~f:(fun n _ -> n + 1) with
        | Error (Truncated {expected_frames= 22050; _}) ->
            ()
        | Error e ->
            failf "unexpected %a" pp_error e
        | Ok _ ->
            fail "a truncated file folded" )
  ; test "read ~sample_rate surfaces Truncated (the error wins over flush)"
      (fun () ->
        match read ~sample_rate:16000 Nx.float32 trunc with
        | Error (Truncated {expected_frames= 22050; _}) ->
            ()
        | Error e ->
            failf "unexpected %a" pp_error e
        | Ok _ ->
            fail "a truncated file decoded" )
  ; test "a resampled Reader on a truncated file errors at the failing chunk"
      (fun () ->
        let r =
          ok ~msg:"open_" (Reader.open_ ~sample_rate:16000 Nx.float32 trunc)
        in
        let rec drain () =
          match Reader.read r ~frames:1024 with
          | Ok (Some _) ->
              drain ()
          | Ok None ->
              fail "EOF without an error on a truncated file"
          | Error (Truncated _) ->
              ()
          | Error e ->
              failf "unexpected %a" pp_error e
        in
        drain () ;
        ( match Reader.read r ~frames:1024 with
        | Ok None ->
            ()
        | _ ->
            fail "after Truncated the reader is at EOF" ) ;
        Reader.close r ) ]

(* {2 Unknown-length streams through the reader} *)

let unknown_tests =
  [ test "an unknown-length Ogg drains in chunks to EOF" (fun () ->
        let path = "corpus/malformed/trunc.ogg" in
        let whole = ok ~msg:"read" (read Nx.float32 path) in
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 path) in
        let rec drain acc =
          match Reader.read r ~frames:777 with
          | Ok (Some c) ->
              drain (Nx.copy c :: acc)
          | Ok None ->
              List.rev acc
          | Error e ->
              failf "drain: %a" pp_error e
        in
        let pieces = drain [] in
        Reader.close r ;
        let total = List.fold_left (fun n c -> n + (Nx.shape c).(1)) 0 pieces in
        equal ~msg:"drained extent" int (Nx.shape whole.data).(1) total ) ]

(* {2 Preconditions} *)

let precondition_tests =
  [ test "read refuses frames < 1" (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 (fixture_path ())) in
        Fun.protect
          ~finally:(fun () -> Reader.close r)
          (fun () ->
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> Reader.read r ~frames:0) ) )
  ; test "read refuses an out of the wrong shape" (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 (fixture_path ())) in
        Fun.protect
          ~finally:(fun () -> Reader.close r)
          (fun () ->
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () ->
                Reader.read ~out:(Nx.zeros Nx.float32 [|2; 100|]) r ~frames:64 ) ) )
  ; test "read refuses a non-contiguous out" (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 (fixture_path ())) in
        Fun.protect
          ~finally:(fun () -> Reader.close r)
          (fun () ->
            let base = Nx.zeros Nx.float32 [|64; 2|] in
            let out = Nx.transpose base in
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> Reader.read ~out r ~frames:64) ) )
  ; test "seek refuses a negative frame" (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 (fixture_path ())) in
        Fun.protect
          ~finally:(fun () -> Reader.close r)
          (fun () ->
            raises_match
              (function Invalid_argument _ -> true | _ -> false)
              (fun () -> Reader.seek r ~frame:(-1)) ) )
  ; test "a closed reader refuses reads and seeks; close is idempotent"
      (fun () ->
        let r = ok ~msg:"open_" (Reader.open_ Nx.float32 (fixture_path ())) in
        Reader.close r ;
        Reader.close r ;
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> Reader.read r ~frames:16) ;
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> Reader.seek r ~frame:0) )
  ; test "open_ refuses ?quality without ?sample_rate" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> Reader.open_ ~quality:`Fast Nx.float32 (fixture_path ())) )
  ; test "open_ refuses a non-positive sample_rate" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> Reader.open_ ~sample_rate:0 Nx.float32 (fixture_path ())) )
  ; test "fold refuses block < 1" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () ->
            fold ~block:0 Nx.float32 (fixture_path ()) ~init:() ~f:(fun () _ ->
                () ) ) )
  ; test "read refuses ?quality without ?sample_rate" (fun () ->
        raises_match
          (function Invalid_argument _ -> true | _ -> false)
          (fun () -> read ~quality:`Best Nx.float32 (fixture_path ())) )
  ; test "open errors are typed, not raised" (fun () ->
        let missing = Filename.concat (tmp_dir ()) "missing.wav" in
        match Reader.open_ Nx.float32 missing with
        | Error (Not_found _) ->
            ()
        | Error e ->
            failf "unexpected %a" pp_error e
        | Ok r ->
            Reader.close r ;
            fail "a missing path opened" ) ]

(* {2 fold cleanup} *)

let cleanup_tests =
  [ test "an exception from f closes the handle and escapes" (fun () ->
        let count = ref 0 in
        raises_match
          (function Failure m -> String.equal m "boom" | _ -> false)
          (fun () ->
            fold ~block:1024 Nx.float32 (fixture_path ()) ~init:()
              ~f:(fun () _ ->
                incr count ;
                if !count = 2 then failwith "boom" ) ) ;
        equal ~msg:"f ran until the raise" int 2 !count ) ]

let suite =
  [ group "reader: chunk protocol" protocol_tests
  ; group "reader: info and format" probe_tests
  ; group "reader: truncation" truncation_tests
  ; group "reader: unknown-length streams" unknown_tests
  ; group "reader: preconditions" precondition_tests
  ; group "reader: fold cleanup" cleanup_tests ]
