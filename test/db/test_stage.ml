(* The pipeline law for [Db.stage (Value _)] over randomized partitionings — the
   stage is stateless, so the law is trivial, and pinned here so it stays
   trivial — plus the offline stages checked against their flat [Convert]
   counterparts through [Pipeline.run]. The PRNG is seeded so CI is
   deterministic. *)

open Windtrap
open Soundml

let seed = 0x5eed

let farray = array (float 0.)

let log_uniform rng low high n =
  Array.init n (fun _ ->
      let u = Random.State.float rng 1. in
      Float.exp (Float.log low +. (u *. (Float.log high -. Float.log low))) )

(* {1 Partitionings} *)

let rec random_sizes rng n =
  if n = 0 then []
  else
    let s = 1 + Random.State.int rng (min n 7) in
    s :: random_sizes rng (n - s)

let partitions rng n =
  let named =
    [ ("ones", List.init n (fun _ -> 1))
    ; ("whole", if n = 0 then [] else [n])
    ; ("empty", []) ]
  in
  let named = List.filter (fun (_, s) -> List.fold_left ( + ) 0 s = n) named in
  named
  @ List.init 5 (fun i -> (Printf.sprintf "random-%d" i, random_sizes rng n))

let slice t off s =
  if s = 0 then Nx.zeros (Nx.dtype t) [|0|] else Nx.shrink [|(off, off + s)|] t

let slices t sizes =
  let rec go off = function
    | [] ->
        []
    | s :: rest ->
        slice t off s :: go (off + s) rest
  in
  go 0 sizes

(* {1 The law for Value stages} *)

let check_law (type b) (dtype : (float, b) Nx.dtype) ~name p x =
  let rng = Random.State.make [|seed|] in
  let source = Pipeline.Format.audio dtype ~sample_rate:1000 ~channels:1 in
  let t = Nx.create dtype [|Array.length x|] x in
  let expected = Nx.to_array (Pipeline.run ~source p t) in
  List.iter
    (fun (pname, sizes) ->
      let max_chunk = List.fold_left max 1 sizes in
      let s = Pipeline.Stream.prepare p ~source ~max_chunk in
      let outs =
        List.filter_map (Pipeline.Stream.push s) (slices t sizes)
        @ Pipeline.Stream.flush s
      in
      let got =
        match outs with
        | [] ->
            [||]
        | chunks ->
            Nx.to_array (Nx.concatenate ~axis:0 chunks)
      in
      equal
        ~msg:(Printf.sprintf "%s/%s/n=%d" name pname (Array.length x))
        farray expected got )
    (partitions rng (Array.length x))

let law_inputs rng =
  [ log_uniform rng 1e-8 1e3 61
  ; log_uniform rng 1e-2 1e2 17
  ; [|1.; 0.1|]
  ; [||] (* empty input, empty partition *) ]

let law_case (type b) (dtype : (float, b) Nx.dtype) ~name mk =
  test name (fun () ->
      let rng = Random.State.make [|seed + 1|] in
      List.iter (fun x -> check_law dtype ~name (mk ()) x) (law_inputs rng) )

let law =
  [ law_case Nx.float32 ~name:"value-1 (float32)" (fun () ->
        Db.stage (Db.Value 1.0) )
  ; law_case Nx.float64 ~name:"value-1 (float64)" (fun () ->
        Db.stage (Db.Value 1.0) )
  ; law_case Nx.float32 ~name:"value-2.5, amin 1e-6 (float32)" (fun () ->
        Db.stage ~amin:1e-6 (Db.Value 2.5) )
  ; law_case Nx.float64 ~name:"value-2.5, amin 1e-6 (float64)" (fun () ->
        Db.stage ~amin:1e-6 (Db.Value 2.5) ) ]

(* {1 Offline stages equal their flat counterparts through run} *)

let source64 () = Pipeline.Format.audio Nx.float64 ~sample_rate:1000 ~channels:1

let maximum_stage_matches_flat () =
  let rng = Random.State.make [|seed + 2|] in
  let t = Nx.create Nx.float64 [|64|] (log_uniform rng 1e-8 1e2 64) in
  let got = Pipeline.run ~source:(source64 ()) (Db.stage Db.Maximum) t in
  let reference = Nx.item [] (Nx.max t) in
  equal ~msg:"maximum reference" farray
    (Nx.to_array (Convert.power_to_db ~reference t))
    (Nx.to_array got) ;
  (* a silent signal is referenced to amin: every floored value sits at 0 dB *)
  let silent = Nx.zeros Nx.float64 [|4|] in
  equal ~msg:"silent signal" farray [|0.; 0.; 0.; 0.|]
    (Nx.to_array
       (Pipeline.run ~source:(source64 ()) (Db.stage Db.Maximum) silent) ) ;
  (* the empty signal passes through *)
  let empty = Nx.zeros Nx.float64 [|0|] in
  equal ~msg:"empty signal" farray [||]
    (Nx.to_array
       (Pipeline.run ~source:(source64 ()) (Db.stage Db.Maximum) empty) )

let clamped_stage_matches_flat () =
  let rng = Random.State.make [|seed + 3|] in
  let t = Nx.create Nx.float64 [|64|] (log_uniform rng 1e-8 1e2 64) in
  let source = source64 () in
  equal ~msg:"clamped, fixed reference" farray
    (Nx.to_array (Convert.power_to_db ~top_db:20. t))
    (Nx.to_array
       (Pipeline.run ~source (Db.clamped_stage ~top_db:20. (Db.Value 1.)) t) ) ;
  let reference = Nx.item [] (Nx.max t) in
  equal ~msg:"clamped, maximum reference" farray
    (Nx.to_array (Convert.power_to_db ~reference ~top_db:20. t))
    (Nx.to_array
       (Pipeline.run ~source (Db.clamped_stage ~top_db:20. Db.Maximum) t) )

let stage_is_instantaneous () =
  let p = Db.stage (Db.Value 1.0) in
  let one = {Pipeline.Rate.num= 1; den= 1} in
  let zero = {Pipeline.Rate.num= 0; den= 1} in
  is_true ~msg:"zero latency"
    (Pipeline.Rate.equal zero
       (Pipeline.latency
          ( p
            : ( (float, Nx.float32_elt) Nx.t
              , (float, Nx.float32_elt) Nx.t
              , Pipeline.offline )
              Pipeline.t ) ) ) ;
  is_true ~msg:"unit rate"
    (Pipeline.Rate.equal one
       (Pipeline.rate
          ( Db.stage (Db.Value 1.0)
            : ( (float, Nx.float32_elt) Nx.t
              , (float, Nx.float32_elt) Nx.t
              , Pipeline.offline )
              Pipeline.t ) ) )

let offline =
  [ test "Maximum stage equals the flat conversion" maximum_stage_matches_flat
  ; test "clamped stage equals the flat conversion" clamped_stage_matches_flat
  ; test "Value stage is instantaneous" stage_is_instantaneous ]

let suite = [group "stage-law" law; group "stage-offline" offline]
