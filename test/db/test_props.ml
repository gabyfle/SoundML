(* Property and pinned-value tests for [Convert]: inverse pairs hold within
   tolerance wherever no floor or clamp bites, the frequency scales round-trip
   and hit librosa's published values, the frame grid inverts exactly on exactly
   representable grids, and every precondition raises the documented
   [Invalid_argument]. The PRNG is seeded so CI is deterministic. *)

open Windtrap
open Soundml

let seed = 0x5eed

let log_uniform rng low high n =
  Array.init n (fun _ ->
      let u = Random.State.float rng 1. in
      Float.exp (Float.log low +. (u *. (Float.log high -. Float.log low))) )

let close64 = Tutils.check_close ~rtol:1e-10 ~atol:1e-10

let close32 = Tutils.check_close ~rtol:1e-4 ~atol:1e-4

(* {1 Pinned decibel values} *)

let floors_before_taking_logarithms () =
  let power = Nx.create Nx.float64 [|4|] [|1.; 1e-5; 1e-10; 1e-12|] in
  close64 ~msg:"floored decibels" ~expected:[|0.; -50.; -100.; -100.|]
    (Convert.power_to_db power)

let floors_the_reference () =
  let power = Nx.create Nx.float64 [|2|] [|0.1; 1.|] in
  close64 ~msg:"reference floored by amin" ~expected:[|0.; 10.|]
    (Convert.power_to_db ~amin:0.1 ~reference:0.01 power)

let clips_dynamic_range () =
  let power = Nx.create Nx.float32 [|4|] [|1.; 0.1; 0.01; 1e-10|] in
  close32 ~msg:"30 dB range" ~expected:[|0.; -10.; -20.; -30.|]
    (Convert.power_to_db ~top_db:30. power)

let clips_globally_across_channels () =
  let power = Nx.create Nx.float64 [|2; 2|] [|1.; 1e-6; 0.1; 1e-3|] in
  close64 ~msg:"global clipping threshold" ~shape:[|2; 2|]
    ~expected:[|0.; -20.; -10.; -20.|]
    (Convert.power_to_db ~top_db:20. power)

let preserves_empty_tensors () =
  let empty = Nx.empty Nx.float32 [|2; 0|] in
  close32 ~msg:"empty" ~shape:[|2; 0|] ~expected:[||]
    (Convert.power_to_db empty)

let pinned =
  [ test "floors before logarithms" floors_before_taking_logarithms
  ; test "floors the reference" floors_the_reference
  ; test "clips dynamic range" clips_dynamic_range
  ; test "clips globally across channels" clips_globally_across_channels
  ; test "preserves empty tensors" preserves_empty_tensors ]

(* {1 Inverse pairs} *)

let roundtrip_power (type b) (dtype : (float, b) Nx.dtype) ~rtol ~atol () =
  let rng = Random.State.make [|seed|] in
  List.iter
    (fun reference ->
      (* every value above the floor: nothing clamps, so the pair inverts *)
      let x = Nx.create dtype [|128|] (log_uniform rng 1e-6 1e3 128) in
      let db = Convert.power_to_db ~reference x in
      Tutils.check_close ~rtol ~atol
        ~msg:(Printf.sprintf "power round-trip, reference %g" reference)
        ~expected:(Nx.to_array x)
        (Convert.db_to_power ~reference db) )
    [1.0; 2.5]

let roundtrip_amplitude (type b) (dtype : (float, b) Nx.dtype) ~rtol ~atol () =
  let rng = Random.State.make [|seed + 1|] in
  List.iter
    (fun reference ->
      let x = Nx.create dtype [|128|] (log_uniform rng 1e-4 1e2 128) in
      let db = Convert.amplitude_to_db ~reference x in
      Tutils.check_close ~rtol ~atol
        ~msg:(Printf.sprintf "amplitude round-trip, reference %g" reference)
        ~expected:(Nx.to_array x)
        (Convert.db_to_amplitude ~reference db) )
    [1.0; 0.5]

let roundtrips =
  [ test "db_to_power inverts power_to_db (float64)"
      (roundtrip_power Nx.float64 ~rtol:1e-12 ~atol:1e-12)
  ; test "db_to_power inverts power_to_db (float32)"
      (roundtrip_power Nx.float32 ~rtol:1e-5 ~atol:1e-6)
  ; test "db_to_amplitude inverts amplitude_to_db (float64)"
      (roundtrip_amplitude Nx.float64 ~rtol:1e-12 ~atol:1e-12)
  ; test "db_to_amplitude inverts amplitude_to_db (float32)"
      (roundtrip_amplitude Nx.float32 ~rtol:1e-5 ~atol:1e-6) ]

(* {1 Frequency scales} *)

let roundtrip_mel (type b) (dtype : (float, b) Nx.dtype) ~rtol ~atol () =
  let rng = Random.State.make [|seed + 2|] in
  List.iter
    (fun scale ->
      (* the grid spans both slaney branches; 0 Hz stays 0 on either scale *)
      let grid = Array.init 65 (fun i -> float_of_int i *. 125.) in
      let random = log_uniform rng 1. 20000. 64 in
      let f = Nx.create dtype [|129|] (Array.append grid random) in
      let mel = Convert.hz_to_mel ~scale f in
      Tutils.check_close ~rtol ~atol ~msg:"mel round-trip"
        ~expected:(Nx.to_array f)
        (Convert.mel_to_hz ~scale mel) )
    [`Slaney; `Htk]

let librosa_mel_values () =
  close64 ~msg:"slaney mels"
    ~expected:[|1.65; 3.3; 6.6; 14.999999999999998; 35.163760314616646|]
    (Convert.hz_to_mel
       (Nx.create Nx.float64 [|5|] [|110.; 220.; 440.; 1000.; 4000.|]) ) ;
  close64 ~msg:"htk mels"
    ~expected:[|549.6386753811499; 999.9855371396244|]
    (Convert.hz_to_mel ~scale:`Htk
       (Nx.create Nx.float64 [|2|] [|440.; 1000.|]) ) ;
  close64 ~msg:"slaney hertz"
    ~expected:
      [| 66.66666666666667
       ; 133.33333333333334
       ; 200.
       ; 1000.0000000000002
       ; 5577.800011749387 |]
    (Convert.mel_to_hz (Nx.create Nx.float64 [|5|] [|1.; 2.; 3.; 15.; 40.|]))

let midi_values_and_roundtrip () =
  close64 ~msg:"hz_to_midi" ~expected:[|60.; 69.|]
    (Convert.hz_to_midi
       (Nx.create Nx.float64 [|2|] [|261.6255653005986; 440.|]) ) ;
  close64 ~msg:"midi_to_hz"
    ~expected:[|261.6255653005986; 440.|]
    (Convert.midi_to_hz (Nx.create Nx.float64 [|2|] [|60.; 69.|])) ;
  let rng = Random.State.make [|seed + 3|] in
  let f = Nx.create Nx.float64 [|64|] (log_uniform rng 20. 20000. 64) in
  close64 ~msg:"midi round-trip" ~expected:(Nx.to_array f)
    (Convert.midi_to_hz (Convert.hz_to_midi f))

let scales =
  [ test "mel scales round-trip (float64)"
      (roundtrip_mel Nx.float64 ~rtol:1e-12 ~atol:1e-9)
  ; test "mel scales round-trip (float32)"
      (roundtrip_mel Nx.float32 ~rtol:1e-5 ~atol:1e-2)
  ; test "librosa mel values" librosa_mel_values
  ; test "MIDI values and round-trip" midi_values_and_roundtrip ]

(* {1 The frame grid} *)

let farray = array (float 0.)

let grid_inverse (type b) (dtype : (float, b) Nx.dtype) ~sample_rate ~hop () =
  (* [hop / sample_rate] is exactly representable on these grids, so the floor
     lands exactly and the pair inverts frame by frame *)
  let frames =
    Nx.create dtype [|1024|] (Array.init 1024 (fun i -> float_of_int (4 * i)))
  in
  let times = Convert.frames_to_time ~sample_rate ~hop frames in
  equal
    ~msg:(Printf.sprintf "grid %d/%d" sample_rate hop)
    farray (Nx.to_array frames)
    (Nx.to_array (Convert.time_to_frames ~sample_rate ~hop times))

let librosa_grid_values () =
  (* librosa's time_to_frames docstring example, floor included *)
  let times =
    Nx.create Nx.float64 [|10|] (Array.init 10 (fun i -> 0.1 *. float_of_int i))
  in
  equal ~msg:"every 100ms at 22050/512" farray
    [|0.; 4.; 8.; 12.; 17.; 21.; 25.; 30.; 34.; 38.|]
    (Nx.to_array (Convert.time_to_frames ~sample_rate:22050 ~hop:512 times)) ;
  let frames = Nx.create Nx.float64 [|3|] [|0.; 1.; 10.|] in
  equal ~msg:"frame times at 22050/512"
    (array (float 1e-15))
    [|0.; 0.02321995464852608; 0.23219954648526078|]
    (Nx.to_array (Convert.frames_to_time ~sample_rate:22050 ~hop:512 frames))

let grid =
  [ test "inverse on the grid (float64, 16384/512)"
      (grid_inverse Nx.float64 ~sample_rate:16384 ~hop:512)
  ; test "inverse on the grid (float64, 48000/750)"
      (grid_inverse Nx.float64 ~sample_rate:48000 ~hop:750)
  ; test "inverse on the grid (float32, 16384/512)"
      (grid_inverse Nx.float32 ~sample_rate:16384 ~hop:512)
  ; test "librosa grid values" librosa_grid_values ]

(* {1 Preconditions} *)

let rejects_invalid_parameters () =
  let raises msg message thunk =
    raises_invalid_arg ~msg message (fun () -> ignore (thunk ()))
  in
  let power = Nx.ones Nx.float32 [|1|] in
  raises "zero amin"
    "Soundml.Convert.power_to_db: amin must be finite and positive" (fun () ->
      Convert.power_to_db ~amin:0. power ) ;
  raises "zero reference"
    "Soundml.Convert.power_to_db: reference must be finite and positive"
    (fun () -> Convert.power_to_db ~reference:0. power ) ;
  raises "negative top_db"
    "Soundml.Convert.power_to_db: top_db must be finite and non-negative"
    (fun () -> Convert.power_to_db ~top_db:(-1.) power ) ;
  raises "nan amplitude amin"
    "Soundml.Convert.amplitude_to_db: amin must be finite and positive"
    (fun () -> Convert.amplitude_to_db ~amin:Float.nan power ) ;
  raises "infinite inverse reference"
    "Soundml.Convert.db_to_power: reference must be finite and positive"
    (fun () -> Convert.db_to_power ~reference:Float.infinity power ) ;
  raises "zero amplitude reference"
    "Soundml.Convert.db_to_amplitude: reference must be finite and positive"
    (fun () -> Convert.db_to_amplitude ~reference:0. power ) ;
  raises "zero sample rate"
    "Soundml.Convert.frames_to_time: sample_rate must be positive" (fun () ->
      Convert.frames_to_time ~sample_rate:0 ~hop:512 power ) ;
  raises "zero hop" "Soundml.Convert.time_to_frames: hop must be positive"
    (fun () -> Convert.time_to_frames ~sample_rate:22050 ~hop:0 power ) ;
  raises "stage amin" "Soundml.Db.stage: amin must be finite and positive"
    (fun () -> Db.stage ~amin:0. (Db.Value 1.) ) ;
  raises "stage reference"
    "Soundml.Db.stage: reference must be finite and positive" (fun () ->
      Db.stage (Db.Value 0.) ) ;
  raises "clamped stage top_db"
    "Soundml.Db.clamped_stage: top_db must be finite and non-negative"
    (fun () -> Db.clamped_stage ~top_db:(-1.) (Db.Value 1.) ) ;
  raises "clamped stage amin"
    "Soundml.Db.clamped_stage: amin must be finite and positive" (fun () ->
      Db.clamped_stage ~amin:Float.nan ~top_db:80. Db.Maximum )

let validation = [test "rejects invalid parameters" rejects_invalid_parameters]

let suite =
  [ group "pinned-values" pinned
  ; group "round-trips" roundtrips
  ; group "frequency-scales" scales
  ; group "frame-grid" grid
  ; group "validation" validation ]
