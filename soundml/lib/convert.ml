let invalid fn what = invalid_arg ("Soundml.Convert." ^ fn ^ ": " ^ what)

let check_reference fn reference =
  if not (Float.is_finite reference && reference > 0.) then
    invalid fn "reference must be finite and positive"

let check_amin fn amin =
  if not (Float.is_finite amin && amin > 0.) then
    invalid fn "amin must be finite and positive"

let check_top_db fn top_db =
  Option.iter
    (fun value ->
      if not (Float.is_finite value && value >= 0.) then
        invalid fn "top_db must be finite and non-negative" )
    top_db

let check_grid fn sample_rate hop =
  if sample_rate < 1 then invalid fn "sample_rate must be positive" ;
  if hop < 1 then invalid fn "hop must be positive"

(* [10 / ln 10]: converts natural logarithms of powers to decibels. *)
let decade = 10. /. Float.log 10.

(* The validated core shared by [power_to_db] and [amplitude_to_db]; [gain] is
   10 for powers and 20 for amplitudes, [reference] and [amin] live in the
   input's own domain. Powers are floored signed — a negative power sits at the
   floor — while amplitudes are magnitudes first; both are librosa's choices. *)
let to_db ~gain ~magnitude ~reference ~amin ~top_db s =
  if Nx.numel s = 0 then Nx.copy s
  else
    let dtype = Nx.dtype s in
    let scale = gain /. 10. *. decade in
    let s = if magnitude then Nx.abs s else s in
    let floored = Nx.maximum s (Nx.scalar dtype amin) in
    let offset = scale *. Float.log (Float.max amin reference) in
    let db = Nx.sub_s (Nx.mul_s (Nx.log floored) scale) offset in
    match top_db with
    | None ->
        db
    | Some range ->
        let maximum = Nx.item [] (Nx.max db) in
        Nx.maximum db (Nx.scalar dtype (maximum -. range))

let power_to_db ?(reference = 1.) ?(amin = 1e-10) ?top_db s =
  check_reference "power_to_db" reference ;
  check_amin "power_to_db" amin ;
  check_top_db "power_to_db" top_db ;
  to_db ~gain:10. ~magnitude:false ~reference ~amin ~top_db s

let amplitude_to_db ?(reference = 1.) ?(amin = 1e-5) ?top_db s =
  check_reference "amplitude_to_db" reference ;
  check_amin "amplitude_to_db" amin ;
  check_top_db "amplitude_to_db" top_db ;
  to_db ~gain:20. ~magnitude:true ~reference ~amin ~top_db s

let from_db ~gain ~reference db =
  let ten = Nx.scalar (Nx.dtype db) 10. in
  Nx.mul_s (Nx.pow ten (Nx.mul_s db (1. /. gain))) reference

let db_to_power ?(reference = 1.) db =
  check_reference "db_to_power" reference ;
  from_db ~gain:10. ~reference db

let db_to_amplitude ?(reference = 1.) db =
  check_reference "db_to_amplitude" reference ;
  from_db ~gain:20. ~reference db

(* Slaney-scale constants (librosa): linear below 1000 Hz at [1 / f_sp] mel per
   hertz, logarithmic above, continuous at the break. *)
let f_sp = 200. /. 3.

let min_log_hz = 1000.

let min_log_mel = min_log_hz /. f_sp

let logstep = Float.log 6.4 /. 27.

let hz_to_mel ?(scale = `Slaney) f =
  match scale with
  | `Htk ->
      Nx.mul_s (Nx.log (Nx.add_s (Nx.div_s f 700.) 1.)) (2595. /. Float.log 10.)
  | `Slaney ->
      let linear = Nx.div_s f f_sp in
      let log_branch =
        Nx.add_s (Nx.div_s (Nx.log (Nx.div_s f min_log_hz)) logstep) min_log_mel
      in
      Nx.where (Nx.less f (Nx.scalar (Nx.dtype f) min_log_hz)) linear log_branch

let mel_to_hz ?(scale = `Slaney) m =
  match scale with
  | `Htk ->
      Nx.mul_s (Nx.sub_s (Nx.exp (Nx.mul_s m (Float.log 10. /. 2595.))) 1.) 700.
  | `Slaney ->
      let linear = Nx.mul_s m f_sp in
      let log_branch =
        Nx.mul_s (Nx.exp (Nx.mul_s (Nx.sub_s m min_log_mel) logstep)) min_log_hz
      in
      Nx.where
        (Nx.less m (Nx.scalar (Nx.dtype m) min_log_mel))
        linear log_branch

let hz_to_midi f = Nx.add_s (Nx.mul_s (Nx.log2 (Nx.div_s f 440.)) 12.) 69.

let midi_to_hz m = Nx.mul_s (Nx.exp2 (Nx.div_s (Nx.sub_s m 69.) 12.)) 440.

let frames_to_time ~sample_rate ~hop frames =
  check_grid "frames_to_time" sample_rate hop ;
  Nx.div_s (Nx.mul_s frames (float_of_int hop)) (float_of_int sample_rate)

let time_to_frames ~sample_rate ~hop times =
  check_grid "time_to_frames" sample_rate hop ;
  Nx.floor
    (Nx.div_s (Nx.mul_s times (float_of_int sample_rate)) (float_of_int hop))
