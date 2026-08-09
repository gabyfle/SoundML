type 'k reference =
  | Value : float -> 'k reference
  | Maximum : Pipeline.offline reference

let invalid fn what = invalid_arg ("Soundml.Db." ^ fn ^ ": " ^ what)

let check_amin fn amin =
  if not (Float.is_finite amin && amin > 0.) then
    invalid fn "amin must be finite and positive"

let check_value fn value =
  if not (Float.is_finite value && value > 0.) then
    invalid fn "reference must be finite and positive"

(* The chunk monoid for spectral tensors. [offline_only] never joins chunks —
   the whole signal is one chunk — but states the monoid for uniformity. *)
let concat = function
  | [] ->
      invalid_arg "Soundml.Db: cannot concatenate zero chunks"
  | chunks ->
      Nx.concatenate ~axis:(-1) chunks

(* Whole-chunk maximum reference, floored by [amin] like librosa floors its
   callable references — so a signal with no positive value is referenced to
   [amin]. Non-finite maxima (a chunk holding NaN) also fall back to [amin]
   rather than raising mid-run: garbage propagates elementwise instead of
   aborting the pipeline. *)
let to_db_with_maximum ~amin ?top_db chunk =
  if Nx.numel chunk = 0 then Nx.copy chunk
  else
    let m = Nx.item [] (Nx.max chunk) in
    let reference =
      if Float.is_finite m && m > 0. then Float.max amin m else amin
    in
    Convert.power_to_db ~reference ~amin ?top_db chunk

let stage : type k a.
       ?amin:float
    -> k reference
    -> ((float, a) Nx.t, (float, a) Nx.t, k) Pipeline.t =
 fun ?(amin = 1e-10) r ->
  check_amin "stage" amin ;
  match r with
  | Value v ->
      check_value "stage" v ;
      Pipeline.stateless (fun chunk ->
          Convert.power_to_db ~reference:v ~amin chunk )
  | Maximum ->
      Pipeline.offline_only (to_db_with_maximum ~amin) ~concat

let clamped_stage ?(amin = 1e-10) ~top_db r =
  check_amin "clamped_stage" amin ;
  if not (Float.is_finite top_db && top_db >= 0.) then
    invalid "clamped_stage" "top_db must be finite and non-negative" ;
  match r with
  | Value v ->
      check_value "clamped_stage" v ;
      Pipeline.offline_only
        (fun chunk -> Convert.power_to_db ~reference:v ~amin ~top_db chunk)
        ~concat
  | Maximum ->
      Pipeline.offline_only (to_db_with_maximum ~amin ~top_db) ~concat
