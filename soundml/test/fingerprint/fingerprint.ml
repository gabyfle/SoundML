(* Bit fingerprints: every public feature, pinned to the bit on each platform,
   and the streaming laws, checked on every platform.

   What a line pins. Each line of the output names one public function, one
   kernel driven directly, or one stage's one-chunk run, at one explicit
   configuration and one dtype, the shapes of its outputs, and the MD5 of those
   outputs' float64 image: float32 values are widened exactly, so a float32 line
   pins float32 bits; complex values are hashed as their (re, im) pairs; integer
   results as themselves. Every NaN hashes as the one quiet NaN, since payload
   propagation differs between x86 and arm64, while -0 and +0 stay distinct. The
   set of lines and their names are pinned too. The input is two seconds of a
   deterministic stereo signal at 22050 Hz (below), and every configuration
   spells out all of its parameters, so the lines pin arithmetic, not defaults:
   the defaults have their own parity suites.

   How the stages are pinned. Through their laws (below): a stage that has the
   bits of a printed row at every chunking moves exactly when that row moves.
   Every law's reference must therefore be a row, and the run fails if one is
   not; where the only reference a stage has is its own one-chunk run (the onset
   envelope, the MFCC stage, which leave a shift or a clamp to the caller), that
   run is a row of its own. Otherwise a change confined to a stage, such as the
   float32 flux kernel onset_strength never calls, would keep every law and
   every line.

   What it does not pin. Correctness: that is the librosa goldens' job, at
   tolerances; this test only answers whether a change moved a bit. Nor equality
   across platforms: the bits of one feature legitimately differ between macOS
   arm64 and Linux x86_64 (the arm64 multiply-add rule, libm, the BLAS Nx calls
   on macOS), so the expectations are recorded per platform, in
   fingerprints.<system>-<architecture><tag>.expected, and compared by dune's
   diff action on the platforms that have one, in the dev profile. The <tag> is
   the environment variable SOUNDML_FINGERPRINT_TAG, empty when unset: two
   machines of one platform need not agree either (another chip, OS or Xcode
   under one key), so CI's macOS runner sets it to -ci and keeps
   fingerprints.macosx-arm64-ci.expected, apart from the developer's
   fingerprints.macosx-arm64.expected. Release builds (dune runtest -p, opam's
   with-test) run the laws but skip the comparison, since the bits belong to the
   recording machines.

   What leaves it unchanged by contract, on every platform. The chunking: every
   stage, streamed at several partitions (one chunk, an odd size, a large size,
   one item at a time for the cheap ones, and once more after a reset), must
   reproduce the offline function's bits; those laws run here on every platform,
   recorded or not, and a violation exits 1 with a report on stderr. The
   documented offline identities (a fused path against its composition, a range
   against a slice, ...) are checked the same way. The number of worker domains
   and of BLAS threads, and repeated runs: the output is the same with
   OCAMLRUNPARAM=d=1, d=2 and the default.

   When bits may legitimately change, and the lines must be re-recorded:

   - a deliberate numeric change: review that only the rows it should move
   moved, then promote;

   - an OCaml compiler upgrade (the arm64 multiply-add fusion rule). CI locks
   its dependencies afresh on every run (dune pkg lock against ocaml >= 5.5.0,
   with no committed lock), so a new compiler release reaches CI by itself and
   can turn its diffs red on a pull request that changes nothing numeric:
   re-record, or pin the compiler in CI;

   - a Raven pin bump (Nx's FFT, its matrix product and the size from which it
   calls the BLAS, its fold constants);

   - a C compiler upgrade on the building machine (contraction in Nx's C
   kernels);

   - an OS or C library update: Apple's libm and Accelerate on macOS, glibc on
   Linux (which is merging correctly rounded functions release by release), as
   when a CI runner image moves;

   - adding, renaming or reconfiguring a row.

   Never for a refactor, a performance change, a change in parallelism, chunking
   or allocation: those must leave every line as it is. The Linux expectation is
   that of glibc on an x86-64 with FMA and AVX2, which every CI runner has: Nx's
   matrix product and glibc's exp and log choose their code path from the
   processor, so an older one, or musl, under the same key, may disagree.

   How to re-record. On the developer's macOS arm64 machine, run the suite,
   which fails and prints the diff:

   dune test

   review which rows moved, and once they are the ones the change meant to move,
   promote the new file:

   dune promote

   In a worktree nested inside another dune project, pass the root to both
   steps, through dune promotion apply since dune promote takes no --root:

   dune test --root .

   dune promotion apply --root .

   From CI, for Linux x86_64 and for CI's macOS runner: the Test job fails on
   the diff, and its "Print the fingerprints this platform produced" step prints
   the whole file, in the log and in the job summary under the file's name
   (fingerprints.linux-amd64.expected, fingerprints.macosx-arm64-ci.expected);
   save that block as the file in this directory, with a final newline, check
   that its row names and shapes match the other files', and push. With no
   expectation file at all, the diff is against the empty file and lists the
   complete content. To record a new platform, add a clause to the enabled_if of
   the diff rule in this directory's dune file and proceed the same way.

   A new public function or stage adds a row here and, for a stage, a law
   against a row: its offline function's, or else its own one-chunk run's. *)

open Soundml

(* {1 Input} *)

let sample_rate = 22050

let n = 2 * sample_rate

(* Silent stretches, on both channels: they give trim and split something to
   find and drive the amin floors and the top_db clamps. *)
let silent i =
  i < sample_rate / 4
  || (i >= sample_rate && i < sample_rate + (sample_rate / 8))
  || i >= n - (sample_rate / 5)

(* The input is the same doubles on every platform: Random.State is a specified
   generator whose floats are 53-bit integers scaled by a power of two, the
   bounds are powers of two, and every other step is one correctly rounded
   operation (the products are exact), with no libm call. Channel 0 is a
   sawtooth at 220.5 Hz with 8-sample clicks every quarter second and a little
   noise, channel 1 is noise; the input rows pin all of it. *)
let signal =
  let rng = Random.State.make [|0x5eed; 2026|] in
  let a = Array.make (2 * n) 0. in
  for i = 0 to n - 1 do
    let u = Random.State.float rng 2. -. 1. in
    let v = Random.State.float rng 2. -. 1. in
    if not (silent i) then begin
      let saw = (Float.of_int (i mod 100) /. 100.) -. 0.5 in
      let click = if i mod 5512 < 8 then 0.75 else 0. in
      let hiss = u *. 0.0625 in
      a.(i) <- saw +. click +. hiss ;
      a.(n + i) <- v *. 0.5
    end
  done ;
  Nx.create Nx.float64 [|2; n|] a

(* Noise with no silence, for the stages whose agreement with their flat
   function holds only where the flat function's 80 dB clamp does not bind
   (mfcc_stage and the logarithmic spectral_contrast_stage): on [signal] the
   silences make it bind. *)
let noise =
  let rng = Random.State.make [|0xd0e5|] in
  Nx.create Nx.float64 [|1; n|]
    (Array.init n (fun _ -> Random.State.float rng 1. -. 0.5))

(* {1 Images and fingerprints} *)

(* An output's image: its shape and one little-endian 64-bit word per value (two
   per complex value). *)
type image = {shape: int array; words: string}

let empty = {shape= [||]; words= ""}

let canonical_nan = 0x7FF8_0000_0000_0000L

let word v = if Float.is_nan v then canonical_nan else Int64.bits_of_float v

let of_floats shape values =
  let b = Bytes.create (8 * Array.length values) in
  Array.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (word v)) values ;
  {shape; words= Bytes.to_string b}

let real (type a) (t : (float, a) Nx.t) =
  of_floats (Nx.shape t) (Nx.to_array (Nx.cast Nx.float64 t))

let complex (type c) (t : (Complex.t, c) Nx.t) =
  let values = Nx.to_array (Nx.cast Nx.complex128 t) in
  let b = Bytes.create (16 * Array.length values) in
  Array.iteri
    (fun i (z : Complex.t) ->
      Bytes.set_int64_le b (16 * i) (word z.re) ;
      Bytes.set_int64_le b ((16 * i) + 8) (word z.im) )
    values ;
  {shape= Nx.shape t; words= Bytes.to_string b}

let ints l =
  let b = Bytes.create (8 * List.length l) in
  List.iteri (fun i v -> Bytes.set_int64_le b (8 * i) (Int64.of_int v)) l ;
  {shape= [|List.length l|]; words= Bytes.to_string b}

let floats l = of_floats [|List.length l|] (Array.of_list l)

let rate_ints (r : Pipeline.Rate.t) = [r.num; r.den]

let shape_string s =
  String.concat "x" (Array.to_list (Array.map Int.to_string s))

let digest images =
  let b = Buffer.create 4096 in
  List.iter
    (fun im ->
      Buffer.add_string b (shape_string im.shape) ;
      Buffer.add_char b ';' ;
      Buffer.add_string b im.words )
    images ;
  Digest.MD5.to_hex (Digest.MD5.string (Buffer.contents b))

(* The digests printed so far, so that a law can be held to rest on a pinned
   value (see [anchor]). *)
let pinned = Hashtbl.create 512

(* Stdout carries the rows and nothing else. *)
let row name images =
  let d = digest images in
  Hashtbl.replace pinned d () ;
  Printf.printf "%-60s %-20s %s\n" name
    (String.concat "+" (List.map (fun im -> shape_string im.shape) images))
    d

(* {1 Laws} *)

type chunking = Whole | Every of int | Reset of int

let describe = function
  | Whole ->
      "one chunk"
  | Every k ->
      Printf.sprintf "chunks of %d" k
  | Reset k ->
      Printf.sprintf "chunks of %d after a reset" k

(* Audio-domain partitions: whole, an odd size, a large one, one sample at a
   time, and the odd size again on a stream that already ran the signal and was
   reset. [coarse] drops the one-sample pushes, for the dearer chains. *)
let audio = [Whole; Every 333; Every 4096; Every 1; Reset 333]

let coarse = [Whole; Every 333; Reset 4096]

let frames = [Whole; Every 7; Every 1; Reset 7]

(* [pieces k x] cuts [x] into chunks of [k] along its last axis, the last one
   shorter. *)
let pieces k x =
  let rank = Nx.ndim x in
  let len = Nx.dim (rank - 1) x in
  let rec go off acc =
    if off >= len then List.rev acc
    else
      let s = Int.min k (len - off) in
      let piece =
        Nx.slice
          (List.init rank (fun i ->
               if i = rank - 1 then Nx.R (off, off + s) else Nx.A ) )
          x
      in
      go (off + s) (piece :: acc)
  in
  go 0 []

(* A stream or a kernel, reduced to what the laws drive. *)
type ('i, 'o) driver =
  {push: 'i -> 'o option; flush: unit -> 'o list; reset: unit -> unit}

let feed d chunks =
  let pushed = List.filter_map d.push chunks in
  (* Sequenced on purpose: OCaml evaluates the right operand of (@) first, which
     would drain the stream before feeding it. *)
  let drained = d.flush () in
  pushed @ drained

let drive make chunking x =
  match chunking with
  | Whole ->
      feed (make (Int.max 1 (Nx.dim (-1) x))) [x]
  | Every k ->
      feed (make k) (pieces k x)
  | Reset k ->
      let d = make k in
      ignore (feed d (pieces k x)) ;
      d.reset () ;
      feed d (pieces k x)

let stream ~source p max_chunk =
  let s = Pipeline.Stream.prepare p ~source ~max_chunk in
  { push= Pipeline.Stream.push s
  ; flush= (fun () -> Pipeline.Stream.flush s)
  ; reset= (fun () -> Pipeline.Stream.reset s) }

let joined image = function
  | [] ->
      empty
  | l ->
      image (Nx.concatenate ~axis:(-1) l)

let checks = ref 0

let failures = ref []

let same_shape a b =
  Array.length a = Array.length b && Array.for_all2 Int.equal a b

let first_difference a b =
  let words = Int.min (String.length a) (String.length b) / 8 in
  let rec go i =
    if i >= words then None
    else if
      Int64.equal
        (String.get_int64_le a (8 * i))
        (String.get_int64_le b (8 * i))
    then go (i + 1)
    else Some i
  in
  go 0

let check name what ~expected got =
  incr checks ;
  if
    not
      ( same_shape expected.shape got.shape
      && String.equal expected.words got.words )
  then
    let detail =
      if not (same_shape expected.shape got.shape) then
        Printf.sprintf "shape %s, expected %s" (shape_string got.shape)
          (shape_string expected.shape)
      else
        match first_difference expected.words got.words with
        | Some i ->
            Printf.sprintf "word %d is %h, expected %h" i
              (Int64.float_of_bits (String.get_int64_le got.words (8 * i)))
              (Int64.float_of_bits (String.get_int64_le expected.words (8 * i)))
        | None ->
            "lengths differ"
    in
    failures := Printf.sprintf "%s, %s: %s" name what detail :: !failures

(* A stage is pinned through its laws: equal to a row's value at every chunking,
   it moves exactly when that row moves. That holds only if the law's reference
   is itself a row; a reference no row prints (the stage's own one-chunk run, a
   function applied to an input no row reads) would let a change confined to the
   stage's arithmetic, such as a float32 kernel only the stage takes, keep every
   law and every row. [anchor name expected] records the reference, and the run
   fails at the end if no single-output row printed its digest. *)
let anchors = ref []

let anchor name expected = anchors := (name, digest [expected]) :: !anchors

let check_anchors () =
  List.iter
    (fun (name, d) ->
      incr checks ;
      if not (Hashtbl.mem pinned d) then
        failures :=
          Printf.sprintf "%s: its reference is printed by no row" name
          :: !failures )
    (List.rev !anchors)

(* [law name ~image ~expected make x chunkings]: the outputs of the driver
   [make] over [x], at every partition in [chunkings], concatenated along the
   time axis, have exactly the bits of [expected], which a row pins. *)
let law name ~image ~expected make x chunkings =
  let expected = image expected in
  anchor name expected ;
  List.iter
    (fun c -> check name (describe c) ~expected (joined image (drive make c x)))
    chunkings

(* [identity name ~image a b]: two documented spellings of one computation have
   the same bits. *)
let identity name ~image a b =
  check name "offline" ~expected:(image a) (image b)

(* {1 Configurations}

   Every parameter explicit. Short names, used in the row names: [hann] is the
   common centered analysis; [odd] an odd transform size, an odd hop, a shorter
   window, left aligned; [right-edge] and [right-reflect] the causal alignment,
   the latter the one stage that declares lookahead. *)

let stft_hann =
  Stft.Config.create ~window:Window.Hann ~win_length:1024 ~hop:256
    ~alignment:`Centered ~pad:`Reflect ~scale:`None ~fft_size:1024 ()

let stft_odd =
  Stft.Config.create ~window:Window.Hamming ~win_length:400 ~hop:111
    ~alignment:`Left ~pad:(`Constant 0.) ~scale:`Magnitude ~fft_size:509 ()

let stft_right_edge =
  Stft.Config.create ~window:(Window.Kaiser 8.6) ~win_length:700 ~hop:160
    ~alignment:`Right ~pad:`Edge ~scale:`Psd ~fft_size:1024 ()

let stft_right_reflect =
  Stft.Config.create ~window:Window.Blackman ~win_length:512 ~hop:128
    ~alignment:`Right ~pad:`Reflect ~scale:`None ~fft_size:512 ()

let stft_left =
  Stft.Config.create ~window:Window.Hann ~win_length:1024 ~hop:256
    ~alignment:`Left ~pad:`Reflect ~scale:`None ~fft_size:1024 ()

let analyses =
  [ ("hann", stft_hann)
  ; ("odd", stft_odd)
  ; ("right-edge", stft_right_edge)
  ; ("right-reflect", stft_right_reflect) ]

let mel_slaney =
  Mel.Config.create ~f_min:0. ~f_max:11025. ~scale:`Slaney ~norm:`Slaney
    ~n_mels:64 ~sample_rate ~fft_size:1024 ()

let mel_htk =
  Mel.Config.create ~f_min:50. ~f_max:8000. ~scale:`Htk ~norm:`None ~n_mels:40
    ~sample_rate ~fft_size:1024 ()

let chroma_c12 =
  Chroma.Config.create ~n_chroma:12 ~tuning:0. ~ctroct:5. ~octwidth:(Some 2.)
    ~base_c:true ~sample_rate ~fft_size:1024 ()

let chroma_a24 =
  Chroma.Config.create ~n_chroma:24 ~tuning:(-0.2) ~ctroct:4.5 ~octwidth:None
    ~base_c:false ~sample_rate ~fft_size:1024 ()

(* [cqt] decimates (through apply_gemm offline), [vqt] is the ERB-gamma
   variable-Q ladder with reflect padding, [cqt-odd] an odd hop and therefore no
   decimation at all. *)
let cqt =
  Cqt.Config.create ~fmin:110. ~bins_per_octave:12 ~gamma:`Constant_q ~tuning:0.
    ~filter_scale:1. ~norm:1. ~window:Window.Hann ~scale:true ~hop:256
    ~pad:(`Constant 0.) ~n_bins:48 ~sample_rate ()

let vqt =
  Cqt.Config.create ~fmin:55. ~bins_per_octave:24 ~gamma:`Erb ~tuning:0.1
    ~filter_scale:0.8 ~norm:2. ~window:Window.Hann ~scale:false ~hop:256
    ~pad:`Reflect ~n_bins:96 ~sample_rate ()

let cqt_odd =
  Cqt.Config.create ~fmin:220. ~bins_per_octave:36 ~gamma:(`Fixed 3.) ~tuning:0.
    ~filter_scale:1. ~norm:1. ~window:Window.Hamming ~scale:true ~hop:301
    ~pad:(`Constant 0.) ~n_bins:108 ~sample_rate ()

let ladders = [("cqt", cqt); ("vqt", vqt); ("cqt-odd", cqt_odd)]

(* One conversion per executor the planner chooses: GEMM (16000, 24000),
   overlap-save FFT (44100), direct (11025 at `Best), the two cascade orders and
   the identity. The input is declared at each conversion's own rate. *)
let conversions =
  [ ("22050-16000 high", 22050, 16000, `High)
  ; ( "22050-24000 custom-90-0.9"
    , 22050
    , 24000
    , `Custom {Resample.attenuation= 90.; passband= 0.9} )
  ; ("22050-44100 fast", 22050, 44100, `Fast)
  ; ("22050-11025 best", 22050, 11025, `Best)
  ; ("48000-8000 high", 48000, 8000, `High)
  ; ("8000-48000 high", 8000, 48000, `High)
  ; ("22050-22050 high", 22050, 22050, `High) ]

let resampler (_, sample_rate, target, quality) =
  Resample.Config.create ~quality ~sample_rate ~target ()

let windows =
  Window.
    [ ("hann", Hann)
    ; ("hamming", Hamming)
    ; ("blackman", Blackman)
    ; ("blackman-harris", Blackman_harris)
    ; ("nuttall", Nuttall)
    ; ("bartlett", Bartlett)
    ; ("kaiser-8.6", Kaiser 8.6)
    ; ("gaussian-50", Gaussian 50.)
    ; ("tukey-0.25", Tukey 0.25)
    ; ("flat-top", Flat_top)
    ; ("rectangular", Rectangular) ]

(* {1 Rows independent of the dtype} *)

let grids () =
  List.iter
    (fun (name, c) ->
      row ("stft.grid " ^ name)
        [ ints
            ( [ Stft.frames c ~n
              ; Stft.first_complete c
              ; Stft.last_complete c ~n
              ; Stft.Config.bins c
              ; Stft.Config.latency c
              ; Stft.Config.synthesis_latency c ]
            @ rate_ints (Pipeline.latency (Stft.stage Nx.complex128 c))
            @ rate_ints (Pipeline.rate (Stft.stage Nx.complex128 c))
            @ rate_ints
                (Pipeline.latency
                   Pipeline.(
                     Stft.stage Nx.complex128 c
                     >> Stft.synthesis_stage Nx.float64 c ) ) ) ] )
    (analyses @ [("left", stft_left)]) ;
  List.iter
    (fun (name, c) ->
      row ("cqt.grid " ^ name)
        [ ints [Cqt.frames c ~n; Cqt.Config.latency c; Cqt.Config.n_octaves c]
        ; floats [Cqt.Config.cutoff c] ] )
    ladders ;
  List.iter
    (fun ((name, _, _, _) as conversion) ->
      let c = resampler conversion in
      row ("resample.grid " ^ name)
        [ ints
            ( [Resample.Config.latency c; Resample.Config.output_frames c ~n]
            @ rate_ints (Resample.Config.rate c)
            @ rate_ints (Resample.Config.output_latency c)
            @ rate_ints (Pipeline.latency (Resample.stage c)) ) ] )
    conversions ;
  row "window.cola"
    [ ints
        (List.map Bool.to_int
           [ Window.cola Window.Hann ~length:1024 ~hop:256
           ; Window.cola (Window.Kaiser 8.6) ~length:1024 ~hop:512
           ; Window.cola Window.Hamming ~length:400 ~hop:111 ] ) ] ;
  row "effects.semitones 12-tet"
    [ ints
        (List.concat_map
           (fun s -> rate_ints (Effects.semitones ~bins_per_octave:12 s))
           [-12.; -7.; -1.; 0.5; 3.; 4.; 7.; 12.; 19.] ) ] ;
  let latency p =
    rate_ints (Pipeline.latency p) @ rate_ints (Pipeline.rate p)
  in
  row "pipeline.latency feature stages"
    [ ints
        (List.concat
           [ latency (rms_stage ~frame_length:1024 ~hop:256 ())
           ; latency
               (zero_crossing_rate_stage ~frame_length:1024 ~hop:256
                  ~threshold:1e-10 () )
           ; latency (delta_stage ~width:9 ~order:1 ~mode:`Interp ())
           ; latency (delta_stage ~width:5 ~order:2 ~mode:`Nearest ())
           ; latency
               Pipeline.(
                 Stft.power_stage ~power:2. stft_hann
                 >> Mel.stage mel_slaney
                 >> mfcc_stage mel_slaney ~n_mfcc:13 ~lifter:22. ()
                 >> delta_stage ~width:9 ~order:1 ~mode:`Interp () )
           ; latency
               Pipeline.(
                 Stft.power_stage ~power:1. stft_right_reflect
                 >> spectral_contrast_stage stft_right_reflect ~n_bands:4
                      ~f_min:200. ~quantile:0.02 ~linear:true ~sample_rate () )
           ; latency
               Pipeline.(
                 rms_stage ~frame_length:1024 ~hop:256 ()
                 >> pcen_stage ~hop:256 ~gain:0.98 ~bias:2. ~power:0.5
                      ~time_constant:0.4 ~eps:1e-6 ~max_size:1 ~sample_rate () )
           ] ) ]

(* {1 Rows and laws at one dtype}

   [features (tag, ctag) dt cdt x noise] prints every row at the real dtype [dt]
   and its complex counterpart [cdt], over [x] and [noise] cast to [dt], and
   runs the laws at that dtype. A row is named by the dtype of its output: [tag]
   for real outputs, [ctag] for complex ones. *)

let features (type a c) (tag, ctag) (dt : (float, a) Nx.dtype)
    (cdt : (Complex.t, c) Nx.dtype) (x : (float, a) Nx.t)
    (noise : (float, a) Nx.t) =
  let name s = s ^ " " ^ tag in
  let r s images = row (name s) images in
  let rc s images = row (s ^ " " ^ ctag) images in
  let source = Pipeline.Format.audio dt ~sample_rate ~channels:2 in
  let source1 = Pipeline.Format.audio dt ~sample_rate ~channels:1 in
  let mono = Nx.slice [Nx.I 0; Nx.A] x in
  r "input" [real x; real noise] ;
  (* Windows. *)
  List.iter
    (fun (wname, w) ->
      r ("window.make " ^ wname)
        [ real (Window.make dt ~periodic:true w 1024)
        ; real (Window.make dt ~periodic:false w 333) ] )
    windows ;
  (* STFT analysis. *)
  let p2 = Stft.power_spectrum ~power:2. stft_hann x in
  let m1 = Stft.power_spectrum ~power:1. stft_hann x in
  let z = Stft.transform cdt stft_hann x in
  r "stft.power_spectrum hann p2" [real p2] ;
  r "stft.power_spectrum hann p1" [real m1] ;
  let powered =
    [ ("odd", stft_odd, 1.5)
    ; ("right-edge", stft_right_edge, 2.)
    ; ("right-reflect", stft_right_reflect, 2.) ]
  in
  List.iter
    (fun (cname, c, power) ->
      r
        (Printf.sprintf "stft.power_spectrum %s p%g" cname power)
        [real (Stft.power_spectrum ~power c x)] )
    powered ;
  List.iter
    (fun (cname, c) ->
      r
        ("stft.coordinates " ^ cname)
        [ real (Stft.frequencies dt c ~sample_rate)
        ; real (Stft.times dt c ~sample_rate ~n) ] )
    analyses ;
  let transforms =
    List.map (fun (cname, c) -> (cname, c, Stft.transform cdt c x)) analyses
  in
  List.iter
    (fun (cname, _, z) -> rc ("stft.transform " ^ cname) [complex z])
    transforms ;
  let range = Stft.transform_range cdt stft_hann ~p0:17 ~p1:61 x in
  rc "stft.transform_range hann 17-61" [complex range] ;
  (* STFT synthesis. *)
  r "stft.invert hann length-n" [real (Stft.invert dt stft_hann ~length:n z)] ;
  let inverses =
    List.map (fun (cname, c, z) -> (cname, c, z, Stft.invert dt c z)) transforms
  in
  List.iter
    (fun (cname, _, _, y) -> r ("stft.invert " ^ cname) [real y])
    inverses ;
  r "stft.griffin_lim hann zero-phase i8 m0.99 length-n"
    [ real
        (Stft.griffin_lim ~n_iter:8 ~momentum:0.99 ~init:`Zero_phase ~length:n
           stft_hann
           (Nx.slice [Nx.I 0; Nx.A; Nx.A] m1) ) ] ;
  let phase =
    let rng = Random.State.make [|7|] in
    Nx.cast dt
      (Nx.create Nx.float64 (Nx.shape m1)
         (Array.init (Nx.numel m1) (fun _ -> Random.State.float rng 4. -. 2.)) )
  in
  r "stft.griffin_lim hann phase i4 m0"
    [ real
        (Stft.griffin_lim ~n_iter:4 ~momentum:0. ~init:(`Phase phase) stft_hann
           m1 ) ] ;
  (* Conversions. *)
  r "convert.power_to_db ref-1 amin-1e-10 top-80"
    [real (Convert.power_to_db ~reference:1. ~amin:1e-10 ~top_db:80. p2)] ;
  r "convert.amplitude_to_db ref-0.5 amin-1e-5 top-60"
    [real (Convert.amplitude_to_db ~reference:0.5 ~amin:1e-5 ~top_db:60. m1)] ;
  r "convert.db_to_power ref-2"
    [ real
        (Convert.db_to_power ~reference:2.
           (Convert.power_to_db ~reference:1. ~amin:1e-10 ~top_db:120. p2) ) ] ;
  r "convert.db_to_amplitude ref-2"
    [ real
        (Convert.db_to_amplitude ~reference:2.
           (Nx.slice [Nx.A; Nx.R (0, 64); Nx.A] m1) ) ] ;
  let freqs = Stft.frequencies dt stft_hann ~sample_rate in
  List.iter
    (fun (sname, scale) ->
      let mels = Convert.hz_to_mel ~scale freqs in
      r
        ("convert.mel_scale " ^ sname)
        [real mels; real (Convert.mel_to_hz ~scale mels)] )
    [("slaney", `Slaney); ("htk", `Htk)] ;
  let midi = Convert.hz_to_midi (Nx.slice [Nx.R (1, 513)] freqs) in
  r "convert.midi" [real midi; real (Convert.midi_to_hz midi)] ;
  let grid =
    Convert.time_to_frames ~sample_rate ~hop:256
      (Stft.times dt stft_hann ~sample_rate ~n)
  in
  r "convert.frame_grid hop-256"
    [real grid; real (Convert.frames_to_time ~sample_rate ~hop:256 grid)] ;
  (* Decibel stages. *)
  let mel_p2 = Mel.apply mel_slaney p2 in
  r "db.stage maximum amin-1e-10 run"
    [real (Pipeline.run ~source (Db.stage ~amin:1e-10 Db.Maximum) mel_p2)] ;
  r "db.clamped_stage maximum amin-1e-10 top-80 run"
    [ real
        (Pipeline.run ~source
           (Db.clamped_stage ~amin:1e-10 ~top_db:80. Db.Maximum)
           mel_p2 ) ] ;
  let db_mel = Convert.power_to_db ~reference:1. ~amin:1e-10 mel_p2 in
  r "db.stage value-1 amin-1e-10 run"
    [real (Pipeline.run ~source (Db.stage ~amin:1e-10 (Db.Value 1.)) mel_p2)] ;
  law
    (name "db.stage value-1 amin-1e-10")
    ~image:real ~expected:db_mel
    (stream ~source (Db.stage ~amin:1e-10 (Db.Value 1.)))
    mel_p2 frames ;
  identity
    (name "db.clamped_stage value-1 = convert.power_to_db")
    ~image:real
    (Convert.power_to_db ~reference:1. ~amin:1e-10 ~top_db:80. mel_p2)
    (Pipeline.run ~source
       (Db.clamped_stage ~amin:1e-10 ~top_db:80. (Db.Value 1.))
       mel_p2 ) ;
  (* Mel and MFCC. *)
  r "mel.filterbank slaney" [real (Mel.filterbank dt mel_slaney)] ;
  r "mel.filterbank htk" [real (Mel.filterbank dt mel_htk)] ;
  r "mel.apply slaney p2" [real mel_p2] ;
  let mel_spec = mel_spectrogram stft_hann mel_slaney ~power:2. x in
  r "mel_spectrogram hann slaney p2" [real mel_spec] ;
  r "mel_spectrogram hann htk p1"
    [real (mel_spectrogram stft_hann mel_htk ~power:1. x)] ;
  let mf = mfcc stft_hann mel_slaney ~n_mfcc:13 ~lifter:22. x in
  r "mfcc hann slaney n13 lifter-22" [real mf] ;
  r "mfcc hann htk n20 lifter-0"
    [real (mfcc stft_hann mel_htk ~n_mfcc:20 ~lifter:0. x)] ;
  identity
    (name "mel_spectrogram = mel.apply . stft.power_spectrum")
    ~image:real mel_p2 mel_spec ;
  (* Chroma from the STFT. *)
  r "chroma.filterbank c12" [real (Chroma.filterbank dt chroma_c12)] ;
  r "chroma.filterbank a24" [real (Chroma.filterbank dt chroma_a24)] ;
  let chroma_p2 = Chroma.apply ~norm:`Inf chroma_c12 p2 in
  r "chroma.apply c12 inf" [real chroma_p2] ;
  let chroma_spec = chroma_stft stft_hann chroma_c12 ~power:2. ~norm:`Inf x in
  r "chroma_stft hann c12 p2 inf" [real chroma_spec] ;
  r "chroma_stft hann c12 p1 l2"
    [real (chroma_stft stft_hann chroma_c12 ~power:1. ~norm:(`P 2.) x)] ;
  r "chroma_stft hann a24 p2 none"
    [real (chroma_stft stft_hann chroma_a24 ~power:2. ~norm:`None x)] ;
  identity
    (name "chroma_stft = chroma.apply . stft.power_spectrum")
    ~image:real chroma_p2 chroma_spec ;
  (* Constant-Q. *)
  List.iter
    (fun (qname, c) ->
      r
        ("cqt.coordinates " ^ qname)
        [real (Cqt.frequencies dt c); real (Cqt.filter_lengths dt c)] )
    ladders ;
  List.iter
    (fun (qname, c) ->
      rc ("cqt.transform " ^ qname) [complex (Cqt.transform cdt c x)] )
    ladders ;
  let cqt_m1 = Cqt.power_spectrum ~power:1. cqt x in
  r "cqt.power_spectrum cqt p1" [real cqt_m1] ;
  r "cqt.power_spectrum vqt p2" [real (Cqt.power_spectrum ~power:2. vqt x)] ;
  r "cqt.power_spectrum cqt-odd p1.5"
    [real (Cqt.power_spectrum ~power:1.5 cqt_odd x)] ;
  r "chroma.cqt_projection cqt c12"
    [real (Chroma.cqt_projection dt ~n_chroma:12 cqt)] ;
  r "chroma.cqt_projection cqt-odd c12"
    [real (Chroma.cqt_projection dt ~n_chroma:12 cqt_odd)] ;
  let of_cqt = Chroma.of_cqt ~n_chroma:12 ~norm:`Inf cqt cqt_m1 in
  r "chroma.of_cqt cqt c12 inf" [real of_cqt] ;
  let ccqt = chroma_cqt cqt ~n_chroma:12 ~norm:`Inf x in
  r "chroma_cqt cqt c12 inf" [real ccqt] ;
  r "chroma_cqt vqt c12 l1" [real (chroma_cqt vqt ~n_chroma:12 ~norm:(`P 1.) x)] ;
  r "chroma_cqt cqt-odd c12 l1"
    [real (chroma_cqt cqt_odd ~n_chroma:12 ~norm:(`P 1.) x)] ;
  identity
    (name "chroma_cqt = chroma.of_cqt . cqt.power_spectrum")
    ~image:real of_cqt ccqt ;
  (* The kernel's partition law is anchored on its own whole-signal instance:
     its agreement with Cqt.transform is documented, not promised. *)
  List.iter
    (fun (qname, c) ->
      let kernel max_block =
        let k = Cqt.Kernel.prepare cdt c dt ~channels:2 ~max_block in
        { push= Cqt.Kernel.step k
        ; flush= (fun () -> Option.to_list (Cqt.Kernel.flush k))
        ; reset= (fun () -> Cqt.Kernel.reset k) }
      in
      let whole = joined complex (drive kernel Whole x) in
      rc ("cqt.kernel " ^ qname) [whole] ;
      anchor (name ("cqt.kernel " ^ qname)) whole ;
      let chunkings =
        [Every 333; Every 4096; Reset 333]
        @ if String.equal qname "cqt" then [Every 1] else []
      in
      List.iter
        (fun ch ->
          check
            (name ("cqt.kernel " ^ qname))
            (describe ch) ~expected:whole
            (joined complex (drive kernel ch x)) )
        chunkings )
    ladders ;
  (* Resampling. *)
  let flat = resample ~quality:`High ~sample_rate ~target:16000 x in
  r "resample 22050-16000 high" [real flat] ;
  List.iter
    (fun ((cname, _, _, _) as conversion) ->
      r
        ("resample.apply " ^ cname)
        [real (Resample.apply (resampler conversion) x)] )
    conversions ;
  (* apply_gemm has a row and never a law: it is numerically distinct from apply
     and carries no partition law. *)
  let dense cname =
    List.exists (String.equal cname) ["22050-16000 high"; "48000-8000 high"]
  in
  List.iter
    (fun ((cname, _, _, _) as conversion) ->
      if dense cname then begin
        r
          ("resample.apply_gemm " ^ cname)
          [real (Resample.apply_gemm (resampler conversion) x)] ;
        r
          ("resample.prototype " ^ cname)
          [real (Resample.Config.prototype dt (resampler conversion))]
      end )
    conversions ;
  identity
    (name "resample = resample.apply")
    ~image:real
    (Resample.apply
       (Resample.Config.create ~quality:`High ~sample_rate ~target:16000 ())
       x )
    flat ;
  identity
    (name "resample.apply identity = input")
    ~image:real x
    (Resample.apply
       (Resample.Config.create ~quality:`High ~sample_rate ~target:sample_rate
          () )
       x ) ;
  List.iter
    (fun ((cname, rate, _, _) as conversion) ->
      let c = resampler conversion in
      law
        (name ("resample.stage " ^ cname))
        ~image:real ~expected:(Resample.apply c x)
        (stream
           ~source:(Pipeline.Format.audio dt ~sample_rate:rate ~channels:2)
           (Resample.stage c) )
        x audio )
    conversions ;
  List.iter
    (fun ((cname, _, _, _) as conversion) ->
      if dense cname then
        let c = resampler conversion in
        let kernel max_block =
          let k = Resample.Kernel.prepare c dt ~channels:2 ~max_block in
          { push= Resample.Kernel.step k
          ; flush= (fun () -> Option.to_list (Resample.Kernel.flush k))
          ; reset= (fun () -> Resample.Kernel.reset k) }
        in
        law
          (name ("resample.kernel " ^ cname))
          ~image:real ~expected:(Resample.apply c x) kernel x coarse )
    conversions ;
  (* Spectral shape. *)
  let centroid = spectral_centroid ~sample_rate m1 in
  let bandwidth = spectral_bandwidth ~p:2. ~sample_rate m1 in
  let rolloff = spectral_rolloff ~roll_percent:0.85 ~sample_rate m1 in
  let flatness = spectral_flatness ~amin:1e-10 ~power:2. m1 in
  r "spectral_centroid hann" [real centroid] ;
  r "spectral_bandwidth hann p2" [real bandwidth] ;
  r "spectral_rolloff hann 0.85" [real rolloff] ;
  r "spectral_flatness hann amin-1e-10 p2" [real flatness] ;
  let custom =
    Nx.cast dt
      (Nx.create Nx.float64 [|513|]
         (Array.init 513 (fun k -> Float.of_int (k * k) /. 24.)) )
  in
  let centroid_f = spectral_centroid ~freqs:custom ~sample_rate m1 in
  let bandwidth_f = spectral_bandwidth ~p:1.5 ~freqs:custom ~sample_rate m1 in
  let rolloff_f =
    spectral_rolloff ~roll_percent:0.5 ~freqs:custom ~sample_rate m1
  in
  r "spectral_centroid custom-freqs" [real centroid_f] ;
  r "spectral_bandwidth custom-freqs p1.5" [real bandwidth_f] ;
  r "spectral_bandwidth custom-freqs p1.5 given-centroid"
    [ real
        (spectral_bandwidth ~p:1.5 ~freqs:custom ~centroid:centroid_f
           ~sample_rate m1 ) ] ;
  r "spectral_rolloff custom-freqs 0.5" [real rolloff_f] ;
  let frame_law s expected p input =
    law (name s) ~image:real ~expected (stream ~source p) input frames
  in
  frame_law "spectral_centroid_stage" centroid
    (spectral_centroid_stage ~sample_rate ())
    m1 ;
  frame_law "spectral_bandwidth_stage p2" bandwidth
    (spectral_bandwidth_stage ~p:2. ~sample_rate ())
    m1 ;
  frame_law "spectral_rolloff_stage 0.85" rolloff
    (spectral_rolloff_stage ~roll_percent:0.85 ~sample_rate ())
    m1 ;
  frame_law "spectral_flatness_stage amin-1e-10 p2" flatness
    (spectral_flatness_stage ~amin:1e-10 ~power:2. ())
    m1 ;
  frame_law "spectral_centroid_stage custom-freqs" centroid_f
    (spectral_centroid_stage ~freqs:custom ~sample_rate ())
    m1 ;
  frame_law "spectral_bandwidth_stage custom-freqs p1.5" bandwidth_f
    (spectral_bandwidth_stage ~p:1.5 ~freqs:custom ~sample_rate ())
    m1 ;
  frame_law "spectral_rolloff_stage custom-freqs 0.5" rolloff_f
    (spectral_rolloff_stage ~roll_percent:0.5 ~freqs:custom ~sample_rate ())
    m1 ;
  (* Energy. *)
  let energy = rms ~frame_length:1024 ~hop:256 x in
  r "rms 1024-256" [real energy] ;
  r "rms_of_spectrogram 1024" [real (rms_of_spectrogram ~frame_length:1024 m1)] ;
  List.iter
    (fun (threshold, chunkings) ->
      let zcr = zero_crossing_rate ~frame_length:1024 ~hop:256 ~threshold x in
      let s =
        Printf.sprintf "zero_crossing_rate 1024-256 threshold-%g" threshold
      in
      r s [real zcr] ;
      law
        (name
           (Printf.sprintf "zero_crossing_rate_stage 1024-256 threshold-%g"
              threshold ) )
        ~image:real ~expected:zcr
        (stream ~source
           (zero_crossing_rate_stage ~frame_length:1024 ~hop:256 ~threshold ()) )
        x chunkings )
    [(1e-10, audio); (0.05, coarse)] ;
  law
    (name "rms_stage 1024-256")
    ~image:real ~expected:energy
    (stream ~source (rms_stage ~frame_length:1024 ~hop:256 ()))
    x audio ;
  (* Delta. *)
  let deltas = [(9, 1, `Interp, "interp"); (5, 2, `Nearest, "nearest")] in
  List.iter
    (fun (width, order, mode, mname) ->
      let s = Printf.sprintf "w%d o%d %s" width order mname in
      let d = delta ~width ~order ~mode mf in
      r ("delta mfcc " ^ s) [real d] ;
      frame_law ("delta_stage " ^ s) d (delta_stage ~width ~order ~mode ()) mf )
    deltas ;
  let energy_delta = delta ~width:9 ~order:1 ~mode:`Interp energy in
  r "delta rms w9 o1 interp" [real energy_delta] ;
  law
    (name "rms_stage >> delta_stage w9 o1 interp")
    ~image:real ~expected:energy_delta
    (stream ~source
       Pipeline.(
         rms_stage ~frame_length:1024 ~hop:256 ()
         >> delta_stage ~width:9 ~order:1 ~mode:`Interp () ) )
    x audio ;
  (* PCEN. *)
  let melm = Mel.apply mel_slaney m1 in
  let pcens =
    [ ("g0.98 b2 p0.5 tc0.4 max1", 0.98, 2., 0.5, 0.4, None, 1)
    ; ("g0.8 b10 p0.25 tc0.06 s0.05 max3", 0.8, 10., 0.25, 0.06, Some 0.05, 3)
    ; ("g0.9 b0 p0 tc0.2 max1", 0.9, 0., 0., 0.2, None, 1) ]
  in
  List.iter
    (fun (s, gain, bias, power, time_constant, b, max_size) ->
      let p =
        pcen ~hop:256 ~gain ~bias ~power ~time_constant ~eps:1e-6 ?b ~max_size
          ~sample_rate melm
      in
      r ("pcen mel " ^ s) [real p] ;
      frame_law ("pcen_stage " ^ s) p
        (pcen_stage ~hop:256 ~gain ~bias ~power ~time_constant ~eps:1e-6 ?b
           ~max_size ~sample_rate () )
        melm )
    pcens ;
  r "pcen band g0.98 b2 p0.5 tc0.4 max1"
    [ real
        (pcen ~hop:256 ~gain:0.98 ~bias:2. ~power:0.5 ~time_constant:0.4
           ~eps:1e-6 ~max_size:1 ~sample_rate
           (Nx.slice [Nx.I 0; Nx.I 10; Nx.A] melm) ) ] ;
  (* Spectral contrast. *)
  let contrast_log =
    spectral_contrast stft_hann ~n_bands:6 ~f_min:200. ~quantile:0.02
      ~linear:false ~sample_rate x
  in
  r "spectral_contrast hann b6 f200 q0.02 log" [real contrast_log] ;
  identity
    (name
       "spectral_contrast = spectral_contrast_of_spectrogram . power_spectrum" )
    ~image:real contrast_log
    (spectral_contrast_of_spectrogram ~n_bands:6 ~f_min:200. ~quantile:0.02
       ~linear:false ~sample_rate m1 ) ;
  let contrast_lin =
    spectral_contrast_of_spectrogram ~n_bands:5 ~f_min:150. ~quantile:0.05
      ~linear:true ~sample_rate m1
  in
  r "spectral_contrast_of_spectrogram hann b5 f150 q0.05 lin" [real contrast_lin] ;
  frame_law "spectral_contrast_stage b5 f150 q0.05 lin" contrast_lin
    (spectral_contrast_stage stft_hann ~n_bands:5 ~f_min:150. ~quantile:0.05
       ~linear:true ~sample_rate () )
    m1 ;
  (* Onset strength. *)
  r "onset_strength hann slaney lag-1"
    [real (onset_strength stft_hann mel_slaney ~lag:1 x)] ;
  r "onset_strength left slaney lag-2"
    [real (onset_strength stft_left mel_slaney ~lag:2 x)] ;
  (* The stage leaves the centered shift and the clamp to the caller, so its law
     is against its own one-chunk run, which a row pins: at float32 the stage
     takes a flux kernel onset_strength never calls. *)
  let onset_stage = onset_strength_stage ~lag:1 () in
  let onset_run = Pipeline.run ~source onset_stage db_mel in
  r "onset_strength_stage lag-1 run" [real onset_run] ;
  frame_law "onset_strength_stage lag-1" onset_run onset_stage db_mel ;
  let onset_chain =
    Pipeline.(
      Stft.power_stage ~power:2. stft_hann
      >> Mel.stage mel_slaney
      >> Db.stage ~amin:1e-10 (Db.Value 1.)
      >> onset_strength_stage ~lag:1 () )
  in
  let onset_chain_run = Pipeline.run ~source onset_chain x in
  r "onset_strength_stage chain hann slaney value-1 lag-1 run"
    [real onset_chain_run] ;
  law
    (name "power_stage >> mel.stage >> db.stage >> onset_strength_stage")
    ~image:real ~expected:onset_chain_run
    (stream ~source onset_chain)
    x audio ;
  (* MFCC stage: against its one-chunk run on the clamped signal, and against
     mfcc itself on noise, where the clamp it omits does not bind; both
     references are rows. *)
  let mfcc_s = mfcc_stage mel_slaney ~n_mfcc:13 ~lifter:22. () in
  let mfcc_run = Pipeline.run ~source mfcc_s mel_p2 in
  r "mfcc_stage slaney n13 lifter-22 run" [real mfcc_run] ;
  frame_law "mfcc_stage slaney n13 lifter-22" mfcc_run mfcc_s mel_p2 ;
  let mfcc_noise = mfcc stft_hann mel_slaney ~n_mfcc:13 ~lifter:22. noise in
  r "mfcc hann slaney n13 lifter-22 noise" [real mfcc_noise] ;
  law
    (name "power_stage >> mel.stage >> mfcc_stage (noise)")
    ~image:real ~expected:mfcc_noise
    (stream ~source:source1
       Pipeline.(
         Stft.power_stage ~power:2. stft_hann
         >> Mel.stage mel_slaney
         >> mfcc_stage mel_slaney ~n_mfcc:13 ~lifter:22. () ) )
    noise coarse ;
  (* Contrast stage on the logarithmic path, on noise for the same reason. *)
  let contrast_noise =
    spectral_contrast stft_hann ~n_bands:6 ~f_min:200. ~quantile:0.02
      ~linear:false ~sample_rate noise
  in
  r "spectral_contrast hann b6 f200 q0.02 log noise" [real contrast_noise] ;
  law
    (name "power_stage >> spectral_contrast_stage log (noise)")
    ~image:real ~expected:contrast_noise
    (stream ~source:source1
       Pipeline.(
         Stft.power_stage ~power:1. stft_hann
         >> spectral_contrast_stage stft_hann ~n_bands:6 ~f_min:200.
              ~quantile:0.02 ~linear:false ~sample_rate () ) )
    noise coarse ;
  (* Harmonic-percussive separation. *)
  let h, p =
    hpss stft_hann ~kernel_size:(31, 31) ~power:2. ~margin:(1., 1.) x
  in
  r "hpss hann k31x31 p2 m1x1" [real h; real p] ;
  let harm =
    harmonic stft_hann ~kernel_size:(17, 31) ~power:1. ~margin:(2., 2.) x
  in
  let perc =
    percussive stft_hann ~kernel_size:(31, 17) ~power:Float.infinity
      ~margin:(1., 3.) x
  in
  r "harmonic hann k17x31 p1 m2x2" [real harm] ;
  r "percussive hann k31x17 pinf m1x3" [real perc] ;
  identity
    (name "harmonic = fst hpss")
    ~image:real
    (fst (hpss stft_hann ~kernel_size:(17, 31) ~power:1. ~margin:(2., 2.) x))
    harm ;
  identity
    (name "percussive = snd hpss")
    ~image:real
    (snd
       (hpss stft_hann ~kernel_size:(31, 17) ~power:Float.infinity
          ~margin:(1., 3.) x ) )
    perc ;
  let h, p =
    hpss stft_hann ~kernel_size:(16, 30) ~power:2. ~margin:(1., 1.) x
  in
  r "hpss hann k16x30 p2 m1x1" [real h; real p] ;
  let h, p =
    hpss_of_spectrogram ~kernel_size:(31, 31) ~power:2. ~margin:(1., 1.) m1
  in
  r "hpss_of_spectrogram hann k31x31 p2 m1x1" [real h; real p] ;
  let h, p = hpss_masks ~kernel_size:(31, 31) ~power:2. ~margin:(1.5, 1.5) m1 in
  r "hpss_masks hann k31x31 p2 m1.5x1.5" [real h; real p] ;
  let h, p = hpss_of_stft ~kernel_size:(31, 31) ~power:2. ~margin:(1., 1.) z in
  rc "hpss_of_stft hann k31x31 p2 m1x1" [complex h; complex p] ;
  (* Effects. *)
  rc "effects.phase_vocoder hann independent 1.5"
    [complex (Effects.phase_vocoder ~phase:`Independent stft_hann ~rate:1.5 z)] ;
  rc "effects.phase_vocoder hann locked 0.75"
    [complex (Effects.phase_vocoder ~phase:`Locked stft_hann ~rate:0.75 z)] ;
  List.iter
    (fun (rate, phase, pname) ->
      r
        (Printf.sprintf "effects.time_stretch hann %s %g" pname rate)
        [real (Effects.time_stretch ~phase stft_hann ~rate x)] )
    [ (1.25, `Independent, "independent")
    ; (0.8, `Locked, "locked")
    ; (1., `Independent, "independent") ] ;
  r "effects.pitch_shift hann locked high +3"
    [ real
        (Effects.pitch_shift ~phase:`Locked ~quality:`High stft_hann
           ~ratio:(Effects.semitones ~bins_per_octave:12 3.)
           x ) ] ;
  r "effects.pitch_shift hann independent fast -5"
    [ real
        (Effects.pitch_shift ~phase:`Independent ~quality:`Fast stft_hann
           ~ratio:(Effects.semitones ~bins_per_octave:12 (-5.))
           x ) ] ;
  let trims =
    [ ("top-60 maximum 1024-256 mono", 60., `Maximum, 1024, 256, mono)
    ; ("top-40 value-0.5 2048-512 stereo", 40., `Value 0.5, 2048, 512, x) ]
  in
  List.iter
    (fun (s, top_db, reference, frame_length, hop, y) ->
      let trimmed, (start, stop) =
        Effects.trim ~top_db ~reference ~frame_length ~hop y
      in
      r ("effects.trim " ^ s) [real trimmed; ints [start; stop]] ;
      let rank = Nx.ndim y in
      identity
        (name ("effects.trim = slice " ^ s))
        ~image:real
        (Nx.slice
           (List.init rank (fun i ->
                if i = rank - 1 then Nx.R (start, stop) else Nx.A ) )
           y )
        trimmed )
    trims ;
  List.iter
    (fun (s, top_db, reference, frame_length, hop, y) ->
      r ("effects.split " ^ s)
        [ ints
            (List.concat_map
               (fun (a, b) -> [a; b])
               (Array.to_list
                  (Effects.split ~top_db ~reference ~frame_length ~hop y) ) ) ] )
    [ ("top-30 maximum 1024-256 mono", 30., `Maximum, 1024, 256, mono)
    ; ("top-20 value-0.25 512-128 stereo", 20., `Value 0.25, 512, 128, x) ] ;
  (* STFT laws. *)
  let range_check =
    Nx.concatenate ~axis:(-1)
      [ Stft.transform_range cdt stft_hann ~p0:0 ~p1:17 x
      ; range
      ; Stft.transform_range cdt stft_hann ~p0:61 ~p1:(Stft.frames stft_hann ~n)
          x ]
  in
  identity
    (name "stft.transform_range = slice of stft.transform")
    ~image:complex
    (Nx.slice [Nx.A; Nx.A; Nx.R (17, 61)] z)
    range ;
  identity
    (name "stft.transform_range adjacent ranges = stft.transform")
    ~image:complex z range_check ;
  List.iter
    (fun (cname, c, power) ->
      law
        (name (Printf.sprintf "stft.power_stage %s p%g" cname power))
        ~image:real
        ~expected:(Stft.power_spectrum ~power c x)
        (stream ~source (Stft.power_stage ~power c))
        x audio )
    (("hann", stft_hann, 2.) :: powered) ;
  List.iter
    (fun (cname, c, z) ->
      law
        (name ("stft.stage " ^ cname))
        ~image:complex ~expected:z
        (stream ~source (Stft.stage cdt c))
        x
        (if String.equal cname "hann" then audio else coarse) )
    transforms ;
  let stft_kernel max_block =
    let k = Stft.Kernel.prepare cdt stft_hann dt ~channels:2 ~max_block in
    { push= Stft.Kernel.step k
    ; flush= (fun () -> Option.to_list (Stft.Kernel.flush k))
    ; reset= (fun () -> Stft.Kernel.reset k) }
  in
  law (name "stft.kernel hann") ~image:complex ~expected:z stft_kernel x coarse ;
  List.iter
    (fun (cname, c, z, y) ->
      law
        (name ("stft.synthesis_stage " ^ cname))
        ~image:real ~expected:y
        (stream ~source (Stft.synthesis_stage dt c))
        z frames )
    inverses ;
  let synthesis max_block =
    let k = Stft.Synthesis.prepare dt stft_hann cdt ~channels:2 ~max_block in
    { push= Stft.Synthesis.step k
    ; flush= (fun () -> Option.to_list (Stft.Synthesis.flush k))
    ; reset= (fun () -> Stft.Synthesis.reset k) }
  in
  law
    (name "stft.synthesis hann")
    ~image:real
    ~expected:(Stft.invert dt stft_hann z)
    synthesis z frames ;
  law
    (name "stft.stage >> stft.synthesis_stage hann")
    ~image:real
    ~expected:(Stft.invert dt stft_hann z)
    (stream ~source
       Pipeline.(Stft.stage cdt stft_hann >> Stft.synthesis_stage dt stft_hann) )
    x audio ;
  (* Projection stages, and the chains that end in them. *)
  frame_law "mel.stage slaney" mel_p2 (Mel.stage mel_slaney) p2 ;
  frame_law "chroma.stage c12 inf" chroma_p2
    (Chroma.stage ~norm:`Inf chroma_c12)
    p2 ;
  frame_law "chroma.of_cqt_stage cqt c12 inf" of_cqt
    (Chroma.of_cqt_stage ~n_chroma:12 ~norm:`Inf cqt)
    cqt_m1 ;
  law
    (name "power_stage >> mel.stage")
    ~image:real ~expected:mel_spec
    (stream ~source
       Pipeline.(Stft.power_stage ~power:2. stft_hann >> Mel.stage mel_slaney) )
    x coarse ;
  law
    (name "power_stage >> chroma.stage")
    ~image:real ~expected:chroma_spec
    (stream ~source
       Pipeline.(
         Stft.power_stage ~power:2. stft_hann
         >> Chroma.stage ~norm:`Inf chroma_c12 ) )
    x coarse

(* {1 Since the rows above}

   The features and stages that landed after the rows above were recorded,
   appended in the order they landed so that every earlier line of the
   expectation files stays as it is: their rows independent of the dtype, then
   their rows and laws at each dtype, as above. *)

(* {2 Images and laws of outputs that are not one tensor} *)

(* A boolean output, as the integers 0 and 1 in its own shape. *)
let bools (t : (bool, Nx.bool_elt) Nx.t) =
  let values = Nx.to_array t in
  let b = Bytes.create (8 * Array.length values) in
  Array.iteri
    (fun i v -> Bytes.set_int64_le b (8 * i) (if v then 1L else 0L))
    values ;
  {shape= Nx.shape t; words= Bytes.to_string b}

(* Frame or sample indices, and sample intervals as their flattened bounds. *)
let indices a = ints (Array.to_list a)

let intervals a =
  ints (List.concat_map (fun (start, stop) -> [start; stop]) (Array.to_list a))

(* The images of a stream of pairs, joined component by component. *)
let pair image l = [joined image (List.map fst l); joined image (List.map snd l)]

(* The image of a stream of index arrays, joined end to end. *)
let concatenated image l = [image (Array.concat l)]

(* [law_by name ~join ~expected make x chunkings] is [law] for a stage whose
   output is not one tensor: [join] maps the outputs of one run, in order, to
   their images, and [expected] is the images a row printed. *)
let law_by name ~join ~expected make x chunkings =
  anchors := (name, digest expected) :: !anchors ;
  let outputs = List.length expected in
  List.iter
    (fun c ->
      let got = join (drive make c x) in
      if List.length got <> outputs then begin
        incr checks ;
        failures :=
          Printf.sprintf "%s, %s: %d outputs, expected %d" name (describe c)
            (List.length got) outputs
          :: !failures
      end
      else
        List.iteri
          (fun i (expected, got) ->
            let what =
              if outputs = 1 then describe c
              else Printf.sprintf "%s, output %d" (describe c) i
            in
            check name what ~expected got )
          (List.combine expected got) )
    chunkings

(* [cut length t] is [t] cut to its first [length] items along its last axis. *)
let cut length t =
  let rank = Nx.ndim t in
  let len = Int.min length (Nx.dim (rank - 1) t) in
  Nx.slice
    (List.init rank (fun i -> if i = rank - 1 then Nx.R (0, len) else Nx.A))
    t

(* {2 Configurations} *)

let q num den = {Pipeline.Rate.num; den}

(* Butterworth designs, as transfer functions and as second-order sections, at
   orders and cutoffs that take every band transform. *)
let designs =
  [ ("lowpass-1000 o4", 4, `Lowpass 1000.)
  ; ("highpass-80 o2", 2, `Highpass 80.)
  ; ("bandpass-300-3400 o3", 3, `Bandpass (300., 3400.))
  ; ("bandstop-50-60 o2", 2, `Bandstop (50., 60.)) ]

let sections =
  [ ("lowpass-2000 o8", 8, `Lowpass 2000.)
  ; ("bandpass-500-2000 o4", 4, `Bandpass (500., 2000.))
  ; ("highpass-40 o5", 5, `Highpass 40.) ]

let butter (_, order, band) = Filter.butter ~order ~sample_rate band

let butter_sos (_, order, band) = Filter.butter_sos ~order ~sample_rate band

let design s = List.find (fun (name, _, _) -> String.equal name s) designs

let section s = List.find (fun (name, _, _) -> String.equal name s) sections

(* A finite impulse response, and a transfer function to normalise by its
   leading denominator coefficient. *)
let fir =
  ( Nx.create Nx.float64 [|5|] [|0.1; 0.2; 0.4; 0.2; 0.1|]
  , Nx.create Nx.float64 [|1|] [|1.|] )

let unnormalised =
  ( Nx.create Nx.float64 [|2|] [|1.; 0.5|]
  , Nx.create Nx.float64 [|3|] [|2.5; -1.; 0.3|] )

let emphasis_state v = Nx.create Nx.float64 [|1|] [|v|]

(* The excerpt the dearer features read, rows and laws alike, so that even the
   one-sample partitions of their stages stay cheap: the 12288 samples (0.56 s)
   from sample 16384 on, which hold the sawtooth, two of its clicks and the
   inner silence. *)
let excerpt_length = 12288

let excerpt t = Nx.slice [Nx.A; Nx.R (16384, 16384 + excerpt_length)] t

(* {2 Rows independent of the dtype} *)

let grids_since () =
  let latency p =
    rate_ints (Pipeline.latency p) @ rate_ints (Pipeline.rate p)
  in
  (* Silence detection. *)
  row "pipeline.latency effects.split_stage"
    [ ints
        (List.concat
           [ latency
               (Effects.split_stage ~top_db:20. ~frame_length:512 ~hop:128
                  ~reference:(`Value 0.25) () )
           ; latency
               (Effects.split_stage ~top_db:35. ~frame_length:1023 ~hop:300
                  ~reference:(`Value 0.1) () ) ] ) ] ;
  (* Harmonic-percussive separation. *)
  row "pipeline.latency hpss stages"
    [ ints
        (List.concat
           [ latency
               (hpss_of_spectrogram_stage ~kernel_size:(31, 31) ~power:2.
                  ~margin:(1., 1.) () )
           ; latency
               (hpss_of_spectrogram_stage ~kernel_size:(16, 30)
                  ~power:Float.infinity ~margin:(1., 2.) () )
           ; latency
               (hpss_masks_stage ~kernel_size:(31, 31) ~power:2.
                  ~margin:(1.5, 1.5) () )
           ; latency
               (hpss_of_stft_stage ~kernel_size:(31, 31) ~power:2.
                  ~margin:(1., 1.) () )
           ; latency
               Pipeline.(
                 Stft.stage Nx.complex128 stft_hann
                 >> map fst
                      (hpss_of_stft_stage ~kernel_size:(31, 31) ~power:2.
                         ~margin:(1., 1.) () )
                 >> Stft.synthesis_stage Nx.float64 stft_hann ) ] ) ] ;
  (* Time-scale and pitch modification. *)
  row "pipeline.latency effects modification stages"
    [ ints
        (List.concat
           [ latency
               (Effects.phase_vocoder_stage ~phase:`Independent stft_hann
                  ~rate:(q 3 2) )
           ; latency
               (Effects.phase_vocoder_stage ~phase:`Locked stft_hann
                  ~rate:(q 3 4) )
           ; latency
               (Effects.time_stretch_stage ~phase:`Independent stft_hann
                  ~rate:(q 5 4) )
           ; latency
               (Effects.time_stretch_stage ~phase:`Locked stft_hann
                  ~rate:(q 4 5) )
           ; latency
               (Effects.time_stretch_stage ~phase:`Independent stft_odd
                  ~rate:(q 1 1) )
           ; latency
               (Effects.pitch_shift_stage ~phase:`Locked ~quality:`High
                  stft_hann
                  ~ratio:(Effects.semitones ~bins_per_octave:12 3.) )
           ; latency
               (Effects.pitch_shift_stage ~phase:`Independent ~quality:`Fast
                  stft_hann
                  ~ratio:(Effects.semitones ~bins_per_octave:12 (-5.)) ) ] ) ] ;
  (* Filtering. *)
  List.iter
    (fun ((name, _, _) as d) ->
      let b, a = butter d in
      row ("filter.butter " ^ name) [real b; real a] ;
      row ("filter.lfilter_zi " ^ name) [real (Filter.lfilter_zi b a)] )
    designs ;
  List.iter
    (fun ((name, _, _) as s) ->
      let sos = butter_sos s in
      row ("filter.butter_sos " ^ name) [real sos] ;
      row ("filter.sosfilt_zi " ^ name) [real (Filter.sosfilt_zi sos)] )
    sections ;
  row "filter.lfilter_zi unnormalised"
    [real (Filter.lfilter_zi (fst unnormalised) (snd unnormalised))] ;
  row "pipeline.latency filter stages"
    [ ints
        (List.concat
           [ (let b, a = butter (design "lowpass-1000 o4") in
              latency (Filter.lfilter_stage b a) )
           ; latency
               (Filter.sosfilt_stage (butter_sos (section "lowpass-2000 o8")))
           ; latency (Filter.preemphasis_stage ~coef:0.97 ())
           ; latency
               (Filter.preemphasis_stage ~coef:0.95 ~zi:(emphasis_state 0.1) ())
           ; latency (Filter.deemphasis_stage ~coef:0.97 ())
           ; latency
               (Filter.deemphasis_stage ~coef:0.95 ~zi:(emphasis_state 0.1) ())
           ] ) ] ;
  (* Fundamental frequency, pitch tracking, onsets, rhythm. *)
  row "pipeline.latency yin_stage"
    [ ints
        (List.concat
           [ latency
               (yin_stage ~frame_length:2048 ~hop:512 ~trough_threshold:0.1
                  ~alignment:`Centered ~pad:(`Constant 0.) ~fmin:60. ~fmax:1000.
                  ~sample_rate () )
           ; latency
               (yin_stage ~frame_length:1023 ~hop:300 ~trough_threshold:0.2
                  ~alignment:`Left ~pad:`Edge ~fmin:80. ~fmax:2000. ~sample_rate
                  () )
           ; latency
               (yin_stage ~frame_length:1500 ~hop:256 ~trough_threshold:0.15
                  ~alignment:`Centered ~pad:`Reflect ~fmin:100. ~fmax:800.
                  ~sample_rate () ) ] ) ] ;
  row "pipeline.latency piptrack_stage"
    [ ints
        (latency
           (piptrack_stage ~fmin:150. ~fmax:4000. ~threshold:0.1
              ~reference:`Maximum ~fft_size:1024 ~sample_rate () ) ) ] ;
  row "pipeline.latency peak_pick_stage"
    [ ints
        (List.concat
           [ latency
               (peak_pick_stage ~pre_max:3 ~post_max:1 ~pre_avg:10 ~post_avg:1
                  ~delta:0.1 ~wait:3 )
           ; latency
               (peak_pick_stage ~pre_max:2 ~post_max:3 ~pre_avg:4 ~post_avg:6
                  ~delta:0.05 ~wait:4 ) ] ) ] ;
  row "pipeline.latency rhythm stages"
    [ ints
        (List.concat
           [ latency
               (tempogram_stage ~win_length:64 ~alignment:`Centered
                  ~window:Window.Hann ~norm:`Inf () )
           ; latency
               (tempogram_stage ~win_length:33 ~alignment:`Left
                  ~window:Window.Hamming ~norm:(`P 2.) () )
           ; latency
               (fourier_tempogram_stage Nx.complex128 ~win_length:64
                  ~alignment:`Centered ~window:Window.Hann () )
           ; latency
               (fourier_tempogram_stage Nx.complex128 ~win_length:33
                  ~alignment:`Left ~window:Window.Hamming () ) ] ) ] ;
  (* Tonal features, analysis and inversion. *)
  row "pipeline.latency tonal stages"
    [ ints
        (List.concat
           [ latency
               (chroma_cens_stage ~norm:(`P 2.) ~smoothing:(Some 41)
                  ~smoothing_window:Window.Hann () )
           ; latency
               (chroma_cens_stage ~norm:`Inf ~smoothing:(Some 10)
                  ~smoothing_window:Window.Hamming () )
           ; latency
               (chroma_cens_stage ~norm:(`P 1.) ~smoothing:None
                  ~smoothing_window:Window.Hann () )
           ; latency (tonnetz_stage ()) ] ) ] ;
  row "pipeline.latency analysis and inverse stages"
    [ ints
        (List.concat
           [ latency (poly_features_stage stft_hann ~order:1 ~sample_rate ())
           ; latency
               Pipeline.(
                 Stft.power_stage ~power:1. stft_odd
                 >> poly_features_stage stft_odd ~order:3 ~sample_rate () )
           ; latency
               (mfcc_to_mel_stage mel_slaney ~n_mfcc:13 ~lifter:22.
                  ~reference:1. () )
           ; latency (mel_to_stft_stage mel_slaney ~power:2. ()) ] ) ]

(* {2 Rows and laws at one dtype}

   As [features], at the real dtype [dt] and its complex counterpart [cdt]. Rows
   on the excerpt end in [excerpt]. *)

let features_since (type a c) (tag, ctag) (dt : (float, a) Nx.dtype)
    (cdt : (Complex.t, c) Nx.dtype) (x : (float, a) Nx.t)
    (noise : (float, a) Nx.t) =
  let name s = s ^ " " ^ tag in
  let r s images = row (name s) images in
  let rc s images = row (s ^ " " ^ ctag) images in
  let source = Pipeline.Format.audio dt ~sample_rate ~channels:2 in
  let source1 = Pipeline.Format.audio dt ~sample_rate ~channels:1 in
  let mono = Nx.slice [Nx.I 0; Nx.A] x in
  let frame_law s expected p input =
    law (name s) ~image:real ~expected (stream ~source p) input frames
  in
  let pair_law s ~image ~expected p input =
    law_by (name s) ~join:(pair image)
      ~expected:[image (fst expected); image (snd expected)]
      (stream ~source p) input frames
  in
  let p2 = Stft.power_spectrum ~power:2. stft_hann x in
  let m1 = Stft.power_spectrum ~power:1. stft_hann x in
  let short = excerpt x in
  let short_noise = excerpt noise in
  let short_m1 = Stft.power_spectrum ~power:1. stft_hann short in
  let short_z = Stft.transform cdt stft_hann short in
  (* Silence detection streamed: against the offline split's row at its
     configuration, and at an odd frame length on mono audio. *)
  let split_law s ~top_db ~frame_length ~hop ~r:level y src chunkings =
    law_by
      (name ("effects.split_stage " ^ s))
      ~join:(concatenated intervals)
      ~expected:
        [ intervals
            (Effects.split ~top_db ~reference:(`Value level) ~frame_length ~hop
               y ) ]
      (stream ~source:src
         (Effects.split_stage ~top_db ~frame_length ~hop
            ~reference:(`Value level) () ) )
      y chunkings
  in
  r "effects.split top-35 value-0.1 1023-300 mono"
    [ intervals
        (Effects.split ~top_db:35. ~reference:(`Value 0.1) ~frame_length:1023
           ~hop:300 mono ) ] ;
  split_law "top-20 value-0.25 512-128 stereo" ~top_db:20. ~frame_length:512
    ~hop:128 ~r:0.25 x source audio ;
  split_law "top-35 value-0.1 1023-300 mono" ~top_db:35. ~frame_length:1023
    ~hop:300 ~r:0.1 mono source1 coarse ;
  (* Harmonic-percussive separation streamed, each stage against its offline
     face, and an even harmonic kernel under the hard mask. *)
  List.iter
    (fun (s, kernel_size, power, margin) ->
      let separated =
        hpss_of_spectrogram ~kernel_size ~power ~margin short_m1
      in
      r
        (Printf.sprintf "hpss_of_spectrogram hann %s excerpt" s)
        [real (fst separated); real (snd separated)] ;
      pair_law
        (Printf.sprintf "hpss_of_spectrogram_stage %s excerpt" s)
        ~image:real ~expected:separated
        (hpss_of_spectrogram_stage ~kernel_size ~power ~margin ())
        short_m1 )
    [ ("k31x31 p2 m1x1", (31, 31), 2., (1., 1.))
    ; ("k16x30 pinf m1x2", (16, 30), Float.infinity, (1., 2.)) ] ;
  let masks =
    hpss_masks ~kernel_size:(31, 31) ~power:2. ~margin:(1.5, 1.5) short_m1
  in
  r "hpss_masks hann k31x31 p2 m1.5x1.5 excerpt"
    [real (fst masks); real (snd masks)] ;
  pair_law "hpss_masks_stage k31x31 p2 m1.5x1.5 excerpt" ~image:real
    ~expected:masks
    (hpss_masks_stage ~kernel_size:(31, 31) ~power:2. ~margin:(1.5, 1.5) ())
    short_m1 ;
  let components =
    hpss_of_stft ~kernel_size:(31, 31) ~power:2. ~margin:(1., 1.) short_z
  in
  rc "hpss_of_stft hann k31x31 p2 m1x1 excerpt"
    [complex (fst components); complex (snd components)] ;
  pair_law "hpss_of_stft_stage k31x31 p2 m1x1 excerpt" ~image:complex
    ~expected:components
    (hpss_of_stft_stage ~kernel_size:(31, 31) ~power:2. ~margin:(1., 1.) ())
    short_z ;
  (* Time-scale and pitch modification streamed, each stage against its offline
     face at the double its exact rate names. *)
  List.iter
    (fun (s, phase, rate, (num, den)) ->
      let stretched = Effects.phase_vocoder ~phase stft_hann ~rate short_z in
      rc
        (Printf.sprintf "effects.phase_vocoder hann %s %g excerpt" s rate)
        [complex stretched] ;
      law
        (name
           (Printf.sprintf "effects.phase_vocoder_stage hann %s %d/%d excerpt" s
              num den ) )
        ~image:complex ~expected:stretched
        (stream ~source
           (Effects.phase_vocoder_stage ~phase stft_hann ~rate:(q num den)) )
        short_z frames )
    [ ("independent", `Independent, 1.5, (3, 2))
    ; ("locked", `Locked, 0.75, (3, 4)) ] ;
  List.iter
    (fun (s, phase, rate, (num, den), chunkings) ->
      let stretched = Effects.time_stretch ~phase stft_hann ~rate short in
      r
        (Printf.sprintf "effects.time_stretch hann %s %g excerpt" s rate)
        [real stretched] ;
      law
        (name
           (Printf.sprintf "effects.time_stretch_stage hann %s %d/%d excerpt" s
              num den ) )
        ~image:real ~expected:stretched
        (stream ~source
           (Effects.time_stretch_stage ~phase stft_hann ~rate:(q num den)) )
        short chunkings )
    [ ("independent", `Independent, 1.25, (5, 4), audio)
    ; ("locked", `Locked, 0.8, (4, 5), coarse)
    ; ("independent", `Independent, 1., (1, 1), coarse) ] ;
  List.iter
    (fun (s, phase, quality, semitones, chunkings) ->
      let ratio = Effects.semitones ~bins_per_octave:12 semitones in
      let shifted =
        Effects.pitch_shift ~phase ~quality stft_hann ~ratio short
      in
      r
        (Printf.sprintf "effects.pitch_shift hann %s %+g excerpt" s semitones)
        [real shifted] ;
      law
        (name
           (Printf.sprintf "effects.pitch_shift_stage hann %s %+g excerpt" s
              semitones ) )
        ~image:real ~expected:shifted
        (stream ~source
           (Effects.pitch_shift_stage ~phase ~quality stft_hann ~ratio) )
        short chunkings )
    [ ("locked high", `Locked, `High, 3., audio)
    ; ("independent fast", `Independent, `Fast, -5., coarse) ] ;
  (* The synthesis drained to an explicit length, against stft.invert at that
     length: short of the default length, past it into the samples the last
     frame reaches beyond the trailing trim and the zeros after them, and past
     it on an uncentered geometry, which trims nothing. *)
  List.iter
    (fun (s, c, length) ->
      let z = Stft.transform cdt c short in
      let inverse = Stft.invert dt c ~length z in
      r
        (Printf.sprintf "stft.invert %s length-%d excerpt" s length)
        [real inverse] ;
      let synthesis max_block =
        let k = Stft.Synthesis.prepare dt c cdt ~channels:2 ~max_block in
        { push= Stft.Synthesis.step k
        ; flush= (fun () -> Option.to_list (Stft.Synthesis.flush ~length k))
        ; reset= (fun () -> Stft.Synthesis.reset k) }
      in
      law
        (name
           (Printf.sprintf "stft.synthesis %s flush length-%d excerpt" s length) )
        ~image:(fun t -> real (cut length t))
        ~expected:inverse synthesis z frames )
    [ ("hann", stft_hann, 8000)
    ; ("hann", stft_hann, 13000)
    ; ("odd", stft_odd, 12500) ] ;
  (* Silence in a batch of clips, each against its own maximum: the second clip
     is the first reversed, 36 dB down, below the threshold against the first
     clip's maximum and above it against its own, where it sounds earlier. *)
  let batch = Nx.stack ~axis:0 [x; Nx.mul_s (Nx.flip ~axes:[1] x) 0x1p-6] in
  let trimmed, (start, stop) =
    Effects.trim ~top_db:30. ~reference:`Maximum ~frame_length:1024 ~hop:256
      batch
  in
  r "effects.trim top-30 maximum 1024-256 batch"
    [real trimmed; ints [start; stop]] ;
  identity
    (name "effects.trim = slice top-30 maximum 1024-256 batch")
    ~image:real
    (Nx.slice [Nx.A; Nx.A; Nx.R (start, stop)] batch)
    trimmed ;
  r "effects.split top-30 maximum 1024-256 batch"
    [ intervals
        (Effects.split ~top_db:30. ~reference:`Maximum ~frame_length:1024
           ~hop:256 batch ) ] ;
  (* Synthetic signals and the mu-law. *)
  r "tone 440 length-4096 phase-0.3"
    [real (tone dt ~length:4096 ~phase:0.3 ~sample_rate 440.)] ;
  r "tone -3000.25 duration-0.3 phase--pi/2"
    [ real
        (tone dt ~duration:0.3 ~phase:(-.Float.pi /. 2.) ~sample_rate (-3000.25))
    ] ;
  r "chirp linear 100-8000 length-8192 phase-0"
    [ real
        (chirp dt ~length:8192 ~linear:true ~phase:0. ~sample_rate ~fmin:100.
           ~fmax:8000. () ) ] ;
  r "chirp exponential 55-7040 duration-0.5 phase-0.7"
    [ real
        (chirp dt ~duration:0.5 ~linear:false ~phase:0.7 ~sample_rate ~fmin:55.
           ~fmax:7040. () ) ] ;
  r "chirp exponential 9000-200 length-5000 phase--pi/2"
    [ real
        (chirp dt ~length:5000 ~linear:false ~phase:(-.Float.pi /. 2.)
           ~sample_rate ~fmin:9000. ~fmax:200. () ) ] ;
  r "clicks times hop-512 f1000 d0.1 length-n"
    [ real
        (clicks dt ~hop:512 ~click_frequency:1000. ~click_duration:0.1 ~length:n
           ~sample_rate
           (`Times [|0.; 0.25; 0.25; 1.; 1.99|]) ) ] ;
  r "clicks frames hop-256 f2500 d0.03"
    [ real
        (clicks dt ~hop:256 ~click_frequency:2500. ~click_duration:0.03
           ~sample_rate
           (`Frames [|40; 0; 3; 170; 41|]) ) ] ;
  r "clicks samples custom-stereo length-30000"
    [ real
        (clicks dt ~hop:512
           ~click:(Nx.slice [Nx.A; Nx.R (5512, 5576)] x)
           ~click_frequency:1000. ~click_duration:0.1 ~length:30000 ~sample_rate
           (`Samples [|100; 50; 22050; 29990; 31000|]) ) ] ;
  let quiet = Nx.mul_s short 0.5 in
  List.iter
    (fun (s, mu, quantize, chunkings) ->
      let compressed = mu_compress ~mu ~quantize quiet in
      let expanded = mu_expand ~mu ~quantize compressed in
      r (Printf.sprintf "mu_compress %s excerpt" s) [real compressed] ;
      r (Printf.sprintf "mu_expand %s excerpt" s) [real expanded] ;
      law
        (name (Printf.sprintf "mu_compress_stage %s excerpt" s))
        ~image:real ~expected:compressed
        (stream ~source (mu_compress_stage ~mu ~quantize ()))
        quiet chunkings ;
      law
        (name (Printf.sprintf "mu_expand_stage %s excerpt" s))
        ~image:real ~expected:expanded
        (stream ~source (mu_expand_stage ~mu ~quantize ()))
        compressed chunkings )
    [ ("mu-255 quantize", 255., true, audio)
    ; ("mu-255 continuous", 255., false, coarse)
    ; ("mu-15 quantize", 15., true, coarse) ] ;
  (* Filtering, on the excerpt. *)
  let lp_b, lp_a = butter (design "lowpass-1000 o4") in
  let hp_b, hp_a = butter (design "highpass-80 o2") in
  let bp_b, bp_a = butter (design "bandpass-300-3400 o3") in
  let bs_b, bs_a = butter (design "bandstop-50-60 o2") in
  let lp_sos = butter_sos (section "lowpass-2000 o8") in
  let bp_sos = butter_sos (section "bandpass-500-2000 o4") in
  let hp_sos = butter_sos (section "highpass-40 o5") in
  let hp_zi = Nx.mul_s (Filter.lfilter_zi hp_b hp_a) 0.25 in
  let bp_zi = Nx.mul_s (Filter.lfilter_zi bp_b bp_a) (-0.5) in
  let sos_zi = Nx.mul_s (Filter.sosfilt_zi bp_sos) 0.5 in
  let lowpassed = Filter.lfilter lp_b lp_a short in
  let highpassed = Filter.lfilter ~zi:hp_zi hp_b hp_a short in
  let firred = Filter.lfilter (fst fir) (snd fir) short in
  let sosfiltered = Filter.sosfilt lp_sos short in
  let sosfiltered_zi = Filter.sosfilt ~zi:sos_zi bp_sos short in
  List.iter
    (fun (s, y) -> r (Printf.sprintf "filter.%s excerpt" s) [real y])
    [ ("lfilter lowpass-1000 o4", lowpassed)
    ; ("lfilter highpass-80 o2 zi-0.25", highpassed)
    ; ("lfilter bandstop-50-60 o2", Filter.lfilter bs_b bs_a short)
    ; ("lfilter fir-5", firred)
    ; ( "lfilter unnormalised"
      , Filter.lfilter (fst unnormalised) (snd unnormalised) short )
    ; ("sosfilt lowpass-2000 o8", sosfiltered)
    ; ("sosfilt bandpass-500-2000 o4 zi-0.5", sosfiltered_zi)
    ; ( "filtfilt lowpass-1000 o4 odd-15"
      , Filter.filtfilt ~pad:`Odd ~padlen:15 lp_b lp_a short )
    ; ( "filtfilt highpass-80 o2 reflect-0"
      , Filter.filtfilt ~pad:`Reflect ~padlen:0 hp_b hp_a short )
    ; ( "filtfilt bandpass-300-3400 o3 edge-100"
      , Filter.filtfilt ~pad:`Edge ~padlen:100 bp_b bp_a short )
    ; ( "sosfiltfilt lowpass-2000 o8 odd-27"
      , Filter.sosfiltfilt ~pad:`Odd ~padlen:27 lp_sos short )
    ; ( "sosfiltfilt highpass-40 o5 reflect-500"
      , Filter.sosfiltfilt ~pad:`Reflect ~padlen:500 hp_sos short ) ] ;
  let y, zf = Filter.lfilter_zf ~zi:bp_zi bp_b bp_a short in
  r "filter.lfilter_zf bandpass-300-3400 o3 zi--0.5 excerpt" [real y; real zf] ;
  let half = excerpt_length / 2 in
  let y1, z1 =
    Filter.lfilter_zf ~zi:bp_zi bp_b bp_a
      (Nx.slice [Nx.A; Nx.R (0, half)] short)
  in
  identity
    (name "filter.lfilter_zf blocks = filter.lfilter excerpt")
    ~image:real y
    (Nx.concatenate ~axis:(-1)
       [ y1
       ; Filter.lfilter ~zi:z1 bp_b bp_a
           (Nx.slice [Nx.A; Nx.R (half, excerpt_length)] short) ] ) ;
  let y, zf = Filter.sosfilt_zf ~zi:sos_zi bp_sos short in
  r "filter.sosfilt_zf bandpass-500-2000 o4 zi-0.5 excerpt" [real y; real zf] ;
  List.iter
    (fun (s, expected, stage, chunkings) ->
      law
        (name (Printf.sprintf "filter.%s excerpt" s))
        ~image:real ~expected (stream ~source stage) short chunkings )
    [ ( "lfilter_stage lowpass-1000 o4"
      , lowpassed
      , Filter.lfilter_stage lp_b lp_a
      , audio )
    ; ( "lfilter_stage highpass-80 o2 zi-0.25"
      , highpassed
      , Filter.lfilter_stage ~zi:hp_zi hp_b hp_a
      , coarse )
    ; ( "lfilter_stage fir-5"
      , firred
      , Filter.lfilter_stage (fst fir) (snd fir)
      , coarse )
    ; ( "sosfilt_stage lowpass-2000 o8"
      , sosfiltered
      , Filter.sosfilt_stage lp_sos
      , audio )
    ; ( "sosfilt_stage bandpass-500-2000 o4 zi-0.5"
      , sosfiltered_zi
      , Filter.sosfilt_stage ~zi:sos_zi bp_sos
      , coarse ) ] ;
  (* The emphasis filters: librosa's start, from the first two samples, on the
     noise, whose first samples are not zero; an explicit state on the
     signal. *)
  List.iter
    (fun (s, coef, zi, y, src, chunkings) ->
      let pre = Filter.preemphasis ~coef ?zi y in
      let de = Filter.deemphasis ~coef ?zi y in
      r (Printf.sprintf "filter.preemphasis %s excerpt" s) [real pre] ;
      r (Printf.sprintf "filter.deemphasis %s excerpt" s) [real de] ;
      let pre', pre_zf = Filter.preemphasis_zf ~coef ?zi y in
      let de', de_zf = Filter.deemphasis_zf ~coef ?zi y in
      r
        (Printf.sprintf "filter.preemphasis_zf %s excerpt" s)
        [real pre'; real pre_zf] ;
      r
        (Printf.sprintf "filter.deemphasis_zf %s excerpt" s)
        [real de'; real de_zf] ;
      law
        (name (Printf.sprintf "filter.preemphasis_stage %s excerpt" s))
        ~image:real ~expected:pre
        (stream ~source:src (Filter.preemphasis_stage ~coef ?zi ()))
        y chunkings ;
      law
        (name (Printf.sprintf "filter.deemphasis_stage %s excerpt" s))
        ~image:real ~expected:de
        (stream ~source:src (Filter.deemphasis_stage ~coef ?zi ()))
        y chunkings ;
      Option.iter
        (fun zi ->
          let one = Nx.create Nx.float64 [|1|] [|1.|] in
          let first = Nx.create Nx.float64 [|2|] [|1.; -.coef|] in
          identity
            (name (Printf.sprintf "filter.preemphasis = filter.lfilter %s" s))
            ~image:real pre
            (Filter.lfilter ~zi first one y) ;
          identity
            (name (Printf.sprintf "filter.deemphasis = filter.lfilter %s" s))
            ~image:real de
            (Filter.lfilter ~zi one first y) )
        zi )
    [ ("coef-0.97 librosa-start", 0.97, None, short_noise, source1, audio)
    ; ( "coef-0.95 zi-0.1"
      , 0.95
      , Some (emphasis_state 0.1)
      , short
      , source
      , coarse ) ] ;
  (* Fundamental frequency, on the excerpt. *)
  List.iter
    (fun ( s
         , (frame_length, hop, trough_threshold, alignment, pad, fmin, fmax)
         , chunkings ) ->
      let f0 =
        yin ~frame_length ~hop ~trough_threshold ~alignment ~pad ~fmin ~fmax
          ~sample_rate short
      in
      r (Printf.sprintf "yin %s excerpt" s) [real f0] ;
      law
        (name (Printf.sprintf "yin_stage %s excerpt" s))
        ~image:real ~expected:f0
        (stream ~source
           (yin_stage ~frame_length ~hop ~trough_threshold ~alignment ~pad ~fmin
              ~fmax ~sample_rate () ) )
        short chunkings )
    [ ( "2048-512 t0.1 centered constant-0 60-1000"
      , (2048, 512, 0.1, `Centered, `Constant 0., 60., 1000.)
      , audio )
    ; ( "1023-300 t0.2 left edge 80-2000"
      , (1023, 300, 0.2, `Left, `Edge, 80., 2000.)
      , coarse )
    ; ( "1500-256 t0.15 centered reflect 100-800"
      , (1500, 256, 0.15, `Centered, `Reflect, 100., 800.)
      , coarse ) ] ;
  let pyin_row s (t : a pyin) =
    r
      (Printf.sprintf "pyin %s excerpt" s)
      [real t.f0; bools t.voiced; real t.voiced_probability]
  in
  pyin_row "2048-512 60-1000 r0.1 nan pruned"
    (pyin ~frame_length:2048 ~hop:512 ~n_thresholds:100
       ~beta_parameters:(2., 18.) ~boltzmann_parameter:2. ~resolution:0.1
       ~max_transition_rate:35.92 ~switch_prob:0.01 ~no_trough_prob:0.01
       ~fill:(`Value Float.nan) ~alignment:`Centered ~pad:(`Constant 0.)
       ~transition_min_prob:1e-4 ~fmin:60. ~fmax:1000. ~sample_rate short ) ;
  pyin_row "1024-256 100-400 r0.25 decoded exact"
    (pyin ~frame_length:1024 ~hop:256 ~n_thresholds:40 ~beta_parameters:(1., 9.)
       ~boltzmann_parameter:1.5 ~resolution:0.25 ~max_transition_rate:20.
       ~switch_prob:0.05 ~no_trough_prob:0.02 ~fill:`Decoded ~alignment:`Left
       ~pad:`Edge ~transition_min_prob:0. ~fmin:100. ~fmax:400. ~sample_rate
       short ) ;
  (* Pitch tracking and tuning, on the excerpt. *)
  List.iter
    (fun (s, fmin, fmax, threshold, reference) ->
      let tracked =
        piptrack ~fmin ~fmax ~threshold ~reference ~fft_size:1024 ~sample_rate
          short_m1
      in
      r
        (Printf.sprintf "piptrack hann %s excerpt" s)
        [real (fst tracked); real (snd tracked)] ;
      pair_law
        (Printf.sprintf "piptrack_stage hann %s excerpt" s)
        ~image:real ~expected:tracked
        (piptrack_stage ~fmin ~fmax ~threshold ~reference ~fft_size:1024
           ~sample_rate () )
        short_m1 )
    [ ("f150-4000 t0.1 maximum", 150., 4000., 0.1, `Maximum)
    ; ("f0-11025 t0.1 value-0.5", 0., 11025., 0.1, `Value 0.5) ] ;
  let pitches, _ =
    piptrack ~fmin:150. ~fmax:4000. ~threshold:0.1 ~reference:`Maximum
      ~fft_size:1024 ~sample_rate short_m1
  in
  r "pitch_tuning piptrack r0.01 b12 excerpt"
    [floats [pitch_tuning ~resolution:0.01 ~bins_per_octave:12 pitches]] ;
  r "pitch_tuning piptrack r0.03 b36 excerpt"
    [floats [pitch_tuning ~resolution:0.03 ~bins_per_octave:36 pitches]] ;
  let tuning =
    estimate_tuning stft_hann ~resolution:0.01 ~bins_per_octave:12 ~fmin:150.
      ~fmax:4000. ~threshold:0.1 ~reference:`Maximum ~sample_rate short
  in
  r "estimate_tuning hann r0.01 b12 t0.1 maximum excerpt" [floats [tuning]] ;
  r "estimate_tuning odd r0.001 b36 t0.2 value-1e-4 excerpt"
    [ floats
        [ estimate_tuning stft_odd ~resolution:0.001 ~bins_per_octave:36
            ~fmin:50. ~fmax:2000. ~threshold:0.2 ~reference:(`Value 1e-4)
            ~sample_rate short ] ] ;
  r "estimate_tuning_of_spectrogram hann r0.01 b12 excerpt"
    [ floats
        [ estimate_tuning_of_spectrogram ~resolution:0.01 ~bins_per_octave:12
            ~fmin:150. ~fmax:4000. ~threshold:0.1 ~reference:`Maximum
            ~fft_size:1024 ~sample_rate short_m1 ] ] ;
  identity
    (name "estimate_tuning = estimate_tuning_of_spectrogram . power_spectrum")
    ~image:(fun t -> floats [t])
    tuning
    (estimate_tuning_of_spectrogram ~resolution:0.01 ~bins_per_octave:12
       ~fmin:150. ~fmax:4000. ~threshold:0.1 ~reference:`Maximum ~fft_size:1024
       ~sample_rate
       (Stft.power_spectrum ~power:1. stft_hann (Nx.cast Nx.float64 short)) ) ;
  (* Peak picking, on the envelope of the noise channel, whose many small peaks
     the selections choose among differently; onset detection, on the envelope
     of the first channel, whose clicks it finds. *)
  let envelopes = onset_strength stft_hann mel_slaney ~lag:1 x in
  let envelope = Nx.slice [Nx.I 0; Nx.A] envelopes in
  let noisy = Nx.slice [Nx.I 1; Nx.A] envelopes in
  let energy = rms ~frame_length:1024 ~hop:256 mono in
  List.iter
    (fun (s, selection, (pre_max, post_max, pre_avg, post_avg, delta, wait)) ->
      let peaks =
        peak_pick ~selection ~pre_max ~post_max ~pre_avg ~post_avg ~delta ~wait
          noisy
      in
      r ("peak_pick noise " ^ s) [indices peaks] ;
      if selection = `Greedy then
        law_by
          (name ("peak_pick_stage noise " ^ s))
          ~join:(concatenated indices)
          ~expected:[indices peaks]
          (stream ~source:source1
             (peak_pick_stage ~pre_max ~post_max ~pre_avg ~post_avg ~delta ~wait) )
          noisy frames )
    [ ("greedy 3-1-10-1 d0.1 w3", `Greedy, (3, 1, 10, 1, 0.1, 3))
    ; ("greedy 2-3-4-6 d0.05 w4", `Greedy, (2, 3, 4, 6, 0.05, 4))
    ; ("dp_count 1-1-3-3 d0 w2", `Dp_count, (1, 1, 3, 3, 0., 2))
    ; ("dp_value 1-1-3-3 d0 w2", `Dp_value, (1, 1, 3, 3, 0., 2)) ] ;
  let onsets =
    onset_detect stft_hann mel_slaney ~normalize:true ~selection:`Greedy
      ~pre_max:1 ~post_max:1 ~pre_avg:4 ~post_avg:5 ~delta:0.02 ~wait:1 mono
  in
  r "onset_detect hann slaney norm greedy 1-1-4-5 d0.02 w1" [indices onsets] ;
  identity
    (name "onset_detect = onset_detect_of_envelope . onset_strength")
    ~image:indices onsets
    (onset_detect_of_envelope ~hop:256 ~normalize:true ~selection:`Greedy
       ~pre_max:1 ~post_max:1 ~pre_avg:4 ~post_avg:5 ~delta:0.02 ~wait:1
       ~sample_rate
       (onset_strength stft_hann mel_slaney ~lag:1 mono) ) ;
  r "onset_detect hann slaney norm rms dp_count 3-2-10-11 w3"
    [ indices
        (onset_detect stft_hann mel_slaney ~normalize:true
           ~backtrack:(`Energy energy) ~selection:`Dp_count ~pre_max:3
           ~post_max:2 ~pre_avg:10 ~post_avg:11 ~delta:0.03 ~wait:3 mono ) ] ;
  r "onset_detect_of_envelope raw envelope dp_value 2-3-4-6"
    [ indices
        (onset_detect_of_envelope ~hop:256 ~normalize:false ~backtrack:`Envelope
           ~selection:`Dp_value ~pre_max:2 ~post_max:3 ~pre_avg:4 ~post_avg:6
           ~delta:0.25 ~wait:2 ~sample_rate envelope ) ] ;
  let raw_greedy =
    onset_detect_of_envelope ~hop:256 ~normalize:false ~selection:`Greedy
      ~pre_max:3 ~post_max:1 ~pre_avg:10 ~post_avg:1 ~delta:0.5 ~wait:3
      ~sample_rate envelope
  in
  identity
    (name "onset_detect_of_envelope raw greedy = peak_pick")
    ~image:indices raw_greedy
    (peak_pick ~selection:`Greedy ~pre_max:3 ~post_max:1 ~pre_avg:10 ~post_avg:1
       ~delta:0.5 ~wait:3 envelope ) ;
  r "onset_backtrack peak_pick greedy 3-1-10-1 d0.5 w3 rms"
    [indices (onset_backtrack raw_greedy energy)] ;
  (* Tempo and beat. *)
  r "tempo_frequencies hop-256 n64"
    [real (tempo_frequencies dt ~hop:256 ~sample_rate 64)] ;
  r "fourier_tempo_frequencies hop-256 win64"
    [real (fourier_tempo_frequencies dt ~hop:256 ~sample_rate 64)] ;
  let tempograms =
    [ ("centered hann w64", 64, `Centered, Window.Hann, ("inf", `Inf))
    ; ("left hamming w33", 33, `Left, Window.Hamming, ("l2", `P 2.)) ]
  in
  List.iter
    (fun (s, win_length, alignment, window, (norm_name, norm)) ->
      let s = s ^ " " ^ norm_name in
      let t = tempogram ~win_length ~alignment ~window ~norm envelopes in
      r ("tempogram " ^ s) [real t] ;
      frame_law ("tempogram_stage " ^ s) t
        (tempogram_stage ~win_length ~alignment ~window ~norm ())
        envelopes )
    tempograms ;
  List.iter
    (fun (s, win_length, alignment, window, _) ->
      let t = fourier_tempogram cdt ~win_length ~alignment ~window envelopes in
      rc ("fourier_tempogram " ^ s) [complex t] ;
      law
        (name ("fourier_tempogram_stage " ^ s))
        ~image:complex ~expected:t
        (stream ~source
           (fourier_tempogram_stage cdt ~win_length ~alignment ~window ()) )
        envelopes frames )
    tempograms ;
  r "tempo hop-256 120-1 ac2 max320 mean"
    [ real
        (tempo ~hop:256 ~start_bpm:120. ~std_bpm:1. ~ac_size:2. ~tempo_max:320.
           ~aggregate:`Mean ~sample_rate envelopes ) ] ;
  r "tempo hop-256 100-0.5 ac1.5 max200 median"
    [ real
        (tempo ~hop:256 ~start_bpm:100. ~std_bpm:0.5 ~ac_size:1.5
           ~tempo_max:200. ~aggregate:`Median ~sample_rate envelopes ) ] ;
  r "tempo hop-256 ac1 max-inf none uniform-prior"
    [ real
        (tempo ~hop:256 ~start_bpm:120. ~std_bpm:1. ~ac_size:1.
           ~tempo_max:Float.infinity ~aggregate:`None
           ~prior:(fun _ -> 0.)
           ~sample_rate envelopes ) ] ;
  let bpm, beats =
    beat_track ~hop:256 ~start_bpm:120. ~tightness:100. ~trim:true ~sample_rate
      envelope
  in
  r "beat_track hop-256 120 t100 trim" [floats [bpm]; indices beats] ;
  let bpm, beats =
    beat_track ~hop:256 ~start_bpm:120. ~tightness:400. ~trim:false ~bpm:150.
      ~sample_rate envelope
  in
  r "beat_track hop-256 bpm-150 t400 no-trim" [floats [bpm]; indices beats] ;
  r "plp hop-256 w64 30-300 reference"
    [ real
        (plp ~hop:256 ~win_length:64 ~tempo_min:30. ~tempo_max:300.
           ~kernels:`Reference ~sample_rate envelopes ) ] ;
  r "plp hop-256 w96 60-240 unit"
    [ real
        (plp ~hop:256 ~win_length:96 ~tempo_min:60. ~tempo_max:240.
           ~kernels:`Unit ~sample_rate envelopes ) ] ;
  (* Chroma energy normalised statistics and the tonal centroid. *)
  let cqt_m1 = Cqt.power_spectrum ~power:1. cqt x in
  let unnormalised_chroma = Chroma.of_cqt ~n_chroma:12 ~norm:`None cqt cqt_m1 in
  let cens =
    chroma_cens cqt ~n_chroma:12 ~norm:(`P 2.) ~smoothing:(Some 41)
      ~smoothing_window:Window.Hann x
  in
  r "chroma_cens cqt c12 l2 s41 hann" [real cens] ;
  identity
    (name "chroma_cens = chroma_cens_of_chroma . chroma_cqt")
    ~image:real cens
    (chroma_cens_of_chroma ~norm:(`P 2.) ~smoothing:(Some 41)
       ~smoothing_window:Window.Hann unnormalised_chroma ) ;
  let stft_chroma_a24 = Chroma.apply ~norm:`None chroma_a24 p2 in
  List.iter
    (fun (s, chroma, norm, smoothing, smoothing_window) ->
      let c = chroma_cens_of_chroma ~norm ~smoothing ~smoothing_window chroma in
      r ("chroma_cens_of_chroma " ^ s) [real c] ;
      frame_law ("chroma_cens_stage " ^ s) c
        (chroma_cens_stage ~norm ~smoothing ~smoothing_window ())
        chroma )
    [ ( "cqt c12 inf s10 hamming"
      , unnormalised_chroma
      , `Inf
      , Some 10
      , Window.Hamming )
    ; ("cqt c12 l1 none", unnormalised_chroma, `P 1., None, Window.Hann)
    ; ("stft a24 l2 s41 hann", stft_chroma_a24, `P 2., Some 41, Window.Hann) ] ;
  let centroid = tonnetz cqt ~n_chroma:12 x in
  r "tonnetz cqt c12" [real centroid] ;
  identity
    (name "tonnetz = tonnetz_of_chroma . chroma_cqt")
    ~image:real centroid
    (tonnetz_of_chroma (Chroma.of_cqt ~n_chroma:12 ~norm:`Inf cqt cqt_m1)) ;
  List.iter
    (fun (s, chroma) ->
      let t = tonnetz_of_chroma chroma in
      r ("tonnetz_of_chroma " ^ s) [real t] ;
      frame_law ("tonnetz_stage " ^ s) t (tonnetz_stage ()) chroma )
    [ ("stft c12 inf", Chroma.apply ~norm:`Inf chroma_c12 p2)
    ; ("stft a24 none", stft_chroma_a24) ] ;
  (* Autocorrelation, linear prediction and polynomial features. *)
  let full = autocorrelate ~max_size:100000 x in
  let lags = autocorrelate ~max_size:2048 x in
  r "autocorrelate max-100000" [real full] ;
  r "autocorrelate max-2048" [real lags] ;
  identity
    (name "autocorrelate max-2048 = slice of autocorrelate")
    ~image:real
    (Nx.slice [Nx.A; Nx.R (0, 2048)] full)
    lags ;
  r "lpc order-16" [real (lpc ~order:16 x)] ;
  r "lpc order-5 noise" [real (lpc ~order:5 noise)] ;
  let slope = poly_features stft_hann ~order:1 ~sample_rate x in
  let cubic = poly_features stft_odd ~order:3 ~sample_rate x in
  r "poly_features hann o1" [real slope] ;
  r "poly_features odd o3" [real cubic] ;
  let custom =
    Nx.cast dt
      (Nx.create Nx.float64 [|513|]
         (Array.init 513 (fun k -> Float.of_int (k * k) /. 24.)) )
  in
  r "poly_features_of_spectrogram hann o2 custom-freqs"
    [real (poly_features_of_spectrogram ~order:2 ~freqs:custom ~sample_rate m1)] ;
  r "poly_features_of_spectrogram hann o0"
    [real (poly_features_of_spectrogram ~order:0 ~sample_rate m1)] ;
  identity
    (name "poly_features = poly_features_of_spectrogram . power_spectrum")
    ~image:real slope
    (poly_features_of_spectrogram ~order:1 ~sample_rate m1) ;
  frame_law "poly_features_stage hann o1" slope
    (poly_features_stage stft_hann ~order:1 ~sample_rate ())
    m1 ;
  law
    (name "power_stage >> poly_features_stage odd o3")
    ~image:real ~expected:cubic
    (stream ~source
       Pipeline.(
         Stft.power_stage ~power:1. stft_odd
         >> poly_features_stage stft_odd ~order:3 ~sample_rate () ) )
    x coarse ;
  (* Feature inversion, on the excerpt and back to its length. *)
  let mel_short =
    Mel.apply mel_slaney (Stft.power_spectrum ~power:2. stft_hann short)
  in
  let mf = mfcc stft_hann mel_slaney ~n_mfcc:13 ~lifter:22. short in
  let mel_of_mf = mfcc_to_mel mel_slaney ~lifter:22. ~reference:1. mf in
  r "mfcc_to_mel slaney n13 lifter-22 ref-1 excerpt" [real mel_of_mf] ;
  r "mfcc_to_mel htk n20 lifter-0 ref-2 excerpt"
    [ real
        (mfcc_to_mel mel_htk ~lifter:0. ~reference:2.
           (mfcc stft_hann mel_htk ~n_mfcc:20 ~lifter:0. short) ) ] ;
  frame_law "mfcc_to_mel_stage slaney n13 lifter-22 ref-1 excerpt" mel_of_mf
    (mfcc_to_mel_stage mel_slaney ~n_mfcc:13 ~lifter:22. ~reference:1. ())
    mf ;
  let fitted = mel_to_stft mel_slaney ~power:2. mel_short in
  r "mel_to_stft slaney p2 excerpt" [real fitted] ;
  frame_law "mel_to_stft_stage slaney p2 excerpt" fitted
    (mel_to_stft_stage mel_slaney ~power:2. ())
    mel_short ;
  let mel_htk_m1 = Mel.apply mel_htk short_m1 in
  let fitted_htk = mel_to_stft mel_htk ~power:1. mel_htk_m1 in
  r "mel_to_stft htk p1 excerpt" [real fitted_htk] ;
  frame_law "mel_to_stft_stage htk p1 excerpt" fitted_htk
    (mel_to_stft_stage mel_htk ~power:1. ())
    mel_htk_m1 ;
  let audio_of_mel =
    mel_to_audio stft_hann mel_slaney ~power:2. ~n_iter:4 ~length:excerpt_length
      mel_short
  in
  r "mel_to_audio hann slaney p2 i4 excerpt" [real audio_of_mel] ;
  identity
    (name "mel_to_audio = stft.griffin_lim . mel_to_stft")
    ~image:real audio_of_mel
    (Stft.griffin_lim ~n_iter:4 ~momentum:0.99 ~init:`Zero_phase
       ~length:excerpt_length stft_hann fitted ) ;
  let audio_of_mfcc =
    mfcc_to_audio stft_hann mel_slaney ~lifter:22. ~reference:1. ~power:2.
      ~n_iter:4 ~length:excerpt_length mf
  in
  r "mfcc_to_audio hann slaney lifter-22 ref-1 p2 i4 excerpt"
    [real audio_of_mfcc] ;
  identity
    (name "mfcc_to_audio = mel_to_audio . mfcc_to_mel")
    ~image:real audio_of_mfcc
    (mel_to_audio stft_hann mel_slaney ~power:2. ~n_iter:4
       ~length:excerpt_length mel_of_mf )

let () =
  grids () ;
  features ("f32", "c64") Nx.float32 Nx.complex64
    (Nx.cast Nx.float32 signal)
    (Nx.cast Nx.float32 noise) ;
  features ("f64", "c128") Nx.float64 Nx.complex128 signal noise ;
  grids_since () ;
  features_since ("f32", "c64") Nx.float32 Nx.complex64
    (Nx.cast Nx.float32 signal)
    (Nx.cast Nx.float32 noise) ;
  features_since ("f64", "c128") Nx.float64 Nx.complex128 signal noise ;
  check_anchors () ;
  match List.rev !failures with
  | [] ->
      ()
  | failed ->
      List.iter (Printf.eprintf "law failed: %s\n") failed ;
      Printf.eprintf "%d of %d law checks failed\n" (List.length failed) !checks ;
      exit 1
