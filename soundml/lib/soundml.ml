module Pipeline = Pipeline
module Window = Window
module Stft = Stft
module Convert = Convert
module Db = Db
module Mel = Mel
module Cqt = Cqt
module Chroma = Chroma
module Resample = Resample
module Effects = Effects

let check_fft_sizes fn stft_config mel_config =
  let stft_size = Stft.Config.fft_size stft_config in
  let mel_size = Mel.Config.fft_size mel_config in
  if stft_size <> mel_size then
    invalid_arg
      (Printf.sprintf
         "%s: cannot project a %d-point STFT through a filterbank built for an \
          FFT of size %d (the two configurations must agree on fft_size)"
         fn stft_size mel_size )

let mel_spectrogram stft_config mel_config ?(power = 2.) x =
  check_fft_sizes "mel_spectrogram" stft_config mel_config ;
  Mel.apply mel_config (Stft.power_spectrum ~power stft_config x)

(* Orthonormal type-II DCT row scales: the raw transform times [1 / sqrt (4 n)]
   on row zero and [1 / sqrt (2 n)] elsewhere. *)
let dct_ortho_scales n_mels n_mfcc =
  Nx.create Nx.float64 [|n_mfcc; 1|]
    (Array.init n_mfcc (fun k ->
         if k = 0 then 1. /. Float.sqrt (4. *. Float.of_int n_mels)
         else 1. /. Float.sqrt (2. *. Float.of_int n_mels) ) )

(* Sinusoidal liftering weights: coefficient [k], from zero, scales by [1 +
   (lifter / 2) * sin (pi * (k + 1) / lifter)]. *)
let lifter_weights lifter n_mfcc =
  Nx.create Nx.float64 [|n_mfcc; 1|]
    (Array.init n_mfcc (fun k ->
         1.
         +. lifter /. 2.
            *. Float.sin (Float.pi *. Float.of_int (k + 1) /. lifter) ) )

let shrink_mel_axis n_mfcc t =
  let nd = Nx.ndim t in
  Nx.slice
    (List.init nd (fun i -> if i = nd - 2 then Nx.R (0, n_mfcc) else Nx.A))
    t

let check_mfcc op ~n_mels ~n_mfcc lifter =
  if n_mfcc < 1 || n_mfcc > n_mels then
    invalid_arg
      (Printf.sprintf
         "%s: cannot keep %d cepstral coefficients of %d mel bands (n_mfcc \
          must lie in [1, n_mels])"
         op n_mfcc n_mels ) ;
  Option.iter
    (fun l ->
      if not (Float.is_finite l && l >= 0.) then
        invalid_arg
          (Printf.sprintf
             "%s: cannot lifter with a coefficient of %g (lifter must be \
              finite and non-negative)"
             op l ) )
    lifter

(* [cepstrum ~n_mels ~n_mfcc lifter db] is the double-precision log-mel [db]
   through the raw type-II DCT along its mel axis, the first [n_mfcc] rows kept,
   the orthonormal row scaling and the optional liftering. The DCT transforms
   each frame's mel column on its own, so a frame's cepstrum does not depend on
   how many frames share the call — which {!mfcc_stage} relies on, and its law
   tests pin. *)
let cepstrum ~n_mels ~n_mfcc lifter db =
  let cepstrum =
    Nx.mul
      (shrink_mel_axis n_mfcc (Nx.dct ~type_:2 ~axis:(-2) db))
      (dct_ortho_scales n_mels n_mfcc)
  in
  match lifter with
  | Some l when l > 0. ->
      Nx.mul cepstrum (lifter_weights l n_mfcc)
  | Some _ | None ->
      cepstrum

(* [empty_cepstrum dtype ~n_mfcc mel] is the broadcast-consistent result for a
   mel spectrogram with no frames or no signals: nothing to transform. *)
let empty_cepstrum dtype ~n_mfcc mel =
  let out = Array.copy (Nx.shape mel) in
  out.(Array.length out - 2) <- n_mfcc ;
  Nx.zeros dtype out

let mfcc stft_config mel_config ?(n_mfcc = 20) ?lifter x =
  check_fft_sizes "mfcc" stft_config mel_config ;
  let n_mels = Mel.Config.n_mels mel_config in
  check_mfcc "mfcc" ~n_mels ~n_mfcc lifter ;
  let mel = mel_spectrogram stft_config mel_config x in
  let dtype = Nx.dtype x in
  if Nx.numel mel = 0 then empty_cepstrum dtype ~n_mfcc mel
  else
    (* The interior runs in double: the log-mel spectrogram ([power_to_db] with
       the 80 dB dynamic-range clamp the parity contract fixes), the cepstrum,
       then one rounding at the boundary. *)
    let db = Convert.power_to_db ~top_db:80. (Nx.cast Nx.float64 mel) in
    Nx.cast dtype (cepstrum ~n_mels ~n_mfcc lifter db)

(* The stage is [mfcc]'s interior on each mel chunk, without the clamp: the 80
   dB floor sits under the whole signal's maximum, which no chunk can see. *)
let mfcc_stage mel_config ?(n_mfcc = 20) ?lifter () =
  let n_mels = Mel.Config.n_mels mel_config in
  check_mfcc "mfcc_stage" ~n_mels ~n_mfcc lifter ;
  Pipeline.stateless (fun mel ->
      let nd = Nx.ndim mel in
      if nd < 2 then
        invalid_arg
          (Printf.sprintf
             "mfcc_stage: cannot transform a rank-%d chunk (the stage needs \
              [...; n_mels; frames])"
             nd ) ;
      let bands = Nx.dim (nd - 2) mel in
      if bands <> n_mels then
        invalid_arg
          (Printf.sprintf
             "mfcc_stage: cannot transform %d mel bands with a stage built for \
              %d"
             bands n_mels ) ;
      let dtype = Nx.dtype mel in
      if Nx.numel mel = 0 then empty_cepstrum dtype ~n_mfcc mel
      else
        let db = Convert.power_to_db (Nx.cast Nx.float64 mel) in
        Nx.cast dtype (cepstrum ~n_mels ~n_mfcc lifter db) )

let chroma_stft stft_config chroma_config ?(power = 2.) ?norm x =
  let stft_size = Stft.Config.fft_size stft_config in
  let chroma_size = Chroma.Config.fft_size chroma_config in
  if stft_size <> chroma_size then
    invalid_arg
      (Printf.sprintf
         "chroma_stft: cannot project a %d-point STFT through a filterbank \
          built for an FFT of size %d (the two configurations must agree on \
          fft_size)"
         stft_size chroma_size ) ;
  Chroma.apply ?norm chroma_config (Stft.power_spectrum ~power stft_config x)

let chroma_cqt cqt_config ?n_chroma ?norm x =
  Chroma.of_cqt ?n_chroma ?norm cqt_config
    (Cqt.power_spectrum ~power:1. cqt_config x)

let resample ?quality ~sample_rate ~target x =
  Resample.(apply (Config.create ?quality ~sample_rate ~target ()) x)

(* Spectral-shape features: flat delegations to the private [Spectral]
   module. *)

let spectral_centroid = Spectral.centroid

let spectral_centroid_stage = Spectral.centroid_stage

let spectral_bandwidth = Spectral.bandwidth

let spectral_bandwidth_stage = Spectral.bandwidth_stage

let spectral_rolloff = Spectral.rolloff

let spectral_rolloff_stage = Spectral.rolloff_stage

let spectral_flatness = Spectral.flatness

let spectral_flatness_stage = Spectral.flatness_stage

(* Energy features: flat delegations to the private [Energy] module. *)

let rms = Energy.rms

let rms_of_spectrogram = Energy.rms_of_spectrogram

let rms_stage = Energy.rms_stage

let zero_crossing_rate = Energy.zero_crossing_rate

let zero_crossing_rate_stage = Energy.zero_crossing_rate_stage

(* Spectral contrast and onset strength: flat delegations to the private
   [Contrast] and [Onset] modules. *)

let spectral_contrast = Contrast.spectral_contrast

let spectral_contrast_of_spectrogram = Contrast.spectral_contrast_of_spectrogram

let spectral_contrast_stage = Contrast.stage

let onset_strength = Onset.onset_strength

let onset_strength_stage = Onset.stage

(* Harmonic/percussive separation: flat delegations to the private [Hpss]
   module. *)

let hpss_of_spectrogram = Hpss.hpss_of_spectrogram

let hpss_of_stft = Hpss.hpss_of_stft

let hpss_masks = Hpss.hpss_masks

let hpss = Hpss.hpss

let harmonic = Hpss.harmonic

let percussive = Hpss.percussive

let version = "0.1.0-dev"
