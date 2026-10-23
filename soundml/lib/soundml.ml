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
module Filter = Filter

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
  match
    if Float.equal power 2. then Mel.power_spectrogram stft_config mel_config x
    else None
  with
  | Some mel ->
      mel
  | None ->
      Mel.apply mel_config (Stft.power_spectrum ~power stft_config x)

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

(* [cepstral_matrix ~n_mels ~n_mfcc lifter] is the first [n_mfcc] rows of the
   orthonormal type-II DCT over [n_mels] points, liftered, prepared for
   {!Ordered.project}: row [k], column [j] is [2 cos (pi k (2j + 1) / (2
   n_mels))] times the row's orthonormal scale ([1 / sqrt (4 n_mels)] on row
   zero, [1 / sqrt (2 n_mels)] elsewhere) and, with a positive [lifter], its
   sinusoidal lifter weight [1 + (lifter / 2) sin (pi (k + 1) / lifter)]. The
   angle's numerator is reduced modulo a full turn, [4 n_mels], first. *)
let cepstral_matrix ~n_mels ~n_mfcc lifter =
  let turn = 4 * n_mels in
  let w = Array.make (n_mfcc * n_mels) 0. in
  for k = 0 to n_mfcc - 1 do
    let scale =
      if k = 0 then 1. /. Float.sqrt (4. *. Float.of_int n_mels)
      else 1. /. Float.sqrt (2. *. Float.of_int n_mels)
    in
    let lift =
      match lifter with
      | Some l when l > 0. ->
          1. +. (l /. 2. *. Float.sin (Float.pi *. Float.of_int (k + 1) /. l))
      | Some _ | None ->
          1.
    in
    for j = 0 to n_mels - 1 do
      let a = k * ((2 * j) + 1) mod turn in
      w.((k * n_mels) + j) <-
        2.
        *. Float.cos (Float.pi *. Float.of_int a /. Float.of_int (2 * n_mels))
        *. scale *. lift
    done
  done ;
  Ordered.matrix (Nx.create Nx.float64 [|n_mfcc; n_mels|] w)

(* [log_mel ?range mel] is [Convert.power_to_db ?top_db:range] of the mel
   spectrogram [mel] widened to float64 — reference [1.], floor [1e-10] — read
   in place when [mel] is float32 or float64. *)
let log_mel ?range mel =
  match
    Loops.decibels64
      ~scale:(10. /. Float.log 10.)
      ~offset:0. ~amin:1e-10 ~magnitude:false ~range mel
  with
  | Some db ->
      db
  | None ->
      Convert.power_to_db ?top_db:range (Nx.cast Nx.float64 mel)

(* [cepstrum matrix db] is the double-precision log-mel [db] through the
   cepstral matrix along its mel axis. Each frame's coefficients are folded from
   its own mel column alone, in a fixed order ({!Ordered.project}), so a frame's
   cepstrum does not depend on how many frames share the call — which
   {!mfcc_stage} relies on, and its law tests pin. *)
let cepstrum matrix db = Ordered.project matrix db

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
    let db = log_mel ~range:80. mel in
    Nx.cast dtype (cepstrum (cepstral_matrix ~n_mels ~n_mfcc lifter) db)

(* The stage is [mfcc]'s interior on each mel chunk, without the clamp: the 80
   dB floor sits under the whole signal's maximum, which no chunk can see. *)
let mfcc_stage mel_config ?(n_mfcc = 20) ?lifter () =
  let n_mels = Mel.Config.n_mels mel_config in
  check_mfcc "mfcc_stage" ~n_mels ~n_mfcc lifter ;
  let matrix = cepstral_matrix ~n_mels ~n_mfcc lifter in
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
      else Nx.cast dtype (cepstrum matrix (log_mel mel)) )

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
  match
    if Float.equal power 2. then
      Chroma.power_chromagram ?norm stft_config chroma_config x
    else None
  with
  | Some chroma ->
      chroma
  | None ->
      Chroma.apply ?norm chroma_config
        (Stft.power_spectrum ~power stft_config x)

let chroma_cqt cqt_config ?n_chroma ?norm x =
  Chroma.of_cqt ?n_chroma ?norm cqt_config
    (Cqt.power_spectrum ~power:1. cqt_config x)

let resample ?quality ~sample_rate ~target x =
  Resample.(apply (Config.cached ?quality ~sample_rate ~target ()) x)

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

(* Delta features: flat delegations to the private [Delta] module. *)

type delta_mode = Delta.mode

let delta = Delta.delta

let delta_stage = Delta.stage

(* Per-channel energy normalization: flat delegations to the private [Pcen]
   module. *)

let pcen = Pcen.pcen

let pcen_stage = Pcen.stage

(* Pitch tracking and tuning estimation: flat delegations to the private
   [Tuning] module. *)

type piptrack_reference = Tuning.reference

let piptrack = Tuning.piptrack

let piptrack_stage = Tuning.stage

let pitch_tuning = Tuning.pitch_tuning

let estimate_tuning = Tuning.estimate_tuning

let estimate_tuning_of_spectrogram = Tuning.estimate_tuning_of_spectrogram

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

let hpss_of_spectrogram_stage = Hpss.hpss_of_spectrogram_stage

let hpss_masks_stage = Hpss.hpss_masks_stage

let hpss_of_stft_stage = Hpss.hpss_of_stft_stage

let hpss = Hpss.hpss

let harmonic = Hpss.harmonic

let percussive = Hpss.percussive

(* Synthetic signals and mu-law companding: flat delegations to the private
   [Synthesis] module. *)

type click_positions = Synthesis.positions

let tone = Synthesis.tone

let chirp = Synthesis.chirp

let clicks = Synthesis.clicks

let mu_compress = Synthesis.mu_compress

let mu_compress_stage = Synthesis.mu_compress_stage

let mu_expand = Synthesis.mu_expand

let mu_expand_stage = Synthesis.mu_expand_stage
(* Fundamental frequency: flat delegations to the private [F0] module. *)

type 'a pyin = 'a F0.pyin =
  { f0: (float, 'a) Nx.t
  ; voiced: (bool, Nx.bool_elt) Nx.t
  ; voiced_probability: (float, 'a) Nx.t }

let yin = F0.yin

let yin_stage = F0.stage

let pyin = F0.pyin

let version = "0.1.0-dev"
