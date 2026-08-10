(** Unit and scale conversions.

    Decibel rescaling of power and amplitude spectra, the mel and MIDI
    frequency scales, and the frame grid. Every function takes its tensor
    last, preserves the element dtype, and returns a fresh tensor. Numerical
    semantics follow librosa 0.11; every deviation is documented on the
    function.

    The decibel conversions are the flat, whole-tensor face of the {!Db}
    pipeline stages; the two agree by construction. *)

(** {1 Decibels} *)

val power_to_db :
     ?reference:float
  -> ?amin:float
  -> ?top_db:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [power_to_db ?reference ?amin ?top_db s] is the power spectrum [s] in
    decibels: [10 * log10 (max amin s) - 10 * log10 (max amin reference)],
    elementwise.

    [reference] defaults to [1.]. [amin] defaults to [1e-10] and floors both
    the input values and the reference before the logarithms, so the result
    is always finite; a power that is not positive — which a power spectrum
    does not contain — sits at the floor. When [top_db] is given, the result is clamped below to
    [m - top_db], where [m] is the maximum decibel value over the {e whole}
    tensor — a whole-tensor reduction, not an elementwise operation: over a
    spectrogram the threshold is global across channels, bins and frames
    alike. Nothing is clamped by default — a deviation from the reference
    implementation, which clamps at [top_db = 80.] unless told otherwise;
    this function applies exactly what
    the caller asks for. For the streaming consequences of that reduction,
    see {!Db.clamped_stage}.

    Raises [Invalid_argument] if [reference] or [amin] is not finite and
    positive, or if [top_db] is not finite and non-negative. *)

val db_to_power : ?reference:float -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [db_to_power ?reference db] is [reference * 10 ** (db / 10)], elementwise
    — the inverse of {!power_to_db} wherever [amin] did not floor and
    [top_db] did not clamp. [reference] defaults to [1.].

    Raises [Invalid_argument] if [reference] is not finite and positive. *)

val amplitude_to_db :
     ?reference:float
  -> ?amin:float
  -> ?top_db:float
  -> (float, 'a) Nx.t
  -> (float, 'a) Nx.t
(** [amplitude_to_db ?reference ?amin ?top_db s] is the amplitude spectrum
    [s] in decibels: {!power_to_db} of [s * s] with the reference and the
    floor squared — [20 * log10 (max amin (abs s)) - 20 * log10 (max amin
    reference)], elementwise, then clamped exactly as {!power_to_db} clamps
    when [top_db] is given.

    [reference] defaults to [1.]. [amin] defaults to [1e-5] — the [1e-10]
    power floor, in the amplitude domain.

    Raises [Invalid_argument] if [reference] or [amin] is not finite and
    positive, or if [top_db] is not finite and non-negative. *)

val db_to_amplitude : ?reference:float -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [db_to_amplitude ?reference db] is [reference * 10 ** (db / 20)],
    elementwise — the inverse of {!amplitude_to_db} wherever [amin] did not
    floor and [top_db] did not clamp. [reference] defaults to [1.].

    Raises [Invalid_argument] if [reference] is not finite and positive. *)

(** {1 Frequency scales} *)

val hz_to_mel : ?scale:[`Slaney | `Htk] -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [hz_to_mel ?scale f] is the frequencies [f], in hertz, on the mel scale,
    elementwise. [scale] defaults to [`Slaney]: linear below
    1000 Hz at 3/200 mel per hertz, logarithmic above, continuous at the
    break. [`Htk] is [2595 * log10 (1 + f / 700)]. *)

val mel_to_hz : ?scale:[`Slaney | `Htk] -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [mel_to_hz ?scale m] is the mel values [m] in hertz, elementwise — the
    inverse of {!hz_to_mel}. [scale] defaults to [`Slaney] and must match the
    scale the mels were built with. *)

val hz_to_midi : (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [hz_to_midi f] is the fractional MIDI note number of each frequency in
    [f], elementwise: [69 + 12 * log2 (f / 440)]. Zero maps to negative
    infinity. *)

val midi_to_hz : (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [midi_to_hz m] is the frequency, in hertz, of each MIDI note number in
    [m], elementwise: [440 * 2 ** ((m - 69) / 12)] — the inverse of
    {!hz_to_midi}. *)

(** {1 The frame grid} *)

val frames_to_time :
  sample_rate:int -> hop:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [frames_to_time ~sample_rate ~hop frames] is the time, in seconds, of the
    first sample of each frame index in [frames]:
    [frames * hop / sample_rate], elementwise.

    Raises [Invalid_argument] if [sample_rate < 1] or [hop < 1]. *)

val time_to_frames :
  sample_rate:int -> hop:int -> (float, 'a) Nx.t -> (float, 'a) Nx.t
(** [time_to_frames ~sample_rate ~hop times] is the index of the frame each
    time in [times] falls in: [floor (times * sample_rate / hop)],
    elementwise.

    It inverts {!frames_to_time} exactly on grids whose [hop / sample_rate]
    ratio is representable in the element dtype — [hop] dividing a
    power-of-two [sample_rate], say. Elsewhere the floor lands one frame
    early for the indices whose time rounded down, matching the reference
    implementation on the same grid.

    Raises [Invalid_argument] if [sample_rate < 1] or [hop < 1]. *)
