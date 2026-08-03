(** Decibel scaling as capability-indexed {!Pipeline} stages.

    The whole-tensor conversions live in {!Convert}; this module is their
    pipeline face. What distinguishes it is that the {e reference} carries the
    capability: a fixed reference is chunk-local and streams, a whole-signal
    maximum reference is unbounded lookahead — so the choice of reference is a
    GADT whose index is the stage's capability, and misuse is a type error at
    the driver, not a runtime surprise. *)

(** The type for decibel references, indexed by the pipeline capability the
    reference needs.

    [Value r] compares against the fixed power [r]: memoryless and
    chunk-local, so a [Value] stage is polymorphic in its capability and
    streams. [Maximum] compares against the maximum of the {e whole} signal,
    an unbounded-lookahead reduction: the constructor itself carries the
    {!Pipeline.offline} index, so a [Maximum] stage only ever types as
    offline and {!Pipeline.Stream.prepare} rejects it at compile time. *)
type 'k reference =
  | Value : float -> 'k reference
  | Maximum : Pipeline.offline reference

val stage :
     ?amin:float
  -> 'k reference
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, 'k) Pipeline.t
(** [stage r] is {!Convert.power_to_db} with reference [r] as a pipeline
    stage, elementwise over each chunk and dtype-preserving. [amin] floors
    magnitudes and reference as there and defaults to [1e-10].

    [stage (Value v)] is stateless: one such value drives both
    {!Pipeline.run} and {!Pipeline.Stream}, and the two agree on every
    partitioning. [stage Maximum] is offline by [Maximum]'s own type;
    offline processing is the one-chunk instance of streaming, so the
    whole-chunk maximum it reads {e is} the whole-signal maximum, and no
    streaming use can typecheck. A signal with no positive value is
    referenced to [amin].

    [top_db] clamping is deliberately absent here. The clamp threshold sits
    [top_db] below the maximum of the {e whole} converted signal — an
    unbounded-lookahead reduction even when the reference is a fixed
    [Value]. A causal stage could only clamp against each chunk's own
    maximum, and the {!Pipeline} law (run equals stream over every
    partitioning) would fail; encoding the clamp as an option here would
    silently turn a streaming-looking stage into a lie. Clamping therefore
    lives in {!clamped_stage}, offline by type whatever the reference.

    Raises [Invalid_argument] if [amin] is not finite and positive, or if a
    [Value] reference is not finite and positive. *)

val clamped_stage :
     ?amin:float
  -> top_db:float
  -> Pipeline.offline reference
  -> ((float, 'a) Nx.t, (float, 'a) Nx.t, Pipeline.offline) Pipeline.t
(** [clamped_stage ~top_db r] is {!stage} with the result clamped below to
    [top_db] under its whole-signal maximum, exactly as
    {!Convert.power_to_db} clamps. Any reference fits — including [Value] —
    but the stage is offline regardless, because the threshold reads the
    whole signal.

    Raises [Invalid_argument] as {!stage} does, and if [top_db] is not
    finite and non-negative. *)
