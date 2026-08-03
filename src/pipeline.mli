(** Streaming and offline chunk pipelines.

    A pipeline is a composition of stages, each a Mealy kernel over chunks:
    [prepare] allocates state from a {!Format.t}, [step] consumes one input
    chunk and produces at most one output chunk, [flush] drains the buffered
    tail, [reset] restores the freshly prepared state. Offline processing is
    the one-chunk instance of streaming: {!run} and {!Stream} drive the same
    kernels, so the two cannot disagree. The governing law, for every
    partitioning of [x] into [chunks] — including one-item chunks, chunks
    larger than any internal window, and inputs shorter than the pipeline's
    total latency — is:

    {[
      run ~source p x
      = concat (List.filter_map (Stream.push s) chunks @ Stream.flush s)
    ]}

    where [s] is [Stream.prepare p ~source ~max_chunk] and [concat] is the
    final stage's chunk monoid. *)

(** The capability of stages that can stream: finite memory and bounded
    lookahead. Bounded lookahead is latency, not offline-ness. *)
type causal

(** The capability of stages that need the whole signal before producing any
    output (peak normalisation, zero-phase filtering, …). *)
type offline

(** The type for pipelines consuming ['a] chunks and producing ['b] chunks.
    ['k] is the capability phantom; causal stages are polymorphic in ['k]. The
    [+] makes ['k] deducible and lets it generalise under the relaxed value
    restriction, so one composed pipeline value serves both {!run} and
    {!Stream.prepare}, while a pipeline containing an {!offline_only} stage is
    typed [offline] and rejected by {!Stream.prepare} at compile time.

    Scope of the static guarantee: the compiler checks capability
    {e propagation}. Assignment happens at stage construction — library stage
    constructors assign it correctly and custom {!kernel}s self-declare. A
    mislabelled custom kernel is a bug at its constructor, not detectable
    downstream.

    Pipelines generalise in ['k]; they do {e not} generalise in a dtype
    parameter — a dtype-polymorphic pipeline value hits the value restriction
    (['_weak]). Build pipelines at a ground element type, or inside a function
    that takes the dtype witness. *)
type ('a, 'b, +'k) t

module Rate : sig
  (** The type for exact rational rates: [num] output items per [den] input
      items, normalised ([den > 0] and [gcd num den = 1]). *)
  type t = {num: int; den: int}

  val ( * ) : t -> t -> t
  (** [a * b] is the product of [a] and [b], normalised. *)

  val identity : t
  (** [identity] is the neutral rate [1/1]. *)

  val equal : t -> t -> bool
  (** [equal a b] is [true] iff [a] and [b] denote the same rational. *)

  val pp : Stdlib.Format.formatter -> t -> unit
  (** [pp ppf r] formats [r] as [num/den], or as [num] alone when [den] is
      [1]. *)
end

module Format : sig
  (** The type for {e dynamic} stream descriptions threaded through stage
      [prepare]s: the element dtype (existentially packed), items per second
      as an exact rational (a post-analysis stream at [44100/512] frames per
      second is representable), the channel count, an upper bound on items per
      chunk, and the {e cumulative upstream latency} accumulated by
      composition as formats thread through the chain — which is what lets a
      stage emit globally positioned, latency-corrected values structurally
      rather than by driver magic.

      Static safety for the data path lives in the pipeline's ['a]/['b];
      [Format.t] is the runtime metadata channel. A stage validates the
      incoming format against what its constructor promised and raises
      [Invalid_argument] at [prepare] time on mismatch — never mid-stream. *)
  type t

  type dtype =
    | Dtype : (float, 'd) Nx.dtype -> dtype
        (** The type for element dtypes, existentially packed. *)

  val audio : (float, 'a) Nx.dtype -> sample_rate:int -> channels:int -> t
  (** [audio dt ~sample_rate ~channels] is the source description for PCM
      chunks — what {!Stream.prepare} and {!run} take. Items per second is
      [sample_rate], the per-chunk bound is unset and the upstream latency is
      zero.

      Raises [Invalid_argument] if [sample_rate < 1] or [channels < 1]. *)

  val dtype : t -> dtype
  (** [dtype f] is [f]'s element dtype, packed. *)

  val items_per_second : t -> Rate.t
  (** [items_per_second f] is the number of items flowing per second, exact. *)

  val channels : t -> int
  (** [channels f] is the number of channels per item. *)

  val max_items : t -> int option
  (** [max_items f] is the upper bound on items per chunk, or [None] when
      chunks are unbounded (the offline case). {!Stream.prepare} sets it to
      [max_chunk] on the source format; stages derive their own output bound
      through [out_format]. *)

  val upstream_latency : t -> Rate.t
  (** [upstream_latency f] is the cumulative involuntary lookahead of every
      stage upstream of the one being prepared, in source-rate samples,
      exact. *)

  val with_dtype : (float, 'd) Nx.dtype -> t -> t
  (** [with_dtype dt f] is [f] with element dtype [dt]. Use it — and the
      other [with_] functions — to build a stage's [out_format] from its
      input format; a format built from scratch is rejected at prepare
      time. *)

  val with_channels : int -> t -> t
  (** [with_channels n f] is [f] with [n] channels.

      Raises [Invalid_argument] if [n < 1]. *)

  val with_items_per_second : Rate.t -> t -> t
  (** [with_items_per_second r f] is [f] flowing at [r] items per second,
      normalised.

      Raises [Invalid_argument] if [r] is not positive. *)

  val with_max_items : int option -> t -> t
  (** [with_max_items bound f] is [f] with per-chunk item bound [bound], or
      unbounded when [bound] is [None].

      Raises [Invalid_argument] if [bound] is negative. *)

  val pp : Stdlib.Format.formatter -> t -> unit
  (** [pp ppf f] formats [f] for inspection. *)
end

(** {1 Constructing stages} *)

val stateless : ('a -> 'b) -> ('a, 'b, 'k) t
(** [stateless f] is the memoryless stage applying [f] to each chunk: latency
    0, rate 1:1, format-preserving. The caller asserts that [f] is memoryless
    and distributes over chunk concatenation; any whole-signal function must
    use {!offline_only} instead. *)

val kernel :
     ?latency:int
  -> ?rate:Rate.t
  -> ?out_format:(Format.t -> Format.t)
  -> ?flush:('s -> 'b list)
  -> ?reset:('s -> unit)
  -> concat:('b list -> 'b)
  -> prepare:(Format.t -> 's)
  -> step:('s -> 'a -> 'b option)
  -> unit
  -> ('a, 'b, 'k) t
(** [kernel ~concat ~prepare ~step ()] is the stage running the Mealy kernel
    [prepare]/[step]/[flush]/[reset].

    [prepare] allocates the stage state from its own input format — the
    source format transformed by every upstream stage's [out_format]. It
    validates the incoming format against what the stage was constructed for
    and raises [Invalid_argument] on mismatch, at prepare time, never
    mid-stream.

    [step] consumes one chunk and emits at most one chunk. Rate-changing
    stages grow the per-chunk {e bound} in [out_format] instead of emitting
    multiple chunks, so downstream allocation is sized correctly at prepare.

    [flush] drains the buffered tail and may emit several chunks; it defaults
    to an empty drain. [reset] restores the freshly prepared state and
    defaults to a no-op, which is correct only for state that never needs
    rewinding — stateful kernels should supply it.

    [concat] is the chunk monoid joining the stage's outputs in {!run}.
    [concat []] must be well-defined and produce an empty chunk — capture
    whatever the empty chunk needs (dtype, shape) at construction.

    [latency] is the stage's involuntary lookahead in input items and
    defaults to [0]. [rate] is the stage's output items per input items and
    defaults to {!Rate.identity}.

    [out_format] turns the stage's input format into its output format
    (dtype, channels, bound changes) and must scale items per second by
    exactly [rate]; it defaults to the identity, with items per second and
    the per-chunk bound scaled by [rate] when a non-identity rate is
    declared. Upstream latency accounting is the library's: [latency] is
    converted to source-rate samples and accumulated into the threaded format
    automatically.

    Raises [Invalid_argument] if [latency < 0] or if [rate] is not positive.
    Threading the formats — inside {!run} or {!Stream.prepare} — raises
    [Invalid_argument] if [out_format] disagrees with the declared [rate] or
    does not derive from its argument. *)

val offline_only : ('a -> 'b) -> concat:('b list -> 'b) -> ('a, 'b, offline) t
(** [offline_only f ~concat] is the whole-signal stage applying [f] to the
    complete input at once: latency 0, rate 1:1, format-preserving. The
    resulting pipeline is typed {!offline} and cannot be given to
    {!Stream.prepare}. [concat] is the chunk monoid for ['b], stated for
    uniformity with {!kernel}. *)

(** {1 Composition} *)

val ( >> ) : ('a, 'b, 'k) t -> ('b, 'c, 'k) t -> ('a, 'c, 'k) t
(** [f >> g] is the pipeline applying [f] then [g]. Formats thread left to
    right at prepare — each stage sees its {e own} input format, transformed
    by the upstream [out_format]s; latency accumulates exactly (rational,
    converted through rate changes); drain runs front-to-back: [f]'s flush
    chunks pass through [g]'s step, then [g] flushes. *)

val fanout :
  ('a, 'b, 'k) t -> ('a, 'c, 'k) t -> ('a, 'b option * 'c option, 'k) t
(** [fanout f g] runs the shared input once through both branches; the
    branches may differ in rate {e and} latency. Each push yields whatever
    each branch yields — pairing and alignment are explicitly {e not}
    promised per-push; consumers align on positions. The reported latency is
    the larger branch latency; the reported rate is {!Rate.identity} (pairs
    flow at push granularity); the downstream format is the shared input
    format with the larger branch latency absorbed. *)

val map : ('b -> 'c) -> ('a, 'b, 'k) t -> ('a, 'c, 'k) t
(** [map f p] is [p >> stateless f]. *)

(** {1 Static queries} *)

val latency : ('a, 'b, _) t -> Rate.t
(** [latency p] is [p]'s cumulative involuntary lookahead in source-rate
    samples, exact. A deliberate delay stage is intent, not latency. *)

val rate : ('a, 'b, _) t -> Rate.t
(** [rate p] is [p]'s cumulative output items per input items, exact. *)

(** {1 Offline driver} *)

val run : source:Format.t -> ('a, 'b, _) t -> 'a -> 'b
(** [run ~source p x] is prepare/step/flush on the single chunk [x], each
    stage's outputs joined by its own [concat] — the result by the final
    stage's. [concat []] is well-defined, so an input shorter than the
    pipeline's total latency yields a well-defined, possibly empty chunk.
    [source] is required — channels and dtype are not recoverable from an
    opaque ['a].

    Semantics, precisely: [run] produces the {e causal} output — identical to
    pushing [x] once and flushing. No hidden latency compensation. *)

(** {1 Online driver} *)

(** [pipeline] equates {!t} so that {!Stream} can refer to it. *)
type ('a, 'b, +'k) pipeline = ('a, 'b, 'k) t

module Stream : sig
  (** The type for prepared plans. Mutable; single-owner; not domain-safe. *)
  type ('a, 'b) t

  val prepare :
    ('a, 'b, causal) pipeline -> source:Format.t -> max_chunk:int -> ('a, 'b) t
  (** [prepare p ~source ~max_chunk] instantiates [p] for chunks of at most
      [max_chunk] items described by [source]. Every buffer is allocated
      here; per-stage output bounds derive from [max_chunk] through each
      stage's [out_format] — there is no divisibility precondition on
      [max_chunk].

      Raises [Invalid_argument] if [max_chunk < 1] or if any stage rejects
      its incoming format. *)

  val push : ('a, 'b) t -> 'a -> 'b option
  (** [push s a] feeds the chunk [a] — at most [max_chunk] items — and is
      the pipeline's output chunk, if any. *)

  val flush : ('a, 'b) t -> 'b list
  (** [flush s] drains every stage front-to-back and is the tail chunks, in
      order. *)

  val reset : ('a, 'b) t -> unit
  (** [reset s] restores [s] to its freshly prepared state. *)

  val latency : ('a, 'b) t -> Rate.t
  (** [latency s] is the prepared pipeline's cumulative involuntary lookahead
      in source-rate samples, exact. *)
end
