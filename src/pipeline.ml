type causal = |

type offline = |

module Rate = struct
  type t = {num: int; den: int}

  let rec gcd a b = if b = 0 then a else gcd b (a mod b)

  let normalize {num; den} =
    if den = 0 then invalid_arg "Soundml.Pipeline.Rate: denominator is zero" ;
    let s = if den < 0 then -1 else 1 in
    let num = s * num and den = s * den in
    let g = gcd (abs num) den in
    if g = 0 then {num= 0; den= 1} else {num= num / g; den= den / g}

  let mul a b = normalize {num= a.num * b.num; den= a.den * b.den}

  let identity = {num= 1; den= 1}

  let zero = {num= 0; den= 1}

  let of_int n = {num= n; den= 1}

  let add a b =
    normalize {num= (a.num * b.den) + (b.num * a.den); den= a.den * b.den}

  let inv {num; den} =
    if num = 0 then
      invalid_arg "Soundml.Pipeline.Rate: zero rate has no inverse"
    else normalize {num= den; den= num}

  let equal a b = a.num * b.den = b.num * a.den

  let max a b = if a.num * b.den >= b.num * a.den then a else b

  let is_positive {num; den} = (num >= 1 && den >= 1) || (num <= -1 && den <= -1)

  let ( * ) = mul

  let pp ppf {num; den} =
    if den = 1 then Stdlib.Format.fprintf ppf "%d" num
    else Stdlib.Format.fprintf ppf "%d/%d" num den
end

module Format = struct
  type dtype = Dtype : (float, 'd) Nx.dtype -> dtype

  type t =
    { fdtype: dtype
    ; ips: Rate.t
    ; chans: int
    ; bound: int option
    ; lat: Rate.t (* upstream latency, source-rate samples *)
    ; src_ips: Rate.t (* the source's items per second, for conversions *) }

  let audio dt ~sample_rate ~channels =
    if sample_rate < 1 then
      invalid_arg "Soundml.Pipeline.Format.audio: sample_rate must be positive" ;
    if channels < 1 then
      invalid_arg "Soundml.Pipeline.Format.audio: channels must be positive" ;
    let ips = Rate.of_int sample_rate in
    { fdtype= Dtype dt
    ; ips
    ; chans= channels
    ; bound= None
    ; lat= Rate.zero
    ; src_ips= ips }

  let dtype t = t.fdtype

  let items_per_second t = t.ips

  let channels t = t.chans

  let max_items t = t.bound

  let upstream_latency t = t.lat

  let with_dtype dt t = {t with fdtype= Dtype dt}

  let with_channels channels t =
    if channels < 1 then
      invalid_arg
        "Soundml.Pipeline.Format.with_channels: channels must be positive" ;
    {t with chans= channels}

  let with_items_per_second r t =
    if not (Rate.is_positive r) then
      invalid_arg
        "Soundml.Pipeline.Format.with_items_per_second: rate must be positive" ;
    {t with ips= Rate.normalize r}

  let with_max_items bound t =
    ( match bound with
    | Some n when n < 0 ->
        invalid_arg
          "Soundml.Pipeline.Format.with_max_items: bound must be non-negative"
    | _ ->
        () ) ;
    {t with bound}

  (* Internal: [true] iff [b] was derived from [a] with the [with_] builders,
     which cannot touch the library-owned bookkeeping fields. *)
  let same_lineage a b =
    Rate.equal a.src_ips b.src_ips && Rate.equal a.lat b.lat

  (* Internal: [items] at [at]'s own rate, converted to source-rate samples. *)
  let latency_to_samples items at = Rate.(items * at.src_ips * inv at.ips)

  let add_latency samples t =
    if samples.Rate.num = 0 then t else {t with lat= Rate.add t.lat samples}

  let pp ppf t =
    let (Dtype dt) = t.fdtype in
    Stdlib.Format.fprintf ppf
      "@[<h>%a, %d channel%s, %a items/s, %s, upstream latency %a@]" Nx.pp_dtype
      dt t.chans
      (if t.chans = 1 then "" else "s")
      Rate.pp t.ips
      ( match t.bound with
      | None ->
          "unbounded chunks"
      | Some n ->
          Printf.sprintf "at most %d items/chunk" n )
      Rate.pp t.lat
end

(* A prepared stage: the closure-packed state of one streaming instance. *)
type ('a, 'b) plan =
  {feed: 'a -> 'b option; drain: unit -> 'b list; restart: unit -> unit}

(* The CCA normal form, folded eagerly at construction: [prep] instantiates
   streaming state, [run1] is the offline normalisation (one whole-chunk pass
   per stage), [thread] maps a stage's input format to its output format
   (out_format plus library-owned latency accounting), [lat]/[rt] are the folded
   cumulative latency and rate. ['k] is a pure phantom. *)
type ('a, 'b, 'k) t =
  { prep: Format.t -> ('a, 'b) plan
  ; run1: Format.t -> 'a -> 'b
  ; thread: Format.t -> Format.t
  ; lat: Rate.t
  ; rt: Rate.t }

let ceil_scale n r = ((n * r.Rate.num) + r.Rate.den - 1) / r.Rate.den

let stage_thread ~latency ~rate ~out_format fmt =
  let out =
    match out_format with
    | Some f ->
        let out = f fmt in
        if not (Format.same_lineage fmt out) then
          invalid_arg
            "Soundml.Pipeline: out_format must be derived from the stage's \
             input format" ;
        let expected = Rate.(Format.items_per_second fmt * rate) in
        if not (Rate.equal (Format.items_per_second out) expected) then
          invalid_arg
            (Stdlib.Format.asprintf
               "Soundml.Pipeline: out_format yields %a items/s but the \
                declared rate %a implies %a items/s"
               Rate.pp
               (Format.items_per_second out)
               Rate.pp rate Rate.pp expected ) ;
        out
    | None ->
        if Rate.equal rate Rate.identity then fmt
        else
          let ips = Rate.(Format.items_per_second fmt * rate) in
          let bound =
            Option.map (fun n -> ceil_scale n rate) (Format.max_items fmt)
          in
          Format.with_max_items bound (Format.with_items_per_second ips fmt)
  in
  Format.add_latency (Format.latency_to_samples (Rate.of_int latency) fmt) out

let kernel ?(latency = 0) ?(rate = Rate.identity) ?out_format
    ?(flush = fun _ -> []) ?(reset = fun _ -> ()) ~concat ~prepare ~step () =
  if latency < 0 then
    invalid_arg "Soundml.Pipeline.kernel: latency must be non-negative" ;
  if not (Rate.is_positive rate) then
    invalid_arg "Soundml.Pipeline.kernel: rate must be positive" ;
  let rate = Rate.normalize rate in
  { prep=
      (fun fmt ->
        let s = prepare fmt in
        {feed= step s; drain= (fun () -> flush s); restart= (fun () -> reset s)} )
  ; run1=
      (fun fmt a ->
        let s = prepare fmt in
        let out = match step s a with Some b -> [b] | None -> [] in
        concat (out @ flush s) )
  ; thread= stage_thread ~latency ~rate ~out_format
  ; lat= Rate.of_int latency
  ; rt= rate }

let stateless f =
  { prep=
      (fun _ ->
        { feed= (fun a -> Some (f a))
        ; drain= (fun () -> [])
        ; restart= (fun () -> ()) } )
  ; run1= (fun _ a -> f a)
  ; thread= (fun fmt -> fmt)
  ; lat= Rate.zero
  ; rt= Rate.identity }

let offline_only f ~concat:_ =
  { prep=
      (fun _ ->
        invalid_arg "Soundml.Pipeline: an offline-only stage cannot stream" )
  ; run1= (fun _ a -> f a)
  ; thread= (fun fmt -> fmt)
  ; lat= Rate.zero
  ; rt= Rate.identity }

let ( >> ) f g =
  { prep=
      (fun fmt ->
        let pf = f.prep fmt in
        let pg = g.prep (f.thread fmt) in
        { feed=
            (fun a -> match pf.feed a with None -> None | Some b -> pg.feed b)
        ; drain=
            (fun () ->
              let through = List.filter_map pg.feed (pf.drain ()) in
              through @ pg.drain () )
        ; restart= (fun () -> pf.restart () ; pg.restart ()) } )
  ; run1= (fun fmt a -> g.run1 (f.thread fmt) (f.run1 fmt a))
  ; thread= (fun fmt -> g.thread (f.thread fmt))
  ; lat= Rate.add f.lat Rate.(g.lat * inv f.rt)
  ; rt= Rate.(f.rt * g.rt) }

let rec zip_pad bs cs =
  match (bs, cs) with
  | [], [] ->
      []
  | b :: bs, [] ->
      (Some b, None) :: zip_pad bs []
  | [], c :: cs ->
      (None, Some c) :: zip_pad [] cs
  | b :: bs, c :: cs ->
      (Some b, Some c) :: zip_pad bs cs

let fanout f g =
  let lat = Rate.max f.lat g.lat in
  { prep=
      (fun fmt ->
        let pf = f.prep fmt in
        let pg = g.prep fmt in
        { feed=
            (fun a ->
              let b = pf.feed a in
              let c = pg.feed a in
              match (b, c) with None, None -> None | _ -> Some (b, c) )
        ; drain= (fun () -> zip_pad (pf.drain ()) (pg.drain ()))
        ; restart= (fun () -> pf.restart () ; pg.restart ()) } )
  ; run1= (fun fmt a -> (Some (f.run1 fmt a), Some (g.run1 fmt a)))
  ; thread=
      (fun fmt -> Format.add_latency (Format.latency_to_samples lat fmt) fmt)
  ; lat
  ; rt= Rate.identity }

let map f p = p >> stateless f

let latency p = p.lat

let rate p = p.rt

let run ~source p x =
  let source = Format.with_max_items None source in
  (* threading the full chain validates every stage's out_format against its
     declared rate, including the final stage's, before any data flows *)
  ignore (p.thread source : Format.t) ;
  p.run1 source x

type ('a, 'b, +'k) pipeline = ('a, 'b, 'k) t

module Stream = struct
  type ('a, 'b) t = {plan: ('a, 'b) plan; lat: Rate.t}

  let prepare p ~source ~max_chunk =
    if max_chunk < 1 then
      invalid_arg "Soundml.Pipeline.Stream.prepare: max_chunk must be positive" ;
    let source = Format.with_max_items (Some max_chunk) source in
    (* threading the full chain validates every stage's out_format against its
       declared rate, including the final stage's, at prepare time *)
    ignore (p.thread source : Format.t) ;
    {plan= p.prep source; lat= p.lat}

  let push s a = s.plan.feed a

  let flush s = s.plan.drain ()

  let reset s = s.plan.restart ()

  let latency s = s.lat
end
