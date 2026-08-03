(* Toy kernels exercising every constructor of Soundml.Pipeline. They are test
   fixtures, not public API: a stateless gain, a stateful block accumulator with
   a flush tail, a 1:4 decimator (rate-changing, shrinking the per-chunk bound),
   a bounded-lookahead stage with nonzero latency, and an offline-only
   whole-chunk normalizer. Both plain [float array] chunks and rank-one float32
   [Nx.t] chunks are covered to prove genericity. *)

open Soundml

let rate_1_4 = {Pipeline.Rate.num= 1; den= 4}

(* {1 Float-array toys} *)

let gain g = Pipeline.stateless (Array.map (fun v -> g *. v))

(* Sums each window of [w] consecutive items; the leftover partial window is
   emitted by [flush]. Validates at prepare time that the stream is mono. *)
let block_sum w =
  if w < 1 then invalid_arg "block_sum: window must be positive" ;
  let sum = Array.fold_left ( +. ) 0. in
  Pipeline.kernel
    ~rate:{Pipeline.Rate.num= 1; den= w}
    ~flush:(fun st ->
      if Array.length !st = 0 then []
      else begin
        let tail = !st in
        st := [||] ;
        [[|sum tail|]]
      end )
    ~reset:(fun st -> st := [||])
    ~concat:Array.concat
    ~prepare:(fun fmt ->
      if Pipeline.Format.channels fmt <> 1 then
        invalid_arg "block_sum: mono input required" ;
      ref [||] )
    ~step:(fun st chunk ->
      let data = Array.append !st chunk in
      let n = Array.length data / w in
      st := Array.sub data (n * w) (Array.length data - (n * w)) ;
      if n = 0 then None
      else Some (Array.init n (fun i -> sum (Array.sub data (i * w) w))) )
    ()

(* Keeps every fourth item, phase carried across chunks. The explicit
   [out_format] shrinks the per-chunk bound along with the item rate. *)
let decim4 () =
  Pipeline.kernel ~rate:rate_1_4
    ~out_format:(fun fmt ->
      let open Pipeline in
      let ips = Rate.(Format.items_per_second fmt * rate_1_4) in
      let bound = Option.map (fun n -> (n + 3) / 4) (Format.max_items fmt) in
      Format.with_max_items bound (Format.with_items_per_second ips fmt) )
    ~reset:(fun ph -> ph := 0)
    ~concat:Array.concat
    ~prepare:(fun _ -> ref 0)
    ~step:(fun ph chunk ->
      let n = Array.length chunk in
      let out = ref [] in
      let i = ref !ph in
      while !i < n do
        out := chunk.(!i) :: !out ;
        i := !i + 4
      done ;
      ph := !i - n ;
      match !out with [] -> None | l -> Some (Array.of_list (List.rev l)) )
    ()

(* Identity with a bounded lookahead of [d] items: emission lags the input by
   [d] items and [flush] releases the withheld tail. Latency, not offline. *)
let lookahead d =
  if d < 0 then invalid_arg "lookahead: lookahead must be non-negative" ;
  Pipeline.kernel ~latency:d
    ~flush:(fun st ->
      if Array.length !st = 0 then []
      else begin
        let tail = !st in
        st := [||] ;
        [tail]
      end )
    ~reset:(fun st -> st := [||])
    ~concat:Array.concat
    ~prepare:(fun _ -> ref [||])
    ~step:(fun st chunk ->
      let data = Array.append !st chunk in
      let n = Array.length data in
      if n <= d then (
        st := data ;
        None )
      else begin
        st := Array.sub data (n - d) d ;
        Some (Array.sub data 0 (n - d))
      end )
    ()

(* Whole-signal peak normalizer: the canonical offline-only stage. *)
let normalize () =
  Pipeline.offline_only
    (fun x ->
      let peak = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. x in
      if peak = 0. then x else Array.map (fun v -> v /. peak) x )
    ~concat:Array.concat

(* {1 Rank-one float32 Nx toys} *)

let nx_concat = function
  | [] ->
      Nx.zeros Nx.float32 [|0|]
  | l ->
      Nx.concatenate ~axis:0 l

let nx_gain g = Pipeline.stateless (fun t -> Nx.mul_s t g)

(* The [lookahead] toy over real tensors. Models the two runtime contracts a
   tensor kernel must honour: [prepare] validates the element dtype against the
   incoming format, and [step] borrows its chunk — the retained tail is copied
   out of the caller's buffer, and the emitted head is copied whenever it would
   otherwise alias it. *)
let nx_lookahead d =
  if d < 0 then invalid_arg "nx_lookahead: lookahead must be non-negative" ;
  let empty () = Nx.zeros Nx.float32 [|0|] in
  Pipeline.kernel ~latency:d
    ~flush:(fun st ->
      if Nx.numel !st = 0 then []
      else begin
        let tail = !st in
        st := empty () ;
        [tail]
      end )
    ~reset:(fun st -> st := empty ())
    ~concat:nx_concat
    ~prepare:(fun fmt ->
      let open Pipeline.Format in
      if not (equal_dtype (dtype fmt) (Dtype Nx.float32)) then
        invalid_arg "nx_lookahead: float32 input required" ;
      ref (empty ()) )
    ~step:(fun st chunk ->
      let data =
        if Nx.numel !st = 0 then chunk else Nx.concatenate ~axis:0 [!st; chunk]
      in
      let n = Nx.numel data in
      if n <= d then (
        st := Nx.copy data ;
        None )
      else begin
        st := Nx.copy (Nx.shrink [|(n - d, n)|] data) ;
        let head = Nx.shrink [|(0, n - d)|] data in
        Some (if data == chunk then Nx.copy head else head)
      end )
    ()
