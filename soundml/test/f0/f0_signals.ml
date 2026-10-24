(* The synthetic signals of the f0 suite, rebuilt from the recipes
   dev/generate_vectors.py records (see F0VectorGenerator): each segment is
   evaluated on its own time t = i / sr, in numpy's order of operations, every
   product let-bound before it is added so that no platform fuses a multiply-add
   numpy rounds twice. The sines are libm's, within ulps of numpy's; the
   generator's decision margins cover the difference. *)

let two_pi = 2. *. Float.pi

let lcg ~seed n =
  let state = ref seed in
  Array.init n (fun _ ->
      state := ((1103515245 * !state) + 12345) mod (1 lsl 31) ;
      (Float.of_int !state /. Float.of_int (1 lsl 30)) -. 1. )

(* [tone ~sr f amp phase i] is [amp * sin (2 pi f t + phase)]. *)
let tone ~sr f amp phase i =
  let t = Float.of_int i /. Float.of_int sr in
  let w = two_pi *. f in
  let arg = w *. t in
  let arg = arg +. phase in
  amp *. Float.sin arg

let harmonic ~sr f partials i =
  let t = Float.of_int i /. Float.of_int sr in
  let acc = ref 0. in
  for h = 1 to partials do
    let hf = Float.of_int h in
    let w = two_pi *. (hf *. f) in
    let arg = w *. t in
    let phase = 0.1 *. hf in
    let arg = arg +. phase in
    let term = 0.6 /. hf *. Float.sin arg in
    acc := !acc +. term
  done ;
  !acc

let vibrato ~sr fc depth rate i =
  let t = Float.of_int i /. Float.of_int sr in
  let slow = two_pi *. rate *. t in
  let wobble = depth /. rate *. Float.sin slow in
  let carrier = two_pi *. fc *. t in
  let arg = carrier +. wobble in
  0.5 *. Float.sin arg

let chirp ~sr ~n f0 f1 i =
  let t = Float.of_int i /. Float.of_int sr in
  let k = (f1 -. f0) /. (Float.of_int n /. Float.of_int sr) in
  let a = f0 *. t in
  let b = 0.5 *. k *. t in
  let b = b *. t in
  let s = a +. b in
  let arg = two_pi *. s in
  0.5 *. Float.sin arg

let offset ~sr c f amp i =
  let t = Float.of_int i /. Float.of_int sr in
  let w = two_pi *. f in
  let arg = w *. t in
  let arg = arg +. 0.3 in
  let v = amp *. Float.sin arg in
  c +. v

let number = function
  | `Int i ->
      Float.of_int i
  | `Float f ->
      f
  | _ ->
      failwith "f0 signal: a recipe argument is not a number"

let int = function
  | `Int i ->
      i
  | _ ->
      failwith "f0 signal: a length is not an integer"

(* [segment ~sr n recipe] is the [n] samples of one recipe. *)
let segment ~sr n (recipe : Yojson.Safe.t) =
  match recipe with
  | `List (`String kind :: args) -> (
      let args = Array.of_list args in
      let num k = number args.(k) in
      match kind with
      | "silence" ->
          Array.make n 0.
      | "tone" ->
          Array.init n (tone ~sr (num 0) (num 1) (num 2))
      | "harmonic" ->
          Array.init n (harmonic ~sr (num 0) (int args.(1)))
      | "vibrato" ->
          Array.init n (vibrato ~sr (num 0) (num 1) (num 2))
      | "chirp" ->
          Array.init n (chirp ~sr ~n (num 0) (num 1))
      | "noise" ->
          let scale = num 0 in
          Array.map (fun v -> scale *. v) (lcg ~seed:(int args.(1)) n)
      | "offset" ->
          Array.init n (offset ~sr (num 0) (num 1) (num 2))
      | other ->
          failwith ("f0 signal: unknown recipe " ^ other) )
  | _ ->
      failwith "f0 signal: malformed recipe"

(* [channel ~sr segments] is one channel: its segments end to end. *)
let channel ~sr (segments : Yojson.Safe.t) =
  match segments with
  | `List l ->
      Array.concat
        (List.map
           (function
             | `List [n; recipe] ->
                 segment ~sr (int n) recipe
             | _ ->
                 failwith "f0 signal: malformed segment" )
           l )
  | _ ->
      failwith "f0 signal: malformed channel"

(* [of_params dtype ~sr ~lead ~channels] is the case's signal, shaped [lead;
   samples] in C order, rounded into [dtype]. *)
let of_params dtype ~sr ~(lead : Yojson.Safe.t) ~(channels : Yojson.Safe.t) =
  let lead =
    match lead with
    | `List l ->
        Array.of_list (List.map int l)
    | _ ->
        failwith "f0 signal: malformed lead"
  in
  let rows =
    match channels with
    | `List l ->
        List.map (channel ~sr) l
    | _ ->
        failwith "f0 signal: malformed channels"
  in
  let n = match rows with r :: _ -> Array.length r | [] -> 0 in
  Nx.create dtype (Array.append lead [|n|]) (Array.concat rows)
