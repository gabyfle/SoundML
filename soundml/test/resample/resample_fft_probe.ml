(* The batched-vs-single FFT bit-identity probe, standing in CI.

   The FFT-executed sharp stage runs every block through one call shape: real
   [channels; N] -> Nx.rfft complex128 -> multiply against the plan-owned
   spectrum -> Nx.irfft dtype. The offline path may additionally batch the
   blocks of a whole signal as extra leading lines of one transform - admitted
   only while nx's FFT transforms lines independently at bit level: the batched
   result must equal the row-by-row loop byte for byte, at every transform
   length the planner can emit, for both dtypes, at any line count. This suite
   is that proof, kept standing so a backend change that breaks per-line
   independence fails CI instead of silently bending the pipeline law. The
   recorded escape is the [ols_batch] switch in resample.ml: flipping it drops
   multi-block steps back to sequential per-block transforms - the exact
   streaming call shapes - losing batching throughput, never the law.

   The comparison is on IEEE bit patterns ([Int64.bits_of_float] on every
   component), never on tolerances; [Nx.to_array] widens float32 and complex64
   exactly, so equal bits here is equal bytes in storage. The length set is the
   superset of every transform length the block rule can emit across the
   standard-rate matrix and the presets (powers of two and 3 * 2^j between the
   smallest ÷F forward length and the largest ×F inverse length); the planner
   agreement test in resample_config pins the shipped plans, whose lengths all
   appear below. *)

open Windtrap
open Soundml

let seed = 0x0f37

(* Every 2^j and 3 * 2^j in the block rule's reach: forward lengths N, inverse
   lengths N/F and N*F, F in {2, 3, 4}. *)
let lengths = [512; 768; 1024; 1536; 2048; 3072; 4096; 6144; 8192; 12288; 16384]

let line_counts = [1; 2; 3; 8; 64]

let bits_of_floats a = Array.map Int64.bits_of_float a

let bits_of_real t = bits_of_floats (Nx.to_array (Nx.cast Nx.float64 t))

let bits_of_complex t =
  let a = Nx.to_array (Nx.cast Nx.complex128 t) in
  Array.init
    (2 * Array.length a)
    (fun i ->
      let z = a.(i / 2) in
      Int64.bits_of_float (if i mod 2 = 0 then z.Complex.re else z.Complex.im) )

let check_bits ~msg expected actual =
  if expected <> actual then begin
    let n = Stdlib.min (Array.length expected) (Array.length actual) in
    let i = ref 0 in
    while !i < n && Int64.equal expected.(!i) actual.(!i) do
      incr i
    done ;
    failf "%s: bit divergence at component %d" msg !i
  end

let signal rng dtype shape =
  let count = Array.fold_left ( * ) 1 shape in
  Nx.reshape shape
    (Nx.create dtype [|count|]
       (Array.init count (fun _ -> (2. *. Random.State.float rng 1.) -. 1.)) )

let rows lines t =
  List.init lines (fun i ->
      let s =
        Array.init (Nx.ndim t) (fun d ->
            if d = 0 then (i, i + 1) else (0, Nx.dim d t) )
      in
      Nx.shrink s t )

let concat_rows parts = Nx.concatenate ~axis:0 parts

(* One random half-spectrum standing in for the plan-owned filter spectrum. *)
let spectrum rng bins =
  Nx.create Nx.complex128 [|bins|]
    (Array.init bins (fun _ ->
         { Complex.re= (2. *. Random.State.float rng 1.) -. 1.
         ; im= (2. *. Random.State.float rng 1.) -. 1. } ) )

let probe_case name dtype () =
  test name (fun () ->
      let rng = Random.State.make [|seed|] in
      List.iter
        (fun len ->
          let h = spectrum rng ((len / 2) + 1) in
          List.iter
            (fun lines ->
              let x = signal rng dtype [|lines; len|] in
              let msg part =
                Printf.sprintf "%s/len=%d/lines=%d" part len lines
              in
              (* forward: rfft of the batch vs the row-by-row loop *)
              let fwd = Nx.rfft Nx.complex128 x in
              let fwd_rows =
                concat_rows (List.map (Nx.rfft Nx.complex128) (rows lines x))
              in
              check_bits ~msg:(msg "rfft") (bits_of_complex fwd)
                (bits_of_complex fwd_rows) ;
              (* inverse: irfft back to the probe dtype *)
              let inv = Nx.irfft dtype fwd in
              let inv_rows =
                concat_rows (List.map (Nx.irfft dtype) (rows lines fwd))
              in
              check_bits ~msg:(msg "irfft") (bits_of_real inv)
                (bits_of_real inv_rows) ;
              (* the full block pipeline: rfft . (mul h) . irfft *)
              let block t =
                Nx.irfft dtype (Nx.mul (Nx.rfft Nx.complex128 t) h)
              in
              check_bits ~msg:(msg "block")
                (bits_of_real (block x))
                (bits_of_real (concat_rows (List.map block (rows lines x)))) )
            line_counts )
        lengths )

let stacked_case () =
  test "a stacked [channels; blocks; N] batch equals every single block"
    (fun () ->
      (* the offline shape: blocks stacked on a second leading axis, channels on
         the first - the transform must still be per-line *)
      let rng = Random.State.make [|seed + 1|] in
      List.iter
        (fun len ->
          let h = spectrum rng ((len / 2) + 1) in
          let x = signal rng Nx.float64 [|2; 3; len|] in
          let block t =
            Nx.irfft Nx.float64 (Nx.mul (Nx.rfft Nx.complex128 t) h)
          in
          let batched = block x in
          for c = 0 to 1 do
            for b = 0 to 2 do
              let one =
                block (Nx.shrink [|(c, c + 1); (b, b + 1); (0, len)|] x)
              in
              let got =
                Nx.shrink [|(c, c + 1); (b, b + 1); (0, len)|] batched
              in
              check_bits
                ~msg:(Printf.sprintf "stacked/len=%d/c=%d/b=%d" len c b)
                (bits_of_real one) (bits_of_real got)
            done
          done )
        [1024; 1536; 4096] )

let coverage_case () =
  test "every transform length the planner emits is probed" (fun () ->
      (* the planner is the authority on which lengths exist: walk the
         standard-rate matrix across the presets, read each FFT-executed stage's
         transform length from [Config.pp] (the only surface that names it), and
         require it — and every companion length its stage can run (the ×F
         forward at N, the ÷F inverse at N/F) — to be in the probed set. A plan
         whose length escapes the probe fails here, not silently. *)
      let rates =
        [ 8000
        ; 11025
        ; 16000
        ; 22050
        ; 24000
        ; 32000
        ; 44100
        ; 48000
        ; 88200
        ; 96000
        ; 192000 ]
      in
      let tags s =
        let out = ref [] in
        let n = String.length s in
        let i = ref 0 in
        while !i + 2 < n do
          if String.sub s !i 2 = "N=" then begin
            let j = ref (!i + 2) in
            while !j < n && s.[!j] >= '0' && s.[!j] <= '9' do
              incr j
            done ;
            out := int_of_string (String.sub s (!i + 2) (!j - !i - 2)) :: !out ;
            i := !j
          end
          else incr i
        done ;
        !out
      in
      List.iter
        (fun quality ->
          List.iter
            (fun sr ->
              List.iter
                (fun target ->
                  if sr <> target then
                    let cfg =
                      Resample.Config.create ~quality ~sample_rate:sr ~target ()
                    in
                    let s = Format.asprintf "%a" Resample.Config.pp cfg in
                    List.iter
                      (fun v ->
                        let need =
                          v
                          :: List.filter_map
                               (fun f ->
                                 if v mod f = 0 && v / f >= 512 then Some (v / f)
                                 else None )
                               [2; 3; 4]
                        in
                        List.iter
                          (fun len ->
                            if not (List.mem len lengths) then
                              failf "%d->%d %s: length %d is not probed" sr
                                target s len )
                          need )
                      (tags s) )
                rates )
            rates )
        [`Fast; `High; `Best] )

let suite =
  [ group "fft-probe"
      [ probe_case "batched == single, float64 lines" Nx.float64 ()
      ; probe_case "batched == single, float32 lines" Nx.float32 ()
      ; stacked_case ()
      ; coverage_case () ] ]
