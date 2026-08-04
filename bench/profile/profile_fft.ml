(* Measurement probe behind the OLS cost-model constants in resample.ml.

   Times, on the pinned nx and this machine: the real transforms ([Nx.rfft
   complex128] / [Nx.irfft dtype]) at every block length the planner can emit,
   the complex frequency-domain multiply ([Nx.mul] at complex128, the __muldc3
   libcall path), and the spectrum-extension ops (shrink + flip + conj +
   concatenate). Emits TSV: FFT (per length x direction x dtype x lines), MUL
   (per bin count), EXT (per length). Wall-clock min/median over n reps after
   warmup; run on a quiet host. The frozen constants derived from a run of this
   probe are committed next to the cost model with the measured numbers quoted;
   they are never probed at run time, so plans stay machine-independent. *)

let now () = Unix.gettimeofday ()

let time ?(warmup = 3) ?(n = 30) f =
  for _ = 1 to warmup do
    ignore (Sys.opaque_identity (f ()))
  done ;
  let ts =
    Array.init n (fun _ ->
        let t0 = now () in
        ignore (Sys.opaque_identity (f ())) ;
        now () -. t0 )
  in
  Array.sort compare ts ;
  (ts.(0), ts.(n / 2))

let lengths = [512; 768; 1024; 1536; 2048; 3072; 4096; 6144; 8192; 12288; 16384]

let fft () =
  Printf.printf
    "# FFT\tlen\tdir\tdtype\tlines\tt_min_us\tt_med_us\tns_per_elt\n" ;
  List.iter
    (fun len ->
      List.iter
        (fun lines ->
          let x32 = Nx.rand Nx.float32 [|lines; len|] in
          let x64 = Nx.rand Nx.float64 [|lines; len|] in
          let s32 = Nx.rfft Nx.complex128 x32 in
          let s64 = Nx.rfft Nx.complex128 x64 in
          let row name f =
            let tmin, tmed = time f in
            Printf.printf "FFT\t%d\t%s\t%d\t%.1f\t%.1f\t%.2f\n" len name lines
              (tmin *. 1e6) (tmed *. 1e6)
              (tmin *. 1e9 /. Float.of_int (len * lines))
          in
          row "rfft\tf32" (fun () -> Nx.rfft Nx.complex128 x32) ;
          row "rfft\tf64" (fun () -> Nx.rfft Nx.complex128 x64) ;
          row "irfft\tf32" (fun () -> Nx.irfft Nx.float32 s32) ;
          row "irfft\tf64" (fun () -> Nx.irfft Nx.float64 s64) )
        [1; 8] )
    lengths

let mul () =
  Printf.printf "# MUL\tbins\tlines\tt_min_us\tt_med_us\tns_per_bin\n" ;
  List.iter
    (fun bins ->
      List.iter
        (fun lines ->
          let a =
            Nx.rfft Nx.complex128 (Nx.rand Nx.float64 [|lines; 2 * (bins - 1)|])
          in
          let h =
            Nx.rfft Nx.complex128 (Nx.rand Nx.float64 [|2 * (bins - 1)|])
          in
          let tmin, tmed = time (fun () -> Nx.mul a h) in
          Printf.printf "MUL\t%d\t%d\t%.1f\t%.1f\t%.2f\n" bins lines
            (tmin *. 1e6) (tmed *. 1e6)
            (tmin *. 1e9 /. Float.of_int (bins * lines)) )
        [1; 8] )
    (List.map (fun n -> (n / 2) + 1) lengths)

let ext () =
  Printf.printf "# EXT\tlen\tt_min_us\tt_med_us\n" ;
  List.iter
    (fun len ->
      let x = Nx.rfft Nx.complex128 (Nx.rand Nx.float64 [|1; len|]) in
      let bins = (len / 2) + 1 in
      let f () =
        (* the ÷F extension: mirror the interior bins with conjugation *)
        let interior = Nx.shrink [|(0, 1); (1, bins - 1)|] x in
        Nx.concatenate ~axis:(-1)
          [x; Nx.Complex.conj (Nx.flip ~axes:[-1] interior)]
      in
      let tmin, tmed = time f in
      Printf.printf "EXT\t%d\t%.1f\t%.1f\n" len (tmin *. 1e6) (tmed *. 1e6) )
    lengths

let () = Nx.Rng.run ~seed:42 @@ fun () -> fft () ; mul () ; ext ()
