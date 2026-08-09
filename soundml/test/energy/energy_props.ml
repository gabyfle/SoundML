(* Contracts of the flat energy features: every precondition raises its
   documented message, the frame grid (defaults, odd frame lengths, hop past the
   frame, the empty signal), the [[...; 1; frames]] shape with leading axes
   broadcasting, dtype preservation as the double interior rounded once, and the
   closed forms the framing must reproduce — rms of a constant signal,
   frame_length-1 rms as |x| sampled on the grid, the zero-crossing rate of an
   alternating sign pattern, and the spectrogram path against its Parseval
   formula. *)

open Windtrap
open Soundml

let farray = array (float 0.)

let signal n = Array.init n (fun i -> Float.sin (0.4 *. Float.of_int i))

(* {2 Preconditions} *)

let validation_tests =
  [ test "frame_length must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"rms"
          "rms: cannot analyse frames of 0 samples (frame_length must be at \
           least 1)" (fun () ->
            ignore (rms ~frame_length:0 (Nx.zeros Nx.float64 [|8|])) ) ;
        raises_invalid_arg ~msg:"zero_crossing_rate"
          "zero_crossing_rate: cannot analyse frames of -3 samples \
           (frame_length must be at least 1)" (fun () ->
            ignore
              (zero_crossing_rate ~frame_length:(-3)
                 (Nx.zeros Nx.float64 [|8|]) ) ) ;
        raises_invalid_arg ~msg:"rms_of_spectrogram"
          "rms_of_spectrogram: cannot analyse frames of 0 samples \
           (frame_length must be at least 1)" (fun () ->
            ignore
              (rms_of_spectrogram ~frame_length:0
                 (Nx.zeros Nx.float64 [|1; 2|]) ) ) ;
        raises_invalid_arg ~msg:"rms_stage"
          "rms_stage: cannot analyse frames of 0 samples (frame_length must be \
           at least 1)" (fun () ->
            ignore
              ( rms_stage ~frame_length:0 ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.causal )
                  Pipeline.t ) ) )
  ; test "hop must be at least 1" (fun () ->
        raises_invalid_arg ~msg:"rms"
          "rms: cannot advance frames by 0 samples (hop must be at least 1)"
          (fun () -> ignore (rms ~hop:0 (Nx.zeros Nx.float64 [|8|])) ) ;
        raises_invalid_arg ~msg:"zero_crossing_rate_stage"
          "zero_crossing_rate_stage: cannot advance frames by -1 samples (hop \
           must be at least 1)" (fun () ->
            ignore
              ( zero_crossing_rate_stage ~hop:(-1) ()
                : ( (float, Nx.float64_elt) Nx.t
                  , (float, Nx.float64_elt) Nx.t
                  , Pipeline.causal )
                  Pipeline.t ) ) )
  ; test "threshold must be finite and non-negative" (fun () ->
        raises_invalid_arg ~msg:"negative"
          "zero_crossing_rate: cannot clamp signs with a threshold of -1 \
           (threshold must be finite and non-negative)" (fun () ->
            ignore
              (zero_crossing_rate ~threshold:(-1.) (Nx.zeros Nx.float64 [|8|])) ) ;
        raises_invalid_arg ~msg:"nan"
          "zero_crossing_rate: cannot clamp signs with a threshold of nan \
           (threshold must be finite and non-negative)" (fun () ->
            ignore
              (zero_crossing_rate ~threshold:Float.nan
                 (Nx.zeros Nx.float64 [|8|]) ) ) )
  ; test "rank-zero tensors are rejected" (fun () ->
        raises_invalid_arg ~msg:"rms"
          "rms: cannot analyse a rank-zero tensor (the time axis must exist)"
          (fun () -> ignore (rms (Nx.zeros Nx.float64 [||])) ) ;
        raises_invalid_arg ~msg:"zero_crossing_rate"
          "zero_crossing_rate: cannot analyse a rank-zero tensor (the time \
           axis must exist)" (fun () ->
            ignore (zero_crossing_rate (Nx.zeros Nx.float64 [||])) ) )
  ; test "the spectrogram path validates rank and bins" (fun () ->
        raises_invalid_arg ~msg:"rank one"
          "rms_of_spectrogram: cannot analyse a rank-1 tensor (the spectrogram \
           path needs [...; bins; frames])" (fun () ->
            ignore (rms_of_spectrogram (Nx.zeros Nx.float64 [|9|])) ) ;
        raises_invalid_arg ~msg:"bins mismatch"
          "rms_of_spectrogram: cannot read 8 frequency bins as frames of 16 \
           samples (a magnitude spectrogram holds frame_length / 2 + 1 bins)"
          (fun () ->
            ignore
              (rms_of_spectrogram ~frame_length:16
                 (Nx.zeros Nx.float64 [|8; 4|]) ) ) ) ]

(* {2 The frame grid and output shapes} *)

let grid_tests =
  [ test "the defaults are frame_length 2048, hop 512" (fun () ->
        let x = Nx.create Nx.float64 [|100|] (signal 100) in
        equal ~msg:"explicit defaults agree" farray
          (Nx.to_array (rms ~frame_length:2048 ~hop:512 x))
          (Nx.to_array (rms x)) ;
        equal ~msg:"zcr defaults agree" farray
          (Nx.to_array
             (zero_crossing_rate ~frame_length:2048 ~hop:512 ~threshold:1e-10 x) )
          (Nx.to_array (zero_crossing_rate x)) ;
        (* 100 samples pad to 2148: frame 0 alone lies within *)
        equal ~msg:"one frame" (array int) [|1; 1|] (Nx.shape (rms x)) )
  ; test "frame counts follow the centered grid" (fun () ->
        let x = Nx.create Nx.float64 [|60|] (signal 60) in
        (* even frame_length: padded to 60 + 16, frames at hop 4 *)
        equal ~msg:"even" (array int) [|1; 16|]
          (Nx.shape (rms ~frame_length:16 ~hop:4 x)) ;
        (* odd frame_length: padded to 60 + 14, 1 + (74 - 15) / 7 frames *)
        equal ~msg:"odd" (array int) [|1; 9|]
          (Nx.shape (rms ~frame_length:15 ~hop:7 x)) ;
        (* hop past the frame: 1 + (68 - 8) / 11 frames *)
        equal ~msg:"gap" (array int) [|1; 6|]
          (Nx.shape (zero_crossing_rate ~frame_length:8 ~hop:11 x)) )
  ; test "the empty signal produces no frames" (fun () ->
        equal ~msg:"rms" (array int) [|1; 0|]
          (Nx.shape (rms ~frame_length:16 ~hop:4 (Nx.zeros Nx.float64 [|0|]))) ;
        equal ~msg:"zcr, batched" (array int) [|3; 1; 0|]
          (Nx.shape
             (zero_crossing_rate ~frame_length:16 ~hop:4
                (Nx.zeros Nx.float32 [|3; 0|]) ) ) )
  ; test "a zero-size leading axis holds no signals" (fun () ->
        equal ~msg:"rms" (array int) [|0; 1; 16|]
          (Nx.shape
             (rms ~frame_length:16 ~hop:4 (Nx.zeros Nx.float64 [|0; 60|])) ) ;
        equal ~msg:"spectrogram path" (array int) [|0; 1; 4|]
          (Nx.shape
             (rms_of_spectrogram ~frame_length:16
                (Nx.zeros Nx.float64 [|0; 9; 4|]) ) ) )
  ; test "leading axes broadcast" (fun () ->
        let batch =
          Nx.init Nx.float64 [|2; 3; 50|] (fun i ->
              Float.sin
                (0.21 *. Float.of_int ((i.(0) * 150) + (i.(1) * 50) + i.(2))) )
        in
        let out = zero_crossing_rate ~frame_length:8 ~hop:3 batch in
        equal ~msg:"batched shape" (array int) [|2; 3; 1; 17|] (Nx.shape out) ;
        for i = 0 to 1 do
          for j = 0 to 2 do
            equal
              ~msg:(Printf.sprintf "slice %d,%d" i j)
              farray
              (Nx.to_array
                 (zero_crossing_rate ~frame_length:8 ~hop:3
                    (Nx.slice [I i; I j; A] batch) ) )
              (Nx.to_array (Nx.slice [I i; I j; A; A] out))
          done
        done ) ]

(* {2 Values: closed forms the framing must reproduce} *)

let value_tests =
  [ test "rms of a constant signal shows the zero border" (fun () ->
        (* ones, frame_length 4, hop 2: frame p holds 4 - |border deficit| ones;
           interior frames are exactly 1 *)
        let x = Nx.full Nx.float64 [|8|] 1. in
        let out = Nx.to_array (rms ~frame_length:4 ~hop:2 x) in
        equal ~msg:"frames" int 5 (Array.length out) ;
        equal ~msg:"left border" (float 1e-15) (Float.sqrt (2. /. 4.)) out.(0) ;
        equal ~msg:"interior" (float 1e-15) 1. out.(1) ;
        equal ~msg:"right border" (float 1e-15)
          (Float.sqrt (2. /. 4.))
          out.(Array.length out - 1) )
  ; test "frame_length 1 rms is |x| on the hop grid" (fun () ->
        let x = Nx.create Nx.float64 [|10|] (signal 10) in
        let out = Nx.to_array (rms ~frame_length:1 ~hop:3 x) in
        equal ~msg:"count" int 4 (Array.length out) ;
        Array.iteri
          (fun p v ->
            equal
              ~msg:(Printf.sprintf "frame %d" p)
              (float 1e-15)
              (Float.abs (Float.sin (0.4 *. Float.of_int (3 * p))))
              v )
          out )
  ; test "an alternating signal crosses at every step" (fun () ->
        (* +1 -1 +1 ...: interior frames cross between every consecutive pair,
           while the border frames open (close) on edge copies of the first
           (last) sample, which never cross each other *)
        let x =
          Nx.create Nx.float64 [|12|]
            (Array.init 12 (fun i -> if i mod 2 = 0 then 1. else -1.))
        in
        let out = Nx.to_array (zero_crossing_rate ~frame_length:4 ~hop:4 x) in
        equal ~msg:"count" int 4 (Array.length out) ;
        (* interior frame [+1; -1; +1; -1]: 3 crossings over 4 samples *)
        equal ~msg:"interior" (float 1e-15) 0.75 out.(1) ;
        (* frame 0 is [+1; +1; +1; -1] after the two edge copies of x0 *)
        equal ~msg:"left border" (float 1e-15) 0.25 out.(0) ;
        (* the last frame is [+1; -1; -1; -1] before the edge copies of the
           final -1 *)
        equal ~msg:"right border" (float 1e-15) 0.25 out.(3) )
  ; test "the threshold clamps small samples to positive" (fun () ->
        let x = Nx.create Nx.float64 [|4|] [|0.5; -0.05; 0.5; -0.5|] in
        (* frame 0 is [0.5; 0.5; 0.5; -0.05] after the edge copies *)
        let strict =
          Nx.item [0; 0] (zero_crossing_rate ~frame_length:4 ~hop:4 x)
        in
        equal ~msg:"default threshold sees -0.05" (float 1e-15) 0.25 strict ;
        let clamped =
          Nx.item [0; 0]
            (zero_crossing_rate ~frame_length:4 ~hop:4 ~threshold:0.1 x)
        in
        (* -0.05 counts positive: frame 0 never crosses *)
        equal ~msg:"threshold 0.1 clamps it" (float 1e-15) 0. clamped )
  ; test "the spectrogram path is its Parseval formula" (fun () ->
        let bins = 9 and frame_length = 16 and frames = 5 in
        let s =
          Nx.init Nx.float64 [|bins; frames|] (fun i ->
              Float.abs
                (Float.sin (0.7 *. Float.of_int ((i.(0) * frames) + i.(1)))) )
        in
        let expected =
          Array.init frames (fun t ->
              let acc = ref 0. in
              for b = 0 to bins - 1 do
                let w = if b = 0 || b = bins - 1 then 0.5 else 1. in
                let v = Nx.item [b; t] s in
                acc := !acc +. (w *. v *. v)
              done ;
              Float.sqrt
                (2. *. !acc /. Float.of_int (frame_length * frame_length)) )
        in
        (* the closed form sums in a different order: allow rounding noise *)
        equal ~msg:"formula"
          (array (float 1e-14))
          expected
          (Nx.to_array (rms_of_spectrogram ~frame_length s)) )
  ; test "float32 results are the double interior rounded once" (fun () ->
        let values = signal 60 in
        let x32 = Nx.create Nx.float32 [|60|] values in
        (* quantize, then compute in double: the reference pipeline *)
        let x64 = Nx.cast Nx.float64 x32 in
        equal ~msg:"rms" farray
          (Nx.to_array (Nx.cast Nx.float32 (rms ~frame_length:16 ~hop:4 x64)))
          (Nx.to_array (rms ~frame_length:16 ~hop:4 x32)) ;
        equal ~msg:"zcr" farray
          (Nx.to_array
             (Nx.cast Nx.float32
                (zero_crossing_rate ~frame_length:16 ~hop:4 x64) ) )
          (Nx.to_array (zero_crossing_rate ~frame_length:16 ~hop:4 x32)) ) ]

let suite =
  [ group "validation" validation_tests
  ; group "grid" grid_tests
  ; group "values" value_tests ]
