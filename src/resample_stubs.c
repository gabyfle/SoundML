/*****************************************************************************/
/*                                                                           */
/*                                                                           */
/*  Copyright (C) 2025                                                       */
/*    Gabriel Santamaria                                                     */
/*                                                                           */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*    http://www.apache.org/licenses/LICENSE-2.0                             */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*****************************************************************************/

/* resample_stubs.c — the one polyphase executor behind Soundml.Resample.

   One call per chunk: the stub receives the phase-major coefficient bank, the
   per-channel history, a scratch lane, the input chunk and the output tensor,
   and does all slicing internally. Each output sample is one independent dot
   product over a deterministic window of `history ++ chunk` with a fixed,
   chunk-independent summation order, which is what makes offline-vs-streaming
   bit-equality structural rather than tested-in.

   Floating-point discipline: the dot-product loop alone is allowed to
   reassociate and contract (multiply-add), scoped with a pragma on clang and a
   function attribute on GCC; any other compiler gets the ordered loop — a
   performance fallback, never a correctness one. The emitted reduction order
   is fixed per build, so chunk-invariance survives. This is not -ffast-math:
   no flush-to-zero, no no-NaN assumptions, nothing outside this one loop.

   The runtime lock is released around the bulk work (an offline apply pushes a
   whole file as one chunk); every pointer is extracted before the release and
   no OCaml value is touched after it. The stub allocates nothing on the OCaml
   heap and therefore raises only before releasing the lock. */

#include <stdint.h>
#include <string.h>

#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if defined(__clang__)
#define SOUNDML_ASSOC_FN
#define SOUNDML_ASSOC_LOOP _Pragma("clang fp reassociate(on) contract(fast)")
#elif defined(__GNUC__)
#define SOUNDML_ASSOC_FN                                              \
  __attribute__((optimize("-fassociative-math", "-fno-signed-zeros", \
                          "-fno-trapping-math")))
#define SOUNDML_ASSOC_LOOP
#else
#define SOUNDML_ASSOC_FN
#define SOUNDML_ASSOC_LOOP
#endif

/* One instantiation per sample type. `dot` carries the scoped floating-point
   pragma (the extra braces are required: the pragma must open a compound
   statement); `run` walks channels serially so the bank and the scratch lane
   stay cache-hot across channels.

   Geometry, in scratch coordinates: scratch[s] is input sample
   `total_fed - 2K + s`. Output i (absolute) reads the taps window starting at
   `floor(i*M/L) + K - total_fed`, phase `(i*M) mod L`; both advance by exact
   integer arithmetic (p += M; s += p / L; p %= L), so no drift is
   representable. Rows of the bank are stored reversed, so the window is read
   forward. */
#define SOUNDML_RESAMPLE_KERNEL(SUFFIX, T)                                   \
  static SOUNDML_ASSOC_FN T soundml_resample_dot_##SUFFIX(                   \
      const T *restrict x, const T *restrict h, int64_t taps) {              \
    {                                                                        \
      SOUNDML_ASSOC_LOOP                                                     \
      T acc = (T)0;                                                          \
      for (int64_t i = 0; i < taps; i++) acc += x[i] * h[i];                 \
      return acc;                                                            \
    }                                                                        \
  }                                                                          \
  static void soundml_resample_run_##SUFFIX(                                 \
      const T *restrict bank, T *restrict hist, T *restrict scratch,         \
      const T *restrict x, T *restrict y, int64_t n, int64_t n_out,          \
      int64_t channels, int64_t k, int64_t l, int64_t m, int64_t phase0,     \
      int64_t s0, int is_flush) {                                            \
    const int64_t taps = (2 * k) + 1, hlen = 2 * k;                          \
    for (int64_t c = 0; c < channels; c++) {                                 \
      memcpy(scratch, hist + (c * hlen), (size_t)hlen * sizeof(T));          \
      if (is_flush)                                                          \
        memset(scratch + hlen, 0, (size_t)n * sizeof(T));                    \
      else                                                                   \
        memcpy(scratch + hlen, x + (c * n), (size_t)n * sizeof(T));          \
      int64_t p = phase0, s = s0;                                            \
      T *out = y + (c * n_out);                                              \
      for (int64_t i = 0; i < n_out; i++) {                                  \
        out[i] =                                                             \
            soundml_resample_dot_##SUFFIX(scratch + s, bank + (p * taps),    \
                                          taps);                             \
        p += m;                                                              \
        s += p / l;                                                          \
        p %= l;                                                              \
      }                                                                      \
      memcpy(hist + (c * hlen), scratch + n, (size_t)hlen * sizeof(T));      \
    }                                                                        \
  }

SOUNDML_RESAMPLE_KERNEL(f32, float)
SOUNDML_RESAMPLE_KERNEL(f64, double)

static int64_t soundml_resample_dim(value v) {
  return (int64_t)Caml_ba_array_val(v)->dim[0];
}

static int soundml_resample_kind(value v) {
  return Caml_ba_array_val(v)->flags & CAML_BA_KIND_MASK;
}

/* Every size below is validated against the actual extents of the arrays the
   OCaml side handed over, in 64-bit arithmetic, before any pointer is formed
   from them; a mismatch means a bug in the OCaml bookkeeping, never silent
   out-of-bounds work. */
CAMLprim value soundml_resample_step(value v_bank, value v_hist,
                                     value v_scratch, value v_x, value v_y,
                                     value v_n, value v_n_out,
                                     value v_channels, value v_k, value v_l,
                                     value v_m, value v_phase0, value v_s0,
                                     value v_is_flush) {
  const int64_t n = Long_val(v_n);
  const int64_t n_out = Long_val(v_n_out);
  const int64_t channels = Long_val(v_channels);
  const int64_t k = Long_val(v_k);
  const int64_t l = Long_val(v_l);
  const int64_t m = Long_val(v_m);
  const int64_t phase0 = Long_val(v_phase0);
  const int64_t s0 = Long_val(v_s0);
  const int is_flush = Bool_val(v_is_flush);
  const int64_t taps = (2 * k) + 1, hlen = 2 * k;

  if (n < 0 || n_out < 0 || channels < 1 || k < 0 || l < 1 || m < 1 ||
      phase0 < 0 || phase0 >= l || s0 < 0)
    caml_failwith("soundml_resample: invalid geometry");
  const int kind = soundml_resample_kind(v_bank);
  if (kind != CAML_BA_FLOAT32 && kind != CAML_BA_FLOAT64)
    caml_failwith("soundml_resample: unsupported dtype");
  if (soundml_resample_kind(v_hist) != kind ||
      soundml_resample_kind(v_scratch) != kind ||
      soundml_resample_kind(v_x) != kind ||
      soundml_resample_kind(v_y) != kind)
    caml_failwith("soundml_resample: mixed dtypes");
  if (soundml_resample_dim(v_bank) < l * taps ||
      soundml_resample_dim(v_hist) < channels * hlen ||
      soundml_resample_dim(v_scratch) < hlen + n ||
      (!is_flush && soundml_resample_dim(v_x) < channels * n) ||
      (n_out > 0 && soundml_resample_dim(v_y) < channels * n_out))
    caml_failwith("soundml_resample: buffer extents disagree with geometry");
  if (n_out > 0) {
    /* the last window must fit inside the scratch lane */
    const int64_t s_last = s0 + ((phase0 + ((n_out - 1) * m)) / l);
    if (s_last > n - 1)
      caml_failwith("soundml_resample: window overruns the scratch lane");
  }

  void *bank = Caml_ba_data_val(v_bank);
  void *hist = Caml_ba_data_val(v_hist);
  void *scratch = Caml_ba_data_val(v_scratch);
  void *x = Caml_ba_data_val(v_x);
  void *y = Caml_ba_data_val(v_y);

  caml_release_runtime_system();
  if (kind == CAML_BA_FLOAT32)
    soundml_resample_run_f32((const float *)bank, (float *)hist,
                             (float *)scratch, (const float *)x, (float *)y, n,
                             n_out, channels, k, l, m, phase0, s0, is_flush);
  else
    soundml_resample_run_f64((const double *)bank, (double *)hist,
                             (double *)scratch, (const double *)x, (double *)y,
                             n, n_out, channels, k, l, m, phase0, s0,
                             is_flush);
  caml_acquire_runtime_system();
  return Val_unit;
}

CAMLprim value soundml_resample_step_bc(value *argv, int argn) {
  (void)argn;
  return soundml_resample_step(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5], argv[6], argv[7], argv[8], argv[9],
                               argv[10], argv[11], argv[12], argv[13]);
}
