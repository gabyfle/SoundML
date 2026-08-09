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

/* resample_stubs.c — the two executors behind Soundml.Resample: the polyphase
   dot-product kernel and the spectrum shaping of the overlap-save blocks.

   One call per chunk: the stub receives the coefficient bank (phase-major or
   in visit order, named by the `visit` flag — see the geometry note below),
   the per-channel history, a scratch lane, the input chunk and the output
   tensor, and does all slicing internally. Each output sample is one
   independent dot product over a deterministic window of `history ++ chunk`
   with a fixed, chunk-independent summation order, which is what makes
   offline-vs-streaming bit-equality structural rather than tested-in.

   The overlap-save stages share the same property through
   `soundml_resample_shape`, which carries one block's half spectrum to the
   inverse transform's half grid: one function on every path, per transform
   line, in a fixed arithmetic order that does not depend on how many lines
   the call carries.

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
   `floor(i*M/L) + K - total_fed` with phase `(i*M) mod L`; the window advance
   is exact integer arithmetic (p += M; s += p / L; p %= L), so no drift is
   representable, and p0 is reconstructed from the caller's row0 = i mod L:
   (row0 * M) mod L equals (i * M) mod L. The bank arrives in one of two
   layouts, named by `visit`. Phase-major (row of output i is its phase p):
   the natural object, kept whenever the bank is L1-resident — there the
   phase walk is free and the loop carries no row counter. Visit order (slot
   j holds the phase-((j*M) mod L) row, row of output i is i mod L): chosen
   by the OCaml side once the bank outgrows L1, so consecutive outputs read
   consecutive rows and the bank streams forward with one wrap per L outputs
   instead of hopping M rows per output — the hop ran the L2-resident
   geometries at 57-60% of the load-bound dot ceiling against 76-94% for
   resident or sequential banks (measured, Apple M4 Pro). Row contents and
   the summation order are identical in both layouts, so the choice cannot
   move a bit. Rows are stored reversed, so the window is read forward. */
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
      int64_t channels, int64_t k, int64_t l, int64_t m, int64_t row0,       \
      int64_t s0, int64_t y_off, int64_t y_stride, int visit,                \
      int is_flush) {                                                        \
    const int64_t taps = (2 * k) + 1, hlen = 2 * k;                          \
    const int64_t p0 = (row0 * m) % l;                                       \
    for (int64_t c = 0; c < channels; c++) {                                 \
      memcpy(scratch, hist + (c * hlen), (size_t)hlen * sizeof(T));          \
      if (is_flush)                                                          \
        memset(scratch + hlen, 0, (size_t)n * sizeof(T));                    \
      else                                                                   \
        memcpy(scratch + hlen, x + (c * n), (size_t)n * sizeof(T));          \
      int64_t p = p0, s = s0;                                                \
      T *out = y + y_off + (c * y_stride);                                   \
      if (visit) {                                                           \
        int64_t j = row0;                                                    \
        for (int64_t i = 0; i < n_out; i++) {                                \
          out[i] =                                                           \
              soundml_resample_dot_##SUFFIX(scratch + s, bank + (j * taps),  \
                                            taps);                           \
          j++;                                                               \
          if (j == l) j = 0;                                                 \
          p += m;                                                            \
          s += p / l;                                                        \
          p %= l;                                                            \
        }                                                                    \
      } else {                                                               \
        for (int64_t i = 0; i < n_out; i++) {                                \
          out[i] =                                                           \
              soundml_resample_dot_##SUFFIX(scratch + s, bank + (p * taps),  \
                                            taps);                           \
          p += m;                                                            \
          s += p / l;                                                        \
          p %= l;                                                            \
        }                                                                    \
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
                                     value v_m, value v_row0, value v_s0,
                                     value v_y_off, value v_y_stride,
                                     value v_visit, value v_is_flush) {
  const int64_t n = Long_val(v_n);
  const int64_t n_out = Long_val(v_n_out);
  const int64_t channels = Long_val(v_channels);
  const int64_t k = Long_val(v_k);
  const int64_t l = Long_val(v_l);
  const int64_t m = Long_val(v_m);
  const int64_t row0 = Long_val(v_row0);
  const int64_t s0 = Long_val(v_s0);
  const int64_t y_off = Long_val(v_y_off);
  const int64_t y_stride = Long_val(v_y_stride);
  const int visit = Bool_val(v_visit);
  const int is_flush = Bool_val(v_is_flush);
  const int64_t taps = (2 * k) + 1, hlen = 2 * k;

  if (n < 0 || n_out < 0 || channels < 1 || k < 0 || l < 1 || m < 1 ||
      row0 < 0 || row0 >= l || s0 < 0 || y_off < 0 || y_stride < n_out)
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
      (n_out > 0 && soundml_resample_dim(v_y) <
                        y_off + ((channels - 1) * y_stride) + n_out))
    caml_failwith("soundml_resample: buffer extents disagree with geometry");
  if (n_out > 0) {
    /* the last window must fit inside the scratch lane */
    const int64_t s_last = s0 + ((((row0 * m) % l) + ((n_out - 1) * m)) / l);
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
                             n_out, channels, k, l, m, row0, s0, y_off,
                             y_stride, visit, is_flush);
  else
    soundml_resample_run_f64((const double *)bank, (double *)hist,
                             (double *)scratch, (const double *)x, (double *)y,
                             n, n_out, channels, k, l, m, row0, s0, y_off,
                             y_stride, visit, is_flush);
  caml_acquire_runtime_system();
  return Val_unit;
}

/* One interleaved complex128 element, the layout Bigarray gives a complex64
   kind and the frequency path's only element type. */
typedef struct {
  double re, im;
} soundml_cx;

/* The plan spectrum is applied with the elementwise complex product, written
   out so the shaping reads the same arithmetic wherever it runs. */
static inline soundml_cx soundml_cx_mul(soundml_cx a, soundml_cx b) {
  soundml_cx r;
  r.re = (a.re * b.re) - (a.im * b.im);
  r.im = (a.re * b.im) + (a.im * b.re);
  return r;
}

/* `soundml_resample_shape` is the block identity between the two transforms,
   per line: from the length-N half spectrum `x` to the half grid the inverse
   transform of length W reads, against the plan spectrum `h`.

     ×L (interpolate): the zero-stuffed block's length-N*L spectrum is the
       periodic extension of the block's, so bin k reads full-grid bin k mod N
       and is multiplied by h[k]. W = N*L.
     ÷M (decimate): multiply on the half grid, then alias-fold the full grid
       onto W = N/M bins — every term kept, summed in ascending fold order.
       The 1/M fold weight lives in the plan spectrum.
     otherwise: the plain product on the half grid, W = N.

   The output line holds W/2 + 1 bins. Every line is computed from its own
   input line alone and in the same order, so a stack of lines and a single
   line agree bit for bit. */
static void soundml_resample_shape_run(const soundml_cx *restrict x,
                                       const soundml_cx *restrict h,
                                       soundml_cx *restrict y, int64_t lines,
                                       int64_t n, int64_t sl, int64_t sm,
                                       int64_t w) {
  const int64_t bins = (n / 2) + 1, obins = (w / 2) + 1, half = n / 2;
  for (int64_t line = 0; line < lines; line++) {
    const soundml_cx *xs = x + (line * bins);
    soundml_cx *ys = y + (line * obins);
    if (sl > 1) {
      int64_t k = 0;
      while (k < obins) {
        for (int64_t j = 0; j <= half && k < obins; j++, k++)
          ys[k] = soundml_cx_mul(xs[j], h[k]);
        for (int64_t j = half + 1; j < n && k < obins; j++, k++) {
          soundml_cx z = xs[n - j];
          z.im = -z.im;
          ys[k] = soundml_cx_mul(z, h[k]);
        }
      }
    } else if (sm > 1) {
      for (int64_t k = 0; k < obins; k++) {
        /* the r = 0 term sits on the kept half grid, W/2 <= N/2 */
        int64_t j = k;
        soundml_cx acc = soundml_cx_mul(xs[k], h[k]);
        for (int64_t r = 1; r < sm; r++) {
          j += w;
          soundml_cx p;
          if (j <= half) {
            p = soundml_cx_mul(xs[j], h[j]);
          } else {
            p = soundml_cx_mul(xs[n - j], h[n - j]);
            p.im = -p.im;
          }
          acc.re += p.re;
          acc.im += p.im;
        }
        ys[k] = acc;
      }
    } else {
      for (int64_t k = 0; k < obins; k++) ys[k] = soundml_cx_mul(xs[k], h[k]);
    }
  }
}

/* Every extent is validated against the arrays the OCaml side handed over
   before any pointer is formed from them. */
CAMLprim value soundml_resample_shape(value v_x, value v_h, value v_y,
                                      value v_lines, value v_n, value v_sl,
                                      value v_sm) {
  const int64_t lines = Long_val(v_lines);
  const int64_t n = Long_val(v_n);
  const int64_t sl = Long_val(v_sl);
  const int64_t sm = Long_val(v_sm);

  if (lines < 0 || n < 2 || (n % 2) != 0 || sl < 1 || sm < 1 ||
      (sl > 1 && sm > 1))
    caml_failwith("soundml_resample_shape: invalid geometry");
  const int64_t w = sl > 1 ? n * sl : (sm > 1 ? n / sm : n);
  if (w < 2 || (sm > 1 && (n % sm) != 0))
    caml_failwith("soundml_resample_shape: invalid geometry");
  if (soundml_resample_kind(v_x) != CAML_BA_COMPLEX64 ||
      soundml_resample_kind(v_h) != CAML_BA_COMPLEX64 ||
      soundml_resample_kind(v_y) != CAML_BA_COMPLEX64)
    caml_failwith("soundml_resample_shape: unsupported dtype");
  const int64_t bins = (n / 2) + 1, obins = (w / 2) + 1;
  if (soundml_resample_dim(v_x) < lines * bins ||
      soundml_resample_dim(v_h) < (sl > 1 ? obins : bins) ||
      soundml_resample_dim(v_y) < lines * obins)
    caml_failwith("soundml_resample_shape: buffer extents disagree");

  const soundml_cx *x = (const soundml_cx *)Caml_ba_data_val(v_x);
  const soundml_cx *h = (const soundml_cx *)Caml_ba_data_val(v_h);
  soundml_cx *y = (soundml_cx *)Caml_ba_data_val(v_y);

  caml_release_runtime_system();
  soundml_resample_shape_run(x, h, y, lines, n, sl, sm, w);
  caml_acquire_runtime_system();
  return Val_unit;
}

CAMLprim value soundml_resample_shape_bc(value *argv, int argn) {
  (void)argn;
  return soundml_resample_shape(argv[0], argv[1], argv[2], argv[3], argv[4],
                                argv[5], argv[6]);
}

CAMLprim value soundml_resample_step_bc(value *argv, int argn) {
  (void)argn;
  return soundml_resample_step(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5], argv[6], argv[7], argv[8], argv[9],
                               argv[10], argv[11], argv[12], argv[13],
                               argv[14], argv[15], argv[16]);
}
