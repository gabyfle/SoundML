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

/* ordered_stubs.c — the fixed-association kernels behind Ordered.

   Every kernel walks a [outer; k; n] view of a C-order float64 operand:
   [outer] planes of [k] rows of [n] columns. Each output accumulates its
   terms in ascending row order, starting from 0.0, and the column loop is
   innermost, so the compiler may vectorise across columns — each lane is a
   different output, its own sum untouched — without reordering any sum. The
   columns are walked in tiles so that the rows a tile reads stay in cache
   across the outputs that reuse them; tiling changes which outputs are
   computed together, never the order of any output's terms.

   The row loops are unrolled by four, so an output is loaded and stored once
   per four terms: [(((d + a) + b) + c) + e] is evaluated left to right, the
   same sequence of additions as four separate steps, and C may not
   reassociate it.

   This file is compiled with -ffp-contract=off: a multiply feeding an add is
   never fused, so the bits are the same on every machine and equal to the
   plain OCaml fold of each output. */

#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

/* Columns per tile: [k] rows of a tile stay resident while every output row
   of the tile reads them. */
#define TILE 256

static inline intnat tile_end(intnat t0, intnat n) {
  return t0 + TILE < n ? t0 + TILE : n;
}

/* soundml_ordered_sum src out dims: out[o][t] = 0 + src[o][0][t] + ... +
   src[o][k-1][t], with dims = [| outer; k; n |]. */
value soundml_ordered_sum(value v_src, value v_out, value v_dims) {
  CAMLparam3(v_src, v_out, v_dims);
  const double *src = (const double *)Caml_ba_data_val(v_src);
  double *out = (double *)Caml_ba_data_val(v_out);
  intnat outer = Long_val(Field(v_dims, 0));
  intnat k = Long_val(Field(v_dims, 1));
  intnat n = Long_val(Field(v_dims, 2));
  for (intnat o = 0; o < outer; o++) {
    const double *plane = src + o * k * n;
    double *dst = out + o * n;
    for (intnat t0 = 0; t0 < n; t0 += TILE) {
      intnat t1 = tile_end(t0, n);
      for (intnat t = t0; t < t1; t++) dst[t] = 0.0;
      intnat j = 0;
      for (; j + 3 < k; j += 4) {
        const double *r0 = plane + j * n, *r1 = r0 + n, *r2 = r1 + n,
                     *r3 = r2 + n;
        for (intnat t = t0; t < t1; t++)
          dst[t] = (((dst[t] + r0[t]) + r1[t]) + r2[t]) + r3[t];
      }
      for (; j < k; j++) {
        const double *row = plane + j * n;
        for (intnat t = t0; t < t1; t++) dst[t] = dst[t] + row[t];
      }
    }
  }
  CAMLreturn(Val_unit);
}

/* soundml_ordered_cumsum src out dims: out[o][j][t] = out[o][j-1][t] +
   src[o][j][t], from 0, with dims = [| outer; k; n |]. */
value soundml_ordered_cumsum(value v_src, value v_out, value v_dims) {
  CAMLparam3(v_src, v_out, v_dims);
  const double *src = (const double *)Caml_ba_data_val(v_src);
  double *out = (double *)Caml_ba_data_val(v_out);
  intnat outer = Long_val(Field(v_dims, 0));
  intnat k = Long_val(Field(v_dims, 1));
  intnat n = Long_val(Field(v_dims, 2));
  for (intnat o = 0; o < outer; o++) {
    const double *plane = src + o * k * n;
    double *dst = out + o * k * n;
    for (intnat t0 = 0; t0 < n; t0 += TILE) {
      intnat t1 = tile_end(t0, n);
      if (k > 0)
        for (intnat t = t0; t < t1; t++) dst[t] = 0.0 + plane[t];
      for (intnat j = 1; j < k; j++) {
        const double *row = plane + j * n;
        const double *prev = dst + (j - 1) * n;
        double *cur = dst + j * n;
        for (intnat t = t0; t < t1; t++) cur[t] = prev[t] + row[t];
      }
    }
  }
  CAMLreturn(Val_unit);
}

/* [fold_row dst w plane j0 j1 n t0 t1] adds w[j] * plane[j][t] to dst[t] for
   j ascending in [j0, j1), four terms per load and store of dst. */
static inline void fold_row(double *dst, const double *w, const double *plane,
                            intnat j0, intnat j1, intnat n, intnat t0,
                            intnat t1) {
  intnat j = j0;
  for (; j + 3 < j1; j += 4) {
    const double w0 = w[j], w1 = w[j + 1], w2 = w[j + 2], w3 = w[j + 3];
    const double *r0 = plane + j * n, *r1 = r0 + n, *r2 = r1 + n, *r3 = r2 + n;
    for (intnat t = t0; t < t1; t++)
      dst[t] = (((dst[t] + (w0 * r0[t])) + (w1 * r1[t])) + (w2 * r2[t]))
               + (w3 * r3[t]);
  }
  for (; j < j1; j++) {
    const double wj = w[j];
    const double *row = plane + j * n;
    for (intnat t = t0; t < t1; t++) dst[t] = dst[t] + (wj * row[t]);
  }
}

/* soundml_ordered_project src out weights bounds dims: out[o][i][t] = 0 +
   sum over j in [first_i, last_i) of weights[i][j] * src[o][j][t], in
   ascending j, with bounds = [| first_0; last_0; ...; first_{m-1};
   last_{m-1} |] and dims = [| outer; m; k; n |]. */
value soundml_ordered_project(value v_src, value v_out, value v_weights,
                              value v_bounds, value v_dims) {
  CAMLparam5(v_src, v_out, v_weights, v_bounds, v_dims);
  const double *src = (const double *)Caml_ba_data_val(v_src);
  double *out = (double *)Caml_ba_data_val(v_out);
  const double *weights = (const double *)Caml_ba_data_val(v_weights);
  intnat outer = Long_val(Field(v_dims, 0));
  intnat m = Long_val(Field(v_dims, 1));
  intnat k = Long_val(Field(v_dims, 2));
  intnat n = Long_val(Field(v_dims, 3));
  for (intnat o = 0; o < outer; o++) {
    const double *plane = src + o * k * n;
    for (intnat t0 = 0; t0 < n; t0 += TILE) {
      intnat t1 = tile_end(t0, n);
      for (intnat i = 0; i < m; i++) {
        double *dst = out + (o * m + i) * n;
        for (intnat t = t0; t < t1; t++) dst[t] = 0.0;
        fold_row(dst, weights + i * k, plane, Long_val(Field(v_bounds, 2 * i)),
                 Long_val(Field(v_bounds, (2 * i) + 1)), n, t0, t1);
      }
    }
  }
  CAMLreturn(Val_unit);
}
