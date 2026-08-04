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

/* Minimal libsndfile ceiling benchmark — the measured ceiling the io
 * performance gates are derived against. Rebuild and re-run it on any new
 * machine before enforcing the gate multipliers there (bench/README.md):
 * every multiplier is a ratio against this harness's numbers on the same
 * corpus, never an absolute.
 *
 * Build: cc -O2 -o bench_sndfile bench_sndfile.c \
 *          $(pkg-config --cflags --libs sndfile)
 * Driver: bench/run_io_ceiling.sh (runs every mode over the io corpus).
 *
 * Modes:
 *   read  <path> <N>
 *       Ceiling read: buffer preallocated once (untimed); each timed
 *       iteration = sf_open + sf_readf_float(whole file) + sf_close.
 *   read_alloc <path> <N>
 *       Same but malloc/free of the destination buffer inside the timed
 *       region (isolates allocation cost vs the pure ceiling).
 *   write <srcpath> <dstpath> <fmt> <N>
 *       Load src fully (untimed), then each timed iteration =
 *       sf_open(dst,WRITE) + sf_writef_float in 65536-frame chunks +
 *       sf_close.  Chunked because libsndfile 1.2.2 segfaults when a
 *       single sf_writef_* call hands the Vorbis encoder more than
 *       roughly 2M frames; chunking is also the realistic production
 *       write pattern.  fmt in {wav16, wav24, wav32, wavf, flac, ogg}.
 *   many <listfile> <R>
 *       Sequential load of every path in listfile (one per line) with a
 *       single reusable preallocated buffer; R repetitions, min total.
 *   open <path> <N>
 *       Per-file fixed overhead: each timed iteration = sf_open + sf_close
 *       only (header parse, no payload decode).
 *
 * Output: one CSV line per run on stdout:
 *   mode,path,n,frames,channels,samplerate,min_ns,median_ns
 */
#include <sndfile.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

static void stats(uint64_t *t, int n, uint64_t *mn, uint64_t *md) {
  qsort(t, (size_t)n, sizeof *t, cmp_u64);
  *mn = t[0];
  *md = t[n / 2];
}

static float *alloc_buf(sf_count_t frames, int channels) {
  if (frames <= 0 || channels <= 0 || channels > 64) {
    fprintf(stderr, "bad frames/channels\n");
    exit(2);
  }
  uint64_t items = (uint64_t)frames * (uint64_t)channels;
  if (items > SIZE_MAX / sizeof(float)) {
    fprintf(stderr, "overflow\n");
    exit(2);
  }
  float *buf = malloc((size_t)items * sizeof(float));
  if (!buf) {
    fprintf(stderr, "oom\n");
    exit(2);
  }
  return buf;
}

static sf_count_t read_once(const char *path, float *buf, sf_count_t frames) {
  SF_INFO info;
  memset(&info, 0, sizeof info);
  SNDFILE *f = sf_open(path, SFM_READ, &info);
  if (!f) {
    fprintf(stderr, "open failed: %s: %s\n", path, sf_strerror(NULL));
    exit(2);
  }
  sf_count_t got = sf_readf_float(f, buf, frames);
  sf_close(f);
  return got;
}

static int mode_read(const char *path, int n, int alloc_inside) {
  SF_INFO info;
  memset(&info, 0, sizeof info);
  SNDFILE *f = sf_open(path, SFM_READ, &info);
  if (!f) {
    fprintf(stderr, "open failed: %s: %s\n", path, sf_strerror(NULL));
    return 2;
  }
  sf_count_t frames = info.frames;
  int channels = info.channels, sr = info.samplerate;
  sf_close(f);

  float *buf = alloc_inside ? NULL : alloc_buf(frames, channels);
  /* warm-up (also warms the page cache) */
  {
    float *w = alloc_inside ? alloc_buf(frames, channels) : buf;
    read_once(path, w, frames);
    if (alloc_inside) free(w);
  }
  uint64_t *t = malloc((size_t)n * sizeof *t);
  if (!t) return 2;
  for (int i = 0; i < n; i++) {
    uint64_t t0 = now_ns();
    float *b = alloc_inside ? alloc_buf(frames, channels) : buf;
    sf_count_t got = read_once(path, b, frames);
    if (alloc_inside) free(b);
    uint64_t t1 = now_ns();
    if (got != frames) {
      fprintf(stderr, "short read: %lld/%lld %s\n", (long long)got,
              (long long)frames, path);
      return 2;
    }
    t[i] = t1 - t0;
  }
  uint64_t mn, md;
  stats(t, n, &mn, &md);
  printf("%s,%s,%d,%lld,%d,%d,%llu,%llu\n",
         alloc_inside ? "read_alloc" : "read", path, n, (long long)frames,
         channels, sr, (unsigned long long)mn, (unsigned long long)md);
  free(t);
  if (!alloc_inside) free(buf);
  return 0;
}

static int mode_open(const char *path, int n) {
  SF_INFO info;
  memset(&info, 0, sizeof info);
  /* warm-up */
  SNDFILE *f = sf_open(path, SFM_READ, &info);
  if (!f) {
    fprintf(stderr, "open failed: %s: %s\n", path, sf_strerror(NULL));
    return 2;
  }
  sf_count_t frames = info.frames;
  int channels = info.channels, sr = info.samplerate;
  sf_close(f);
  uint64_t *t = malloc((size_t)n * sizeof *t);
  if (!t) return 2;
  for (int i = 0; i < n; i++) {
    memset(&info, 0, sizeof info);
    uint64_t t0 = now_ns();
    f = sf_open(path, SFM_READ, &info);
    if (!f) {
      fprintf(stderr, "open failed: %s\n", path);
      return 2;
    }
    sf_close(f);
    uint64_t t1 = now_ns();
    t[i] = t1 - t0;
  }
  uint64_t mn, md;
  stats(t, n, &mn, &md);
  printf("open,%s,%d,%lld,%d,%d,%llu,%llu\n", path, n, (long long)frames,
         channels, sr, (unsigned long long)mn, (unsigned long long)md);
  free(t);
  return 0;
}

static int fmt_code(const char *s) {
  if (!strcmp(s, "wav16")) return SF_FORMAT_WAV | SF_FORMAT_PCM_16;
  if (!strcmp(s, "wav24")) return SF_FORMAT_WAV | SF_FORMAT_PCM_24;
  if (!strcmp(s, "wav32")) return SF_FORMAT_WAV | SF_FORMAT_PCM_32;
  if (!strcmp(s, "wavf")) return SF_FORMAT_WAV | SF_FORMAT_FLOAT;
  if (!strcmp(s, "flac")) return SF_FORMAT_FLAC | SF_FORMAT_PCM_16;
  if (!strcmp(s, "ogg")) return SF_FORMAT_OGG | SF_FORMAT_VORBIS;
  fprintf(stderr, "unknown fmt %s\n", s);
  exit(2);
}

static int mode_write(const char *src, const char *dst, const char *fmt,
                      int n) {
  SF_INFO info;
  memset(&info, 0, sizeof info);
  SNDFILE *f = sf_open(src, SFM_READ, &info);
  if (!f) {
    fprintf(stderr, "open failed: %s: %s\n", src, sf_strerror(NULL));
    return 2;
  }
  sf_count_t frames = info.frames;
  int channels = info.channels, sr = info.samplerate;
  float *buf = alloc_buf(frames, channels);
  if (sf_readf_float(f, buf, frames) != frames) {
    fprintf(stderr, "short read on src\n");
    return 2;
  }
  sf_close(f);

  SF_INFO out;
  uint64_t *t = malloc((size_t)(n + 1) * sizeof *t);
  if (!t) return 2;
  for (int i = 0; i < n + 1; i++) { /* first run is warm-up, dropped */
    memset(&out, 0, sizeof out);
    out.samplerate = sr;
    out.channels = channels;
    out.format = fmt_code(fmt);
    uint64_t t0 = now_ns();
    SNDFILE *g = sf_open(dst, SFM_WRITE, &out);
    if (!g) {
      fprintf(stderr, "write open failed: %s: %s\n", dst, sf_strerror(NULL));
      return 2;
    }
    for (sf_count_t off = 0; off < frames;) {
      sf_count_t want = frames - off;
      if (want > 65536) want = 65536;
      if (sf_writef_float(g, buf + (size_t)off * (size_t)channels, want) !=
          want) {
        fprintf(stderr, "short write\n");
        return 2;
      }
      off += want;
    }
    sf_close(g);
    uint64_t t1 = now_ns();
    t[i] = t1 - t0;
  }
  unlink(dst);
  uint64_t mn, md;
  stats(t + 1, n, &mn, &md);
  printf("write,%s,%d,%lld,%d,%d,%llu,%llu\n", src, n, (long long)frames,
         channels, sr, (unsigned long long)mn, (unsigned long long)md);
  free(t);
  free(buf);
  return 0;
}

static int mode_many(const char *listfile, int reps) {
  FILE *lf = fopen(listfile, "r");
  if (!lf) {
    perror("listfile");
    return 2;
  }
  char line[4096];
  char **paths = NULL;
  int npaths = 0, cap = 0;
  sf_count_t max_items = 0;
  long long total_frames = 0;
  while (fgets(line, sizeof line, lf)) {
    size_t len = strlen(line);
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
      line[--len] = 0;
    if (!len) continue;
    if (npaths == cap) {
      cap = cap ? cap * 2 : 256;
      paths = realloc(paths, (size_t)cap * sizeof *paths);
    }
    paths[npaths++] = strdup(line);
  }
  fclose(lf);
  /* pre-scan for max buffer size; also warms page cache via warm-up pass */
  for (int i = 0; i < npaths; i++) {
    SF_INFO info;
    memset(&info, 0, sizeof info);
    SNDFILE *f = sf_open(paths[i], SFM_READ, &info);
    if (!f) {
      fprintf(stderr, "open failed: %s\n", paths[i]);
      return 2;
    }
    sf_count_t items = info.frames * info.channels;
    if (items > max_items) max_items = items;
    total_frames += info.frames;
    sf_close(f);
  }
  float *buf = alloc_buf(max_items, 1);
  /* warm-up pass */
  for (int i = 0; i < npaths; i++) {
    SF_INFO info;
    memset(&info, 0, sizeof info);
    SNDFILE *f = sf_open(paths[i], SFM_READ, &info);
    sf_readf_float(f, buf, info.frames);
    sf_close(f);
  }
  uint64_t *t = malloc((size_t)reps * sizeof *t);
  for (int r = 0; r < reps; r++) {
    uint64_t t0 = now_ns();
    for (int i = 0; i < npaths; i++) {
      SF_INFO info;
      memset(&info, 0, sizeof info);
      SNDFILE *f = sf_open(paths[i], SFM_READ, &info);
      if (!f) {
        fprintf(stderr, "open failed: %s\n", paths[i]);
        return 2;
      }
      if (sf_readf_float(f, buf, info.frames) != info.frames) {
        fprintf(stderr, "short read: %s\n", paths[i]);
        return 2;
      }
      sf_close(f);
    }
    uint64_t t1 = now_ns();
    t[r] = t1 - t0;
  }
  uint64_t mn, md;
  stats(t, reps, &mn, &md);
  printf("many,%s,%d,%lld,%d,0,%llu,%llu\n", listfile, npaths, total_frames, reps,
         (unsigned long long)mn, (unsigned long long)md);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s read|read_alloc|write|many ...\n", argv[0]);
    return 2;
  }
  if (!strcmp(argv[1], "read") && argc == 4)
    return mode_read(argv[2], atoi(argv[3]), 0);
  if (!strcmp(argv[1], "read_alloc") && argc == 4)
    return mode_read(argv[2], atoi(argv[3]), 1);
  if (!strcmp(argv[1], "write") && argc == 6)
    return mode_write(argv[2], argv[3], argv[4], atoi(argv[5]));
  if (!strcmp(argv[1], "many") && argc == 4)
    return mode_many(argv[2], atoi(argv[3]));
  if (!strcmp(argv[1], "open") && argc == 4)
    return mode_open(argv[2], atoi(argv[3]));
  fprintf(stderr, "bad args\n");
  return 2;
}
