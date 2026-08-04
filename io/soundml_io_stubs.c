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

/* soundml_io_stubs.c — the libsndfile seam behind Soundml_io.

   The stubs own every sf_* call and nothing else: format mapping, error
   typing and buffer bookkeeping live on the OCaml side. Failures are data,
   not exceptions — each stub returns counts and captured error state and the
   OCaml side turns them into the typed error; the only raises here are
   geometry guards that fire before the runtime lock is released and mean a
   bug in the OCaml bookkeeping, never a file condition.

   The runtime lock is released around all file I/O (open, read, write,
   close); every pointer (bigarray data, the copied path, the SNDFILE*) is
   extracted before the release, errno and sf_strerror are captured into C
   locals before the lock is reacquired, and no OCaml value is touched in
   between. No OCaml callback is reachable from any I/O path.

   Decode is single-copy (the design's §3 account): a mono destination is
   decoded into directly, 65536 frames per sf_readf call; a multichannel
   destination goes through one cache-resident staging block —
   clamp(262144 / (channels * elt_size), 4096, 65536) frames, at most 512 KB
   — whose deinterleave (or downmix) pass is the only full-size copy beyond
   libsndfile's own decode write. Writes mirror the same staging for the
   planar-to-interleaved pass and never hand libsndfile more than 65536
   frames per sf_writef call, unconditionally: a single multi-megaframe
   Ogg/Vorbis write segfaults libsndfile 1.2.2 (measured threshold 2.2 M
   frames), and the cap is a structural loop bound, not an advisory.

   Size arithmetic is overflow-checked with __builtin_mul_overflow /
   __builtin_add_overflow at every site; staging is freed on every path. */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sndfile.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

/* Frames libsndfile is handed per sf_readf/sf_writef call, unconditionally
   (the Vorbis write segfault cap; symmetric on reads so no call ever hands
   libsndfile an unbounded count). */
#define SOUNDML_IO_CALL_FRAMES ((int64_t) 65536)

/* Staging block budget in bytes: L2-resident on every current target. */
#define SOUNDML_IO_STAGING_BYTES ((int64_t) 262144)

/* Decode/encode layout modes. Kept in sync with the OCaml side
   (soundml_io.ml). */
#define SOUNDML_IO_MODE_DIRECT 0
#define SOUNDML_IO_MODE_PLANAR 1
#define SOUNDML_IO_MODE_DOWNMIX 2

/* Filesystem classification of a failed open, probed with open(2) so the
   OCaml side never interprets raw errno values. Kept in sync with
   soundml_io.ml. */
#define SOUNDML_IO_FS_NONE 0
#define SOUNDML_IO_FS_NOENT 1
#define SOUNDML_IO_FS_ACCES 2
#define SOUNDML_IO_FS_NOTDIR 3
#define SOUNDML_IO_FS_OTHER 4
#define SOUNDML_IO_FS_REFUSED 5 /* sf_format_check refusal, no I/O attempted */

/* Stub-level failure reported in the error slot of a read/write result when
   the failure is the stub's own (staging allocation), not libsndfile's. */
#define SOUNDML_IO_ERR_STAGING (-1)

static int soundml_io_classify_errno(int err) {
  switch (err) {
    case ENOENT:
      return SOUNDML_IO_FS_NOENT;
    case EACCES:
      return SOUNDML_IO_FS_ACCES;
    case ENOTDIR:
      return SOUNDML_IO_FS_NOTDIR;
    default:
      return SOUNDML_IO_FS_OTHER;
  }
}

/* {1 The handle}

   A custom block owning the SNDFILE*. Closed handles hold NULL; the
   finalizer is a GC backstop for leaked handles only — the OCaml side closes
   eagerly and checks the closed state before every call. */

typedef struct {
  SNDFILE *file;
} soundml_io_handle;

#define Handle_val(v) ((soundml_io_handle *) Data_custom_val(v))

static void soundml_io_handle_finalize(value v) {
  soundml_io_handle *h = Handle_val(v);
  if (h->file != NULL) {
    sf_close(h->file);
    h->file = NULL;
  }
}

static struct custom_operations soundml_io_handle_ops = {
    "soundml.io.handle",        soundml_io_handle_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

static SNDFILE *soundml_io_file(value v_handle) {
  SNDFILE *file = Handle_val(v_handle)->file;
  if (file == NULL) caml_failwith("soundml_io: handle used after close");
  return file;
}

/* [Error (fs, sf_err, details)] payload. */
static value soundml_io_error(int fs, int sf_err, const char *details) {
  CAMLparam0();
  CAMLlocal2(v_payload, v_res);
  v_payload = caml_alloc_tuple(3);
  Store_field(v_payload, 0, Val_int(fs));
  Store_field(v_payload, 1, Val_int(sf_err));
  Store_field(v_payload, 2, caml_copy_string(details));
  v_res = caml_alloc(1, 1);
  Store_field(v_res, 0, v_payload);
  CAMLreturn(v_res);
}

/* [(count, err, details)] outcome of a read/write loop. */
static value soundml_io_outcome(int64_t count, int err, const char *details) {
  CAMLparam0();
  CAMLlocal1(v_res);
  v_res = caml_alloc_tuple(3);
  Store_field(v_res, 0, Val_long((intnat) count));
  Store_field(v_res, 1, Val_int(err));
  Store_field(v_res, 2, caml_copy_string(details));
  CAMLreturn(v_res);
}

/* {1 Opening} */

CAMLprim value soundml_io_open_read(value v_path) {
  CAMLparam1(v_path);
  CAMLlocal3(v_res, v_payload, v_handle);
  char *path = caml_stat_strdup(String_val(v_path));
  SF_INFO info;
  memset(&info, 0, sizeof info);
  SNDFILE *file;
  int sf_err = 0, fs = SOUNDML_IO_FS_NONE;
  char details[256] = {0};

  caml_release_runtime_system();
  file = sf_open(path, SFM_READ, &info);
  if (file == NULL) {
    sf_err = sf_error(NULL);
    snprintf(details, sizeof details, "%s", sf_strerror(NULL));
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0)
      close(fd);
    else
      fs = soundml_io_classify_errno(errno);
  }
  caml_acquire_runtime_system();
  caml_stat_free(path);

  if (file == NULL) CAMLreturn(soundml_io_error(fs, sf_err, details));

  /* sf_count_t is int64; saturate the (malformed-header) tail that a 63-bit
     OCaml int cannot carry — the OCaml allocation guard rejects any such
     claim against the file's byte size anyway. */
  int64_t frames = (int64_t) info.frames;
  if (frames < 0) frames = 0;
  if (frames > (((int64_t) 1 << 62) - 1)) frames = ((int64_t) 1 << 62) - 1;

  v_handle =
      caml_alloc_custom(&soundml_io_handle_ops, sizeof(soundml_io_handle), 0, 1);
  Handle_val(v_handle)->file = file;
  v_payload = caml_alloc_tuple(6);
  Store_field(v_payload, 0, v_handle);
  Store_field(v_payload, 1, Val_long((intnat) frames));
  Store_field(v_payload, 2, Val_long(info.channels));
  Store_field(v_payload, 3, Val_long(info.samplerate));
  Store_field(v_payload, 4, Val_long(info.format));
  Store_field(v_payload, 5, Val_bool(info.seekable));
  v_res = caml_alloc(1, 0);
  Store_field(v_res, 0, v_payload);
  CAMLreturn(v_res);
}

CAMLprim value soundml_io_open_write(value v_path, value v_format,
                                     value v_channels, value v_rate) {
  CAMLparam4(v_path, v_format, v_channels, v_rate);
  CAMLlocal2(v_res, v_handle);
  SF_INFO info;
  memset(&info, 0, sizeof info);
  info.samplerate = Int_val(v_rate);
  info.channels = Int_val(v_channels);
  info.format = Int_val(v_format);

  /* Encoder refusal is knowable without touching the filesystem. */
  if (!sf_format_check(&info))
    CAMLreturn(soundml_io_error(SOUNDML_IO_FS_REFUSED, 0, ""));

  char *path = caml_stat_strdup(String_val(v_path));
  SNDFILE *file;
  int sf_err = 0, fs = SOUNDML_IO_FS_NONE;
  char details[256] = {0};

  caml_release_runtime_system();
  file = sf_open(path, SFM_WRITE, &info);
  if (file == NULL) {
    sf_err = sf_error(NULL);
    snprintf(details, sizeof details, "%s", sf_strerror(NULL));
    /* O_EXCL so the probe never clobbers an existing file: if creation
       succeeds the filesystem was fine (remove the probe artifact), and
       EEXIST means the same. */
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd >= 0) {
      close(fd);
      unlink(path);
    } else if (errno != EEXIST) {
      fs = soundml_io_classify_errno(errno);
    }
  } else {
    /* Float data into a PCM subtype saturates at full scale rather than
       wrapping (python-soundfile's behavior, pinned by the clipping golden). */
    int subtype = info.format & SF_FORMAT_SUBMASK;
    if (subtype == SF_FORMAT_PCM_S8 || subtype == SF_FORMAT_PCM_U8 ||
        subtype == SF_FORMAT_PCM_16 || subtype == SF_FORMAT_PCM_24 ||
        subtype == SF_FORMAT_PCM_32)
      sf_command(file, SFC_SET_CLIPPING, NULL, SF_TRUE);
  }
  caml_acquire_runtime_system();
  caml_stat_free(path);

  if (file == NULL) CAMLreturn(soundml_io_error(fs, sf_err, details));

  v_handle =
      caml_alloc_custom(&soundml_io_handle_ops, sizeof(soundml_io_handle), 0, 1);
  Handle_val(v_handle)->file = file;
  v_res = caml_alloc(1, 0);
  Store_field(v_res, 0, v_handle);
  CAMLreturn(v_res);
}

/* {1 Geometry helpers} */

static int64_t soundml_io_block_frames(int64_t channels, size_t elt) {
  int64_t per_frame = channels * (int64_t) elt; /* channels <= 1024, no ovf */
  int64_t block = SOUNDML_IO_STAGING_BYTES / per_frame;
  if (block < 4096) block = 4096;
  if (block > SOUNDML_IO_CALL_FRAMES) block = SOUNDML_IO_CALL_FRAMES;
  return block;
}

static void *soundml_io_staging(int64_t block, int64_t channels, size_t elt) {
  int64_t elts, bytes;
  if (__builtin_mul_overflow(block, channels, &elts) ||
      __builtin_mul_overflow(elts, (int64_t) elt, &bytes) ||
      bytes > (int64_t) (SIZE_MAX / 4))
    return NULL;
  return malloc((size_t) bytes);
}

/* {1 Decode kernels}

   One instantiation per sample type. The planar deinterleave walks the
   staging block once — one sequential read stream, [channels] sequential
   write cursors into the destination — so the destination's full-size write
   happens exactly once and the staging traffic never leaves cache. The
   downmix replaces the deinterleave with the per-frame channel mean, summed
   in channel order in the element type and multiplied by 1/channels. */
#define SOUNDML_IO_READ_KERNEL(SUFFIX, T, SF_READF)                           \
  static int64_t soundml_io_read_direct_##SUFFIX(SNDFILE *file, T *dst,       \
                                                 int64_t frames) {            \
    int64_t done = 0;                                                         \
    while (done < frames) {                                                   \
      int64_t want = frames - done;                                           \
      if (want > SOUNDML_IO_CALL_FRAMES) want = SOUNDML_IO_CALL_FRAMES;       \
      int64_t got = SF_READF(file, dst + done, want);                         \
      if (got <= 0) break;                                                    \
      done += got;                                                            \
      if (got < want) break;                                                  \
    }                                                                         \
    return done;                                                              \
  }                                                                           \
  static int64_t soundml_io_read_planar_##SUFFIX(                             \
      SNDFILE *file, T *restrict dst, int64_t dst_off, int64_t dst_total,     \
      int64_t frames, int64_t channels, int mode, T *restrict staging,        \
      int64_t block) {                                                        \
    int64_t done = 0;                                                         \
    const T inv = (T) 1 / (T) channels;                                       \
    while (done < frames) {                                                   \
      int64_t want = frames - done;                                           \
      if (want > block) want = block;                                         \
      int64_t got = SF_READF(file, staging, want);                            \
      if (got <= 0) break;                                                    \
      if (mode == SOUNDML_IO_MODE_PLANAR) {                                   \
        T *out = dst + dst_off + done;                                        \
        for (int64_t i = 0; i < got; i++) {                                   \
          const T *fr = staging + (i * channels);                             \
          for (int64_t c = 0; c < channels; c++)                              \
            out[(c * dst_total) + i] = fr[c];                                 \
        }                                                                     \
      } else {                                                                \
        T *out = dst + dst_off + done;                                        \
        for (int64_t i = 0; i < got; i++) {                                   \
          const T *fr = staging + (i * channels);                             \
          T acc = (T) 0;                                                      \
          for (int64_t c = 0; c < channels; c++) acc += fr[c];                \
          out[i] = acc * inv;                                                 \
        }                                                                     \
      }                                                                       \
      done += got;                                                            \
      if (got < want) break;                                                  \
    }                                                                         \
    return done;                                                              \
  }                                                                           \
  static int64_t soundml_io_write_direct_##SUFFIX(SNDFILE *file, const T *src, \
                                                  int64_t frames) {           \
    int64_t done = 0;                                                         \
    while (done < frames) {                                                   \
      int64_t want = frames - done;                                           \
      if (want > SOUNDML_IO_CALL_FRAMES) want = SOUNDML_IO_CALL_FRAMES;       \
      int64_t put = SF_WRITEF_##SUFFIX(file, src + done, want);               \
      if (put <= 0) break;                                                    \
      done += put;                                                            \
      if (put < want) break;                                                  \
    }                                                                         \
    return done;                                                              \
  }                                                                           \
  static int64_t soundml_io_write_planar_##SUFFIX(                            \
      SNDFILE *file, const T *restrict src, int64_t src_off,                  \
      int64_t src_total, int64_t frames, int64_t channels,                    \
      T *restrict staging, int64_t block) {                                   \
    int64_t done = 0;                                                         \
    while (done < frames) {                                                   \
      int64_t want = frames - done;                                           \
      if (want > block) want = block;                                         \
      const T *in = src + src_off + done;                                     \
      for (int64_t i = 0; i < want; i++) {                                    \
        T *fr = staging + (i * channels);                                     \
        for (int64_t c = 0; c < channels; c++) fr[c] = in[(c * src_total) + i]; \
      }                                                                       \
      int64_t put = SF_WRITEF_##SUFFIX(file, staging, want);                  \
      if (put <= 0) break;                                                    \
      done += put;                                                            \
      if (put < want) break;                                                  \
    }                                                                         \
    return done;                                                              \
  }

#define SF_WRITEF_f32 sf_writef_float
#define SF_WRITEF_f64 sf_writef_double

SOUNDML_IO_READ_KERNEL(f32, float, sf_readf_float)
SOUNDML_IO_READ_KERNEL(f64, double, sf_readf_double)

/* {1 Read and write}

   Both stubs share the argument shape: the destination (source) bigarray is
   planar with channel stride [total] frames, the transfer covers [frames]
   frames starting at frame [off], and [mode] names the layout pass. Every
   extent is validated against the actual bigarray dimension, in 64-bit
   arithmetic, before any pointer is formed; a mismatch means a bug in the
   OCaml bookkeeping, never silent out-of-bounds work. */

static void soundml_io_check_geometry(value v_ba, int mode, int64_t off,
                                      int64_t total, int64_t frames,
                                      int64_t channels) {
  if (frames < 0 || off < 0 || total < 0 || channels < 1)
    caml_failwith("soundml_io: invalid geometry");
  if (mode != SOUNDML_IO_MODE_DIRECT && mode != SOUNDML_IO_MODE_PLANAR &&
      mode != SOUNDML_IO_MODE_DOWNMIX)
    caml_failwith("soundml_io: invalid mode");
  if (mode == SOUNDML_IO_MODE_DIRECT && channels != 1)
    caml_failwith("soundml_io: direct mode is single-channel");
  int64_t end, needed;
  int64_t width = (mode == SOUNDML_IO_MODE_PLANAR) ? channels : 1;
  if (__builtin_add_overflow(off, frames, &end) || end > total ||
      __builtin_mul_overflow(width, total, &needed))
    caml_failwith("soundml_io: buffer extents disagree with geometry");
  if ((int64_t) Caml_ba_array_val(v_ba)->dim[0] < needed)
    caml_failwith("soundml_io: buffer extents disagree with geometry");
  int kind = Caml_ba_array_val(v_ba)->flags & CAML_BA_KIND_MASK;
  if (kind != CAML_BA_FLOAT32 && kind != CAML_BA_FLOAT64)
    caml_failwith("soundml_io: unsupported dtype");
}

CAMLprim value soundml_io_readf(value v_handle, value v_ba, value v_mode,
                                value v_off, value v_total, value v_frames,
                                value v_channels) {
  CAMLparam2(v_handle, v_ba);
  const int mode = Int_val(v_mode);
  const int64_t off = Long_val(v_off);
  const int64_t total = Long_val(v_total);
  const int64_t frames = Long_val(v_frames);
  const int64_t channels = Long_val(v_channels);
  soundml_io_check_geometry(v_ba, mode, off, total, frames, channels);

  SNDFILE *file = soundml_io_file(v_handle);
  const int kind = Caml_ba_array_val(v_ba)->flags & CAML_BA_KIND_MASK;
  const size_t elt = (kind == CAML_BA_FLOAT32) ? sizeof(float) : sizeof(double);
  void *data = Caml_ba_data_val(v_ba);
  void *staging = NULL;
  int64_t block = 0;
  if (mode != SOUNDML_IO_MODE_DIRECT) {
    block = soundml_io_block_frames(channels, elt);
    staging = soundml_io_staging(block, channels, elt);
    if (staging == NULL)
      CAMLreturn(soundml_io_outcome(0, SOUNDML_IO_ERR_STAGING,
                                    "staging allocation failed"));
  }

  int64_t done;
  int err;
  char details[256] = {0};

  caml_release_runtime_system();
  if (kind == CAML_BA_FLOAT32) {
    float *dst = (float *) data;
    if (mode == SOUNDML_IO_MODE_DIRECT)
      done = soundml_io_read_direct_f32(file, dst + off, frames);
    else
      done = soundml_io_read_planar_f32(file, dst, off, total, frames, channels,
                                        mode, (float *) staging, block);
  } else {
    double *dst = (double *) data;
    if (mode == SOUNDML_IO_MODE_DIRECT)
      done = soundml_io_read_direct_f64(file, dst + off, frames);
    else
      done = soundml_io_read_planar_f64(file, dst, off, total, frames, channels,
                                        mode, (double *) staging, block);
  }
  err = sf_error(file);
  if (err != SF_ERR_NO_ERROR)
    snprintf(details, sizeof details, "%s", sf_strerror(file));
  caml_acquire_runtime_system();
  free(staging);

  CAMLreturn(soundml_io_outcome(done, err, details));
}

CAMLprim value soundml_io_readf_bc(value *argv, int argn) {
  (void) argn;
  return soundml_io_readf(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                          argv[6]);
}

CAMLprim value soundml_io_writef(value v_handle, value v_ba, value v_mode,
                                 value v_off, value v_total, value v_frames,
                                 value v_channels) {
  CAMLparam2(v_handle, v_ba);
  const int mode = Int_val(v_mode);
  const int64_t off = Long_val(v_off);
  const int64_t total = Long_val(v_total);
  const int64_t frames = Long_val(v_frames);
  const int64_t channels = Long_val(v_channels);
  if (mode == SOUNDML_IO_MODE_DOWNMIX)
    caml_failwith("soundml_io: invalid mode");
  soundml_io_check_geometry(v_ba, mode, off, total, frames, channels);

  SNDFILE *file = soundml_io_file(v_handle);
  const int kind = Caml_ba_array_val(v_ba)->flags & CAML_BA_KIND_MASK;
  const size_t elt = (kind == CAML_BA_FLOAT32) ? sizeof(float) : sizeof(double);
  void *data = Caml_ba_data_val(v_ba);
  void *staging = NULL;
  int64_t block = 0;
  if (mode == SOUNDML_IO_MODE_PLANAR) {
    block = soundml_io_block_frames(channels, elt);
    staging = soundml_io_staging(block, channels, elt);
    if (staging == NULL)
      CAMLreturn(soundml_io_outcome(0, SOUNDML_IO_ERR_STAGING,
                                    "staging allocation failed"));
  }

  int64_t done;
  int err;
  char details[256] = {0};

  caml_release_runtime_system();
  if (kind == CAML_BA_FLOAT32) {
    const float *src = (const float *) data;
    if (mode == SOUNDML_IO_MODE_DIRECT)
      done = soundml_io_write_direct_f32(file, src + off, frames);
    else
      done = soundml_io_write_planar_f32(file, src, off, total, frames,
                                         channels, (float *) staging, block);
  } else {
    const double *src = (const double *) data;
    if (mode == SOUNDML_IO_MODE_DIRECT)
      done = soundml_io_write_direct_f64(file, src + off, frames);
    else
      done = soundml_io_write_planar_f64(file, src, off, total, frames,
                                         channels, (double *) staging, block);
  }
  err = sf_error(file);
  if (err != SF_ERR_NO_ERROR)
    snprintf(details, sizeof details, "%s", sf_strerror(file));
  caml_acquire_runtime_system();
  free(staging);

  CAMLreturn(soundml_io_outcome(done, err, details));
}

CAMLprim value soundml_io_writef_bc(value *argv, int argn) {
  (void) argn;
  return soundml_io_writef(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                           argv[6]);
}

/* {1 Closing} */

CAMLprim value soundml_io_close(value v_handle) {
  CAMLparam1(v_handle);
  CAMLlocal1(v_res);
  soundml_io_handle *h = Handle_val(v_handle);
  SNDFILE *file = h->file;
  int err = 0;
  char details[256] = {0};
  if (file != NULL) {
    /* Marked closed before the lock is released: no concurrent stub can
       reach the SNDFILE* once the close is committed. */
    h->file = NULL;
    caml_release_runtime_system();
    err = sf_close(file);
    if (err != 0) {
      const char *msg = sf_error_number(err);
      snprintf(details, sizeof details, "%s", msg == NULL ? "" : msg);
    }
    caml_acquire_runtime_system();
  }
  v_res = caml_alloc_tuple(2);
  Store_field(v_res, 0, Val_int(err));
  Store_field(v_res, 1, caml_copy_string(details));
  CAMLreturn(v_res);
}

/* {1 Format queries} — pure table lookups, no I/O, no lock dance. */

CAMLprim value soundml_io_format_check(value v_format, value v_channels,
                                       value v_rate) {
  SF_INFO info;
  memset(&info, 0, sizeof info);
  info.samplerate = Int_val(v_rate);
  info.channels = Int_val(v_channels);
  info.format = Int_val(v_format);
  return Val_bool(sf_format_check(&info));
}

CAMLprim value soundml_io_format_name(value v_format) {
  CAMLparam1(v_format);
  const int format = Int_val(v_format);
  SF_FORMAT_INFO major, sub;
  char buf[192];
  memset(&major, 0, sizeof major);
  memset(&sub, 0, sizeof sub);
  major.format = format & SF_FORMAT_TYPEMASK;
  sub.format = format & SF_FORMAT_SUBMASK;
  int have_major =
      sf_command(NULL, SFC_GET_FORMAT_INFO, &major, sizeof major) == 0 &&
      major.name != NULL;
  int have_sub = sf_command(NULL, SFC_GET_FORMAT_INFO, &sub, sizeof sub) == 0 &&
                 sub.name != NULL;
  if (have_major && have_sub)
    snprintf(buf, sizeof buf, "%s, %s", major.name, sub.name);
  else if (have_major)
    snprintf(buf, sizeof buf, "%s", major.name);
  else
    snprintf(buf, sizeof buf, "format 0x%08x", (unsigned) format);
  CAMLreturn(caml_copy_string(buf));
}
