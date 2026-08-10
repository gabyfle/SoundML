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

   Decode is single-copy: a mono destination is decoded into directly, one
   sf_readf call over the whole extent the caller's destination already
   sizes; a multichannel destination goes through one staging block —
   clamp(262144 / (channels * elt_size), 4096, 65536) frames — whose
   deinterleave (or downmix) pass is the only full-size copy beyond
   libsndfile's own decode write. The block is <= 256 KB while the byte
   budget sets the count (up to 64 bytes per frame: 16 channels of float32,
   8 of float64); past that the 4096-frame floor dominates and the block
   grows linearly with the channel count, to 32 MB at libsndfile's
   1024-channel cap with float64 — allocation overflow- and failure-checked
   (SOUNDML_IO_ERR_STAGING), never assumed cache-resident. Writes mirror the
   same staging for the planar-to-interleaved pass and never hand libsndfile
   more than 65536 frames per sf_writef call, unconditionally: a single
   multi-megaframe Ogg/Vorbis write segfaults libsndfile 1.2.2 (measured
   threshold 2.2 M frames), and the cap is a structural loop bound, not an
   advisory.

   Size arithmetic is overflow-checked with __builtin_mul_overflow /
   __builtin_add_overflow at every site; staging is freed on every path.

   On arm64 macOS a bounded worker pool carries two overlaps. A whole-file
   decode of a seekable, exactly-seekable format is split into W <= 4
   contiguous frame extents: the workers take the leading ones through
   duplicate decoders — sf_open_virtual over one shared descriptor, each with
   its own read cursor and pread — and the calling handle takes the last, so
   it ends the call at EOF exactly as the sequential decode leaves it. A
   planar write interleaves the next block while libsndfile encodes the
   current one. Both are opportunistic: no worker, no descriptor, a format
   outside the whitelist, a chunked (non-whole-file) request, or any anomaly
   mid-flight and the call runs the sequential body over the full request, so
   every short-read, error and position semantic is the sequential code's.
   The pool owns threads and nothing else — at most min(3, ncpu-1) of them,
   created lazily and joined at exit; all staging stays per-call and is freed
   on every path, so parallel decoding adds no retained memory. */

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__GNUC__)
#include <arm_neon.h>
#define SOUNDML_IO_NEON_PAIR 1
#endif

#include <sndfile.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

/* Frames libsndfile is handed per sf_writef call, unconditionally (the
   Vorbis write segfault cap), and the ceiling of the staging block that
   bounds every staged call. A direct decode carries no such bound: it is one
   sf_readf over the destination the caller already sized. */
#define SOUNDML_IO_CALL_FRAMES ((int64_t) 65536)

/* Staging block byte budget. Sets the block's frame count only down to the
   4096-frame floor of soundml_io_block_frames — beyond 64 bytes per frame
   the floor wins and the block outgrows this budget (see the overview). */
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

/* libsndfile's failed-open error state — sf_error(NULL) / sf_strerror(NULL)
   — is one process-global that every sf_open writes (clearing it on success,
   setting it on failure). The runtime lock is released around opens, so two
   threads opening concurrently would race on it and cross-attribute
   failures: measured on libsndfile 1.2.2, a garbage file's failed open can
   report "No Error." or another thread's message. Every sf_open and its
   error capture therefore run under this one mutex; the critical section is
   bounded by the open itself (~20-250 us — header I/O), the errno-based
   filesystem probe is per-thread and stays outside, and decode/encode never
   touch the global, so reads and writes stay fully parallel. */
static pthread_mutex_t soundml_io_open_mutex = PTHREAD_MUTEX_INITIALIZER;

/* A plain malloc'd copy of an OCaml string: unlike caml_stat_strdup it stays
   valid with the runtime lock released and is freed with free(). NULL on
   allocation failure — every reader of it treats NULL as "not available". */
static char *soundml_io_strdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *copy = malloc(n);
  if (copy != NULL) memcpy(copy, s, n);
  return copy;
}

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

   A custom block owning one pointer, to the open file's state off the OCaml
   heap. Closed handles hold NULL; the finalizer is a GC backstop for leaked
   handles only — the OCaml side closes eagerly and checks the closed state
   before every call. Keeping the state out of line holds the custom block to
   a single word whatever the state grows to carry.

   Besides the SNDFILE*, a read handle records what a parallel decode needs to
   reach the same bytes through a second decoder: the file's path (this
   structure's own malloc'd copy, so a worker can use it with the runtime lock
   released), the frame count the header advertises, the format word and the
   seekable flag. A write handle holds NULL and zeroes. */

typedef struct {
  SNDFILE *file;
  char *path;
  int64_t frames;
  int format;
  int seekable;
} soundml_io_state;

typedef struct {
  soundml_io_state *state;
} soundml_io_handle;

#define Handle_val(v) ((soundml_io_handle *) Data_custom_val(v))
#define State_val(v) (Handle_val(v)->state)

static void soundml_io_state_free(soundml_io_state *s) {
  if (s == NULL) return;
  if (s->file != NULL) {
    sf_close(s->file);
    s->file = NULL;
  }
  free(s->path);
  free(s);
}

static void soundml_io_handle_finalize(value v) {
  soundml_io_handle *h = Handle_val(v);
  soundml_io_state_free(h->state);
  h->state = NULL;
}

/* The open file's state, or NULL when allocation fails; the caller closes the
   file and reports the failure. */
static soundml_io_state *soundml_io_state_alloc(SNDFILE *file, const char *path,
                                                int64_t frames, int format,
                                                int seekable) {
  soundml_io_state *s = malloc(sizeof *s);
  if (s == NULL) return NULL;
  s->file = file;
  s->path = (path == NULL) ? NULL : soundml_io_strdup(path);
  s->frames = frames;
  s->format = format;
  s->seekable = seekable;
  return s;
}

static struct custom_operations soundml_io_handle_ops = {
    "soundml.io.handle",        soundml_io_handle_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

static SNDFILE *soundml_io_file(value v_handle) {
  soundml_io_state *s = State_val(v_handle);
  if (s == NULL || s->file == NULL)
    caml_failwith("soundml_io: handle used after close");
  return s->file;
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
  pthread_mutex_lock(&soundml_io_open_mutex);
  file = sf_open(path, SFM_READ, &info);
  if (file == NULL) {
    sf_err = sf_error(NULL);
    snprintf(details, sizeof details, "%s", sf_strerror(NULL));
  }
  pthread_mutex_unlock(&soundml_io_open_mutex);
  if (file == NULL) {
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

  soundml_io_state *state = soundml_io_state_alloc(
      file, String_val(v_path), frames, info.format, info.seekable);
  if (state == NULL) {
    caml_release_runtime_system();
    sf_close(file);
    caml_acquire_runtime_system();
    CAMLreturn(soundml_io_error(SOUNDML_IO_FS_OTHER, 0, "out of memory"));
  }

  v_handle =
      caml_alloc_custom(&soundml_io_handle_ops, sizeof(soundml_io_handle), 0, 1);
  Handle_val(v_handle)->state = state;
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
  pthread_mutex_lock(&soundml_io_open_mutex);
  file = sf_open(path, SFM_WRITE, &info);
  if (file == NULL) {
    sf_err = sf_error(NULL);
    snprintf(details, sizeof details, "%s", sf_strerror(NULL));
  }
  pthread_mutex_unlock(&soundml_io_open_mutex);
  if (file == NULL) {
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

  soundml_io_state *state = soundml_io_state_alloc(file, NULL, 0, 0, 0);
  if (state == NULL) {
    caml_release_runtime_system();
    sf_close(file);
    caml_acquire_runtime_system();
    CAMLreturn(soundml_io_error(SOUNDML_IO_FS_OTHER, 0, "out of memory"));
  }

  v_handle =
      caml_alloc_custom(&soundml_io_handle_ops, sizeof(soundml_io_handle), 0, 1);
  Handle_val(v_handle)->state = state;
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

/* {1 The worker pool}

   Present on arm64 macOS only; everywhere else none of this compiles and the
   read stub is its sequential body alone. The split whole-file decode is the
   pool's only client: a write stages and encodes on the calling thread.

   The pool owns threads and nothing else. At most SOUNDML_IO_POOL_SLOTS of
   them exist — further bounded at first use by min(3, ncpu-1), zero on a
   single-core machine, which switches the feature off — each created on the
   first call that needs it and never destroyed until the atexit handler joins
   them. A worker holds no buffer between tasks: every staging block belongs
   to the call that submitted the task and is freed on that call's every exit
   path, so nothing the pool touches outlives the call and parallel work adds
   no retained memory. Workers never touch the OCaml runtime, allocate no
   OCaml value and run no OCaml callback; they call sf_* and memcpy-shaped
   kernels only.

   Slots are handed out by a non-blocking try-take under one mutex, so a call
   that finds the pool busy — including a second OCaml domain already
   decoding — runs its sequential body instead of queueing. The per-slot
   rendezvous is a sequence counter with a parked bit: a bounded yield spin
   covers the common handoff and a dispatch semaphore absorbs the rest, so a
   handoff costs well under a microsecond against tasks measured in tens.

   Every rendezvous word is accessed sequentially consistently. Each waiting
   side stores its parked bit and then re-reads the counter it waits on,
   while the waking side stores that counter and then reads the parked bit;
   only one total order over those four accesses rules out both sides reading
   the value from before the other's store, which is the shape that drops a
   wakeup and parks the pair forever. Weaker orderings order each side's own
   pair and leave that window open. A parked bit is only ever cleared with an
   exchange, so exactly one side observes it set: every park is answered by
   exactly one signal and the semaphore's count never drifts. */

#if defined(__APPLE__) && defined(__aarch64__)
#define SOUNDML_IO_PARALLEL 1
#endif

#ifdef SOUNDML_IO_PARALLEL

#include <dispatch/dispatch.h>
#include <stdatomic.h>

/* Hard ceiling on live workers, hence on the extra file descriptors and
   staging blocks a single call can hold: the split width never exceeds
   SOUNDML_IO_POOL_SLOTS + 1. */
#define SOUNDML_IO_POOL_SLOTS 3

/* Yield iterations before a rendezvous parks on its semaphore. */
#define SOUNDML_IO_SPIN 20000

typedef void (*soundml_io_task)(void *);

typedef struct {
  _Atomic uint64_t posted, done;
  _Atomic int parked_worker, parked_caller, shutdown;
  dispatch_semaphore_t sem_worker, sem_caller;
  pthread_t thread;
  int started;
  soundml_io_task run;
  void *arg;
} soundml_io_slot;

static struct {
  pthread_mutex_t lock;
  soundml_io_slot slot[SOUNDML_IO_POOL_SLOTS];
  int taken[SOUNDML_IO_POOL_SLOTS];
  int cap;   /* usable slots; 0 disables every parallel path */
  int width; /* forced split width from the environment, 0 = automatic */
} soundml_io_pool = {PTHREAD_MUTEX_INITIALIZER, {{0}}, {0}, 0, 0};

/* Wait for the slot's next task. Returns 1 with a task pending, 0 once the
   slot is shutting down and nothing is left to run. A pending task outranks
   shutdown, so every submitted task runs and publishes [done] whatever the
   teardown does: a caller in soundml_io_join is never left without an
   answer. Shutdown is re-read after the parked bit is stored for the same
   reason the posted counter is — a teardown that lands in between would
   otherwise never be observed and the atexit join would hang. */
static int soundml_io_worker_wait(soundml_io_slot *s, uint64_t seen) {
  int spin = 0;
  for (;;) {
    if (atomic_load_explicit(&s->posted, memory_order_seq_cst) != seen)
      return 1;
    if (atomic_load_explicit(&s->shutdown, memory_order_seq_cst)) return 0;
    if (++spin <= SOUNDML_IO_SPIN) {
      __asm__ volatile("yield");
      continue;
    }
    atomic_store_explicit(&s->parked_worker, 1, memory_order_seq_cst);
    if (atomic_load_explicit(&s->posted, memory_order_seq_cst) != seen ||
        atomic_load_explicit(&s->shutdown, memory_order_seq_cst)) {
      /* A waker that took the bit first owes this slot a signal: consume it
         rather than leave it to fall through the next park. */
      if (!atomic_exchange_explicit(&s->parked_worker, 0,
                                    memory_order_seq_cst))
        dispatch_semaphore_wait(s->sem_worker, DISPATCH_TIME_FOREVER);
    } else {
      dispatch_semaphore_wait(s->sem_worker, DISPATCH_TIME_FOREVER);
    }
    spin = 0;
  }
}

static void *soundml_io_worker(void *p) {
  soundml_io_slot *s = p;
  uint64_t seen = 0;
  while (soundml_io_worker_wait(s, seen)) {
    seen++;
    if (s->run != NULL) s->run(s->arg);
    atomic_store_explicit(&s->done, seen, memory_order_seq_cst);
    if (atomic_exchange_explicit(&s->parked_caller, 0, memory_order_seq_cst))
      dispatch_semaphore_signal(s->sem_caller);
  }
  return NULL;
}

static void soundml_io_pool_shutdown(void) {
  for (int i = 0; i < SOUNDML_IO_POOL_SLOTS; i++) {
    soundml_io_slot *s = &soundml_io_pool.slot[i];
    if (!s->started) continue;
    atomic_store_explicit(&s->shutdown, 1, memory_order_seq_cst);
    if (atomic_exchange_explicit(&s->parked_worker, 0, memory_order_seq_cst))
      dispatch_semaphore_signal(s->sem_worker);
    pthread_join(s->thread, NULL);
    s->started = 0;
  }
}

/* No thread but the caller survives fork(2), so the child inherits slots whose
   workers no longer exist. Zeroing the pool makes the child start over: no
   join of a dead thread, no slot marked taken forever. */
static void soundml_io_pool_atfork_child(void) {
  int cap = soundml_io_pool.cap, width = soundml_io_pool.width;
  memset(&soundml_io_pool, 0, sizeof soundml_io_pool);
  pthread_mutex_init(&soundml_io_pool.lock, NULL);
  soundml_io_pool.cap = cap;
  soundml_io_pool.width = width;
}

/* SOUNDML_IO_PARALLEL selects the library's parallel decode: unset or empty
   leaves the width to the work model, "0" switches every parallel path off,
   and "1".."4" pins the split width (still subject to the format and
   seekability conditions). */
static void soundml_io_pool_init(void) {
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  int cap = (ncpu > 1) ? (int) (ncpu - 1) : 0;
  if (cap > SOUNDML_IO_POOL_SLOTS) cap = SOUNDML_IO_POOL_SLOTS;
  const char *env = getenv("SOUNDML_IO_PARALLEL");
  int width = 0;
  if (env != NULL && env[0] != '\0') {
    if (env[0] == '0' && env[1] == '\0')
      cap = 0;
    else if (env[1] == '\0' && env[0] >= '1' && env[0] <= '4')
      width = env[0] - '0';
  }
  soundml_io_pool.cap = cap;
  soundml_io_pool.width = width;
  if (cap > 0) {
    atexit(soundml_io_pool_shutdown);
    pthread_atfork(NULL, NULL, soundml_io_pool_atfork_child);
  }
}

static pthread_once_t soundml_io_pool_once = PTHREAD_ONCE_INIT;

/* Take up to [want] idle slots, spawning threads on demand. Returns how many
   were taken and fills [out]; 0 means the caller runs sequentially. */
static int soundml_io_pool_acquire(int want, soundml_io_slot **out) {
  pthread_once(&soundml_io_pool_once, soundml_io_pool_init);
  if (want <= 0 || soundml_io_pool.cap == 0) return 0;
  int got = 0;
  pthread_mutex_lock(&soundml_io_pool.lock);
  for (int i = 0; i < soundml_io_pool.cap && got < want; i++) {
    soundml_io_slot *s = &soundml_io_pool.slot[i];
    if (soundml_io_pool.taken[i]) continue;
    if (!s->started) {
      if (s->sem_worker == NULL) s->sem_worker = dispatch_semaphore_create(0);
      if (s->sem_caller == NULL) s->sem_caller = dispatch_semaphore_create(0);
      if (s->sem_worker == NULL || s->sem_caller == NULL) break;
      if (pthread_create(&s->thread, NULL, soundml_io_worker, s) != 0) break;
      s->started = 1;
    }
    soundml_io_pool.taken[i] = 1;
    out[got++] = s;
  }
  pthread_mutex_unlock(&soundml_io_pool.lock);
  return got;
}

static void soundml_io_pool_release(soundml_io_slot **slots, int n) {
  pthread_mutex_lock(&soundml_io_pool.lock);
  for (int k = 0; k < n; k++) {
    for (int i = 0; i < SOUNDML_IO_POOL_SLOTS; i++)
      if (&soundml_io_pool.slot[i] == slots[k]) soundml_io_pool.taken[i] = 0;
  }
  pthread_mutex_unlock(&soundml_io_pool.lock);
}

static void soundml_io_submit(soundml_io_slot *s, soundml_io_task run,
                              void *arg) {
  s->run = run;
  s->arg = arg;
  atomic_fetch_add_explicit(&s->posted, 1, memory_order_seq_cst);
  if (atomic_exchange_explicit(&s->parked_worker, 0, memory_order_seq_cst))
    dispatch_semaphore_signal(s->sem_worker);
}

static void soundml_io_join(soundml_io_slot *s) {
  uint64_t want = atomic_load_explicit(&s->posted, memory_order_relaxed);
  int spin = 0;
  for (;;) {
    if (atomic_load_explicit(&s->done, memory_order_seq_cst) >= want) return;
    if (++spin <= SOUNDML_IO_SPIN) {
      __asm__ volatile("yield");
      continue;
    }
    atomic_store_explicit(&s->parked_caller, 1, memory_order_seq_cst);
    if (atomic_load_explicit(&s->done, memory_order_seq_cst) >= want) {
      if (!atomic_exchange_explicit(&s->parked_caller, 0,
                                    memory_order_seq_cst))
        dispatch_semaphore_wait(s->sem_caller, DISPATCH_TIME_FOREVER);
      return;
    }
    dispatch_semaphore_wait(s->sem_caller, DISPATCH_TIME_FOREVER);
    spin = 0;
  }
}

/* {1 Duplicate decoders}

   A split extent is decoded by a second SNDFILE over the same bytes. It is
   built with sf_open_virtual on one descriptor the calling thread opens for
   the call and every worker shares: each duplicate keeps its own cursor in
   this structure and reads with pread, which neither moves nor consults the
   descriptor's own offset, so duplicates are independent without a descriptor
   each. Opening this way costs a header parse rather than a second path
   resolution. Writing is refused — these handles are read-only by
   construction. */

typedef struct {
  int fd;
  sf_count_t pos, len;
} soundml_io_vfile;

static sf_count_t soundml_io_vio_len(void *user) {
  return ((soundml_io_vfile *) user)->len;
}

static sf_count_t soundml_io_vio_seek(sf_count_t offset, int whence,
                                      void *user) {
  soundml_io_vfile *v = user;
  switch (whence) {
    case SEEK_SET:
      v->pos = offset;
      break;
    case SEEK_CUR:
      v->pos += offset;
      break;
    case SEEK_END:
      v->pos = v->len + offset;
      break;
    default:
      break;
  }
  if (v->pos < 0) v->pos = 0;
  if (v->pos > v->len) v->pos = v->len;
  return v->pos;
}

static sf_count_t soundml_io_vio_read(void *ptr, sf_count_t count, void *user) {
  soundml_io_vfile *v = user;
  if (count <= 0) return 0;
  if (count > v->len - v->pos) count = v->len - v->pos;
  if (count <= 0) return 0;
  ssize_t got = pread(v->fd, ptr, (size_t) count, (off_t) v->pos);
  if (got <= 0) return 0;
  v->pos += got;
  return (sf_count_t) got;
}

static sf_count_t soundml_io_vio_write(const void *ptr, sf_count_t count,
                                       void *user) {
  (void) ptr;
  (void) count;
  (void) user;
  return 0;
}

static sf_count_t soundml_io_vio_tell(void *user) {
  return ((soundml_io_vfile *) user)->pos;
}

static SF_VIRTUAL_IO soundml_io_vio = {
    soundml_io_vio_len, soundml_io_vio_seek, soundml_io_vio_read,
    soundml_io_vio_write, soundml_io_vio_tell};

/* {1 The split work model}

   Splitting pays a descriptor, one header parse per duplicate and a pair of
   rendezvous; it earns a share of the decode. The estimate below prices the
   decode from the sample count and the subtype's measured cost, and buys one
   more extent per SOUNDML_IO_SPLIT_STEP_US of work above a floor that keeps
   short reads — every one-second row, every small file of an ingest loop —
   strictly sequential.

   Admission has two conditions and only uncompressed formats meet both. A
   duplicate must seek exactly, landing on the frame the sequential decode
   would be at rather than near it. And a split that is abandoned must be
   replayable: the calling handle runs the whole request again from frame
   zero, which a decoder that latches an error after reading into a truncated
   tail can no longer serve — it would report a shortfall the sequential path
   never had. Everything else decodes sequentially. */

#define SOUNDML_IO_SPLIT_MIN_US 180
#define SOUNDML_IO_SPLIT_STEP_US 130

/* Picoseconds of decode per sample, per subtype; 0 for a format the split
   does not admit. */
static int soundml_io_decode_cost(int format) {
  switch (format & SF_FORMAT_TYPEMASK) {
    case SF_FORMAT_WAV:
    case SF_FORMAT_WAVEX:
    case SF_FORMAT_AIFF:
    case SF_FORMAT_CAF:
      break;
    default:
      return 0;
  }
  switch (format & SF_FORMAT_SUBMASK) {
    case SF_FORMAT_PCM_16:
      return 450;
    case SF_FORMAT_PCM_24:
      return 670;
    case SF_FORMAT_PCM_32:
      return 600;
    case SF_FORMAT_FLOAT:
    case SF_FORMAT_DOUBLE:
      return 150;
    default:
      return 0;
  }
}

static int soundml_io_split_width(int format, int64_t frames,
                                  int64_t channels) {
  int cost = soundml_io_decode_cost(format);
  if (cost == 0) return 1;
  int64_t samples;
  if (__builtin_mul_overflow(frames, channels, &samples)) return 1;
  int64_t est = samples / 1000 * cost / 1000; /* microseconds */
  if (est < SOUNDML_IO_SPLIT_MIN_US) return 1;
  int64_t width = 1 + est / SOUNDML_IO_SPLIT_STEP_US;
  if (width > SOUNDML_IO_POOL_SLOTS + 1) width = SOUNDML_IO_POOL_SLOTS + 1;
  return (int) width;
}

#endif /* SOUNDML_IO_PARALLEL */

/* {1 The two-channel layout pass}

   Stereo carries the bulk of the layout traffic, so on arm64 it is spelled
   out: paired lane loads (ld2) split the staging block into its two channels
   and each channel cursor is written with a non-temporal paired store. The
   destination is written exactly once here and is not read again before the
   call returns, so keeping it out of the caches leaves the staging block —
   which the next decoder call overwrites and reads again — resident. STNP
   addresses Normal memory with no alignment requirement, so neither cursor
   has to be aligned; the tail below the unrolled width is stored plainly.
   Everywhere else the same movement as the plain loop, left to the
   vectorizer. */

#ifdef SOUNDML_IO_NEON_PAIR
#define SOUNDML_IO_DEINTERLEAVE2(SUFFIX, T, VEC2, VLD2, LANES)                \
  static void soundml_io_deinterleave2_##SUFFIX(                              \
      const T *restrict s, T *restrict o0, T *restrict o1, int64_t n) {       \
    int64_t i = 0;                                                            \
    for (; i + (2 * (LANES)) <= n; i += 2 * (LANES)) {                        \
      VEC2 a = VLD2(s + (2 * i));                                             \
      VEC2 b = VLD2(s + (2 * i) + (2 * (LANES)));                             \
      __asm__ volatile("stnp %q0, %q1, [%2]"                                  \
                       :                                                      \
                       : "w"(a.val[0]), "w"(b.val[0]), "r"(o0 + i)            \
                       : "memory");                                           \
      __asm__ volatile("stnp %q0, %q1, [%2]"                                  \
                       :                                                      \
                       : "w"(a.val[1]), "w"(b.val[1]), "r"(o1 + i)            \
                       : "memory");                                           \
    }                                                                         \
    for (; i < n; i++) {                                                      \
      o0[i] = s[2 * i];                                                       \
      o1[i] = s[(2 * i) + 1];                                                 \
    }                                                                         \
  }
#else
#define SOUNDML_IO_DEINTERLEAVE2(SUFFIX, T, VEC2, VLD2, LANES)                \
  static void soundml_io_deinterleave2_##SUFFIX(                              \
      const T *restrict s, T *restrict o0, T *restrict o1, int64_t n) {       \
    for (int64_t i = 0; i < n; i++) {                                         \
      o0[i] = s[2 * i];                                                       \
      o1[i] = s[(2 * i) + 1];                                                 \
    }                                                                         \
  }
#endif

SOUNDML_IO_DEINTERLEAVE2(f32, float, float32x4x2_t, vld2q_f32, 4)
SOUNDML_IO_DEINTERLEAVE2(f64, double, float64x2x2_t, vld2q_f64, 2)

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
    /* one call over the whole extent; a short delivery is the stream's end */\
    int64_t got = SF_READF(file, dst, frames);                                \
    return got > 0 ? got : 0;                                                 \
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
        if (channels == 2) {                                                  \
          soundml_io_deinterleave2_##SUFFIX(staging, out, out + dst_total,   \
                                            got);                            \
        } else {                                                             \
          for (int64_t i = 0; i < got; i++) {                                \
            const T *fr = staging + (i * channels);                          \
            for (int64_t c = 0; c < channels; c++)                           \
              out[(c * dst_total) + i] = fr[c];                              \
          }                                                                  \
        }                                                                    \
      } else {                                                                \
        T *out = dst + dst_off + done;                                        \
        if (channels == 2) {                                                  \
          const T *restrict s = staging;                                     \
          for (int64_t i = 0; i < got; i++)                                  \
            out[i] = (s[2 * i] + s[(2 * i) + 1]) * inv;                      \
        } else {                                                             \
          for (int64_t i = 0; i < got; i++) {                                \
            const T *fr = staging + (i * channels);                          \
            T acc = (T) 0;                                                   \
            for (int64_t c = 0; c < channels; c++) acc += fr[c];             \
            out[i] = acc * inv;                                              \
          }                                                                  \
        }                                                                    \
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
  /* The encode-side layout pass: [want] frames of [channels] planar cursors  \
     [src_total] apart, laced into one interleaved staging block. */          \
  static void soundml_io_interleave_##SUFFIX(                                 \
      const T *restrict in, int64_t src_total, int64_t channels,              \
      T *restrict staging, int64_t want) {                                    \
    if (channels == 2) {                                                      \
      const T *restrict i0 = in;                                              \
      const T *restrict i1 = in + src_total;                                  \
      T *restrict s = staging;                                                \
      for (int64_t i = 0; i < want; i++) {                                    \
        s[2 * i] = i0[i];                                                     \
        s[(2 * i) + 1] = i1[i];                                               \
      }                                                                       \
    } else {                                                                  \
      for (int64_t i = 0; i < want; i++) {                                    \
        T *fr = staging + (i * channels);                                     \
        for (int64_t c = 0; c < channels; c++)                                \
          fr[c] = in[(c * src_total) + i];                                    \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  static int64_t soundml_io_write_planar_##SUFFIX(                            \
      SNDFILE *file, const T *restrict src, int64_t src_off,                  \
      int64_t src_total, int64_t frames, int64_t channels,                    \
      T *restrict staging, int64_t block) {                                   \
    int64_t done = 0;                                                         \
    while (done < frames) {                                                   \
      int64_t want = frames - done;                                           \
      if (want > block) want = block;                                         \
      soundml_io_interleave_##SUFFIX(src + src_off + done, src_total,         \
                                     channels, staging, want);                \
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

/* The decode of a whole request by the calling thread alone: the kernel the
   sample type and layout mode select, over the extent the caller asked for.
   Every read that is not split runs exactly this, and so does every split
   that does not complete. */
static int64_t soundml_io_read_body(SNDFILE *file, void *data, int kind,
                                    int mode, int64_t off, int64_t total,
                                    int64_t frames, int64_t channels,
                                    void *staging, int64_t block) {
  if (kind == CAML_BA_FLOAT32) {
    float *dst = (float *) data;
    if (mode == SOUNDML_IO_MODE_DIRECT)
      return soundml_io_read_direct_f32(file, dst + off, frames);
    return soundml_io_read_planar_f32(file, dst, off, total, frames, channels,
                                      mode, (float *) staging, block);
  }
  double *dst = (double *) data;
  if (mode == SOUNDML_IO_MODE_DIRECT)
    return soundml_io_read_direct_f64(file, dst + off, frames);
  return soundml_io_read_planar_f64(file, dst, off, total, frames, channels,
                                    mode, (double *) staging, block);
}

#ifdef SOUNDML_IO_PARALLEL

/* {1 Extent decoding}

   One contiguous frame range of a whole-file decode, carried out by a
   duplicate decoder over the shared descriptor and laid into the caller's
   destination by the same kernels the sequential body uses. The extents of a
   split partition the request, so the destination regions written here are
   disjoint and every one of them is written exactly once.

   [ok] is set only when the duplicate agreed with the original about the
   file's geometry, seeked exactly, delivered the full extent and reported no
   error. Anything else leaves it clear and the whole split is abandoned. */

typedef struct {
  int fd;
  int64_t file_bytes;
  int64_t adv_frames;
  int64_t start, count;
  void *dst;
  void *staging;
  int64_t dst_off, dst_total, channels, block;
  int mode, kind;
  int ok;
} soundml_io_extent;

static void soundml_io_extent_decode(void *arg) {
  soundml_io_extent *e = arg;
  SF_INFO info;
  soundml_io_vfile vf;
  SNDFILE *file;
  int64_t got;

  e->ok = 0;
  vf.fd = e->fd;
  vf.pos = 0;
  vf.len = e->file_bytes;
  memset(&info, 0, sizeof info);
  pthread_mutex_lock(&soundml_io_open_mutex);
  file = sf_open_virtual(&soundml_io_vio, SFM_READ, &info, &vf);
  pthread_mutex_unlock(&soundml_io_open_mutex);
  if (file == NULL) return;
  if ((int64_t) info.frames != e->adv_frames ||
      (int64_t) info.channels != e->channels ||
      sf_seek(file, (sf_count_t) e->start, SEEK_SET) != (sf_count_t) e->start) {
    sf_close(file);
    return;
  }

  if (e->mode == SOUNDML_IO_MODE_DIRECT) {
    if (e->kind == CAML_BA_FLOAT32)
      got = soundml_io_read_direct_f32(
          file, (float *) e->dst + e->dst_off + e->start, e->count);
    else
      got = soundml_io_read_direct_f64(
          file, (double *) e->dst + e->dst_off + e->start, e->count);
  } else if (e->kind == CAML_BA_FLOAT32) {
    got = soundml_io_read_planar_f32(file, (float *) e->dst,
                                     e->dst_off + e->start, e->dst_total,
                                     e->count, e->channels, e->mode,
                                     (float *) e->staging, e->block);
  } else {
    got = soundml_io_read_planar_f64(file, (double *) e->dst,
                                     e->dst_off + e->start, e->dst_total,
                                     e->count, e->channels, e->mode,
                                     (double *) e->staging, e->block);
  }
  e->ok = (got == e->count) && sf_error(file) == SF_ERR_NO_ERROR;
  sf_close(file);
}

/* Decode [frames] frames as [width] extents in parallel, the calling handle
   taking the last one so it ends the call at the position a sequential decode
   would leave it: end of file. Returns 1 only when every extent was delivered
   in full, and then [*out_done] is [frames]. A 0 return means nothing about
   the destination can be assumed and the caller must run its sequential body
   over the whole request — which is what makes every partial, truncated and
   error outcome the sequential code's to define. */
static int soundml_io_read_split(SNDFILE *file, const char *path,
                                 int64_t adv_frames, void *data, int kind,
                                 int mode, int64_t off, int64_t total,
                                 int64_t frames, int64_t channels, size_t elt,
                                 void *staging, int64_t block, int width,
                                 int64_t *out_done) {
  soundml_io_slot *slots[SOUNDML_IO_POOL_SLOTS];
  soundml_io_extent extent[SOUNDML_IO_POOL_SLOTS];
  void *stage[SOUNDML_IO_POOL_SLOTS];
  struct stat st;
  int64_t per, last_start, last_count, done = 0;
  int workers, i, fd, ok = 0;

  if (sf_seek(file, 0, SEEK_CUR) != 0) return 0;

  workers = soundml_io_pool_acquire(width - 1, slots);
  if (workers < 1) return 0;
  width = workers + 1;

  for (i = 0; i < workers; i++) stage[i] = NULL;
  fd = open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) goto done;
  if (fstat(fd, &st) != 0 || st.st_size <= 0) goto done;

  if (mode != SOUNDML_IO_MODE_DIRECT) {
    for (i = 0; i < workers; i++) {
      stage[i] = soundml_io_staging(block, channels, elt);
      if (stage[i] == NULL) goto done;
    }
  }

  /* Every extent must be non-empty: the calling handle has to decode a last
     extent that ends at the file's end, which is where a sequential decode
     leaves it. */
  per = (frames + width - 1) / width;
  last_start = per * workers;
  last_count = frames - last_start;
  if (last_count <= 0) goto done;

  for (i = 0; i < workers; i++) {
    extent[i].fd = fd;
    extent[i].file_bytes = (int64_t) st.st_size;
    extent[i].adv_frames = adv_frames;
    extent[i].start = per * i;
    extent[i].count = per;
    extent[i].dst = data;
    extent[i].staging = stage[i];
    extent[i].dst_off = off;
    extent[i].dst_total = total;
    extent[i].channels = channels;
    extent[i].block = block;
    extent[i].mode = mode;
    extent[i].kind = kind;
    extent[i].ok = 0;
    soundml_io_submit(slots[i], soundml_io_extent_decode, &extent[i]);
  }

  if (sf_seek(file, (sf_count_t) last_start, SEEK_SET) ==
      (sf_count_t) last_start) {
    if (mode == SOUNDML_IO_MODE_DIRECT)
      done = (kind == CAML_BA_FLOAT32)
                 ? soundml_io_read_direct_f32(
                       file, (float *) data + off + last_start, last_count)
                 : soundml_io_read_direct_f64(
                       file, (double *) data + off + last_start, last_count);
    else
      done = (kind == CAML_BA_FLOAT32)
                 ? soundml_io_read_planar_f32(
                       file, (float *) data, off + last_start, total,
                       last_count, channels, mode, (float *) staging, block)
                 : soundml_io_read_planar_f64(
                       file, (double *) data, off + last_start, total,
                       last_count, channels, mode, (double *) staging, block);
  }

  for (i = 0; i < workers; i++) soundml_io_join(slots[i]);

  ok = (done == last_count) && (sf_error(file) == SF_ERR_NO_ERROR);
  for (i = 0; i < workers; i++)
    if (!extent[i].ok) ok = 0;

done:
  for (i = 0; i < workers; i++) free(stage[i]);
  if (fd >= 0) close(fd);
  soundml_io_pool_release(slots, workers);
  if (!ok) {
    /* Hand the request back exactly as it arrived: at frame zero, with the
       sequential body free to define what it delivers and what it reports. */
    sf_seek(file, 0, SEEK_SET);
    return 0;
  }
  *out_done = frames;
  return 1;
}

/* The split width this call would use: the environment's pin when it names
   one, otherwise the work model. Either way the format must be one whose
   seek is exact and the pool must have a worker to give. */
static int soundml_io_split_plan(int format, int64_t frames,
                                 int64_t channels) {
  pthread_once(&soundml_io_pool_once, soundml_io_pool_init);
  if (soundml_io_pool.cap == 0) return 1;
  if (soundml_io_decode_cost(format) == 0) return 1;
  int width = soundml_io_pool.width;
  if (width == 0) width = soundml_io_split_width(format, frames, channels);
  if (width > SOUNDML_IO_POOL_SLOTS + 1) width = SOUNDML_IO_POOL_SLOTS + 1;
  return width < 1 ? 1 : width;
}

#endif /* SOUNDML_IO_PARALLEL */

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

#ifdef SOUNDML_IO_PARALLEL
  /* The custom block can move once the runtime lock is released, so the
     split's inputs are read out of it here; the path is a malloc'd copy the
     handle owns for its whole life, stable to dereference either side of the
     release. A split covers a whole-file decode: the request must fill the
     destination from its first frame and span exactly the frames the header
     advertises, and the handle must still be at the file's start (checked in
     the split itself). A chunked read qualifies only when its one chunk is
     the whole file; a downmix never does, and neither does any request that
     leaves frames behind for a later call. */
  const soundml_io_state *st = State_val(v_handle);
  const char *par_path = st->path;
  const int64_t par_frames = st->frames;
  int par_width = 1;
  if (mode != SOUNDML_IO_MODE_DOWNMIX && st->seekable && par_path != NULL &&
      off == 0 && par_frames > 0 && frames == par_frames)
    par_width = soundml_io_split_plan(st->format, frames, channels);
#endif

  caml_release_runtime_system();
#ifdef SOUNDML_IO_PARALLEL
  if (par_width < 2 ||
      !soundml_io_read_split(file, par_path, par_frames, data, kind, mode, off,
                             total, frames, channels, elt, staging, block,
                             par_width, &done))
#endif
    done = soundml_io_read_body(file, data, kind, mode, off, total, frames,
                                channels, staging, block);
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

/* {1 Seeking} */

CAMLprim value soundml_io_seek(value v_handle, value v_frame) {
  CAMLparam1(v_handle);
  CAMLlocal1(v_res);
  SNDFILE *file = soundml_io_file(v_handle);
  const sf_count_t frame = (sf_count_t) Long_val(v_frame);
  sf_count_t pos;
  char details[256] = {0};

  caml_release_runtime_system();
  pos = sf_seek(file, frame, SEEK_SET);
  if (pos < 0) snprintf(details, sizeof details, "%s", sf_strerror(file));
  caml_acquire_runtime_system();

  v_res = caml_alloc_tuple(2);
  Store_field(v_res, 0, Val_long((intnat) pos));
  Store_field(v_res, 1, caml_copy_string(details));
  CAMLreturn(v_res);
}

/* {1 Closing} */

CAMLprim value soundml_io_close(value v_handle) {
  CAMLparam1(v_handle);
  CAMLlocal1(v_res);
  soundml_io_handle *h = Handle_val(v_handle);
  soundml_io_state *state = h->state;
  SNDFILE *file = (state == NULL) ? NULL : state->file;
  int err = 0;
  char details[256] = {0};
  /* Detached before the lock is released: no concurrent stub can reach the
     SNDFILE* once the close is committed, and the state is this call's to
     free. Closing again finds NULL and does nothing. */
  h->state = NULL;
  if (file != NULL) {
    state->file = NULL;
    caml_release_runtime_system();
    err = sf_close(file);
    if (err != 0) {
      const char *msg = sf_error_number(err);
      snprintf(details, sizeof details, "%s", msg == NULL ? "" : msg);
    }
    caml_acquire_runtime_system();
  }
  soundml_io_state_free(state);
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
