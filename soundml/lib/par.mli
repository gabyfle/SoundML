(*****************************************************************************)
(*                                                                           *)
(*                                                                           *)
(*  Copyright (C) 2025                                                       *)
(*    Gabriel Santamaria                                                     *)
(*                                                                           *)
(*                                                                           *)
(*  Licensed under the Apache License, Version 2.0 (the "License");          *)
(*  you may not use this file except in compliance with the License.         *)
(*  You may obtain a copy of the License at                                  *)
(*                                                                           *)
(*    http://www.apache.org/licenses/LICENSE-2.0                             *)
(*                                                                           *)
(*  Unless required by applicable law or agreed to in writing, software      *)
(*  distributed under the License is distributed on an "AS IS" BASIS,        *)
(*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *)
(*  See the License for the specific language governing permissions and      *)
(*  limitations under the License.                                           *)
(*                                                                           *)
(*****************************************************************************)

(** Data parallelism over short-lived domains.

    The library's own kernels split their outputs into disjoint ranges and
    compute each range in a fixed arithmetic order, so which domain computes
    a range never changes a bit of the result: parallelism here is a matter
    of speed alone.

    Workers are domains spawned on demand. An idle domain makes every minor
    collection of the program stop for it, so none is kept alive between
    uses: a worker that finds no work spins for some tens of microseconds —
    long enough to catch a kernel issued right after the last one, too short
    to compete with nx's own pool for the cores — and then exits. A kernel
    never waits for the workers to start: the calling domain claims chunks
    from the first moment, and each worker joins in as it comes up. *)

val run : chunks:int -> (int -> unit) -> unit
(** [run ~chunks f] calls [f i] once for every [i] in [\[0, chunks)], on the
    calling domain and on as many workers as the chunks can use, and returns
    when every call has returned. Calls on different chunks may run
    concurrently, so they must write disjoint memory. When a call raises, the
    remaining unstarted chunks are skipped and the first exception is
    re-raised once the running ones have returned.

    A [run] issued while another is in progress — from inside a chunk, or
    from another domain — computes its chunks on the calling domain. *)

val ranges : n:int -> grain:int -> (int -> int -> unit) -> unit
(** [ranges ~n ~grain f] covers [\[0, n)] with consecutive ranges of at least
    [grain] elements (one range when [n < 2 * grain]) and calls [f lo hi] on
    each through {!run}. [grain] is the smallest range worth a chunk, sized
    by the caller so that a chunk costs some tens of microseconds. *)
