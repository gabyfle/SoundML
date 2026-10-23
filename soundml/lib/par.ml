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

(* One job is published at a time, in [current]. Its chunks are claimed by
   [fetch_and_add] on [next], so the calling domain and every worker draw from
   one counter and a slow worker only delays the chunk it holds; [finished]
   counts the chunks whose call has returned, and the caller spins on it once
   the counter is exhausted.

   Workers are reserved in [alive] before they are spawned, so concurrent
   reservations never exceed the target. Each new worker spawns the next one
   before it starts claiming, while its job still has unclaimed chunks, which
   keeps the calling domain out of the serialized domain creation after the
   first spawn. A worker that sees no new job spins for [linger] relaxations:
   long enough to catch a kernel issued right after the last one, short enough
   that spinning workers never compete with the cores nx's own pool is about to
   use. It then parks on [wake] until the next job. The publisher only takes
   [lock] to wake the pool when [parked] counts a sleeper: a worker counts
   itself before it reads [current] under the lock, and the publisher writes
   [current] before it reads [parked], so with sequentially consistent atomics
   either the worker sees the job or the publisher sees the worker.

   A parked domain still makes every minor collection stop for it, so the pool
   retires. The first publisher sets a GC alarm, and when a whole major cycle
   has passed without a new job id the alarm moves [epoch] on: every worker
   spawned in an older epoch exits, parked ones as soon as they are woken and
   spinning ones at their next look, all together, so that the collections their
   exits force mostly merge. The alarm never blocks on [lock]: when the lock is
   taken it tries again at the next cycle. It runs on the domain that set it and
   ends with that domain, which then retires the pool and leaves the alarm to
   the next publisher, unless it is the main domain, whose exit ends the
   program. Spawned domains are never joined, as nothing waits for their
   result. *)

type job =
  { f: int -> unit
  ; chunks: int
  ; next: int Atomic.t
  ; finished: int Atomic.t
  ; failure: exn option Atomic.t
  ; id: int }

let current : job option Atomic.t = Atomic.make None

let alive = Atomic.make 0

let ids = Atomic.make 0

(* Workers asleep on [wake], and the generation they belong to: a worker exits
   once [epoch] has moved past the one it was spawned in. *)
let parked = Atomic.make 0

let epoch = Atomic.make 0

let lock = Mutex.create ()

let wake = Condition.create ()

(* Whether a domain's alarm watches the pool. *)
let watched = Atomic.make false

(* Workers beyond the calling domain. *)
let capacity = Int.max 0 (Domain.recommended_domain_count () - 1)

(* Some tens of microseconds of [Domain.cpu_relax] on current hardware. *)
let linger = 8_000

let claim job =
  let rec loop () =
    let i = Atomic.fetch_and_add job.next 1 in
    if i < job.chunks then begin
      ( match Atomic.get job.failure with
      | Some _ ->
          ()
      | None -> (
        try job.f i
        with e -> ignore (Atomic.compare_and_set job.failure None (Some e)) ) ) ;
      Atomic.incr job.finished ; loop ()
    end
  in
  loop ()

let rec reserve target =
  let a = Atomic.get alive in
  a < target && (Atomic.compare_and_set alive a (a + 1) || reserve target)

let pending job = Atomic.get job.next < job.chunks

let fresh last =
  match Atomic.get current with Some job -> job.id <> last | None -> false

let rec spawn job target =
  if pending job && reserve target then
    match
      Domain.spawn (fun () ->
          let era = Atomic.get epoch in
          spawn job target ; idle era (-1) 0 )
    with
    | (_ : unit Domain.t) ->
        ()
    | exception _ ->
        (* The runtime refused another domain: work with those running. *)
        Atomic.decr alive

and idle era last spins =
  match Atomic.get current with
  | Some job when job.id <> last ->
      claim job ; idle era job.id 0
  | Some _ | None ->
      if Atomic.get epoch <> era then Atomic.decr alive
      else if spins < linger then begin
        Domain.cpu_relax () ;
        idle era last (spins + 1)
      end
      else park era last

and park era last =
  Atomic.incr parked ;
  Mutex.lock lock ;
  while (not (fresh last)) && Atomic.get epoch = era do
    Condition.wait wake lock
  done ;
  Mutex.unlock lock ;
  Atomic.decr parked ;
  idle era last 0

(* Under [lock]: every worker alive belongs to an older epoch from now on. *)
let retire () = Atomic.incr epoch ; Condition.broadcast wake

let watch () =
  if Atomic.compare_and_set watched false true then begin
    let seen = ref (-1) in
    let alarm () =
      let id = Atomic.get ids in
      if id <> !seen then seen := id
      else if
        Option.is_none (Atomic.get current)
        && Atomic.get alive > 0
        && Mutex.try_lock lock
      then begin
        retire () ; Mutex.unlock lock
      end
    in
    ignore (Gc.create_alarm alarm : Gc.alarm) ;
    if not (Domain.is_main_domain ()) then
      Domain.at_exit (fun () ->
          Atomic.set watched false ;
          Mutex.lock lock ;
          retire () ;
          Mutex.unlock lock )
  end

let sequential ~chunks f =
  for i = 0 to chunks - 1 do
    f i
  done

let run ~chunks f =
  if chunks <= 1 || capacity = 0 then sequential ~chunks f
  else
    let job =
      { f
      ; chunks
      ; next= Atomic.make 0
      ; finished= Atomic.make 0
      ; failure= Atomic.make None
      ; id= Atomic.fetch_and_add ids 1 }
    in
    if not (Atomic.compare_and_set current None (Some job)) then
      sequential ~chunks f
    else begin
      if not (Atomic.get watched) then watch () ;
      if Atomic.get parked > 0 then begin
        Mutex.lock lock ; Condition.broadcast wake ; Mutex.unlock lock
      end ;
      spawn job (Int.min capacity (chunks - 1)) ;
      claim job ;
      while Atomic.get job.finished < chunks do
        Domain.cpu_relax ()
      done ;
      Atomic.set current None ;
      Option.iter raise (Atomic.get job.failure)
    end

let ranges ~n ~grain f =
  let grain = Int.max 1 grain in
  let most = 4 * (capacity + 1) in
  let chunks = Int.max 1 (Int.min most (n / grain)) in
  if chunks = 1 then (if n > 0 then f 0 n)
  else run ~chunks (fun c -> f (c * n / chunks) ((c + 1) * n / chunks))
