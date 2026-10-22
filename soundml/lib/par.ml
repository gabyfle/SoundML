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
   first spawn. A worker that sees no new job spins for [linger] relaxations and
   exits: long enough to catch a kernel issued right after the last one, short
   enough that spinning workers never compete with the cores nx's own pool is
   about to use. Spawned domains are never joined, as nothing waits for their
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

(* Workers beyond the calling domain. *)
let capacity = Stdlib.max 0 (Domain.recommended_domain_count () - 1)

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

let rec spawn job target =
  if pending job && reserve target then
    match Domain.spawn (fun () -> spawn job target ; idle (-1) 0) with
    | (_ : unit Domain.t) ->
        ()
    | exception _ ->
        (* The runtime refused another domain: work with those running. *)
        Atomic.decr alive

and idle last spins =
  match Atomic.get current with
  | Some job when job.id <> last ->
      claim job ; idle job.id 0
  | Some _ | None ->
      if spins < linger then begin
        Domain.cpu_relax () ;
        idle last (spins + 1)
      end
      else Atomic.decr alive

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
      spawn job (Stdlib.min capacity (chunks - 1)) ;
      claim job ;
      while Atomic.get job.finished < chunks do
        Domain.cpu_relax ()
      done ;
      Atomic.set current None ;
      Option.iter raise (Atomic.get job.failure)
    end

let ranges ~n ~grain f =
  let grain = Stdlib.max 1 grain in
  let most = 4 * (capacity + 1) in
  let chunks = Stdlib.max 1 (Stdlib.min most (n / grain)) in
  if chunks = 1 then (if n > 0 then f 0 n)
  else run ~chunks (fun c -> f (c * n / chunks) ((c + 1) * n / chunks))
