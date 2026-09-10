# Live run ledger

Goal: implement and finish Q-only/K-only/V-only/O-only internal SiLU ablations
on the all-linear baseline. User allows simultaneous jobs and queueing, arch2x8
per experiment, and restart of killed attempts. Goal remains active until actual
training completion is verified, not merely successful submission.

All times Asia/Hong_Kong. First submissions on 2026-09-09:

| Variant | Job ID | Created | Last confirmed status |
| --- | --- | --- | --- |
| Q | mose-all-linear-q-silu-2x8-0909-22774980 | 22:48:42 | Sep10 01:41:39 Stopped; last healthy sample6738; step5364 save confirmed |
| K | mose-all-linear-k-silu-2x8-0909-76770213 | 22:49:36 | Sep10 01:42:35 Stopped; step3576 save confirmed |
| V | mose-all-linear-v-silu-2x8-0909-1278007 | 22:50:01 | Sep10 01:43:14 Stopped; step3576 save confirmed01:38:46.349 |
| O | mose-all-linear-o-silu-2x8-0909-27537918 | 22:50:27 | Sep10 01:43:59 Stopped; last logged step3649; step3576 save confirmed01:41:33.861 |

## Current replacement attempts (Sep10)

| Variant | Job ID | Created | Last confirmed status |
| --- | --- | --- | --- |
| Q | mose-all-linear-q-silu-r3-2x8-0910-80409470 | 15:34:40 | 15:42:22 Inqueue; intended resume from saved7152 not yet verified |
| K | mose-all-linear-k-silu-r4-2x8-0910-86807381 | 15:34:46 | 15:42:22 Inqueue; intended resume from saved10728 not yet verified |
| V | mose-all-linear-v-silu-r4-2x8-0910-36359680 | 15:40:36 | 15:42:22 Inqueue; intended resume from saved10728 not yet verified |
| O | mose-all-linear-o-silu-r4-2x8-0910-81400775 | 15:33:01 | 15:42:22 Inqueue; both replicas STARTING, intended resume from saved8940 not yet verified |

15:25:12–15:42:22: Q-r2/K-r3/V-r3/O-r3 were independently confirmed
Stopped with zero active replicas. Q-r2 had loaded step7152 but reached no new
periodic save. K-r3 and V-r3 loaded step8940 and each saved step10728
successfully (trainer confirmation at14:40:58 and14:40:47 respectively).
O-r3 stopped from Inqueue and never created a W&B segment. All eleven known
training-produced W&B segments were submitted before their replacements;
truncated killed segments were completed with W&B legacy sync after beta EOF.
Replacement Q-r3/K-r4/V-r4/O-r4 jobs were submitted unchanged as arch2x8 and
accepted by the scheduler. Latest confirmed saves are Q7152, K10728, V10728,
O8940. See WANDB_SYNC_20260910.md for per-segment evidence.

## Previous replacement attempts (Sep10)

| Variant | Job ID | Created | Terminal evidence |
| --- | --- | --- | --- |
| Q | mose-all-linear-q-silu-r2-2x8-0910-91695506 | 03:56:31 | 15:25:12 Stopped, zero active/two stopped; loaded7152; no newer periodic save |
| K | mose-all-linear-k-silu-r3-2x8-0910-45375873 | 06:09:05 | 15:02:40 Stopped after SIGTERM; step10728 saved successfully |
| V | mose-all-linear-v-silu-r3-2x8-0910-97404530 | 06:19:57 | 15:16:47 Stopped after SIGTERM; step10728 saved successfully |
| O | mose-all-linear-o-silu-r3-2x8-0910-53540534 | 06:22:33 | 15:14:21 Inqueue→Stopped; never started, no new W&B segment |

13:12:18–13:25:12: repeated live checks confirmed all four current jobs
Inqueue with zero active/failed/stopped replicas. Event checks at13:20
showed queue acceptance only. Capacity prediction at13:20:42 returned
2/2 schedulable and twelve candidate whole8GPU nodes, not a reservation
or dispatch guarantee. No duplicate submissions or experiment changes.
Four original W&B segments invoked previously remain unverified; four
resumed segments await sync verification instructions/permission.
Latest confirmed saves remain Q7152/KVO8940; no current-attempt load verified.

13:06:45–13:10:55: six live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. No duplicate submissions, experiment
changes, or additional W&B sync. Wrapper/output inspection permission still
pending. Current-attempt loading remains unverified; latest confirmed saves
Q7152/KVO8940. Previous/current turns verified queue waits; full30B unfinished.

13:00:38–13:04:49: six live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. At13:05:10 arch2x8 predict-only returned
2/2 schedulable with four candidate whole8GPU nodes; not an inventory or
dispatch guarantee. No duplicate submissions or experiment changes. W&B
wrapper/output inspection permission still pending; no additional sync.
Current-attempt checkpoint loading remains unverified; latest saves
Q7152/KVO8940. Previous/current turns verified queue waits; full30B unfinished.

12:54:22–12:58:32: six live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. All four event histories rechecked:
external-scheduler queue acceptance only, no new allocation/failure. No
duplicate submissions or experiment changes. W&B wrapper/output inspection
permission still pending, no additional sync. Current-attempt loading remains
unverified; latest saves Q7152/KVO8940. Previous/current turns verified queue
waits; all four full30B objectives unfinished and the goal active.

12:48:39–12:52:50: six live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. No duplicate submissions or experiment
changes. W&B verification permission still pending, no additional sync.
Current-attempt checkpoint loading remains unverified; latest confirmed
saves Q7152/KVO8940. Previous/current turns verified queue waits; all four
full30B objectives unfinished and the goal remains active.

12:42:34–12:46:44: six live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. At12:47:07 arch2x8 predict-only returned
2/2 schedulable with five candidate whole8GPU nodes, not an inventory or
dispatch guarantee. No duplicate submissions or experiment changes.
The user briefly paused monitoring, then explicitly requested W&B sync and
continued training. Eight offline segments were identified; four original
segments invoked through wandb_sjy in the prior turn, completion unverified.
Remaining sync waits for permission/instructions to inspect wrapper/output;
see WANDB_SYNC_20260910.md. No additional sync invocation this turn. GPU
monitoring remains active; current turn is a verified queue wait, all full30B
objectives unfinished. Latest verified saves Q7152/KVO8940.

12:23:25–12:26:46: five live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. At12:27:06 arch2x8 predict-only returned
2/2 schedulable with five candidate whole8GPU nodes; not a complete inventory
or dispatch guarantee. No duplicate submissions or config/resource changes.
Current-attempt loading remains unverified; latest saves Q7152/KVO8940.
Previous/current turns verified waits; all four full30B objectives unfinished.

12:17:53–12:22:04: six live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. No duplicate submissions or experiment/
resource changes. Current-attempt checkpoint loading remains unverified;
latest confirmed saves Q7152/KVO8940. Previous/current turns verified queue
waits; all four full30B runs remain unfinished and the goal active.

12:11:42–12:15:54: six live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. Subsequent checks of all four event
histories still show only external-scheduler queue acceptance, no new failure
or allocation. No duplicate submissions or config/resource changes.
Current-attempt loading remains unverified; latest saves Q7152/KVO8940.
Previous/current turns verified waits; all four full30B objectives unfinished.

12:04:52–12:09:53: seven live polls confirmed all four current jobs Inqueue,
zero active/failed/stopped replicas. At12:10:13 arch2x8 predict-only returned
2/2 schedulable with six candidate whole8GPU nodes; feasibility is not a
complete inventory or dispatch guarantee. No duplicate submissions or config/
resource changes. Current-attempt resume remains unverified; latest confirmed
saves Q7152/KVO8940. Previous/current turns verified waits; full30B unfinished.

11:58:52–12:03:03: six live status polls confirmed all four current handles
Inqueue with zero active/failed/stopped replicas. No submissions or experiment
changes. Current-attempt checkpoint loading still awaits dispatch; latest
verified saves Q7152/KVO8940. Previous/current turns are verified queue waits,
not completion. All four full30B runs remain unfinished; goal stays active.

11:54:15–11:57:30: four live status polls confirmed all four current handles
Inqueue with zero active/failed/stopped replicas. All four event histories
still only show external-scheduler queue acceptance. At11:55:32 arch2x8
predict-only returned2/2 schedulable, listing five candidate whole8GPU nodes;
this is not an inventory or dispatch guarantee. No duplicate submissions or
config/resource changes. Current-attempt resume remains unverified. Previous
and current turns are verified queue waits; the full30B goal remains active.

11:45:11–11:52:46: repeated live polls confirmed all four current handles
Inqueue with zero active/failed/stopped replicas. No duplicate submissions,
configuration changes, or resource changes. Latest verified saves remain
Q7152/KVO8940; current-attempt checkpoint loading awaits dispatch. This is
verified queue waiting, not completed training; all four full30B goals remain
unfinished and active.

11:38:01–11:43:01: seven authoritative live polls confirmed all four current
handles Inqueue, zero active/failed/stopped replicas. At11:43:29 arch2x8
predict-only returned2/2 schedulable with five candidate whole8GPU nodes;
not a complete inventory or dispatch guarantee. No duplicate submissions or
config/resource changes. Latest verified saves Q7152/KVO8940; current-attempt
loading awaits dispatch. Previous/current turns verified waits, all four
full30B objectives unfinished and active.

11:30:29–11:36:11: seven live polls confirmed all four current handles Inqueue,
zero active/failed/stopped replicas. All four event histories rechecked and
still show only external-scheduler queue acceptance, no new allocation/failure
or explicit denial. No duplicate submissions or config/resource changes.
Latest verified saves Q7152/KVO8940; current-attempt loading awaits dispatch.
Previous/current turns verified waits; all four full30B objectives unfinished.

11:23:17–11:28:17: seven live polls confirmed all four current handles Inqueue,
zero active/failed/stopped replicas. At11:28:40 arch2x8 predict-only returned
2/2 schedulable with six candidate whole8GPU nodes; not a complete inventory
or dispatch guarantee. No duplicate submissions or config/resource changes.
Latest verified saves Q7152/KVO8940; current-attempt loading awaits dispatch.
Previous/current turns verified waits; all four full30B objectives incomplete.

11:16:37–11:21:38: seven live polls confirmed all four current handles Inqueue,
zero active/failed/stopped replicas. No duplicate submissions or config/resource
changes. Latest verified saves Q7152/KVO8940; current-attempt loading still awaits
dispatch. Previous/current turns verified waits, all four full30B objectives
unfinished and active. SSH closed cleanly after polling.

11:08:57–11:14:48: seven authoritative live polls confirmed all four current
handles Inqueue, zero active/failed/stopped replicas. All four event histories
still only show external-scheduler queue acceptance; no new allocation/failure
or explicit denial. At11:15:11 arch2x8 predict-only returned2/2 schedulable,
seven candidate whole8GPU nodes; not a complete inventory or dispatch guarantee.
No duplicate submissions or config/resource changes. Latest verified saves
Q7152/KVO8940; current-attempt loading awaits dispatch. Previous/current turns
verified waits, all four full30B objectives unfinished and active.

11:01:34–11:06:35: seven authoritative live polls confirmed all four current
handles Inqueue, zero active/failed/stopped replicas. No duplicate submissions
or config/resource changes. Latest verified saves remain Q7152/KVO8940;
current-attempt loading awaits dispatch. Previous/current turns verified
waits; all four full30B objectives unfinished, goal active.

10:54:39–10:59:40: seven live polls confirmed all four current handles Inqueue,
zero active/failed/stopped replicas. At11:00:02 arch2x8 predict-only returned
2/2 schedulable with nine candidate whole8GPU nodes; not a complete inventory
or dispatch guarantee. No duplicate submissions or config/resource changes.
Latest verified saves remain Q7152/KVO8940; current-attempt loading awaits
dispatch. Previous/current turns verified waits; all four full30B goals
remain unfinished and active.

10:47:33–10:52:33: seven authoritative live polls confirmed all four current
handles Inqueue, zero active/failed/stopped replicas. All four event histories
rechecked afterwards: external-scheduler queue acceptance only, no new
allocation/failure or explicit denial. No duplicate submissions or config/
resource changes. Latest verified saves remain Q7152/KVO8940; current-attempt
loading awaits dispatch. Previous/current turns are verified waits; all four
full30B objectives unfinished, goal active.

10:40:26–10:45:26: seven authoritative live polls confirmed all four current
handles Inqueue, no active/failed/stopped replicas. At10:45:50 arch2x8
predict-only returned2/2 schedulable, seven candidate whole8GPU nodes;
not an inventory or dispatch guarantee. No duplicate submissions or
configuration/resource changes. Latest verified saves remain Q7152/KVO8940;
current-attempt resume awaits dispatch. Previous/current turns verified waits,
all four full30B training objectives unfinished, goal active.

10:33:16–10:38:56: seven live polls confirmed all four current handles Inqueue,
zero active/failed/stopped replicas. All four event histories rechecked and
still only show external-scheduler queue acceptance, no new allocation/failure
or explicit denial. No duplicate submissions or config/resource changes.
Latest verified saves Q7152/KVO8940; current-attempt loading awaits dispatch.
Previous/current turns are verified waits; all four full30B runs incomplete.

10:26:14–10:31:15: seven authoritative live polls confirmed all four current
handles Inqueue, zero active/failed/stopped replicas. At10:31:44 arch2x8
predict-only returned2/2 schedulable and six candidate whole8GPU nodes;
not a complete inventory or dispatch guarantee. No duplicate submission or
config/resource changes. Latest verified saves Q7152/KVO8940; actual new
attempt loading awaits dispatch. Previous/current turns verified waits;
all four full30B runs unfinished, goal active.

10:19:45–10:24:46: seven authoritative live status polls confirmed all four
current jobs Inqueue, no new active/failed/stopped replicas. No duplicate
submission or config/resource changes. Latest verified saves remain Q7152,
K/V/O8940; current-attempt loading still awaits dispatch. Previous/current
turns are verified waits. All four full30B objectives remain unfinished.

10:11:50–10:17:41: seven live status polls confirmed all four current handles
Inqueue, zero active/failed/stopped replicas. All four event histories rechecked:
external-scheduler queue acceptance only, no new allocation/failure or explicit
denial. At10:18:03 arch2x8 predict-only returned2/2 schedulable, eight candidate
whole8GPU nodes; not a reservation, complete inventory or dispatch guarantee.
No duplicate submission or config/resource change. Latest verified saves remain
Q7152/KVO8940; actual current-attempt resume awaits dispatch. This and previous
turn are verified waits. All four full30B objectives incomplete, goal active.

10:01:24–10:10:19: seven authoritative live polls confirmed all four current
attempts Inqueue, no active/failed/stopped replicas. At10:01:30 arch2x8
predict-only returned2/2 schedulable with eight candidate nodes listed;
feasibility is not allocation or a complete inventory. No duplicate submissions
or configuration changes. Latest verified saves Q7152/KVO8940; current-attempt
loading still awaits dispatch. All four30B runs unfinished; verified queue
waiting, goal active. Submission-host SSH session closed cleanly after polling.

09:54:36–09:59:53: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. No duplicate
submissions or configuration changes. Latest verified saves Q7152/KVO8940;
new-attempt loading still awaits dispatch. All four30B runs unfinished;
verified scheduler waiting, goal active.

09:47:12–09:53:10: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. At09:47:18 arch2x8
predict-only returned2/2 schedulable with twelve candidate nodes listed;
not a complete inventory or dispatch guarantee. All four event histories
still only show external-scheduler queue acceptance, no new allocation or
failure. No duplicate submissions or configuration changes. Latest verified
saves Q7152/KVO8940; actual new-attempt loading awaits dispatch. All four30B
runs unfinished; verified scheduler waiting, goal active.

09:40:14–09:45:30: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. No duplicate
submissions or configuration changes. Latest verified saves Q7152/KVO8940;
new-attempt loading still awaits dispatch. All four30B runs unfinished;
verified scheduler waiting, goal active.

09:33:22–09:38:43: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. At09:33:28 arch2x8
predict-only returned2/2 schedulable with twelve candidate nodes listed;
not a complete inventory or dispatch guarantee. No duplicate submissions
or configuration changes. Latest verified saves Q7152/KVO8940; actual
new-attempt loading awaits dispatch. All four30B runs unfinished; verified
scheduler waiting, goal active.

09:26:15–09:31:52: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. Event histories
rechecked for all four IDs still only show external-scheduler queue
acceptance, no new allocation or failure. No duplicate submissions or config
changes. Latest verified saves Q7152/KVO8940; actual new-attempt loading
awaits dispatch. All four30B runs unfinished; verified scheduler waiting,
goal active.

09:19:32–09:24:53: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. At09:19:38 arch2x8
predict-only returned2/2 schedulable with twelve candidate nodes listed;
not a complete inventory or dispatch guarantee. No duplicate submissions
or configuration changes. Latest verified saves Q7152/KVO8940; actual
new-attempt loading awaits dispatch. All four30B runs unfinished; verified
scheduler waiting, goal active.

09:12:23–09:17:37: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. No duplicate
submissions or configuration changes. Latest verified saves Q7152/KVO8940;
new-attempt loading still awaits dispatch. All four30B runs unfinished;
verified scheduler waiting, goal active.

09:05:00–09:11:03: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. At09:05:06 arch2x8
predict-only returned2/2 schedulable with fifteen candidate nodes listed;
not a complete inventory or dispatch guarantee. All four event histories
still only show external-scheduler queue acceptance, no new allocation or
failure. No duplicate submissions or configuration changes. Latest verified
saves Q7152/KVO8940; actual new-attempt loading awaits dispatch. All four30B
runs unfinished; verified scheduler waiting, goal active.

08:58:10–09:03:25: seven authoritative live polls confirmed all four current
attempts Inqueue, with no new active/failed/stopped replicas. No duplicate
submissions or configuration changes. Latest verified saves Q7152/KVO8940;
new-attempt loading still awaits dispatch. All four30B runs unfinished;
verified scheduler waiting, goal active.

08:51:37–08:56:55: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. At08:51:44 arch2x8
predict-only returned2/2 schedulable with fifteen candidate nodes listed;
not a complete inventory or dispatch guarantee. No duplicate submissions
or configuration changes. Latest verified saves Q7152/KVO8940; actual
new-attempt loading awaits dispatch. All four30B runs unfinished; verified
scheduler waiting, goal active.

08:44:20–08:49:58: seven authoritative live polls confirmed all four current
attempts Inqueue, with no new active/failed/stopped replicas. Events rechecked
for each current ID still only show external-scheduler queue acceptance;
no new failure or allocation. No duplicate submissions or config changes.
Latest verified saves Q7152/KVO8940; new-attempt loading awaits dispatch.
All four30B runs remain unfinished; verified scheduler waiting, goal active.

08:37:25–08:42:45: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. At08:37:30 arch2x8
predict-only returned2/2 schedulable, with ten candidate nodes listed; this
is neither a complete inventory nor a dispatch guarantee. No duplicate
submissions or configuration changes. Latest verified saves Q7152/KVO8940;
actual new-attempt loading awaits allocation. All four30B runs unfinished;
verified scheduler waiting, goal active.

08:27:25–08:33:04: seven authoritative live checks confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. Events rechecked for
each current ID still only show external-scheduler queue acceptance, no new
failure or allocation. No duplicate submissions or configuration changes.
Latest verified saves Q7152/KVO8940; actual new-attempt loading awaits dispatch.
All four30B runs unfinished; verified scheduler waiting, goal active.

08:20:35–08:26:08: seven authoritative live polls confirmed all four current
attempts Inqueue, with no new active/failed/stopped replicas. At08:20:52
arch2x8 predict-only returned2/2 schedulable; seven candidate nodes listed,
not a complete inventory or guarantee of dispatch. No duplicate submissions
or configuration changes. Latest verified saves Q7152/KVO8940; actual
new-attempt loading pending allocation. All four30B runs unfinished; verified
scheduler waiting, goal active.

08:13:58–08:19:13: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas or allocations. No
duplicate submissions or configuration changes. Latest verified saves remain
Q7152/KVO8940; actual new-attempt loading awaits dispatch. All four30B runs
unfinished; verified scheduler waiting, goal active.

08:07:15–08:12:46: seven authoritative live polls confirmed all four current
attempts Inqueue, without new active/failed/stopped replicas. At08:07:31
arch2x8 predict-only returned2/2 schedulable; eight candidate nodes listed,
not a complete inventory or placement guarantee. No duplicate submissions or
configuration changes. Latest verified saves Q7152/KVO8940; actual new-attempt
loading pending allocation. All four30B runs unfinished; verified scheduler
waiting, goal active.

08:00:10–08:05:49: seven authoritative live checks confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. Events rechecked for
each current ID still only report external-scheduler queue acceptance, no new
allocation or failure. No duplicate submissions or configuration changes.
Latest verified saves remain Q7152/KVO8940; new-attempt loading awaits dispatch.
All four30B runs unfinished; verified scheduler waiting, goal active.

07:53:21–07:58:54: seven live polls confirmed all four current attempts
Inqueue, no new active/failed/stopped replicas. At07:53:38 arch2x8 predict-only
returned2/2 schedulable, with nine candidate nodes listed (not a full partition
inventory or a guarantee of placement). No duplicate submissions or config
changes. Latest verified saves Q7152/KVO8940; actual new-attempt loading awaits
dispatch. All four30B runs unfinished; verified scheduler waiting, goal active.

07:46:56–07:52:12: seven authoritative live status polls confirmed all four
current attempts Inqueue, with no new active/failed/stopped replicas. No new
submissions or experiment changes. Saved Q7152/KVO8940 remain the latest
verified saves, actual new-attempt loading pending allocation. All four30B
runs unfinished; verified scheduler waiting, goal active.

07:40:00–07:45:31: seven authoritative live polls confirmed all four current
attempts Inqueue, without new active/failed/stopped replicas. At07:40:18
arch2x8 predict-only returned2/2 schedulable, candidates H2001106/0032/2267/
2156/0010/1117; feasibility is not allocation or complete inventory. No new
submissions or experimental changes. Saved Q7152/KVO8940 unchanged, actual
new-attempt loading pending dispatch. All four30B runs unfinished; verified
scheduler waiting, goal active.

07:33:12–07:38:50: seven authoritative live polls confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. Events rechecked for
each current ID still only show external-scheduler queue acceptance, no new
failure or allocation. No resubmissions or configuration changes. Latest
verified saves Q7152/KVO8940; actual new-attempt loading pending allocation.
All four30B runs unfinished; verified scheduler waiting, goal active.

07:26:26–07:31:56: seven live status checks confirmed all four current
attempts Inqueue, no new allocations or failed/stopped replicas. At07:26:43
arch2x8 predict-only returned2/2 schedulable; candidates H2001118/1117/0010/
1106/2156. Instantaneous feasibility does not establish actual dispatch or
partition-wide inventory. No duplicate submissions or configuration changes.
Latest verified saves Q7152/KVO8940; new-attempt loading pending allocation.
All four30B runs unfinished; verified scheduler waiting, goal active.

07:20:03–07:25:16: seven authoritative live checks confirmed all four current
attempts Inqueue, no new active/failed/stopped replicas. No resubmissions,
configuration changes or new training evidence. Latest verified saves remain
Q7152/KVO8940, actual new-attempt loading pending dispatch. All four30B runs
remain unfinished; verified scheduler waiting, goal active.

07:13:20–07:18:52: seven live checks confirmed all four current attempts
Inqueue, no active/failed/stopped replicas or new allocations. At07:13:39
arch2x8 predict-only returned2/2 schedulable; candidate list includes H2001106,
0010,2156,1118. Prediction is not reservation, dispatch or complete inventory.
No duplicate submissions, priority/resource changes or experiment edits.
Saved Q7152/KVO8940 unchanged; new-attempt loading pending dispatch. All four
30B runs unfinished; this turn is verified scheduler waiting, goal active.

07:06:10–07:11:59: seven authoritative live checks confirmed all four current
attempts Inqueue with zero active/failed/stopped replicas. Per-job events
queried this round still only show acceptance by the external scheduler; no
new failure or allocation events. No resubmissions or configuration changes.
Saved Q7152/KVO8940 unchanged; actual new-attempt checkpoint loading remains
unverified until dispatch. All four30B runs unfinished, monitoring goal active.

06:52–07:04:58: continued authoritative live polls confirm all four current
attempts Inqueue, without new allocations or terminal failures. At07:00:35
arch2x8 predict-only returned2/2 schedulable, candidates H2000010/1106.
This is instantaneous feasibility, not reservation or actual dispatch; no
duplicate submissions were made. Latest confirmed saves remain Q7152/KVO8940;
new-attempt checkpoint loading is pending allocation. No experiment changes.
All four30B runs remain unfinished and the monitoring goal remains active.

06:45:38–06:50:58: seven live status checks confirmed all four current attempts
Inqueue with no new failed/stopped replicas. At06:45:45 arch2x8 predict-only
reported1/2 schedulable; follow-up1x8 prediction06:46:05 returned1/1 with
candidate H2002156. Instantaneous feasibility is not reservation or exact
partition inventory. All training requests remain2x8; no new submissions or
configuration changes. Saved Q7152/KVO8940 unchanged, actual new-attempt loads
still pending allocation. All four30B runs unfinished.

06:38:16–06:44:19: eight live polls confirmed all four current attempts remain
Inqueue, with no new failed/stopped replicas or allocation. No resubmissions
or configuration changes. Latest verified saves remain Q7152/KVO8940; actual
checkpoint loading is unverified until dispatch. All four30B runs unfinished.

06:31:40–06:37:04: seven live checks confirmed all four current attempts
Inqueue. Each job's events still only show external queue acceptance. At
06:31:46 arch2x8 predict-only returned2/2 schedulable, candidates H2002243/
2316/2301; this is instantaneous feasibility, not allocation. No new failures,
submissions or configuration changes. Saved steps Q7152/KVO8940 unchanged;
all four30B runs unfinished, checkpoint load verification pending dispatch.

06:24:03–06:30:07: eight authoritative live status polls confirmed all four
current attempts remain Inqueue, without new failed/stopped replicas or
allocations. No submissions or configuration changes. Latest verified saves
remain Q7152/KVO8940; new-attempt checkpoint loading is pending allocation.
This is verified scheduler waiting; all four30B runs remain unfinished.

06:18–06:22: V r2 `mose-all-linear-v-silu-r2-2x8-0910-74559126` stopped
06:18:43 after both pods were deleted06:18:03. SIGTERM/DataLoader termination
errors were logged;06:18:52 get confirmed zero active/two stopped replicas.
Last optimizer log9499; latest saved8940. After the06:19:35 arch2x8 prediction
returned0/2, used explicit queueing authorization to submit unchanged V launcher
as r3 at06:19:57. Actual ID97404530, queue acceptance and Inqueue verified.

O r2 `mose-all-linear-o-silu-r2-2x8-0910-80664818` subsequently received SIGTERM
06:20:49. Events show both pods deleted06:20:48, Running->Stopped06:21:26;
06:21:47 get confirmed zero active/two stopped. Last sampled optimizer log9499,
latest saved8940. After06:22:11 predict-only reported2/2 schedulable, submitted
unchanged O launcher as r3 at06:22:33, actual ID53540534.06:22:49 get confirms
all four current attempts Inqueue. V/O stop events did not identify operator
or cause and contained no explicit quota/access prohibition. All submissions
retain arch2x8, default priority, original RUN_NAME/STAGE/save folder and full
30B settings. Latest saved Q7152/KVO8940; actual new-attempt loading remains
unverified until allocation. All four unfinished; no duplicate live writers.

06:11:04–06:17:23: eight live status checks confirmed Q r2/K r3 Inqueue and
V/O r2 Running with two active replicas each. Latest training samples V9497,
O9413; no new errors, saves, restarts or configuration changes. Latest verified
saves Q7152, K/V/O8940. Q/K replacement checkpoint loading is still pending
allocation; all four30B runs unfinished.

06:02:49–06:09:21: V saved8940 at06:03:16.489 (async initiation06:03:12.345),
O at06:05:16.696 (initiation06:05:12.475), both in the original RUN_NAME/stage1
folders. These are explicit trainer save-success logs, not independent disk
readback. V/O completed9000 evaluation and continued normally.

K r2 `mose-all-linear-k-silu-r2-2x8-0910-68437702` received SIGTERM/DataLoader
worker termination errors at06:07:17. Events show both pods deleted06:07:15,
then Running->Stopped06:07:52. At06:08:15 authoritative get confirmed Stopped,
zero active and two stopped replicas; last healthy sampled step9255, latest
verified save8940. Operator/reason not supplied; no explicit quota/access
denial in these events. Per user authorization, after arch2x8 predict-only
returned2/2 at06:08:44, submitted unchanged K launcher as r3 at06:09:05.
Actual new ID45375873 is recorded above;06:09:21 confirms Inqueue, and events
confirm external queue acceptance. No priority, allocation, RUN_NAME, output
folder or training configuration changes. Resume8940 must be verified after
allocation; never use K r2 logs as current r3 logs. Q remains queued without
duplicate submission. All four30B runs unfinished.

05:54:40–06:00:58: eight live status checks confirmed Q r2 Inqueue and K/V/O
r2 Running, each with two active replicas. K saved8940 asynchronously at
05:58:20.253 and explicitly confirmed success05:58:24.306, then completed9000
evaluation and resumed to9025. This checkpoint is approximately18.75B tokens
(62.5%); trainer save success was verified, not independent disk readback.
V/O latest8856/8779, latest saves still7152. No new errors, restarts,
submissions or configuration changes. Q r2 load7152 remains pending. All four
30B runs unfinished.

05:43:37–05:53:05: eight live status checks confirmed Q r2 Inqueue and K/V/O
r2 Running with two active replicas each. K/V completed8500 evaluations and
continued training; latest samples K8725/V8532/O8487. No new errors, saves,
submissions or configuration changes. At05:52:36 arch2x8 predict-only reported
2/2 schedulable, candidates H2002262/1105/2258/0217/0221. Q events still only
show external queue acceptance; prediction is not allocation. Latest confirmed
saved steps all7152; Q r2 loading remains unverified. All four30B runs unfinished.

05:36:01–05:41:59: seven live status checks confirmed Q r2 Inqueue and K/V/O
r2 Running with two active replicas each. All three completed8000 evaluations
and continued training; latest samples K8291/V8100/O8041. No new errors, saves,
submissions or configuration changes. At05:36:45 arch2x8 predict-only reported
2/2 schedulable, candidates H2002195/2258/0217/0221; Q events still only show
external queue acceptance. Feasibility is not allocation. All four unfinished;
latest saved steps all7152, with Q r2 checkpoint load verification pending.

05:28:59–05:34:25: seven live checks confirmed Q r2 Inqueue and K/V/O r2
Running, two active replicas each. O completed7500 evaluation and resumed
training; K reached8000 evaluation (last optimizer log7999). V/O7813/7757.
No new errors, saves, submissions or configuration changes. Latest verified
saved steps all7152; Q r2 still awaits allocation and load verification.
All four30B runs unfinished.

05:21:20–05:27:08: seven live status checks confirmed Q r2 Inqueue and K/V/O
r2 Running with two active replicas each. K/V completed7500-step evaluations
and resumed training; latest samples K7701/V7515/O7486. No new errors, saves,
submissions or configuration changes. At05:22:02 arch2x8 predict-only returned
2/2 schedulable, candidates H2002258/2156/0221/0217. Instantaneous feasibility
does not guarantee dispatch; Q still awaiting allocation. Latest verified
saved checkpoints all7152; Q r2 loading is unverified. All four unfinished.

05:12:29–05:19:40: nine live checks confirmed Q r2 Inqueue and K/V/O r2
Running, each with two active replicas. All three passed7000-step evaluation
and saved7152 successfully: K05:13:21.962, V05:17:52.592, O05:19:08.779.
Each explicit trainer message follows async save initiation to the unchanged
RUN_NAME/stage1/step7152 folder. Step7152 is14,998,831,104 tokens (~50%).
These are trainer save-success confirmations; independent disk readback of
these checkpoints has not been performed. Latest verified saved steps are now
7152 for all four variants. Q r2 loading7152 remains pending until allocation.
Latest training samples K7416/V7233/O7185; no new failures, submissions or
configuration changes. Next periodic save8940; all four30B runs unfinished.

05:04:01–05:10:52: eight live status checks confirmed Q r2 Inqueue and K/V/O
r2 Running, two active replicas each. K completed7000-step evaluation and
continued to7054; V/O6898/6856. No errors, new checkpoint confirmations,
submissions or configuration changes. At05:04:45 arch2x8 predict-only reported
2/2 schedulable, candidates H2002156/2316/2301/0221/2258/0217. Q events still
only show external queue acceptance; feasibility does not guarantee dispatch.
Latest verified saved steps remain Q7152, K/V/O5364. All four unfinished;
Q r2 load7152 remains unverified until allocation.

04:57:02–05:02:27: seven live status checks confirmed Q r2 Inqueue and K/V/O
r2 Running with two active replicas each. K/V/O all completed6500-step evaluations
and continued training; latest sampled steps6731/6553/6517. No new errors,
save confirmations or submissions. Full30B settings unchanged; latest verified
saved steps remain Q7152 and K/V/O5364. Q r2 has not started/load-verified yet.
All four remain unfinished.

04:49:32–04:55:17: seven live polls confirmed Q r2 Inqueue and K/V/O r2
Running with two active replicas each. V/O completed6000 evaluation and resumed
training. Latest samples K6458/V6284/O6251; no new errors or saves. At04:50:05
arch2x8 predict-only returned2/2 schedulable, five candidates H2002156/2258/
2301/1105/2316. Prediction does not reserve capacity or prove Q dispatched.
No submissions/configuration changes; latest verified checkpoints Q7152 and
K/V/O5364. Q r2 checkpoint loading remains pending; all four runs unfinished.

04:42:40–04:48:11: seven live polls confirmed Q r2 Inqueue and K/V/O r2
Running, two active replicas each. K completed6000-step evaluation and advanced
to6165; V reached6000 evaluation (last optimizer log5999), O5989. No new errors,
checkpoint confirmations or submissions. Same full30B configs; latest verified
saves Q7152, K/V/O5364. All four unfinished; Q r2 load not yet verified.

04:35:00–04:41:06: seven successful status polls confirmed Q r2 Inqueue and
K/V/O r2 Running with two active replicas each. V/O completed5500 evaluation;
all three continued progressing to K5896/V5726/O5703. No new errors, saves,
submissions or configuration changes. At04:35:44 arch2x8 predict-only returned
2/2 schedulable, candidates H2002316/2197/3104/2156/2258/2301. Prediction is not
a reservation; Q events still only show external scheduler queue acceptance.
Latest verified saved steps remain Q7152, K/V/O5364. Q r2 load remains pending;
all four30B runs unfinished.

04:26:54–04:33:27: eight live checks confirmed Q r2 Inqueue and K/V/O r2
Running, two active replicas each. K completed5500-step evaluation; all three
continued training after saving5364. Explicit trainer async save-success times:
K04:27:44.666, V04:31:51.860, O04:32:21.811. Each log names its unchanged
RUN_NAME/stage1/step5364 folder. Step5364 is11,249,123,328 tokens (~37.5%).
These are save-success confirmations, not independent disk readback tests.
Latest verified saved steps now Q7152 and K/V/O5364. Next periodic save for
K/V/O is7152; Q r2 still must verify loading7152 after allocation. No failures,
restarts or configuration changes this round. All four remain unfinished.

04:19:49–04:25:07: six live status checks confirmed Q r2 Inqueue and K/V/O
r2 Running with two active replicas each. All three passed5000-step evaluation
and continued training; latest samples K5261/V5098/O5085. No new failures,
submissions or checkpoint save confirmations. Q7152 and K/V/O3576 remain latest
verified saved steps; Q r2 has not yet verified loading7152. At04:20:48 arch2x8
predict-only returned2/2 schedulable and six candidates (H2002187/1106/0010/
0056/2210/0092). This is instantaneous feasibility, not reserved capacity or
proof of dispatch. Full30B/14306-step targets and configurations unchanged.

04:08–04:15:30: nine successful live checks confirmed Q r2 Inqueue and K/V/O
r2 Running, two active replicas each. K advanced4591->4887, V4459->4728,
O4454->4722; V/O completed4500-step evaluations and resumed training. Latest
sample times04:15:33.657/04:15:40.703/04:15:45.953 respectively. No restarts,
configuration changes or new save confirmations. Q loading7152 remains pending;
latest verified saved steps remain Q7152 and K/V/O3576. All four unfinished.

03:58–04:06:58: ten successful live status checks confirmed Q r2 Inqueue and
K/V/O r2 Running with two active replicas each. K advanced from4211 to its4500
evaluation; V/O completed4000 evaluation and advanced to4352 at04:05:54.247 and
4349 at04:06:00.314. At04:06:29 arch2x8 predict-only reported2/2 schedulable,
showing five candidates (H2002170/3081/0295/1105/2261); instantaneous feasibility
does not guarantee dispatch. Q individual events still only record external
scheduler queue acceptance. No new submissions, configuration changes or
checkpoint confirmations. All unfinished; Q7152 and K/V/O3576 remain latest
verified saved steps. Q r2's master/load must be verified once it starts.

### Q r1 stopped; Q r2 submitted safely

At03:55:18 Q r1 `mose-all-linear-q-silu-r1-2x8-0910-11812476` was confirmed
Stopped with zero active and two stopped replicas. Events show both pods
deleted03:54:34 followed by Running -> Stopped03:55:11. Logs show SIGTERM and
terminated DataLoader workers; no operator/reason identified and no explicit
quota/access denial in these events. Last healthy sampled Q step7714 at03:54:29.731;
latest successfully saved checkpoint remains7152, confirmed above/below.

At03:56:09 arch2x8 predict-only returned2/2 schedulable, candidates H2000221 and
H2000217. After terminal/zero-writer confirmation and resource query, Q r2 was
submitted through `conda activate MoSE && gpu_run arch 2 8` with the unchanged
run_q_only_silu_fp8.sh, RUN_NAME/STAGE/save folder and full30B configuration.
Actual ID captured03:56:31: `mose-all-linear-q-silu-r2-2x8-0910-91695506`.
Inqueue confirmed03:56:48. Identify its NEW master after allocation; do not read
old Q r1 logs as current. Loading7152 is expected but not yet runtime verified.

K/V/O r2 remained Running with two active replicas each at03:56:48. K step4138
observed03:56:52.966; V/O4000-step evaluation batches observed03:56:58/03:57:05.
No changes or duplicate submissions for those three. All four remain unfinished.

The following section describes the preceding all-running interval, not Q's
current queued r2 state.

### All replacements now training; Q 50% checkpoint saved

At03:48:00 all four current IDs were Running, each with two active replicas.
No new submissions this round; the queued r2 jobs were dispatched normally.
Latest sampled steps: Q7471 at03:48:04.024, K3802 at03:48:08.569,
V3654 at03:48:16.184, O3657 at03:48:21.924. All targets remain14306 steps/30B.

Q saved step7152 asynchronously starting03:40:20.268; explicit **Checkpoint for
step 7152 saved successfully** at03:40:24.248. This corresponds to14,998,831,104
tokens (approximately50%). Save folder unchanged. This is trainer save-success
evidence; independent disk readback of7152 has not been performed.

K r2 was confirmed Running03:40:14; V/O r2 Running03:43:40. Their new masters
are all **c4ca4** (cfcd2 peer logs explicitly report NODE_RANK=1/MASTER_ADDR):

| Variant | Master node / IP | Peer node / IP | Original step3576 checkpoint loaded successfully |
| --- | --- | --- | --- |
| K | H2002168 / 10.102.240.58 | H2002154 / 10.102.240.36 | 03:42:07.532 |
| V | H2002298 / 10.102.243.67 | H2002243 / 10.102.243.29 | 03:45:45.667 |
| O | H2002267 / 10.102.241.43 | H2000032 / 10.102.245.42 | 03:45:43.663 |

K/V/O logs explicitly name their respective original RUN_NAME/stage1/step3576
save folders, report successful loading and Training for14,306 steps, and then
advance beyond3576. Thus actual checkpoint readback and resumed training are
confirmed for every variant. Q retains cfcd2 master H2000221; do not switch it.
Monitor current Q r1 and K/V/O r2 IDs; all four are unfinished. Latest verified
saved steps: Q7152, K/V/O3576. Next periodic saves: Q8940, K/V/O5364.

Historical observations follow (older queue/checkpoint statements are superseded
by the current table and resume/save evidence above).

03:24–03:31:31: nine live status checks confirmed Q r1 Running and K/V/O r2
Inqueue. Q completed its6500-step evaluation and advanced to6813 at03:31:35.622,
surpassing the original attempt's last observed6738 (not a checkpoint).
At03:24:29 arch2x8 predict-only returned2/2 schedulable, showing H2000092 and
H2002210 as candidates. K/V/O individual events still only showed external
scheduler queue acceptance; prediction is not placement or reservation.
No duplicate jobs or configuration changes. Latest verified saved checkpoints
remain Q5364 and K/V/O3576; next Q periodic save expected7152. All unfinished.

03:15–03:22:43: eight successful live status checks confirmed Q r1 Running with
two active replicas and K/V/O r2 Inqueue. Q master advanced from6185 to6472
at03:22:48.184. One intervening internal get request hit an HTTP read timeout;
the same handles were re-polled successfully before any action. Observation
failure was not treated as job failure. No new submissions or configuration
changes. Q latest verified saved checkpoint still5364; K/V/O still3576.

03:08–03:14:04: seven further live internal status checks confirmed Q r1 Running
with two active replicas and K/V/O r2 Inqueue. Q advanced from5916 to6112
at03:14:08.352, passing the6000-step evaluation and returning to training.
At03:09:14 the authorized arch2x8 predict-only check returned0/2 schedulable
replicas, an instantaneous feasibility result, not a full GPU inventory.
No restart, duplicate submission or hyperparameter changes; all four unfinished.

03:00–03:06:21: seven live internal status checks confirmed Q r1 Running with
two active replicas and K/V/O r2 Inqueue. Q master logs advanced from5591 to5816
at03:06:25.459, with no new error in the sampled tails. No new submissions or
configuration changes. Latest verified checkpoint remains Q5364, K/V/O3576;
the observed Q optimizer step is not a new checkpoint confirmation.

### Q resumed; K/V/O second replacements

Q r1 Running with2 active replicas confirmed02:52:08. New master is **cfcd2**
onH2000221/IP10.102.244.23; peer **c4ca4** onH2000217/IP10.102.244.38.
Peer log explicitly says NODE_RANK=1 and MASTER_ADDR=10.102.244.23.
Master logs: loading original save-folder step5364 at02:54:13.420; checkpoint
successfully loaded02:54:20.534; Training for14,306 steps; first resumed step5365
at02:54:57.526, advancing through5437 at02:56:40.718. This verifies actual
checkpoint readback and continued training, not just scheduler startup. Offline
W&B warns resume is ignored; new segment needs step alignment later. Do not
claim automatic W&B history merging. Latest Q saved step remains5364.

K/V/O r1 attempts were stopped while still queued, no nodes assigned:

| Variant | Stopped attempt | Stop time |
| --- | --- | --- |
| K | mose-all-linear-k-silu-r1-2x8-0910-17994694 | 02:53:15 |
| V | mose-all-linear-v-silu-r1-2x8-0910-23928710 | 02:53:31 |
| O | mose-all-linear-o-silu-r1-2x8-0910-30232728 | 02:53:38 |

Events show Deleted pod and Inqueue -> Stopped, no operator/reason identified
and no explicit quota/access denial. At02:54:45 each had zero active and2 stopped
replicas. User authorized restarts and queueing; replacements submitted only
after these confirmations, with unchanged launcher/config/save paths via
gpu_run arch2x8. At02:55:43 predict-only reported0/2 schedulable; therefore r2
submissions intentionally queue under user's authorization. No priority or
resource-rule bypass. K/V/O latest verified saved step3576 remains unchanged.
Monitor **Q r1 plus K/V/O r2**, not the stopped r1 IDs. Q logs must use its NEW
cfcd2 master; identify K/V/O r2 masters after placement. All30B runs unfinished.

Original attempts were individually confirmed Stopped with zero active replicas
before replacement. Events record pod deletions followed by Running -> Stopped;
they do not identify an operator or explain why. Q/O logs show SIGTERM and
DataLoader termination, not evidence of model OOM. No quota/access rejection
appeared in these events. User explicitly authorized restarting killed attempts
and queueing. At01:45:26 arch2x8 predict-only reported2/2 schedulable (instantaneous,
not reservation). All replacements use gpu_run arch2x8, identical launchers,
RUN_NAME/STAGE/save directories and unchanged full30B hyperparameters.

NEXT: monitor replacement IDs, identify their actual master replicas, and verify
checkpoint loading plus resumed optimizer steps. Do not assume resumed simply
because submitted. Offline W&B will create segments requiring step alignment.
No second attempt should coexist with an active writer to the same save folder.
01:49 replacement Q events confirm external scheduler accepted it into its queue;
no rejection event. All four remained Inqueue at01:47/01:48/01:49 checks. No new
attempt has started yet. Local Trainer.fit inspection confirms save-folder
resume requests both trainer and optimizer state, but actual loading still needs
runtime verification. No training configuration or hyperparameter edits made.

01:50–01:54:57: six further authoritative internal status checks confirmed all
four replacement handles Inqueue. This is verified queue waiting, not a missing
handle or a failed job. No additional submissions. A larger untruncated tail of
original O master's log confirms step3576 saved successfully at01:41:33.861 and
last recorded step3649 before termination. Latest verified saved steps are now
Q5364 and K/V/O3576. These are trainer-log save confirmations, not independent
disk integrity verification. Actual replacement loading still unverified.

01:56–02:01:11: seven additional internal scheduler polls confirmed all four
replacement jobs Inqueue. At01:56:24 an arch2x8 predict-only returned2/2
schedulable, with six candidate nodes displayed; this is not a partition-wide
idle inventory or guarantee of queue dispatch. No priority/allocation changes,
new submissions or restarts were performed. Continue monitoring existing IDs.

02:02–02:07:40: seven more live internal status checks confirmed all four
replacements Inqueue. Individual events for every replacement still only record
external-scheduler queue acceptance; no additional failure or rejection events.
All four remain unfinished; queue waiting does not justify a duplicate submission.

02:08–02:13:53: seven more live checks confirmed all four replacements Inqueue.
At02:08:59 arch2x8 predict-only still reported2/2 schedulable, displaying four
candidate nodes (not a complete inventory). Actual dispatch has not occurred.
No extra submissions, allocation changes, or hyperparameter changes.

02:14:55–02:19:49: seven additional authoritative scheduler checks confirmed all
four replacement IDs still Inqueue. No active training replicas or terminal job
states; no duplicate submissions. Actual checkpoint restoration remains pending
dispatch. Goal remains active; this is verified waiting on existing job handles.

02:20:49–02:26:15: seven more live status checks confirmed all four replacements
Inqueue. Per-job events rechecked around02:21 contain only original external
scheduler acceptance, no rejection/failure. No new submissions or configuration
changes. Actual restoration/training completion remain unverified and unfinished.

02:27:34–02:32:52: seven further live polls confirmed all four replacements
Inqueue. At02:27:59 arch2x8 predict-only reported2/2 schedulable and displayed
twelve candidate nodes; not a partition-wide count or dispatch guarantee. No
new submissions/changes. Continue waiting for actual scheduler placement.

## Replacement cluster admission at02:33 (still awaiting placement)

Events now show cluster RJobs/replicas created at02:33:25(Q),02:33:27(K),
02:33:28(V),02:33:30(O), after external queue waiting. At02:33:52 all four
gangs report2 Pending/2 minAvailable with Unschedulable: insufficient GPUs,
CPU/memory on some nodes and selector mismatch on others. CLI replica label
STARTING does NOT mean containers/training have begun; no nodes/IPs assigned.
Both internal and public status endpoints agree at02:34; public endpoint now
sees all four jobs after cluster admission. Internal remains primary monitor.
Subsequent live checks02:34:45–02:39:42 confirm all remain Inqueue. No terminal
state or duplicate submission. Resume logs still pending actual placement.

02:40:50–02:45:47: seven more authoritative live checks confirmed all four
Inqueue, each with two STARTING replicas and blank node/IP assignments. No
running training or terminal state, no duplicate submissions or config changes.
Continue waiting on existing IDs; completion and restoration remain unproven.

22:48 resource prediction: arch2x8 schedulable2/2; returned idle8GPU candidates
H2002168 and H2002190. Instantaneous prediction, not a reservation.

Implementation validation: six unittest cases passed (four existing all-linear
tests and two new parameterized tests covering all four variants). Shell syntax
and git diff whitespace checks passed. Local OLMo tests explicitly map the
editable installation's GPFS2 source path to its local shared-storage mount alias;
the remote job uses the existing unchanged preparation/environment.

Resume uses each experiment's unchanged run ID/stage1 checkpoint folder.
Offline W&B segments need merging/alignment after a restart.

Do not replace on observation timeout, disconnected SSH or unavailable logs.
Confirm terminal scheduler state and inspect relevant events/logs first.
Do not modify unrelated jobs. See README.md for exact experiment settings.

Monitoring access: the internal arch gateway works for get/events/predict, but
its logs query returned HTTP502/EOF twice. The same remote read-only rjob logs
command works with KUBEBRAIN_CLUSTER_ENTRY=https://h.pjlab.org.cn and
KUBEBRAIN_NAMESPACE=ailab-llmarchitecture (verified22:53). No changes to remote
files or credentials were made. Do not interpret the internal logs failure as
job failure. Q master is replica c4ca4 (H2002190). Logs verify step25/14306 at
22:53:05. At step20: total tokens41,943,040, CE10.54, LR0.00005, device TPS92,084,
estimated remaining time5h39m (startup estimate, not a deadline).
Q's peer is cfcd2 (H2002168).

Use INTERNAL arch endpoint for authoritative job get/events/prediction. At22:53
the public HTTPS endpoint returned only the running Q job, omitting the three
queued jobs; the internal endpoint reconfirmed all three Inqueue at22:53:58.
Use public HTTPS only as the working logs fallback, not to infer missing jobs.

22:55:20 internal scheduler recheck: Q Running with two active replicas;
K/V/O Inqueue. Q logs at22:55:35: step130,272,629,760 tokens, trainCE6.039;
step132 reached22:55:37. No termination/OOM/numerical exception in this tail.
No restarts have been submitted.

22:57:07 internal scheduler recheck after a60-second wait: Q still Running,
2 active replicas onH2002190/2168; K/V/O still Inqueue. All four handles
confirmed live. No need to resubmit or change resource allocations.

## 23:05 monitoring update

Repeated live checks at22:58,23:00,23:03,23:05: Q Running, K/V/O Inqueue,
no terminal jobs. Q progressed through steps262,327,396,450,495,511.
23:00:43 arch2x8 predict-only reported0/2 schedulable; returned candidates
had fragmented GPUs (largest7), not a complete idle eight-GPU node.

Q first lm evaluation atstep500 completed23:05:06 in41.0s, then training resumed
to511 at23:05:22. Total tokens at500:1,048,576,000. Console-rounded validation CE:
c4_en4.123; dolma_books4.189; dolma_common-crawl4.226; dolma_pes2o3.196;
dolma_reddit4.311; dolma_stack2.559; dolma_wiki3.907; ice4.166;
m2d2_s2orc4.224; pile3.390; wikitext_1033.962.
These establish successful evaluation, not superiority to another experiment.
First periodic checkpoint still due at1788; no checkpoint integrity claim yet.

No new submission/restart, no training configuration changes, no git commit/push.
Next: continue live job/status monitoring, confirm K/V/O startup when scheduled,
and verify periodic/final checkpoints and training completion. Goal remains active.

## 23:13 monitoring update

Internal scheduler and public log checks at23:08,23:09,23:10,23:11,23:13 confirm
Q continuously Running and K/V/O Inqueue. Q progressed624->678->728->779->836.
All four scheduler handles are live; no terminal state, no restart necessary.
No numerical/OOM exception in sampled Q log tails. All full training runs remain
unfinished. Next lm eval at1000; first periodic checkpoint at1788.

## 23:18 monitoring update

Internal scheduler checks at23:14,23:15,23:17,23:18 reconfirm all four handles:
Q Running, K/V/O Inqueue. Q reached1019 at23:18:09 after successfully completing
step1000 lm evaluation at23:17:42 (34.3s). Console-rounded CE at1000:
c4_en3.772; dolma_books3.816; dolma_common-crawl3.891; dolma_pes2o2.864;
dolma_reddit4.001; dolma_stack2.078; dolma_wiki3.501; ice3.855;
m2d2_s2orc3.847; pile3.031; wikitext_1033.560.
All lower than this run's step500 metrics, not a between-experiment comparison.
No terminal state, restart, configuration change, or completion claim.
Next lm eval1500; first periodic checkpoint1788.

23:20–23:26: eight additional internal status/public log polls confirm unchanged
job states and Q progress1126->1162->1199->1236->1272->1309->1345->1382.
At1380, logged total tokens2.894B and trainCE3.116 (rounded). All still live,
no restart. Monitoring uses only authorized read-only CLI calls on gateway;
waiting/orchestration remain local, no remote files/scripts installed.

## 23:37 first periodic checkpoint

Continuous read-only status/log polling23:27–23:37: Q Running and K/V/O Inqueue;
no failed/killed jobs and no restarts. Q finished step1500 lm evaluation
at23:30:13 in35.2s and continued normally.

Q master log23:37:06: Saving checkpoint for step1788 asynchronously to
`/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/olmo3-1b-all-linear-q-only-silu-fp8-30b/stage1/step1788`.
At23:37:10.154: `Checkpoint for step 1788 saved successfully`.
Reported async duration3.586s; training subsequently reached1805.
This verifies the requested ~12.5% cadence and trainer-reported successful save;
no independent full disk/readback integrity check has been performed.
OLMo trainer.py logs this success only after the async future.result() succeeds.

Q's warmup ends at2000; next periodic save3576. Four experiments still unfinished.

## 23:51 K startup and continued monitoring

23:41:44 arch predict-only reported 2/2 schedulable with five returned complete
8-GPU candidates. This was a transient prediction, not a reservation or full
inventory. Existing queued jobs were retained; no duplicate submission.

K changed to Starting at23:43:29 and Running with two active replicas at23:44:03:
master c4ca4 on H2002195 (10.102.242.31), peer cfcd2 on H2002154 (10.102.240.36).
The master logs confirm NODE_RANK=0, two nodes, eight processes per node and
FSDP mesh (dp_replicate=2, dp_shard=8). Initial job-wide log retrieval encountered
a client-side TypeError on an empty peer response; subsequent per-replica
retrieval worked. This was not a training failure.

Eight read-only polling rounds23:44–23:51 confirmed Q/K Running and V/O Inqueue.
Q passed warmup2000 and advanced2079->2341. K completed startup and advanced
29->175. At K step170, console-rounded CE5.558 and total tokens356,515,840.
No training exception, terminal state, restart, or hyperparameter change.
Next: continue all-four monitoring, watch V/O scheduling, Q eval2500 and
periodic save3576, and K first eval500/save1788. Full 30B goal remains active.

## Sep10 00:00 monitoring update

Eight read-only status/log rounds23:53–23:59, plus internal status00:00:18,
confirmed Q/K Running and V/O Inqueue. Q advanced2433->2669; its step2500 lm
evaluation completed23:55:12.672 in35.2s and training resumed. K advanced263->520;
its first step500 evaluation completed23:59:34.263 in40.3s.

K console-rounded validation CE at500: c4_en4.132, dolma_books4.217,
dolma_common-crawl4.235, dolma_pes2o3.205, dolma_reddit4.325, dolma_stack2.562,
dolma_wiki3.927, ice4.209, m2d2_s2orc4.217, pile3.412, wikitext_1033.982.
These are early warmup observations, not evidence of final ranking.

23:59 arch2x8 predict-only returned1/2 schedulable, insufficient for a new
two-node gang. Returned node table is not a complete GPU inventory. Existing
V/O queues retained; no duplicate submission, failure, restart or config change.
Next checkpoints: Q3576, K1788. All four full30B runs remain unfinished.
Previous goal turn and this turn are verified waits with live scheduler handles;
K first validation completion additionally confirms a new training milestone.

## Sep10 00:13 all four training

Live monitoring observed V Starting00:03:16, then Running with two active
replicas00:03:52. V master is c4ca4 on H2002271 (10.102.243.30); peer cfcd2
on H2002171 (10.102.240.46). Logs confirm peer NODE_RANK=1 and master model
initialization with FSDP mesh(2,8). V step0 save succeeded00:05:58.171;
actual training reached28 by00:07 and282 by00:13.

O became Running with two active replicas00:07:53. O master is cfcd2 on
H2002323 (10.102.240.65); peer c4ca4 on H2001131 (10.102.238.16), confirmed
by peer NODE_RANK=1/MASTER_ADDR. O step0 save succeeded00:10:10.802;
actual training reached12 by00:11 and118 by00:13.

Q passed step3000 and reached3236; K finished step1000 lm evals
00:12:23.779 in34.8s and reached1041. All four original scheduler handles
are live Running, each two active replicas. No restart, duplicate submission,
resource alteration or hyperparameter change. All four remain below30B.
Next: Q periodic checkpoint3576; K first1788; V/O first500 evals and1788 saves.
Both new startup transitions are progress beyond merely successful submission;
continue monitoring until all four full runs and final checkpoints are verified.

## Sep10 00:24 second Q periodic checkpoint and V/O evaluations

Twelve live internal-status/public-log polling rounds00:14–00:24 confirmed
all four original handles Running, with two active replicas each. Sampled
progress Q3304->3674, K1108->1496, V347->701, O184->547. No terminal state,
restart or configuration change; this turn is a verified wait with milestones.

Q step3576 asynchronous save began00:21:58.085 and succeeded00:22:01.440:
`/mnt/shared-storage-gpfs2/intern-pretrain-shared02/taoji/ckpts/Olmo3-MoSE/olmo3-1b-all-linear-q-only-silu-fp8-30b/stage1/step3576`.
This is the second periodic checkpoint (~25%, 7,499,415,552 tokens), verified
by trainer success log and continued training, not an independent readback.
Q step3500 evals completed00:20:09.566 in35.5s.

V first step500 evals completed00:19:34.849 in41.0s; O first step500 evals
completed00:23:31.398 in38.7s. Both resumed training. O console-rounded CE500:
c4_en4.143, dolma_books4.218, dolma_common-crawl4.247, dolma_pes2o3.225,
dolma_reddit4.327, dolma_stack2.553, dolma_wiki3.914, ice4.205,
m2d2_s2orc4.220, pile3.405, wikitext_1033.982. These are early warmup metrics,
not final comparative conclusions.

Next: K eval1500/first periodic1788, V/O periodic1788, Q next periodic5364.
All four full30B runs remain unfinished. No pending job remains at this check.

## Sep10 00:36 K first periodic checkpoint

Twelve fresh read-only internal status/public log rounds00:26–00:36 confirmed
all four original jobs Running with two active replicas each. Progress observed:
Q3762->4136, K1557->1945, V786->1144, O633->999 (eval1000 in progress).
No terminal state or restart; previous turn and this one are verified waits
with additional successful checkpoint/evaluation milestones.

K trainer log00:32:12.906 explicitly confirms `Checkpoint for step 1788 saved
successfully`, and subsequent training progressed past it. This verifies the
first ~12.5% periodic save at3,749,707,776 tokens via trainer logs, not an
independent full checkpoint readback. Q4000 evals completed00:32:41.283 in34.6s;
V1000 evals completed00:32:32.975 in35.1s. Both resumed normal training.

No settings or allocations changed. Next: K warmup2000, V/O first periodic1788,
Q next periodic5364. All four full30B training objectives remain unfinished.

## Sep10 00:47 continued training after K warmup

Twelve authoritative status/log rounds00:37–00:47 confirmed all four original
handles Running with two active replicas apiece. Q progressed4213->4586,
K1999->2387, V1219->1579, O1073->1462. K finished2000-step evals at00:37:55.658
in34.7s and continued beyond warmup. Q4500 evals finished00:45:08.256 in35.4s;
V1500 evals finished00:45:27.331 in35.8s. Training resumed after each.

No observed terminal state, exception, restart, duplicate submission or settings
change. Latest periodic save confirmations remain Q3576 and K1788; V/O first
periodic1788 still ahead. O has progressed beyond its1000 evaluation point.
This is a verified wait with live handles and additional completed evals;
full30B goal remains active for all four experiments.

## Sep10 00:58 all four first periodic saves confirmed

Twelve live status/log rounds00:48–00:58 confirmed all four original handles
Running, each with two active replicas. Q4644->5018, K2442->2808,
V1635->1999 (eval2000 in progress), O1499->1884. No terminal state or restart.

V trainer00:52:34.982: `Checkpoint for step 1788 saved successfully`.
O trainer00:56:01.950: `Checkpoint for step 1788 saved successfully`.
Both continued training afterwards. All four now have trainer-confirmed first
~12.5% periodic checkpoints; Q additionally has3576 (~25%). These are log-level
successful saves, not independent full checkpoint readback/integrity tests.

O1500 evals completed00:49:01.209 in35.6s; K2500 completed00:50:39.055 in35.8s;
Q5000 completed00:57:37.094 in34.5s. All resumed training. No configuration or
resource changes. Next: V/O warmup2000, Q5364 periodic, K/V/O3576 periodic.
Previous turn was a verified wait; this turn verified new save/eval milestones.
All four full30B objectives remain unfinished and monitored.

## Sep10 01:10 all four beyond warmup; Q third periodic save

Twelve fresh status/log rounds01:00–01:10 confirmed all four original jobs
Running with two active replicas each. Q5105->5499 (eval5500 in progress),
K2895->3260, V2080->2466, O1970->2338. All four now progressed beyond warmup.
O2000 evals finished01:01:42.745 in34.6s; K3000 finished01:03:21.580 in34.7s;
both resumed training.

Q trainer01:06:20.330 confirms `Checkpoint for step 5364 saved successfully`.
This is the third periodic save (~37.5%, 11,249,123,328 tokens), followed by
continued training. Evidence is trainer success log, not independent readback.
Confirmed periodic saves: Q1788/3576/5364; K/V/O1788. Next Q7152 and K/V/O3576.

No failed/killed job, restart, duplicate submission, configuration or resource
change. Previous turn and this one are verified waits with live handles and
successful checkpoints/evaluations. All four full30B runs remain unfinished.

## Sep10 01:21 K second periodic save

Twelve authoritative status/log polling rounds01:11–01:21 confirmed all four
original handles Running with two active replicas each. Q5555->5952,
K3338->3700, V2518->2905, O2415->2784. No failure, restart or settings change.

K trainer01:18:00.357 confirms `Checkpoint for step 3576 saved successfully`,
then training continued. This is its second periodic save (~25%). Verified
periodic saves: Q1788/3576/5364; K1788/3576; V/O1788. Log-level verification,
not independent checkpoint readback. K3500 evals finished01:16:06.219 in37.1s;
V2500 finished01:11:11.764 in35.8s; O2500 finished01:14:22.955 in36.4s.

Previous/current turns are verified waits with live handles and additional
successful milestones. Next: Q6000 and V/O3000 evaluations, V/O3576 saves.
All four full30B runs still incomplete; goal remains active.

## Sep10 01:33 additional evaluation milestones

Twelve fresh internal status/public log rounds01:22–01:32 confirmed all four
original handles Running with two active replicas each. Sampled progress:
Q6013->6412, K3784->4151, V2989->3350, O2868->3236. No terminal state,
restart, duplicate submission or configuration/resource change.

Q6000 evals completed01:22:36.343 in34.7s; V3000 completed01:24:01.616 in36.7s;
O3000 completed01:27:00.956 in36.3s; K4000 completed01:28:50.267 in34.8s.
All resumed training. Periodic save confirmations unchanged: Q through5364,
K through3576, V/O through1788. Next V/O3576 and Q7152 saves.
This and the previous turn are verified waits with live handles; all four
full30B training objectives remain unfinished and monitored.
