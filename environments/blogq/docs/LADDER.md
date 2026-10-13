# LADDER — checkpoints of record and the anchors they are measured against

Rules: a number carries its judge version and decode temperature or it is
not on this ladder. Same judge for any comparison. Greedy (t=0) is the
serving number; t=0.7 is the harvest/sampling number. Absolutes understate
(strict judge: ~10% false-accept, ~21% false-reject measured 2026-09-03);
deltas are fair.

## Anchors (carried in from the closed campaign — the only legacy numbers kept)

Judge: judge_v2 on the current 35B box (Qwen3.6-35B FP8), unless noted.

| model | lineage | agentq-200 (default, t0.7) | blogq held-out default | blogq held-out strict | notes |
|---|---|---|---|---|---|
| stock Qwen3.5-0.8B | — | .120 | — | — | |
| v3.4 | 0.8B, gazetteer+legacy traj (unlaundered) | .245 | .047 (ho1k, t0.7) | .025 | pre-blogq |
| v34rr | v3.4 recipe, typed-citation legacy, from stock (vast) | .210 | .045 (ho200) | .010 | |
| **v3.5** = v34rr + blogq p1 (19K traces, continue-train) | 2×3090, lr 1e-5, cap 8192 | **.275** | **.270** (ho200, t0.7) / **.310** (greedy) | .140 / .195 (greedy) | reaches gold evidence on ~60% of ho200; passes ~40% of those |
| teacher Qwen3.6-35B | — | — | .299 (ho1k, t0.7) | .050 | strict/default 17%: the decoration gap |
| Gemini Flash (prod) | — | agentq: FlashLite .790 / 3.7 Flash .765 (bank v2, judge v2 final) | see the v2.1 table below | see below | the product bar |

ho1k = the full held-out 1,000. **ho200 = the 200 ids in
`data/banks/ho200_ladder.jsonl`** (the set the legacy ho200 runs used — NOT
the first 200 lines of the file; a bank built that way shares only 43 ids
and is not ladder-comparable).

### ho200-ladder, judge v2.1 (full evidence, tool-evidence grounding), greedy — 2026-09-03

Same 200 held-out questions (`data/banks/ho200_ladder.jsonl`, the ids the
legacy ho200 runs used), same harness/tools/prompt, same judge.

| model | default | strict | strict passes with clean citations | markers missing | identity fail | web_fetch |
|---|---|---|---|---|---|---|
| gemini-3.8-flash | **.605** (121) | .100 (20) | 20 | 0 | 0 | 70 |
| v3.5 (0.8B) | .310 (62) | **.205** (41) | 3 | 11 | 27 | 181 |

Read: the product bar wins coverage 2x; **the 0.8B is 2x more honest under
strict grounding**. The 0.8B's citation discipline is the gap — 27 of its
41 strict passes never cite a retrieved places-DB ref for the main place
(the app-join key), where Gemini is clean on all 20. v4's job is coverage
plus citation discipline WITHOUT giving up the honesty edge.
⚠ The earlier "Gemini .680/.095" figure used a different 200 questions
(first 200 lines of the held-out 1K, only 43 ids in common) — not ladder-
comparable, superseded by the table above.


### v3.5c vs v3.5 — the clean-data A/B (running overnight 2026-09-03)

Same v34rr start, same 48,000 replay rows verbatim, lr 1e-5, eff batch 64,
cap 8192, 1 epoch, 2x3090. **Only the blogq slice differs.**

| | v3.5 | v3.5c |
|---|---|---|
| blogq traces | 19,131 (pass-1 keeps, laundered, ungated) | 8,227 (judge v2.1 full-evidence + identity + contract) |
| blogq rows | 78,839 | 38,290 |
| content rewriting | yes (qwen launder) | none; markers inserted text-preservingly |
| of its traces, kept under today's gates | ~47% | 100% by construction |

Results land in `runs/cleanprobe_eval/SUMMARY.txt`; paired per-question
diffs in `runs/reports/v35c_vs_v35_{ho200,agentq}.html`. Judge for BOTH:
`qwen/qwen3.6-35b-a3b` on OpenRouter (the rented box is off), so v3.5's
stored rollouts are re-judged rather than compared against their old
.310/.205 numbers.

## Checkpoints produced in model/

| id | date | recipe | mix manifest | agentq-200 | blogq ho1k default (t0.7 / greedy) | strict | citation validity | battery vs stock | notes |
|---|---|---|---|---|---|---|---|---|---|
| (v4) | | from stock, full-FT lr 2e-5 eff 64 cap 8192 | | | target ≥ .25 (beat v3.5 .270) | small gap | ~1.000 | ≥ | |

### ho200-ladder, 2026-09-05 05:30 — v3.5++ (first contract-v2 probe) vs v3.5c, greedy

Judge: rented Qwen3.6-35B FP8, judge_v2.1, 1 vote, SAME for both (v3.5c's
stored rollouts re-judged; the OpenRouter judge ran out of credits mid-eval).
Each model under its own contract's harness (v3.5c: v1 summaries; v3.5++:
v2 handles + 400-tok chunks). strict+identity is the contract-agnostic headline.

| model | default | strict | strict+identity | strict+gates | protocol ok |
|---|---|---|---|---|---|
| v3.5c = v34rr + 38K v1 clean rows | .255 (51) | .100 (20) | .090 (18) | .085 (17) | 194 |
| **v3.5++** = v34rr + 54K v2 rows (A/C/Gemini/Qwen-p4) | .240 (47) | .092 (18) | .082 (16) | .071 (14) | 183 |

Paired strict on 196 shared ids: v3.5++ wins 9, v3.5c wins 11. **No
difference at n=200** (±~.04). The v2 FORMAT was learned (14 of 18 strict
passes clean on the v2 gates; v3.5c's were mostly clean on v1) — the gap
between these models and the product bar is coverage/correctness, not
format. v3.5++ goes over budget more (183 vs 194 ok; 4.9 vs 4.1 tools/q).
Both copy the example ref W12903132 from the place_details tool description
into a few finals (5 / 3) — placeholder that description.

agentq-200, greedy (v3.5++ judged on the rented box; v3.5c's stored verdicts
are from the OpenRouter judge, same model family — a near-same-judge read):

| model | default | strict | strict+identity | strict+gates |
|---|---|---|---|---|
| v3.5c | .322 (64) | .141 (28) | .126 (25) | .121 (24) |
| v3.5++ | .327 (65) | .141 (28) | .136 (27) | .090 (18) |

Identical. The strict+gates gap is v2-format gate drops on agentq finals
(9 of 27) — agentq's legacy bank has no web evidence for many questions, so
the "web used and numbers but no page credited" rule bites; not a quality
difference.

### ho200-ladder + agentq, 2026-09-05 17:30 — v3.5+++ (clean replay) vs v3.5++ vs v3.5c, greedy

Judge: then-active rental, Qwen3.6-35B FP8, judge_v2.1, 1 vote, SAME
for all three (stored rollouts re-judged). v3.5+++ = v3.5++ recipe with the
48K replay replaced by 33K chat/fc-only rows (the old replay carried 15K
rows on the places prompt in the v1 citation format) and the crawl-2 gates.

| model | ho200 default | ho200 strict | strict+identity | strict+gates | agentq default | agentq strict | agentq strict+identity |
|---|---|---|---|---|---|---|---|
| v3.5c | .250 | .090 | .085 | .080 | .327 | .131 | .121 |
| v3.5++ | .245 | .117 | .102 | .092 | .317 | .146 | .136 |
| **v3.5+++** | **.275** | **.120** | **.115** | **.105** | .327 | .101 | .101 |

Read: on the blogq ladder the v2 package now sits +3 to +6 points over
v3.5c on every column (strict 24 vs 18), the clean replay adding a little
over v3.5++ — still inside the ±.04 noise band at n=200 per column, but the
direction is consistent across four columns. On agentq (legacy bank, mostly
no web evidence) v3.5+++ lost ~3 points strict vs v3.5c. Coverage (default
~.25-.28) is unchanged by any of these runs; the v2 package buys grounding
and format, not reach. Judge-box drift: v3.5c ho200 strict re-judged on the
replica = .090 vs .100 on the rented box yesterday (1 vote).

### LADDER RE-BASE 2026-09-06 23:30 — every checkpoint, ONE harness, ONE judge (judge_v2.2, majority of 3)

Harness = pass-6 settings (places 2 / web 2 / fetch 1 / locality column /
2600-char chunks, contract v2, greedy). Judge = judge_v2.2 (`JUDGE_V22=1`:
memory-first, unresolved-primary, compile, DB spatial checks, locator pass),
3 votes, Qwen3.6-35B FP8 on the rental. Fresh rollouts for the 0.8B
checkpoints and the teacher; Gemini = its 2026-09-03 rollouts re-judged (old
harness). Files: `runs/ladder_v22/`, script `evals/run_ladder_v22.sh`.
**These supersede every earlier ho200 number above for comparisons.**

| model | default | strict | strict+identity | strict+gates | in budget |
|---|---|---|---|---|---|
| gemini 3.8 flash (old harness, re-judged) | .610 (122) | .125 (25) | .125 | .125 | 199 |
| teacher Qwen3.6-35B | .450 (90) | **.240** (48) | .240 | .155 (31) | 122 |
| **v3.5+++** (0.8B) | .295 (59) | **.180** (36) | .170 (34) | .150 (30) | 170 |
| v3.5++ | .245 (49) | .155 (31) | .150 (30) | .120 (24) | 160 |
| v3.5c | .225 (45) | .090 (18) | .045 (9) | .015 (3) | 197 |

Read: the v2-package checkpoints hold their order (v3.5+++ > v3.5++ > v3.5c) with
a wider spread than under v2.1; v3.5c collapses on identity/gates because its
v1-contract finals have no map object. The teacher forfeits 78/200 to budget
overruns (same 38% as pass 6) and still leads strict; on gates it is level with
v3.5+++ (31 vs 30). Gemini's strict is depressed by the old harness (web 1, no
locality) — its default .610 is the coverage bar, its strict is not comparable.

Judge-vs-harness split for v3.5+++ (same model): old harness + v2.1 ×1 = .120;
old harness + v2.2 ×3 = **.160** (`runs/ladder_v22/v35ppp_oldharness.strict.jsonl`);
new harness + v2.2 ×3 = .180. So ~+.04 is the judge (3-vote majority + the
locator recovering overlooked specifics; it is stricter elsewhere) and ~+.02 the
harness — each inside the ±.04 band at n=200.

### Reach / localization columns (2026-09-07 01:00) — from the re-based ladder rollouts

"Can't find the place" measured mechanically: distance from the model's FIRST
search_places anchor to the gold coords, and whether a gold ref ever appears in
its tool results. Same harness for all (Gemini = old-harness rollouts).

| model | median anchor err | <5 km | <25 km | gold ref retrieved | strict pass / retrieved |
|---|---|---|---|---|---|
| gemini 3.8 flash | 0.7 km | .78 | .93 | .82 | — |
| teacher 35B | 6.8 km | .39 | .65 | .53 | 48/105 |
| v3.5+++ | 61 km | .28 | .41 | .27 | 36/53 |
| v3.5++ | 95 km | .27 | .41 | .26 | 31/51 |
| v3.5c | 136 km | .26 | .37 | .21 | 18/42 |

Read: the 0.8B passes ~2/3 of the questions where it retrieves the gold place
and retrieves it ~1/4 of the time — reach IS the gap. 38% of v3.5+++'s anchors
are >200 km off (Kisdon Force → Sydney; Dromana Beach → North Carolina).

Facts retention probes on v3.5+++ (no tools, greedy, `runs/` ad hoc 2026-09-07):
- ladder places in the facts set vs not: anchored <25 km **.42 vs .43** — coverage made no difference.
- name only → <25 km .14 / <200 km .26 (covered = uncovered); name + type + town/country
  → **.59 / .80** (median 10 km); covered facts rows with their own prompt → .61 / .94.
  ⇒ the retained geography is SETTLEMENT-level (the facts prompt leaks the town in
  parentheses); place-name→coords did not stick at 0.8B / v34rr recipe.
- coordinate-token logprob vs error (name-only): most-confident third <200 km .41,
  least-confident .16 — a usable but weak targeting signal (Jon's idea).
Open (Jon 2026-09-07): scale knowledge injection to ~10M places from blog/wiki/business
text vs harness features (anchor-free name search; "anchor on the town in the
question" skill). Decision gate proposed: controlled exposure sweep on a fixed place
set with NO town leak, name-only recall probe, before any 10M-scale generation.

### v4 (from stock, 2026-09-07) — ho200 ladder, same harness + judge as the 23:30 re-base

v4 = stock Qwen3.5-0.8B + `data/mixes/v4` (574,859 rows: blogq 91K requalified, routing 84K
oracle-RIGHT, facts 250K, chat 60K, fc 29K, nemotron 60K; ONE prompt, places 3 / web 2 / fetch 1 /
locality / route tool), full FT lr 2e-5, eff 64, cap 16K, 1 epoch, 8,982 steps on 1× RTX Pro 6000
(11.5 h, loss 0.72 → ~0.54 plateau from step 3,500). Checkpoint `runs/train_v4_rental/{final,serve}`.

| model | default | strict | strict+gates | reach | anchor <25 km | in budget | strict / in-budget |
|---|---|---|---|---|---|---|---|
| v3.5+++ (budget 2) | .295 | **.180** | .150 | .27 | .41 | 170 | 36/170 |
| v4 (budget 2, as the others) | .280 | .105 | .095 | .29 | .39 | 142 | 21/142 |
| v4 (budget 3 = its trained prompt) | .275 | .090 | .080 | .28 | — | 150 | 18/150 |
| agentq-200 strict: v4 **.185** (v3.5c .131 / v3.5++ .146 / v3.5+++ .101 under v2.1 ×1 — not same-judge) |

Read: v4 is BEHIND the v3.5 line on blogq strict (−7 to −9 pts, outside the ±.04 band) at equal
reach and anchoring; it over-calls tools (54 budget kills at budget 2, 50 at budget 3 — web_search
30 / search_places 24) and puts web handles or names into the map object (3 compile fails the old
line never makes). agentq strict is its best number. Paired vs v3.5+++: both 9, v4-only 12,
v3.5+++-only 27 (6 budget, 3 compile, rest fabricated specifics).
Hypothesis (not yet tested): EXPOSURE — v3.5+++'s lineage saw agentic rows ~3× (v34rr 19.6K steps
+ three continue-trains); v4 saw its 175K agentic rows once, at half the supervised tokens.
Reach unchanged in either direction confirms 2026-09-07 01:00: neither facts nor traces move
localization at 0.8B.

### GRPO round 1 on v4 (prime-rl, geogym-blogq env) — 2026-09-09 06:50, ho200, Gemma 4 31B v2.2 ×1 for ALL rows

20 steps × (128 q × 8 samples), lr 1e-6, IPO eps 0.1, kl 0.001; reward −1 mechanical / 0 judged fail /
+1 pass; pool = 86K untrained questions; run `runs/grpo/v4_iter1/iter1`, checkpoint
`runs/grpo/eval_iter1/step20_serve`. Training reward −0.42 → +0.13; on-policy pass rate 3% → ~12%.

| model | default | strict | strict+gates | reach | anchor<25 km | in budget | chars |
|---|---|---|---|---|---|---|---|
| v4 (iter 0) | .210 | .090 | .080 | .29 | .39 | 142 | 663 |
| **v4rl (round 1)** | .295 | **.260** | **.260** | .32 | .41 | 199 | 436 |
| v3.5+++ | .195 | .155 | .120 | .27 | .41 | 170 | 686 |
| teacher 35B | .395 | .230 | .130 | .53 | .65 | 122 | 690 |

Read: +17 pts strict on held-out questions in one round, all of it conversion (reach flat): budget kills
gone, every strict pass clears the gates. Caveats: single Gemma vote (±.04 at n=200) and the eval judge
IS the reward judge — Sonnet review of the 52 strict passes pending (`runs/grpo/eval_iter1/pass_review_*`).
Answers 34% shorter. Probe 1K×4 (mechanical-failure rate vs iteration 0) pending.
Sonnet audit of the 52 v4rl strict passes (07:00): 41 agree / 8 minor / 3 major (6% false-accept,
same as pre-RL) — no evasive/terse answers; residual = credits pointing at the wrong page.

### GRPO round 2 (v4rl2) + search radius 100 km — 2026-09-08 20:32, ho200 greedy, Gemma 4 31B v2.2 ×1 for all rows

Round 2 (`runs/grpo/eval_iter2/step20_serve`) was never written here; both rows below are tallied with
the same snippet (`rl/overnight.sh` ladder block) so they are comparable to each other. `rl/ROUND3.md`
quotes round-2 strict as .345 from a different tally of the same verdict files; the snippet gives .390.

The second row is the SAME model re-run with `WPLACES_SEARCH_RADIUS_KM=100` (explicit 100 km
`/places/search` radius instead of the server's dynamic 5–100 km, median 30). Probe replay showed the
0.8B's anchors are routinely 10–300 km off and a wider radius surfaces the gold with ~1% precision loss.
Files: `runs/grpo/eval_iter2_r100/`.

| model | default | strict | strict+gates | reach | anchor <25 km | in budget | mean tools |
|---|---|---|---|---|---|---|---|
| v4rl2 (round 2), dynamic radius | .370 | .390 | .390 | .44 | .54 | 199 | 3.61 |
| **v4rl2, radius 100 km** | **.500** | **.510** | **.510** | **.55** | .55 | 200 | 3.44 |

Paired strict: both 72, dynamic-only 6, 100 km-only 30. Read: +12 pts strict from a harness knob, with
no training — all of it reach (.44 → .55) at unchanged anchoring (.54 → .55). This is a HARNESS change:
every later rung must run with the same radius, and earlier rungs are not comparable on reach until
re-run. Judge unchanged (single Gemma vote, ±.04 at n=200).

### Judge of record moved to Gemma 4 31B NVFP4 on the RTX 5090 — 2026-09-08 21:25, ho200 re-judged, every row below on the SAME judge

Rows are the existing rollouts (no re-run) re-judged on `[judge_gemma_nvfp4]` (single vote, v2.2 strict /
default). Files: `runs/ladder_nvfp4/`. The NVFP4 judge is a shade more lenient than the FP8 box on every
rung (+1 to +6 pts, same direction each time); per-row strict agreement 190–197/200; ordering unchanged.
Throughput 62 verdicts/min at concurrency 48 vs 78 on the FP8 box, at $0.41/h vs $0.89/h.

| model | default | strict | strict+gates | reach | in budget | strict agreement w/ FP8 |
|---|---|---|---|---|---|---|
| v4 (pre-RL) | .225 | .115 | .085 | .29 | 142 | 193/200 |
| v4rl (round 1) | .360 | .280 | .270 | .32 | 199 | 194/200 |
| v4rl2 (round 2) | .415 | .405 | .405 | .44 | 199 | 197/200 |
| **v4rl2 + search radius 100 km** | **.545** | **.530** | **.530** | **.55** | 200 | 194/200 |
| teacher 35B | .415 | .280 | .105 | .53 | 122 | 190/200 |

Use THIS table as the baseline for v4rl3. The FP8 rows above it are retired for comparison purposes.

### GRPO round 4 (v4rl4) — 2026-09-10 10:24, ho200 greedy, Gemma 4 31B NVFP4 v2.2 ×1 (same judge as every row above)

Init = **v4rl3s** (the style SFT, `runs/train_style_v1`), 20 steps, same recipe as round 3 plus: the
"According to" opener as a cosmetic style flag (−0.25 on a pass), 4,421 SFT-burned questions excluded,
seed 50, radius 100, pin gate on judged fails. First attempt lost 10 blind steps to a wsearch outage and
a result-line format change (REWARD_HACKS §18); this is the clean relaunch. Files `runs/grpo/eval_iter4/`.
**strict+gates now means HARD gates only** — the new opener flag is cosmetic and is excluded here (it
fires on 97% of pre-style rungs and would zero their column; `rl/overnight.sh` snippet updated to match).

| model | default | strict | strict+gates | reach | in budget | chars | opens "According to" | "according to" anywhere | "you" |
|---|---|---|---|---|---|---|---|---|---|
| v4rl3 (round 3) | .585 | .580 | .580 | .60 | 195 | 177 | 97% | 98% | 18% |
| v4rl3s (style SFT) | .540 | .530 | .510 | .59 | 188 | 280 | 45% | 88% | 54% |
| **v4rl4 (round 4)** | .540 | **.545** | .540 | .56 | 200 | 224 | **2%** | 88% | 50% |

Paired strict: vs v4rl3 both 99 / only-v4rl3 17 / only-v4rl4 10 (−3.5 pts); vs v4rl3s both 90 /
only-v4rl3s 16 / only-v4rl4 19 (+1.5 pts). Read: RL from the styled init recovered 1.5 pts over its
start and kept roughly half the voice (224 chars, 50% second person, opener gone) but did not get back
to round 3's accuracy, and "according to" merely moved mid-sentence (88% still contain it). In-training
pass rate matched round 3 by step 16. Length eroded 298 → 220 chars across the run: the reward still
pays nothing for a second sentence. Budget back to 200/200 (the SFT's search-loop habit trained out).

### Gemini 3.8 Flash on the CURRENT harness — 2026-09-10 11:48 (replaces the stale 2026-09-03 row)

Rolled out via OpenRouter, greedy, radius 100, locality, chunks 2600, budgets 3/2/1, context OFF; judged on
the NVFP4 ladder judge like every row above. Files `runs/gemini_ho200/`.

| model | default | strict | reach | in budget | calls | chars |
|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | **.720** | .100 | **.85** | 199 | 3.68 | 835 |
| v4rl4 (round 4) | .540 | .545 | .56 | 200 | 2.53 | 224 |
| v4rl3 (round 3) | .585 | .580 | .60 | 195 | 3.33 | 177 |

Read: Flash is 14-18 pts ahead on "found and answered" (default) and its strict .100 is almost entirely
"unsupported specifics" — a 1T model stating true things it did not read, which the strict judge (by
design) refuses to credit. **The gap that matters is reach: .85 vs .56-.60 with identical tools.**

**Blind Sonnet grading, format ignored** (right place + grounded in the system's OWN evidence + answers
the question; X/Y randomised per row, key held back; `runs/gemini_ho200/blind/`): v4rl4 148/200 = 74%,
Gemini 104/200 = 52% (61 of Gemini's misses were right-place-but-unsourced); preference v4rl4 79 /
Gemini 61 / tie 60. Jon's read (2026-09-10): this measures the grounding discipline the 0.8B MUST have
because it cannot memorise, not answer quality vs Flash — do not sell it as "beats Flash". Gold flagged
by graders: bq-098438, bq-011661, bq-041171, bq-014930 (Migration Series is at the Phillips, not SAM).

### 1K×4 probe, v4rl4 — 2026-09-10 12:20 (`runs/grpo/eval_iter4/probe.*`, NVFP4 judge, radius 100, t=0.7)

| model | per-rollout pass | pass@4 (any of 4) | all 4 pass | over budget | median chars |
|---|---|---|---|---|---|
| v4rl (round 1) | 14.2% | 31.1% | 2.2% | 20 | 381 |
| v4rl2 (round 2) | 31.2% | 47.7% | 15.0% | 17 | 174 |
| **v4rl4 (round 4)** | **48.6%** | **63.6%** | **31.5%** | 1 | 206 |

Rounds 1-2 were judged on the FP8 box under the old dynamic radius, so part of the round-4 jump is the
100 km radius (worth ~12 pts strict on ho200 to the same model), not RL alone; v4rl3 and v4rl3s have no
probe row (skipped / never run). Read: sampling ceiling pass@4 = 63.6% → the greedy target for round 5;
the model now passes ALL four samples on 31.5% of questions (was 15%), i.e. consolidation is real.
Over-budget is gone (1/4,000).

## 1K×4 probe, v4rl4, with audited bad questions removed (2026-09-10)
Same rows/verdicts as the recorded v4rl4 probe; only the denominator changes. Bad ids from the Sonnet
question audit (`runs/grpo/eval_iter4/qaudit`, covers all 364 all-fail + 100 all-pass questions) plus the
author-second-person regex. Caveat: the 536 questions with 1-3 passes were not audited, so this removes
bad questions only where they hurt — a slight flattering bias.

| denominator | n | per-rollout | pass@4 | all-4 |
|---|---|---|---|---|
| all (recorded) | 1000 | .486 | .636 | .315 |
| minus nonsense (author voice / anchorless / unanswerable) — `data/banks/probe1k_nonsense_ids.txt` | 971 | .494 | .645 | .320 |
| minus all Sonnet BAD (+ ambiguous, gold) — `data/banks/probe1k_bad_ids.txt` | 943 | .506 | .661 | .327 |

Use the "minus nonsense" row as the comparable for round 5 (same 971 ids for every later probe).

## ho200 Sonnet question audit (2026-09-10) — ladder rescored without bad questions
Sonnet labelled all 200 from question + gold only (`data/banks/ho200_sonnet_labels.jsonl`): 9 BAD = anchorless 7,
author voice 1, ambiguous 1, gold 1 (bq-046798). Of the 4 regex author flags Sonnet kept 2 as GOOD (present-tense
hypotheticals); of the 4 grader-flagged golds it caught 2 as anchorless. Id files: `ho200_nonsense_ids.txt` (Sonnet
nonsense ∪ regex, 9) and `ho200_bad_ids.txt` (all Sonnet BAD ∪ regex ∪ grader golds, 13). Same verdict files as the
recorded rows; denominator only.

| model | strict, all 200 | strict, −nonsense (191) | strict, −all bad (187) |
|---|---|---|---|
| v4rl | .260 | .262 | .262 |
| v4rl2 | .390 | .403 | .406 |
| v4rl2 r100 | .510 | .524 | .529 |
| v4rl3 | .580 | .597 | .604 |
| v4rl4 | .545 | .560 | .567 |

Ordering unchanged; the bad questions are ~all-fail for every model, so removal lifts each row by 1-2 points.
Report the 191-denominator alongside the 200 from round 5 on (keep 200 for comparability with older rows).

## v4rl5 — round 5 (clean Sonnet-vetted bank, init v4rl4, seed 53), 2026-09-10 20:37
Same judge (`[judge_gemma_nvfp4]`, v2.2), greedy, ho200. Config `rl/configs/rl_v5_clean.toml`; run `runs/grpo/v4_iter5/iter5`; eval `runs/grpo/eval_iter5`.
| model | default | strict (200) | strict −nonsense (191) | strict −all bad (187) | reach | median chars | protocol ok |
|---|---|---|---|---|---|---|---|
| v4rl3 | .585 | .580 | .597 | .604 | .60 | 177 | |
| v4rl4 | .540 | .545 | .560 | .567 | .56 | 224 | |
| **v4rl5** | **.540** | **.550** | **.565** | **.572** | **.55** | **166** | 196/200 |
Read: flat on strict (+.005), reach flat, answers 26% shorter (median 224 → 166; the round's length drift). Training
reward rose .37 → .58 almost entirely from the wrong-place-pin tier shrinking (26% → ~3% of rollouts at t0.7) — fewer
catastrophic pins, not more correct answers; at greedy that does not show on strict. Bank hygiene removed a confound
(no all-fail groups from bad questions) and did not lift accuracy, as the 5% bad rate predicted. Probe 1K×4 pending.
Probe 1K×4 (t0.7, strict v2.2, judged split across `judge_gemma_nvfp4` + `judge_gemma_nvfp4_b`):
| model | per-rollout | pass@4 | all-4 | per-rollout (971) | pass@4 (971) | all-4 (971) |
|---|---|---|---|---|---|---|
| v4rl4 | .486 | .636 | .315 | .494 | .645 | .320 |
| **v4rl5** | **.516** | **.662** | **.368** | **.524** | **.671** | **.374** |
Read: +3 per-rollout, +2.6 pass@4, **+5.3 all-4** — the round's gain is consistency at t0.7 (the far-pin tier collapsing),
which greedy ho200 does not see. Files `runs/grpo/eval_iter5/probe.strict.jsonl` (`rl/probe_judge_split.sh`).

## v4rl6 — round 6 (second Sonnet-vetted bank, init v4rl5, seed 56), 2026-09-11 04:16
Same judge, greedy, ho200. Config `rl/configs/rl_v6_clean.toml`; run `runs/grpo/v4_iter6/iter6`; eval `runs/grpo/eval_iter6`.
| model | default | strict (200) | strict −nonsense (191) | reach | median chars | clarifying finals |
|---|---|---|---|---|---|---|
| v4rl3 | .585 | .580 | .597 | .60 | 177 | |
| v4rl5 | .540 | .550 | .565 | .55 | 166 | |
| **v4rl6** | **.600** | **.615** | **.634** | **.62** | 156 | 32/200 |
Read: **new strict high (+6.5 over v4rl5, +3.5 over v4rl3)**, reach back to .62. 40 of the 77 strict fails are
"incomplete: main place not resolved" = honest non-resolutions ("I couldn't find X in the places database… could you
confirm the city?"), 32 of them phrased as clarifying questions — the model stopped far-pinning and started asking, with
no SFT. Training reward gain came from the far-pin tier (11%→2.5%) moving into the VOCAB-LEAK format tier (17%→26%,
`core/gates.py`), which punishes "places database" wording harder than the honest tier — fix before round 7 (HANDOFF).
Probe 1K×4 (t0.7, strict v2.2, single judge `judge_gemma_nvfp4`):
| model | per-rollout | pass@4 | all-4 | per-rollout (971) | pass@4 (971) | all-4 (971) |
|---|---|---|---|---|---|---|
| v4rl5 | .516 | .662 | .368 | .524 | .671 | .374 |
| **v4rl6** | **.531** | **.682** | **.373** | **.539** | **.691** | **.377** |
Read: +1.5 per-rollout, +2 pass@4, all-4 flat — this round's gain shows at greedy (strict +6.5) more than at t0.7.
Sonnet hack audit of all 200 v4rl6 ho200 rows (2026-09-11; 4 auditors, tool results in view; `runs/grpo/eval_iter6/audit/`):
verdicts RIGHT 116 / WRONG 44 / NO_ANSWER 40. Strict judge: 123 passes → Sonnet disagreed on 7, but 6 of those were the
audit's own 1,500-char truncation (the "unsupported" claim is present in the full tool result) → **1 real false accept
(bq-075207, wrong Zizzi branch 2.2 km off) = 0.8%; 0 false rejects.** Hacks across all 200: wrong entity 17 (1 among
passes), ref laundering 6 (all caught by the gate — strict FAIL), false decline 3 (declined although the web text named
the answer), vague pass 2, attribute spread 2. Nothing systematic; the gates are doing their job.

## v4rl6a — ask-SFT v1 on v4rl6 (2026-09-11 10:47) — NOT adopted
Continue-train (lr 1e-5, 1 epoch, 12,635 rows: 339 spliced miss→re-search continuations, 41 ask continuations, 301 own
no-ask passes, 8,845 keepers replay; no general-chat replay). `runs/train_ask_v1/`. Same judge, greedy, ho200.
| model | default | strict (200) | strict (191) | reach | median chars | "according to" | pass / wrong / ask / decline | tools/q |
|---|---|---|---|---|---|---|---|---|
| v4rl6 | .600 | .615 | .634 | .62 | 156 | 77% | 123 / 37 / 32 / 8 | 2.6 |
| v4rl6a | .550 | .545 | .555 | .60 | 195 | 22% | 109 / 90 / 0 / 1 | 3.9 |
Paired strict: both 100 / only v4rl6 23 / only v4rl6a 9. Of v4rl6's 32 asks → 29 wrong, 2 pass, 1 decline; its 8 declines → all
wrong. 8 of the 23 losses are over_budget (search until the budget runs out). Read: the set taught "always answer" — asking
and declining vanished and became wrong answers (wrong 18.5% → 45%). Voice moved as intended (tic 77% → 22%, +25% length).
**Do not use as the round-7 init** (no asks → no gradient for the ask branch); v4rl6 stays the init. Lessons: the SFT needs
honest-decline/ask exemplars at the real rate (~15%) rather than 0, and the general-chat replay back in; the miss→re-search
splices at 19% over-taught persistence (tools/q 2.6 → 3.9, over-budget ×8). Samples: https://claude.ai/code/artifact/bf43e239-7bef-4aed-bee3-c8cff227e2b0

## MMLU (lm_eval, local-completions, 5-shot default) — 2026-09-11
| model | MMLU acc |
|---|---|
| v4rl3 | .501 |
| v4rl3s | .499 |
| v4rl6 | .499 |
| v4rl6a | .500 |
Six RL rounds without chat replay: general capability unchanged (.501 → .499, within ±.004).

## v4rl7 — round 7c (ClarifyEnv: ask → one traveller reply; bank v2 15% ambiguous; init v4rl6, seed 59), 2026-09-12 05:33
Config `rl/configs/rl_v7_clarify.toml`; run `runs/grpo/v4_iter7/iter7c` (resumed from step 5 after a backend stall corrupted batch 7);
eval `runs/grpo/eval_iter7`. **Budget confound:** round 7 trained at BENCH budgets 5/3/2 (rounds ≤6: 3/2/1). v4rl7 cannot be scored at
3/2/1 (94/200 over_budget, strict .260), and greedy @5/3/2 burns calls for BOTH models (v4rl6 .260, v4rl7 .460, ~8 calls/q; a prompt
sensitivity — one extra tool line restores ~3 calls). Every row below is therefore PAIRED with v4rl6 re-run under the same condition.
| condition (ho200) | v4rl6 | v4rl7 | Δ |
|---|---|---|---|
| greedy @3/2/1 (old ladder) | **.615** | .260 (over-budget) | n/a |
| t=0.7 ×3 seeds @5/3/2, no reply | .558 | .550 | −.8 |
| greedy @5/3/2, ask_user tool + simulated reply turn (`ho200_reply2*`) | .610 | **.625** | +1.5 |
| └ rollouts that got a reply → passed after it | 33 → 2 | 15 → 4 | conversion 6% → 27% |
| └ terminal asks / declines | 26 / 10 | 5 / 25 | |
Probe 1K×4 (t=0.7, strict, single judge), paired at 5/3/2:
| model | per-rollout | pass@4 | all-4 |
|---|---|---|---|
| v4rl6 @3/2/1 | .531 | .682 | .373 |
| v4rl6 @5/3/2 | .543 | .694 | .388 |
| **v4rl7 @5/3/2** | **.552** | **.707** | **.391** |
Read: direct-answer accuracy unchanged (ho200 t=0.7 ±0, probe +1 at the same budget). The round's product is the ask branch: on 477 unseen
Sonnet twins v4rl7 asks on ≥80% of rollouts for half of them and converts 258 after a reply (v4rl6's pipeline: 85); on ho200 with a
reply turn it asks less (20 vs 59 asks) and converts 27% vs 6%, strict .625 vs .610. MMLU not yet run. Ladder rule from here: state the
budget and temperature on every row; greedy @5/3/2 is not a valid condition for these models.

## HARNESS: search radius 100 → 200 km (2026-09-12 15:19) — **+4.5 strict on v4rl6, a new high, with no training**
Paired ho200, greedy, budgets 3/2/1, judge `judge_gemma_nvfp4` v2.2 ×1; only `WPLACES_SEARCH_RADIUS_KM` differs.
| model | radius | default | strict | strict/191 | reach | pass / wrong / ask / decline | mean calls |
|---|---|---|---|---|---|---|---|
| v4rl6 | 100 | .600 | .615 | .634 | 62.0% | 123 / 37 / 20 / 20 | 2.6 |
| **v4rl6** | **200** | **.665** | **.660** | **.681** | **67.5%** | 132 / 39 / 18 / 11 | 2.5 |
| v4rl7d | 100 | .565 | .550 | .560 | 55.5% | 110 / 35 / 16 / 39 | 2.6 |
| v4rl7d | 200 | .605 | .615 | .634 | 62.0% | 123 / 36 / 15 / 26 | 2.3 |
Read: the win is reach (+5.5 / +6.5) converting declines into passes (20→11, 39→26). **Wrong answers do NOT rise** (37→39,
35→36), so the wider radius is not crowding the right answer out of the top 6 — the risk that kept radius at 100 since round 5.
Mean calls fall, so it also finds things sooner. Anchor analysis (2026-09-12) predicted this: 27 ho200 questions have the gold
100-300 km from the agent's nearest query anchor, and 19 of those retrieve at rank 1-2 with the agent's OWN query string once
the radius admits them; 7 more searched for the WRONG place entirely (all 7 golds ARE in the index, findable by their own name) — a reading failure, not a retrieval one; the remaining 9 need radius 300.
**CONFIRMED on the 1K×4 probe (2026-09-12 18:22), 4,000 rollouts per arm, t=0.7:**
| model | radius | per-rollout | pass@4 | all-4 | over-budget | calls |
|---|---|---|---|---|---|---|
| v4rl6 | 100 | .531 | .682 | .373 | .014 | 2.75 |
| **v4rl6** | **200** | **.598** | **.736** | **.456** | .011 | 2.54 |
| v4rl7d | 100 | .499 | .661 | .324 | .002 | 2.44 |
| v4rl7d | 200 | .559 | .702 | .384 | .001 | 2.22 |
+6.6 / +6.0 per-rollout — LARGER than the ho200 estimate, and **all-4 rises 22% relative (.373 → .456)**, i.e. consistency not
sampling luck. Tool calls FALL at the wider radius on both models. The checkpoint gap (~4 pts, v4rl6 ahead) is unchanged by radius,
so radius does not rescue 7d's deficit. **Radius 200 is adopted for training from round 7e on.**

## v4rl7e — round 7e (= round 7d verbatim, ONE variable changed: search radius 100 → 200), 2026-09-13 05:00
Config `rl/configs/rl_v7d_clarify.toml --run.name iter7e`; init v4rl6; bank v2; seed 59; ClarifyEnv; budgets 3/2/1 + reply
allowance 2/1/1; `WPLACES_SEARCH_RADIUS_KM=200`. Run `runs/grpo/v4_iter7/iter7e`, eval `runs/grpo/eval_iter7e`.
ho200, greedy, budgets 3/2/1, judge `judge_gemma_nvfp4` v2.2 ×1 — every row paired at the SAME radius:
| run | strict | /191 | reach | pass / wrong / ask / decline | med chars | calls |
|---|---|---|---|---|---|---|
| v4rl6 @200 | **.660** | .681 | .68 | 132 / 39 / 18 / 11 | 188 | 2.50 |
| v4rl6 @200 +reply | .655 | .675 | .67 | 131 / 44 / 11 / 4 | 190 | 2.84 |
| v4rl7d @200 | .615 | .634 | .62 | 123 / 36 / 15 / 26 | 208 | 2.34 |
| v4rl7e @200 | .625 | .644 | .60 | 125 / 46 / 12 / 17 | 237 | 2.02 |
| **v4rl7e @200 +reply** | **.665** | .681 | .64 | 133 / 57 / 6 / 4 | 239 | 2.35 |
Clarification conversion (identical denominators, 23 asks each): **v4rl7e 8/23 (35%) vs v4rl6 2/23 (9%)**.
1K×4 probe (t=0.7) @200, all arms verified free of tool errors:
| model | per-rollout | pass@4 | all-4 | reach |
|---|---|---|---|---|
| v4rl6 @100 | .531 | .682 | .373 | .572 |
| **v4rl6 @200** | **.598** | **.736** | **.456** | **.644** |
| v4rl7d @200 | .559 | .702 | .384 | .591 |
| v4rl7e @200 | .568 | .710 | .402 | .589 |
MMLU: v4rl7e **.500**, v4rl7d .4998 (v4rl6 .499, v4rl3 .501) — nine RL rounds, general capability still flat.
Read: **the recipe costs ~3 points of aggregate accuracy and buys a 4× better ask branch, and that has now replicated three
times** (7c, 7d, 7e) across two budgets and two radii. 7e is the best of the three and the first to beat v4rl6 under any
condition (+reply: .665 vs .655), but only 23/200 ho200 questions involve an ask, so the ask branch cannot pay for the
deficit. The deficit is REACH (.589 vs .644 on the probe); conversion-given-reach is identical across all four checkpoints.
Training (steps 16-20): pass 54.7%, honest tier 9.0% (7d: 26.4%), far pin 12.0% (7d: 4.5%), tools 2.17 direct / 2.68 asked,
clear-row ask 5.0%, twins ask 74.5% with 68/152 converting. Better reach turned 7d's declines into attempts — some right,
some wrong.
Twin pools from the same 477 leak-checked candidates, showing that pool size tracks the filter model's conversion ability:
v4rl7 @100 → 258 resolvable / 185 band; v4rl6 @200 → 86 / 80; v4rl7e @200 → 235 / 163.


## RADIUS 300 (2026-09-13 08:09) — v4rl6 ho200 strict **.695** (191-q .717)
| radius | default | strict | /191 | pass / wrong / ask / decline | calls |
|---|---|---|---|---|---|
| 100 | .600 | .615 | .634 | 123 / 37 / 20 / 20 | 2.6 |
| 200 | .665 | .660 | .681 | 132 / 39 / 18 / 11 | 2.5 |
| **300** | **.680** | **.695** | **.717** | 139 / 43 / 11 / 7 | 2.3 |
8 of the 9 ho200 questions whose gold sat 200-300 km from the agent's anchor now pass. Diminishing: 100→200 bought 9 passes for
2 wrong, 200→300 bought 7 for 4, and 4 questions that passed at 200 fail at 300. **Standard is 300; measure before going wider.** (The 1K×4 probe at 300 was not run — ho200 taken as sufficient.)

## Program verdict, rounds 7c-7e (2026-09-13)
Best model is still v4rl6. The clarification objective cost 3-6 strict per round and bought an ask branch that fires on ~17% of
questions; the ceiling is reach (26% of probe questions never reached: 65% wrong entity, 30% invented anchor, 5% ranking, 1 question
missing from the index). Radius 100→300 = +8 strict from the harness. Next levers: `anchor_name` resolution, place context in
training, gold equivalence sets. Full post-mortem: `rl/ROUND7_SPEC.md`.

## v4rl8 — round 8 "reach" (init v4rl6, bank r8 2,771 clear, seed 62, radius 300, budgets 3/2/1, **BENCH_PLACE_CONTEXT=1**), 2026-09-13 18:16
Config `rl/configs/rl_v8_reach.toml`; run `runs/grpo/v4_iter8/iter8` (relaunch 12:16 on the rebuilt backends — wplaces 4-8 gunicorn
workers, wsearch uvicorn-wsgi, **wiki :8003 live for the first time in any round**; first attempt void at step 5). 669 rollouts
dropped to backend timeouts (excluded from training; 242 = one stall, 246 = my restart+warm, rest = checkpoint-write cache eviction).
Training: reward .61 → .72-.90; pass 59% → 64%; honest/ask tier 9.3% → 2.9% (that IS the pass gain: ~2/3 of drained declines → PASS, 1/3 → wrong).
ho200 greedy 3/2/1 @300. **Every row below has wiki ON except the 08:09 row (wiki was down all day until 10:05).**
| condition | default | strict | reach | pass / wrong / decline / over | calls | first search names gold |
|---|---|---|---|---|---|---|
| v4rl6, context OFF (08:09, no wiki) | .680 | **.695** | .71 | 139 / 43 / 18 / 0 | 2.35 | 68.5% |
| v4rl6, context ON (paired baseline) | .635 | .630 | .71 | 126 / 56 / 18 / 0 | 2.25 | 67.5% |
(files: `eval_iter6/ho200_r300ctx_old3wiki.*` — the `ho200_r300ctx.*` names get overwritten by launch_iter8.sh step 7 under the NEW harness defaults.)
| **v4rl8, context ON** | .675 | **.685** | .73 | 137 / 53 / 3 / 7 | 2.58 | 69.0% |
Paired strict: v4rl8 vs v4rl6+ctx **20 / 9** (both 117) → training with context on is worth ~+5.5 under that condition. But
v4rl6+ctx vs v4rl6 (08:09) is **7 / 20**: the new observation (context lines and/or wiki lines) costs v4rl6 ~6.5 zero-shot, so
v4rl8 with context ≈ v4rl6 without it. Entity selection on the first search did NOT move (67.5-69.0% across all three rows):
place context does not change what the model searches for; the round's gain is the honest tier draining (declines 18→3),
which also bought 7 over-budget loops (same query repeated 3×). Separation + full observation sweep: see the next block.
**Separation (18:35):** v4rl6 @300 **wiki ON, context OFF** = default .640 / strict **.650** / reach .70 / 130-51-18-1 / 2.31 calls.
Paired: wiki-only vs neither **5 / 14** (−4.5); wiki+ctx vs wiki-only **9 / 13** (−2). So the Wikipedia lines (3 extra entries, ~1.2K chars
per web result) cost v4rl6 more than the context lines do; the 14 lost questions are ordinary misattribution / wrong-entity / unanswered-
detail failures (only 1 cites Wikipedia at all) — the longer observation crowds the blog snippets for a 0.8B. **Verdict so far:** round 8 =
+5.5 vs its own condition, ≈0 vs the ladder condition; keep v4rl6 as init unless the probe says otherwise; fix the observation (fewer/shorter
wiki lines, or wiki only when blog results are thin) before training on it again.

## OBSERVATION SWEEP (2026-09-13 19:18, complete) — v4rl6 zero-shot pays a FIXED penalty for any unfamiliar observation
All rows: v4rl6, ho200, greedy, budgets 3/2/1, radius 300, same backends. "ctx" = place-context line under each search result;
"wiki" = Wikipedia entries inside WEB RESULTS (live on ripper only since 10:05 — every earlier round and ladder row had wiki DOWN).
| observation | strict | reach | tool chars (Δ vs clean) | paired vs clean (only variant / only clean) |
|---|---|---|---|---|
| **clean (ladder condition)** | **.695** | .71 | 8,053 | — |
| 1 wiki entry (200 chars) | .665 | .69 | 8,184 (+131) | 4 / 10 |
| ctx only (no filler lines) | .655 | .71 | 8,830 (+777) | 4 / 12 |
| 3 wiki entries (400 chars) | .650 | .70 | 9,275 (+1,222) | 5 / 14 |
| ctx + 1 wiki | .640 | .70 | 9,043 (+990) | 4 / 15 |
| ctx + 3 wiki (**round 8's**) | .630 | .71 | 9,827 (+1,774) | 7 / 20 |
Two effects, both real, and the minimal case separates them: **+131 chars (+1.6% text) costs 3 strict** — volume cannot explain that —
while more text costs incrementally more on top (3 wiki < 1 wiki, ctx+wiki < either). So there is a FIXED distribution-shift penalty
for any departure from the format v4rl6 trained on through eight GRPO rounds, plus a smaller dose effect; the two dimensions are
roughly additive. Reach is flat at .69-.71 in every row — the losses are misattribution and wrong-entity picks, not retrieval.
Consequences: (1) a checkpoint may only be compared to another under the SAME observation, and the ladder's condition is the clean one;
(2) round 8's "+5.5 vs v4rl6 under its own condition" was recovery of this penalty, not new capability — consistent with its flat
first-search entity selection (67-69% in all rows); (3) enriching the observation is a multi-round investment, never a free harness
win like radius was, and must not be mixed with a reward change in the same round. Harness knobs added: `BENCH_CONTEXT_EMPTY_LINE`
(filler context lines, now off by default), `BENCH_WIKI_MAX` / `BENCH_WIKI_CHARS` / `BENCH_WIKI_ONLY_IF_THIN` (defaults 1 / 200 / off).
**Round 9 therefore runs on the clean observation, changing only the reward.**

**v4rl8 probe, 1K×4 @300 +ctx+wiki (2026-09-13 19:50):** per-rollout **.653** | pass@4 **.783** | all-4 **.513** | reach .708 |
over-budget **.056** | first search names gold 67.7% (any search 70.3%; pass|first-names-gold 76.9%).
**No paired baseline exists at this condition** — v4rl6's radius-300 probe was killed on 2026-09-13 and voided, so the nearest row is
v4rl6 @200 CLEAN (.598 / .736 / .456 / reach .644 / over-budget .011), which differs in BOTH radius and observation. Do not quote the
delta as a round-8 gain; the ho200 pairing (.685 vs .695 clean, .685 vs .630 same-observation) is the trustworthy comparison.
Two findings that do survive: **over-budget 1.1% → 5.6%** (224 of 4,000 rollouts; the honest tier drained into repeated identical
searches — the round-9 `GEOGYM_REPEAT_QUERY` term targets exactly this), and **entity selection still flat** (67.7% here, 65.5% on
v4rl6's @200 probe, 67.5-69.0% across all ho200 rows): 4,000 rollouts confirm place context does not change what the model searches for.

## v4rl9 — round 9 "reach, grounded" (init v4rl6, CLEAN observation, fresh qgate bank 2,800, seed 65), 2026-09-14 01:00
Config `rl/configs/rl_v9_reach.toml`; run `runs/grpo/v4_iter9/iter9`. ONE change vs the ladder recipe: `GEOGYM_REPEAT_QUERY=-0.1`
(per repeated `search_places` query, cap 3) + `GEOGYM_GROUNDED_SEARCH=0.1` (first search grounded in already-retrieved text AND the
rollout reached a verdict), REWARD_HACKS §21. Backends: 1 infra drop in 20 steps (vs 669 in round 8).
| ho200 @300 clean, greedy 3/2/1 | default | strict | reach | pass / wrong / ask / decline / over | calls | first search names gold |
|---|---|---|---|---|---|---|
| **v4rl6 (bar)** | .680 | **.695** | .71 | 139 / 43 / 0 / 18 / 0 | 2.35 | 68.5% |
| v4rl9 | .660 | **.650** | .70 | 130 / **63** / 0 / **6** / 1 | 2.45 | 70.0% |
Paired strict: both 121 / only v4rl9 9 / only v4rl6 18. **Both new terms behaved** — repeat-query rollouts 9.5% → 3.4% over the
round (the round-8 loop regression is fixed), grounded-first-search 61% → 67% with no runaway (farm check: it never approached 100%
and entity selection rose 68.5% → 70.0%, so the term tracked the behaviour instead of being gamed). Training reward .80 → .85.
**But strict fell 4.5**, and the shape is identical to round 8: declines 18 → 6, wrong 43 → **63**.

## THE STRUCTURAL BUG behind rounds 8 and 9 (found 2026-09-14 01:50): the reward paid for guessing
`_score_base` scored a judged-WRONG answer that cleared every gate at **0.0**, and `+REACH_BONUS` (0.25) when it had reached the
gold, while an honest non-resolution scored `HONEST_FAIL` **−0.5** with no reach credit (`HONEST_REACH=0`). So the ordering was
**wrong-but-reached (+0.25) > wrong (0.0) > honest (−0.5)**: a 0.75 gradient toward answering confidently rather than admitting a
miss. −0.5 was set in round 7b for a real reason (at −0.25-with-reach the honest tier was farmed to 36% declines) but it inverted
the ordering instead of just damping it. Every round since has drained the decline tier into wrong answers with no strict gain
(round 8: declines 18→3, wrong 43→53; round 9: 18→6, 43→63). Not the ask cost, not the observation — this.
Fix: `GEOGYM_WRONG_FAIL` (new, default 0.0 = old behaviour) added to the judged-fail return in `reward.py`. Round 10 runs it at
−0.3 with `GEOGYM_HONEST_FAIL=-0.15`, giving PASS +1 > wrong-reached-and-pinned +0.2 > wrong-reached −0.05 > honest −0.15 >
wrong −0.3 > far-pin/identity −1.
**v4rl9 probe, 1K×4 @300 CLEAN (2026-09-14 02:19):** per-rollout **.641** | pass@4 .763 | all-4 .495 | reach .684 |
**over-budget .007** | first search names gold 69.0% (pass|first-names-gold 76.0%). The repeat-query term is confirmed at scale:
over-budget .056 (v4rl8) → **.007**, the lowest of any round. Entity selection 69.0% — still inside the 67-70% band every checkpoint
has occupied since round 6, so neither place context (round 8) nor the grounding term (round 9) has moved what the model searches for.
**Probe baseline gap:** there is still no v4rl6 1K×4 probe at radius 300 clean (the 2026-09-13 run was killed and voided), so probe
rows across rounds 8-9 cannot be paired to the init. Nearest is v4rl6 @200 clean (.598/.736/.456/.644). Worth one GPU-hour when idle:
`runs/twins_r8/radius300.sh` is the template.

## v4rl10 — round 10 "honest ordering" (init v4rl6, CLEAN observation, fresh bank 2,800, seed 68), 2026-09-14 07:4x
Config `rl/configs/rl_v10_honest.toml`. ONE new variable vs round 9: `GEOGYM_WRONG_FAIL=-0.3` with `GEOGYM_HONEST_FAIL=-0.15`
(REWARD_HACKS §22) so the ordering became PASS > honest > confident-wrong; round-9 search terms kept. 3 infra drops in 20 steps.
| ho200 @300 clean, greedy 3/2/1 | default | strict | reach | pass / wrong / ask / decline / over | calls |
|---|---|---|---|---|---|
| **v4rl6 (bar)** | .680 | **.695** | .71 | 139 / 43 / 0 / 18 / 0 | 2.35 |
| v4rl10 | .660 | **.680** | .71 | 136 / **47** / 0 / **16** / 1 | 2.27 |
| v4rl9 (old ordering) | .660 | .650 | .70 | 130 / 63 / 0 / 6 / 1 | 2.45 |
Paired strict vs v4rl6: both 125 / only v4rl10 11 / only v4rl6 14 — a tie at n=200.
**The fix worked on its own terms and that is the finding.** Round 9's collapse reversed cleanly: wrong 63 → 47, declines 6 → 16,
pass 130 → 136 — i.e. the tier balance returned to v4rl6's own (43 / 18 / 139) almost exactly. The training stream agreed
(honest tier by 5-step block: round 9 11.0 → 4.0%, round 10 9.8 → 11.9%). And the result is v4rl6's score. Correcting a broken
shaping term reproduced the init; it did not beat it.
**Six rounds of process reward (5-10), best outcome = a tie with v4rl6.** Rounds 5-7 clarification, 8 place context, 9 search
shaping, 10 ordering: none beat .695. The only real gain of the week was a harness constant (radius 100 → 300, +8). Entity
selection — the 65% wrong-entity ceiling — sat at 67-70% in every checkpoint and every condition measured.
**Jon's call (2026-09-14): stop process rewards; round 11 is pure binary** (1.0 for a strict pass, 0.0 otherwise). See
`rl/configs/rl_v11_binary.toml`, REWARD_HACKS §23.
**v4rl10 probe, 1K×4 @300 CLEAN (2026-09-14 08:43):** per-rollout **.625** | pass@4 .746 | all-4 .496 | reach .666 |
**over-budget .000** (0 of 4,000 — the repeat-query term from round 9, kept here, has eliminated the loop class entirely) |
first search names gold 68.8%. Against v4rl9 (.641/.763/.495/.684/.007) it is 1.6 lower per-rollout, the mirror of its +3.0 on
ho200 — both inside the noise of a 1,000-question sample. Entity selection 68.8%: the ninth consecutive measurement inside 67-70%.

## v4rl11 — round 11 "BINARY" (init v4rl6, CLEAN observation, fresh qgate bank 2,800, seed 71), 2026-09-14 19:22
Config `rl/configs/rl_v11_binary.toml`. The ENTIRE reward is `GEOGYM_BINARY=1`: **1.0 for a strict pass** (judge correct AND hard
format gates clean AND identity ok) **0.0 for everything else**. No reach, pinned-fail, honest tier, format tiers, efficiency, ask
cost/bonus, wrong-fail, or round-9 search terms. 4 infra drops in 20 steps.
| ho200 @300 clean | default | strict | reach | pass / wrong / ask / decline / over | calls | chars |
|---|---|---|---|---|---|---|
| **v4rl11** | .680 | **.700** | .69 | 140 / 45 / 0 / 10 / 5 | 2.46 | 199 |
| v4rl6 (bar) | .680 | .695 | .71 | 139 / 43 / 0 / 18 / 0 | 2.35 | 186 |
| v4rl10 | .660 | .680 | .71 | 136 / 47 / 0 / 16 / 1 | 2.27 | 216 |
Paired vs v4rl6: both 125 / only v4rl11 15 / only v4rl6 14 — a tie at n=200.
**1K×4 probe @300 clean is the discriminating test and binary WINS it:**
| model | per-rollout | pass@4 | all-4 | reach | over-budget | pass \| first-search-names-gold |
|---|---|---|---|---|---|---|
| **v4rl11 (binary)** | **.662** | **.773** | **.533** | .684 | .038 | **79.6%** |
| v4rl10 (shaped) | .625 | .746 | .496 | .666 | .000 | 74.5% |
| v4rl9 (shaped) | .641 | .763 | .495 | .684 | .007 | 76.0% |
+3.7 per-rollout over v4rl10 and **all-4 .533, the highest any checkpoint has recorded** (previous best .513) — consistency, not
sampling luck. The gain is CONVERSION, exactly what the reward now pays for: entity selection is unchanged (66.4%, the tenth
consecutive measurement in 66-70%) but the pass rate GIVEN the first search named the gold went 74.5% → 79.6% on the probe and
74.5% → 84.1% on ho200. Costs, both expected from deleting the terms that suppressed them: declines 18 → 10 and over-budget
.000 → .038 (the repeat-query loops return; under binary they are already punished by failing, so no term was added back).
Also contra my prediction: the ASK branch did not collapse (5-10% all round) — with the ask cost gone, asking is simply free.
**Six rounds of shaping produced .650-.685; deleting all of it produced the best probe of the program.**

## v4rl12 — round 12 "binary + difficulty-matched bank" (init **v4rl11**, first chained round, seed 74), 2026-09-15 07:08
Config `rl/configs/rl_v12_mixed.toml`. Reward unchanged from round 11 (`GEOGYM_BINARY=1`). Bank `rl_bank_mixed_r12.jsonl`:
2,800 questions that EACH produced both a pass and a fail across rounds 8-11 (8 judged rollouts apiece, v4rl6-line), 2,187 in the
25-75% band, built free from past training streams. Attempt 1 died at step 10 — root disk 100% full, DCP write truncated; 270 GB
reclaimed from stale trainer checkpoints and the chain now aborts below 120 GB free. Relaunched 22:35, 1 infra drop.
| ho200 @300 clean | strict | reach | pass / wrong / ask / decline / over | calls | chars |
|---|---|---|---|---|---|---|
| **v4rl12** | **.720** | **.74** | 144 / 44 / 0 / 2 / 10 | 2.58 | 235 |
| v4rl11 | .700 | .69 | 140 / 45 / 0 / 10 / 5 | 2.46 | 199 |
| v4rl6 | .695 | .71 | 139 / 43 / 0 / 18 / 0 | 2.35 | 186 |
Paired: v4rl12 vs v4rl6 **15 / 10**; vs v4rl11 **15 / 11**.
| 1K×4 probe @300 clean | per-rollout | pass@4 | all-4 | reach | over-budget | pass \| first-names-gold |
|---|---|---|---|---|---|---|
| **v4rl12** | **.671** | .769 | **.550** | **.699** | .065 | 78.9% |
| v4rl11 | .662 | .773 | .533 | .684 | .038 | 79.6% |
| v4rl10 | .625 | .746 | .496 | .666 | .000 | 74.5% |
**ho200 .720 and all-4 .550 are program records**, and reach finally moved (.71 → .74 ho200, .684 → .699 probe) after sitting flat
all week. **The bank did what it was built to do:** zero-advantage groups (all-8-pass + all-8-fail, which give NO gradient under
binary) fell from **63% → 36%** at step 1, so ~82 of 128 groups per step carried gradient instead of ~48.
Entity selection 68.0% — the eleventh consecutive measurement in 66-70%; still untouched by anything.
**Over-budget is the one regression of the binary line** (.000 → .038 → .065). Diagnosed 2026-09-15 and NOT fixed (Jon's call):
all 259 over-budget probe rollouts emit an EMPTY final, and **0 of them are judged correct**, so they are not discarded wins.
Long episodes are a symptom, not a cause — judge-correct by tool-call count: 2 calls **82.1%** (n=2,776), 3 → 51.4%, 4 → 30.0%,
5 → 22.8%, 6+ → **17.7%**. Raising the budget would feed calls to already-lost episodes. The available fix (force a final answer
when the budget is exhausted, converting ~6.5% guaranteed zeros into attempts) is recorded here, deliberately unspent.
**v4rl11 and v4rl12 backed up to HF** (`topoloop/wanderfugl-0_8b-v4rl11`, `-v4rl12`, each with `serve/`).

## v4rl13 — round 13 "binary + bank re-matched to the current model" (init **v4rl12**, seed 77), 2026-09-15 16:37
Config `rl/configs/rl_v13_mixed.toml`. Reward unchanged (`GEOGYM_BINARY=1`). Bank = 1,960 round-12 questions still mixed under
the round-12 policy + 840 unused mixed from rounds 8-11. Evaluated ho200 ONLY (SKIP_PROBE=1 — chaining rounds, Jon 2026-09-15).
| ho200 @300 clean | strict | reach | pass / wrong / ask / decline / over | calls |
|---|---|---|---|---|
| **v4rl13** | **.735** | **.75** | 148 / 46 / 0 / 0 / 6 | 2.51 |
| v4rl12 | .720 | .74 | 144 / 44 / 0 / 2 / 10 | 2.58 |
| v4rl11 | .700 | .69 | 140 / 45 / 0 / 10 / 5 | 2.46 |
| v4rl6 | .695 | .71 | 139 / 43 / 0 / 18 / 0 | 2.35 |
Paired: vs v4rl12 **11 / 8** (noise at n=200); vs v4rl6 **18 / 10**. NOTE the rows file holds 400 rows = 2 rollouts per question:
an orphaned `post_round.sh` from the retired `launch_iter13.sh` (it spawns post_round after `sleep 300`, so killing the parent
reparents the child to init) woke when training ended and ran ho200 concurrently with the chain. Both halves are genuine rollouts
against the live v4rl13 server; .735 is the conservative unique-question figure, .740 the 400-row mean. Its stray probe files were
deleted. Lesson: kill the process GROUP, and never leave a script whose child has its own wait loop.
**The binary line, four rounds: .700 → .720 → .735, reach .69 → .74 → .75, declines 10 → 2 → 0.** Entity selection 67.8% — the
twelfth consecutive measurement in 66-70%. Everything gained has come from conversion and reach, never from what it searches for.

## NVMe write stalls — the real cause of "wplaces stalling" all week (2026-09-15)
Round 13 lost step 12 (reward .396 vs .68, 386 rollouts dropped, 46% of groups all-fail). The kernel ring buffer had it:
`nvme nvme1: I/O tag ... opcode 0x1 (I/O Cmd) QID 6 timeout, aborting req_op:WRITE(1) size:131072` at 12:57 and 12:59, and the
stall ran 12:59-13:06. When the device stops acknowledging writes every process on that filesystem blocks in D-state — which is
why all 8 wplaces workers froze together, accepted no new connections (request rate 957/min → 6/min: the CLIENT stopped sending
because every connection was stuck), and recovered on their own. Root is `/dev/nvme1n1p2` = **Crucial P2 (CT2000P2SSD8)**, a
DRAM-less SSD whose write speed collapses once its SLC cache fills — exactly what four 9.4 GB checkpoint writes per round do.
This also fits the older "box hard-crashes during sustained NVMe writes" incident. Earlier explanations this week (worker memory
growth, page-cache eviction, my own warm-up script competing) fit the surface facts but were wrong: **check `journalctl -k` FIRST**.
Mitigations applied for rounds 14+: `[ckpt] interval` 5 → 10 (only step_20 is ever used; we have never resumed mid-round) and
checkpoints symlinked to `/media/jon/linuxdata/wanderfugl_ckpt/iterN` on the Samsung 860 EVO, so zero checkpoint bytes touch the
P2. Logs/monitors/streams stay on root. Not done: `sudo smartctl -a /dev/nvme1` (needs a password) — worth running, it is the root disk.

## v4rl14 — round 14 (init v4rl13, first THREE-WAY bank: exploit 1,400 / retry 560 / explore 840; seed 80), 2026-09-16 03:09
ho200 @300 clean: strict **.740** reach **.76** | pass/wrong/ask/decline/over 148/45/0/2/5 | calls **3.31** (v4rl13 .735/.75/2.51).
Flat vs round 13 — the first binary round that did not clearly advance. Checkpoints on linuxdata (interval 10, [ckpt] output_dir);
9 infra drops. Post-round failed once (post_round.sh looked in the run dir; now takes CKPT_ROOT) and re-ran 4 h later.
**Bank mix, full round:** exploit 78.6% mixed (67% of all gradient) | retry 36.4% mixed, 62% all-fail (12.5%) | explore 39.3% mixed,
46.5% all-pass (20%) | TOTAL 58.4% mixed vs 37% on round 11's uniform bank. Retry earns ~half an exploit slot: a third of questions a
weaker checkpoint failed 8/8 are now reachable. Explore has been OUTGROWN (71% pass on fresh vetted questions) and is now a label
factory rather than a challenge. Shares for 16/17: exploit 1,500 / retry 400 / explore 900.
**Length anatomy (round 14 stream, 43,272 rollouts):** pass by calls 2 → 77%, 3 → 69%, 4 → 51%, 5 → 35%, 6+ → 9%; 4-call episodes are
2s/1w/1f (62%) or 2s/2w (26%), all inside the 3/2/1 budget; the growth 3.14 → 4.11 calls across the round is almost all a SECOND
places search (1.21 → 1.69). Of 6,434 four-call PASSES, 51% already held the gold by call 2, 31% first retrieved it on call 4
(≈8% of all passes exist only because of a 4th call), 15% passed without the gold ref in any result. A 4th search_places is never
attempted (0.0-0.7%); the 7% over-budget deaths are a 3rd web_search, unlearned because of REWARD_HACKS §24.

## v4rl15 — round 15 (init v4rl14, bank re-matched to the round-14 policy: exploit 1,400 / retry 560 / explore 840; seed 83), 2026-09-16 13:27
ho200 @300 clean: strict **.755** reach .73 | pass/wrong/ask/decline/over **151 / 42** / 0 / 1 / 6 | calls 3.34. **Program record**;
wrong answers at their lowest ever (v4rl6 43, v4rl14 45). Reach dipped .76 → .73 while strict rose — conversion, not retrieval.
Binary line: .700 → .720 → .735 → .740 → **.755**. Bank mix: exploit 78.4% mixed / retry 37.5% / explore 38.2% / total 58.2% —
identical to round 14's, i.e. the three-way recipe is stable round to round. 9 infra drops; steps ~28 min (5.3 calls/episode).
**Chain incident:** `chain_r16_r17.sh` aborted at 13:26 — it waited on `ho200.strict.jsonl` existing + 60 s, but judge_v2 writes
that file incrementally, so it existed a minute before `POST_ROUND_DONE` (13:27). Round 16 launched by hand at 14:37 on the
canonical `runs/grpo/chain.sh 16 17` (retry 400 / explore 900). Rule: wait on `POST_ROUND_DONE` in the post log, never on an output file.

## v4rl16 — round 16 (init v4rl15, bank re-matched to the round-15 policy: exploit 1,500 / retry 400 / explore 900; seed 86), 2026-09-16 23:55
ho200 @300 clean: strict **.735** reach .70 | pass/wrong/ask/decline/over 147 / **27** / 0 / 0 / **26** | calls 3.96. NOT adopted.
Wrong answers fell to the lowest ever (42 → 27) but over-budget deaths quadrupled (6 → 26 = 13% of ho200); net −4 on v4rl15.
In-training calls/episode 4.6 at step 10 → 6.0 at step 20 (the §24 dynamic: the exhausted-message episode is scored 0 so the
policy never sees the cost of a 3rd web_search). Bank mix stable: exploit 82.5% mixed / retry 37.0% / explore 39.3% / total 62.1%.

## v4rl17 — round 17 (init v4rl16, bank re-matched to the round-16 policy: exploit 1,500 / retry 400 / explore 900; seed 89), 2026-09-17 09:30
ho200 @300 clean: strict **.670** reach .69 | pass/wrong/ask/decline/over 134 / 48 / 0 / 0 / 18 | calls 4.53. NOT adopted; eval clean (0 errors, 200 rows).
Below v4rl6 (.695). Wrong back to 48, over-budget 18, and 10 `[locator] unsupported` (v4rl16: 3). Training opened at 6.5 calls/episode
(highest ever) and drifted down to 5.3 by step 20 while reward rose .52 → .58, i.e. the policy started paying for length only
after the ho200-visible damage was done. Bank mix: exploit 78.7% / retry 28.1% / explore 39.4% / total 58.9%.
**Read:** two chained rounds after the record went .755 → .735 → .670. The three-way binary recipe has run out at v4rl15 unless the
§24 inversion is fixed; every further chained round compounds length drift. **v4rl15 stays the checkpoint of record.** Resume from v4rl15, not v4rl17.

## v4rl18 — round 18, the REWARD_HACKS §24 single-variable test (init **v4rl15**, round-16 bank, seed 86, `GEOGYM_CAP_ANSWER=1`), 2026-09-17 20:55
Round 16 replayed with ONE change: after a search cap the env refuses further tools and scores the answer that follows (was: 0 regardless).
ho200 @300 clean, standard harness: strict **.745** reach **.76** | pass/wrong/ask/decline/over 149 / 39 / 0 / 0 / **12** | calls 3.72.
Cap harness (`BENCH_CAP_ANSWER=1`, the model may answer after a cap): strict **.760** | 152 / 45 / 0 / 1 / 2 | capped 11 (2 pass) | calls 3.94.
Paired baselines on the cap harness: v4rl15 .755 (capped 6, 0 pass) — identical to its standard score.
**Read vs its twin v4rl16 (.735, over 26, calls 3.96):** +1 strict, over-budget halved, reach the highest of any checkpoint, length drift
contained (training calls 5.3-5.4 through steps 14-19 where round 16 went 5.2 → 6.0). vs v4rl15: −1 standard / +0.5 cap harness — a tie
at n=200. In training the cap rate did NOT fall (7.8% → 8-10%): scoring the post-cap answer makes the 3rd web_search free, so the
fix stops the bleeding but does not teach stopping. Post-cap answers pass ~10% by step 18 (up from 3%), not the 15-30% §24 guessed.
Decision (Jon: "make a judgment call"): chain 19-20 from v4rl18 with cap scoring ON (`runs/grpo/chain_cap.sh 19 20`) to test whether
cap-on chaining holds where cap-off chaining collapsed (16 → 17). Report both harnesses per round from here; the standard number stays
the ladder figure until a decision to switch.

## v4rl19 — round 19 (init v4rl18, bank re-matched to the round-18 policy, cap scoring ON), 2026-09-18 08:25 — NOT adopted
Standard: strict **.680** reach .76 | 136 / 35 / 0 / 0 / **29** | calls 4.08. Cap harness: **.725** | 145 / 44 / 0 / 0 / 11 | capped 28 (11 pass) | calls 4.97.
**The cap got farmed.** With the post-cap answer scored, a refused call costs nothing under binary reward; training cap-rate went 15% → 24%
over the round (round 18: 8-10%), post-cap passes ~100/step. Reach .76 and wrong 35 are both good — the loss is entirely rollouts the
standard harness kills at the 3rd web_search. Chain held before round 20 (`hold_r20.sh`). Round 20 = v4rl18 + the SAME round-19 bank +
cap scoring + **GEOGYM_MAX_TOOLS=5** (total-call ceiling, same refusal message, harness --max-tools 5 for the cap ho200) — one variable vs round 19.
Launch gotcha: chain_cap.sh's config sed ignored INIT_OVERRIDE (first attempt inited from v4rl19; killed at 08:30, dirs removed, relaunched
08:31 on the fixed `chain_cap2.sh`). Log runs/grpo/chain_cap_20.log.

## v4rl20 — round 20 (init **v4rl18**, the round-19 bank unchanged, cap scoring ON + **GEOGYM_MAX_TOOLS=5** total ceiling), 2026-09-18 19:55 — NOT adopted
Standard: strict **.735** reach .71 | 147 / 40 / 0 / 0 / 13 | calls 3.92. Cap harness (--max-tools 5): **.750** | 150 / 46 / 0 / 3 / 1 | capped 26 (5 pass) | calls 3.55.
One variable vs round 19 (.680 / .725, over 29): the ceiling bounds the farming (over-budget 29 → 13, cap harness +2.5) but does not
teach stopping — training ceiling-hit rate flat at ~29% from step 1 to 20, calls pinned at 5.7-6.0 (refusals counted). Reach fell
.76 → .71: the ceiling cuts off some searches that would have reached. Net: below v4rl18 (.745/.760) on both harnesses.
**Cap-era verdict (rounds 18-20):** scoring the post-cap answer is honest and fixes the never-answer behaviour, but under binary reward
a refused call is free, so persistence never falls. Bounding it (ceiling) trades reach for containment. Fewer calls needs a COST the
policy can see — either a per-call term (shaping, retired) or the old zero-on-cap rule (dishonest but held 8% for 7 rounds).
**v4rl15 (.755) remains the checkpoint of record; v4rl18 (.745/.760) is the best cap-era checkpoint.** Nothing running.

## v4rl21 — round 21, FIRST FOLLOW-UP ROUND (init v4rl18; bank = 1,400 follow-ups balanced in-context/needs-call + 1,400 turn-1; cap scoring on; two follow-up rewards), 2026-09-19 06:19
Turn 1 (standard ho200): strict **.735** reach .71 | 147 / 45 / 0 / 1 / 7 | calls 3.11. Cap harness .730 | calls 2.83. Turn-1 skill intact (v4rl18 .745).
**Follow-up ladder** (frozen v4rl18 prefixes, n=253): plain **.755** (v4rl18 .684) | scenario **.640** (v4rl18 .213) | calls 0.44 (2.38) | wrong call/no-call **19.8%** (61.7%).
In-context: .841 scenario, no-call 98.7% (was 0.6%). Needs-call: plain .615 / scenario .312 — the model flipped its default from
always-call to never-call (48/96 needs-call answered with no tool; logistics 12/12, detail 5/6). Nearby still calls (web_search 39,
never search_places). Training: in-context no-call 3% → 96% by step 18; needs-call pass 54% → 13%; turn-1 58-68% throughout.
Bootstrapping worry did not materialise (in-context mixed groups 4/83 at step 2 → 17/66 by step 8).
**Round 22** launched 06:30 from v4rl21: follow-ups re-matched to round-21 outcomes (mixed 1,003 + all-fail needs-call 186 + fresh, needs-call weighted), turn-1 = r21 mixed ids.

## v4rl22 — round 22 (init v4rl21; follow-ups re-matched to r21 outcomes, needs-call weighted; turn-1 = r21 mixed ids), 2026-09-19 16:29
Turn 1: standard **.735** (over 15, calls 3.00) | cap harness **.750** (v4rl18 .760). Turn-1 skill intact after two follow-up rounds.
**Follow-up ladder** (frozen v4rl18 prefixes, n=253): plain **.763** | scenario **.664** (v4rl21 .640, v4rl18 .213) | calls 1.35 | wrong call/no-call **15.8%** (19.8%).
Needs-call recovered .312 → **.562** (nearby .698, called on 90%) while in-context gave back .841 → .726 — mostly attribute (no-call 95% → 52%;
detail held .826). Net +2.4 scenario: the model is starting to discriminate rather than flip, but attribute follow-ups ("what food do they
serve") are where it still cannot tell "already in my fetched page" from "need to look". Both halves now mixed → same recipe continues.
Label note: logistics in-context tagging is weak (key = anchor name only); bq-003899#fu1 answered correctly from context and was gated as needs-call.

## v4rl23 — round 23, FIRST PARTIAL-MATCH ROUND (init v4rl22; bank 884 Sonnet twins + 1,000 turn-1 + 800 follow-ups), 2026-09-20 08:20
Turn 1: standard **.745** reach .74 (over 9, calls 3.20) | cap .735. Follow-up ladder: plain .771 | scenario **.680** (v4rl22 .664) | wrong choice 15.8%.
**Partial ladder** (frozen 126 ho200 twins, `partial_ho200.FROZEN.jsonl`): pass **.190** (v4rl22 .127) | gold cited .74 (.72) | mismatch stated .33 (.24) | overclaim .34 (.42).
Small real movement on all three partial flags, but in training the partial flags were FLAT step 2 → 18 (pass 10-14%, mismatch ~28%,
overclaim ~42%) and only 22% of partial groups carried gradient (follow-ups 34%, turn-1 62%): a 3-condition pass at a 10% base rate
leaves GRPO mostly all-zero groups. Turn-1 and follow-up both improved alongside (no interference). Nothing running after 08:20.
**Next (proposed):** two-tier partial reward (gold+mismatch 1.0 / gold+no-overclaim 0.5) on the 401 r23-mixed twins, or a small SFT
seed of Sonnet's `ideal_answer` hedges before another RL round.

## v4rl24 — round 24 (init v4rl23; two-tier partial reward GEOGYM_PM_TIER2=0.5; bank 586 partial (r23-mixed + gold-citing fails) + 800 follow-ups + 1,000 turn-1), 2026-09-20 19:10
Turn 1: standard **.710** (v4rl23 .745) | cap .705 | wrong 46, over 12, declines 6 (first non-zero since v4rl15). **Regression of ~3.5 on both harnesses.**
Follow-up ladder: plain .787 | scenario **.696** (v4rl23 .680) | wrong choice **11.9%** (15.8%). Best yet.
Partial ladder: pass **.286** (v4rl23 .190, baseline .127) | gold cited .75 | mismatch stated **.41** (.33) | overclaim **.20** (.34). The tier worked.
Turn-1 flips v4rl23 → v4rl24: pass→wrong 12, wrong→pass 5, wrong→over 5. Of the 13 lost passes only 4 finals carry hedge words — the loss
is NOT the hedge leaking into perfect-match answers; it looks like ordinary entity/selection churn on a bank whose turn-1 share fell to 42%.
Read: three families now trade off at 0.8B when the turn-1 share drops; the next round must restore turn-1 gradient (≥50% share, r24-mixed).

## v4rl25 — round 25 (init v4rl24; same rewards; turn-1 restored to 50% of the bank), 2026-09-21 06:10 — NOT adopted
Turn 1: standard **.665** (v4rl24 .710, v4rl23 .745) | over-budget **32** | calls 4.17 | cap harness .715 (capped 30, 6 pass) | reach .73-.75.
Follow-up: scenario .696 (=v4rl24) | wrong choice 11.1%. Partial: pass .254 (v4rl24 .286) | mismatch stated **.48** (.41) | overclaim .20 | over-budget 22.
**Diagnosis: cap-farming, not bank share.** Turn-1 cap-hit rate in training 10% → 21% across the round; calls/episode 4.0 → 5.1; the
standard-vs-cap harness gap (5 pts) is the round-19 signature. Three chained rounds under cap scoring (23 → 24 → 25) re-inflated
length exactly as 18 → 19 did. Reach is fine (.75); every lost point is a rollout killed at the 3rd web_search on the standard harness.
The follow-up and partial gains are intact (mismatch-stated still rising), so the family mix is not the problem — the cap rule is.
**v4rl23 remains best all-round; v4rl24 best on follow-up + partial.**

**Backed up to HF 2026-09-28 (private):** `topoloop/wanderfugl-0_8b-v4rl15`, `-v4rl23`, `-v4rl24` — `serve/` composites only
(the evaluated step20_serve; trainer-loadable text roots were not retained for these rounds). v4rl25 not backed up (not adopted).
Also `topoloop/wplaces-e5-small-graded-v1` = wplaces encoder `model_B_all_graded_v1`.

## v4rl23 MLX quant ladder — 2026-09-28 (laptop mlx_lm.server 0.31.3 via reverse tunnel; judge = google/gemma-4-31b-it via OpenRouter for ALL rows incl. a re-judge of the vLLM reference; NVFP4/FP8 judges offline)
ho200, greedy, budgets 3/2/1, radius 300, contract v2. Standard = strict correct rate. Paired vs the re-judged vLLM reference.
| serve | strict | default | tools/q | over-budget | lost/gained vs ref |
|---|---|---|---|---|---|
| vLLM bf16 (eval_iter23 rows, re-judged) | .740 (LADDER .745) | .600 | 3.24 | 8 | — |
| MLX bf16 | **.755** | .610 | 3.29 | 8 | 5 / 8 |
| MLX 8-bit g64 | .720 | .575 | 3.25 | 7 | 8 / 4 |
| MLX 6-bit g64 | .700 | .550 | 3.27 | 8 | 14 / 6 |
| MLX 4-bit g64 | **.570** | .425 | **4.23** | **14** | 42 / 8 |
MLX bf16 ≈ vLLM (within paired noise), so the drops are quantization, not the serving stack. 8-bit costs ~2-3 pts, 6-bit ~4-5, 4-bit collapses (-17 pts; +1 tool/q, 35% more prompt tokens, over-budget ×2) — the agent loop, not just recall, breaks at 4-bit. Same shape as the gazetteer findings (q6 ≈ q8 there; here 6 is a real but small step down). Ship 8-bit if 782 MB fits the phone budget, else 6-bit; do not ship 4-bit without QAT. Rows: runs/mlx_quant_v4rl23/{vllm_ref,bf16,8bit,6bit,4bit}/. Recipe: rl/mlx_quant_eval.sh.

**Qwen3.8-27B LoRA fits on ripper (2026-10-06):** prime-rl RL trainer, fake data, 3-way FSDP over the three 3090s, LoRA r16 on 18.8B of 27.4B params (vision tower frozen), seq 16384, one sequence per GPU per step. In-VRAM OOMs (18 GB bf16 shard + working set > 24 GB); with `fsdp_cpu_offload = true` (+ `optim_cpu_offload = false`, bf16 optimization/reduce dtypes, LM-head chunk 1024) it runs: 4 steps, **1m16s per step steady (640 tok/s, peak 16.2 GiB/GPU, host RAM 188/220 GB used)**. Scaling: a 6M-token RL step ≈ 2.6 h with all three cards training, ≈ 8 h on one card. Config + logs: `runs/lora27b_smoke/` (trainer_lora27b_3gpu_offload.toml). Gotchas: default `optimization_dtype` is float32 (110 GB host materialisation → OOM-killed); `[data.fake] batch_size` is global and must be ≥ world size; `fsdp_cpu_offload` and `optim_cpu_offload` are mutually exclusive. Mismatch KL 6.5 is meaningless on fake data.

**Local judge, one card (2026-10-06):** three 3090s now; GPU0 has no NVLink partner, so it hosts the judge alone and GPU1+2 (NV4) stay free for training. Gemma 4 26B-A4B MoE as cyankiwi's QAT-based AWQ-INT4 (compressed-tensors int4 g32 → Marlin for linear + MoE on sm86), loaded text-only via `--hf-overrides architectures=Gemma4ForCausalLM` (vision tower never instantiated), bf16, 16 GiB weights, 64.8K-token KV at 20K ctx (`rl/serve_judge_gemma26.sh`, section `judge_gemma26_ripper`). Same 200 8-bit strict rows: agreement 185/200 with OpenRouter 31B and 186/200 with local 31B (the local 31B had 193), pass .725 vs .720/.735; fp16 vs bf16 identical on 199/200. Disagreements are leniency — it passes wrong-entity answers the 31B fails (Pelican Brewing wrong site, Cannon Beach for Cannon River, Vancouver hotel for Cape Canaveral), matching the 2026-09-05 calibration (26B FP 39 vs 31B FP 25). 200 rows in 4.5 min at concurrency 12 (31B TP2: 10.6 min at 6). Use it as the cheap always-on judge for probes and in-loop reward; keep the 31B for ladder scores of record.

**Local judge validation (2026-09-30):** ripper Gemma 4 31B QAT-W4A16 (judge_gemma_ripper) re-judged the 8-bit strict rows: agreement 193/200 with OpenRouter gemma-4-31b-it, pass 147 vs 144 (.735 vs .720). 200 rows in 10.6 min at concurrency 6. Good enough to be the default free judge; the NVFP4 box had 194/200 and 197/200 vs the FP8 box.

## Evidence shim A/B — can v4rl23 use web evidence inside the places result zero-shot? (2026-09-30) — NO
Setup: v4rl23 MLX bf16 on the laptop (:8010 via reverse tunnel), ho200, greedy, budgets 3/2/1, radius 300, contract v2,
BENCH_NO_JUDGE=1 (inline judge off — it was gating rollouts on Gemma); judge of record for ALL arms = local Gemma 4 31B
QAT-W4A16 (judge_gemma_ripper), strict. Only WPLACES_API_ENDPOINT differs: :8001 plain, :8009 evidence shim
(`/media/jon/linuxdata/evidence_lab/shim.py` — corpus-wide arctic-l MRL-256 index of 3.5M chunks, page-tag geo gate,
chunks vote for places they name, presence-positive adds, `blog_quote/blog_source/blog_title` on vouched results),
:8012 same with the 174 ho200 source pages held out. Shim stats over the runs: vouched 55% of searches, added a place on 23%.
| arm | model saw | strict | default | paired vs baseline | saw quote | cited it | web calls/q |
|---|---|---|---|---|---|---|---|
| baseline | plain wplaces | **150** | 114 | — | — | — | 1.78 |
| shim / holdout | reranked order + added places, no text (renderer dropped the fields) | 146 / 145 | 110 / 109 | −6+2 / −8+3 | 0 | 0 | 1.82 / 1.84 |
| shim_q / holdout_q | `↳ blog: "quote" — url` context line | 142 / 140 | 102 / 98 | −10+2 / −11+1 | 122 / 115 | 15 / 7 | 1.80 / 1.81 |
| shim_h / holdout_h | `↳ blog [w1] site: "quote"` (handle registered, citable) | 143 / 139 | 107 / 102 | −11+4 / −12+1 | 120 / 115 | 13 / 8 | 1.81 / 1.82 |
- The model ignores evidence it was not trained on: cites the vouched source ~1 in 10 times it sees one, keeps the
  web_search step at the same rate, and the extra text slightly HURTS (−4 to −11 rows, noise-band but one-directional).
  Lost-question judge reasons are attribution cross-wiring (Pacific City given Siletz Bay's decor; Royal Irish YC named,
  Royal St George's ref cited; "no bar mentioned" when the quote said gin bar) — denser result blocks, same 0.8B.
- Holdout tracks the full arm within 2 rows everywhere: no leakage effect at this uptake level.
- Reranking alone (no text) is also slightly negative: in a 6-item list the gold is usually already there, so reordering /
  adding `from blogs` places can only displace it.
- **Native use needs training, not plumbing**: next RL round with the shim as the environment's wplaces tool and the
  citation gate accepting evidence handles (BENCH_EVIDENCE_HANDLE path is in core/harness.py). Also fix the shim's
  precision first (name-match heuristics → enrichment-pass chunk→place links) — once the agent trusts it, errors propagate.
- Harness plumbing added (both copies): `blog_*` pass-through in wp_search (BENCH_EVIDENCE=0 strips), `↳ blog` context
  line in fmt_results, BENCH_EVIDENCE_HANDLE registers the source in the handle table. Rows: runs/evidence_shim/{baseline,
  shim,holdout,shim_q,holdout_q,shim_h,holdout_h}/.
- Infra notes: Gemma judge at 20 tok/s aggregate is the eval bottleneck (judge_v2 200 rows ≈ 35 min/arm when alone);
  laptop MLX bf16 0.8B serves ~1.2K prompt tok/s aggregate for 12 interleaved streams (prefill-bound, prompt cache thrash)
  — serve 8-bit for future runs (Jon). Re-judged bf16 baseline: Gemma 150 vs OpenRouter 151 — same-judge within noise.

### Evidence shim, round 2 (2026-09-30 evening): two shim bugs, the one-search metric, and Qwen3.6-35B
**Metric that matters for the phone — "win on one search"** (`evidence_lab/one_search.py`, no judge): is the gold ref in the
FIRST search_places result, and does it carry a quote? Baseline v4rl23: gold in first search **.710**, in any search .755.
| shim variant | gold in 1st search | gold in 1st search WITH quote | strict | paired vs baseline |
|---|---|---|---|---|
| rerank + dedupe (shim_h) | .620 | .155 | 143 | −11 +4 |
| append (no displacement) + dedupe (shim_a) | .620 | .155 | (judging) | |
| **append, no dedupe, name-guard fixed (shim_b)** | **.710** | **.410** | **148** | −6 +4 |
| same, ho200 source pages held out (holdout_b) | .705 | .330 | 148 | −4 +2 |
- Bug 1: shim dedupe-by-name dropped the GOLD OSM object (wplaces returns relation+way with one name; the bank's gold is
  one specific ref; "nearer wins" threw it away) — 19/200 first searches lost the gold, 18 with an identical query. The
  whole −9 pt first-search drop and most of the strict loss. Fix: never dedupe upstream results.
- Bug 2: the generic-name guard skipped any place name contained in the query — which is exactly the ho200 pattern
  (query = the place name) — so the gold could never be vouched. Fix: guard single-word names only.
- Append mode (upstream top-6 kept, ≤3 blog places appended) is the right merge: baseline recall by construction.
- With the fixes the shim is neutral on strict (148 vs 150, noise) and the first result list carries the gold WITH a
  citable quote on 41% of questions (33% when the question's own source page is excluded). The model still ignores it
  (cites 26/147 sightings, web calls 1.73/q vs 1.78) — that 41% is the ceiling an RL round would chase, and the latency
  prize is ~1 fewer tool call on those questions.
**Qwen3.6-35B-A3B (teacher) gets no uplift either** — plain vs quotes-with-handles, strict, Gemma judge, budgets 8/4/2,
max-tools 12 (its parallel tool calls blow the 3/2/1 budgets instantly): OpenRouter thinking-on 65 vs 61 (22–31 truncations
at 4K tokens; paired churn 29/25 — nondeterministic); LOCAL FP8 TP=2 no-think 67 vs 69 (paired 23/25). Cited the quote 2–4
times per ~110 sightings. Its failures are contract (locator "main place not resolved" 22–38, map compile 11–18, over-budget
25–34 empty finals), i.e. it is untrained on the v2 contract — the harness measures contract compliance, not knowledge.
Gold in first search only .13 (it opens broad and narrows).
**Conclusion:** nobody uses the quote until trained to. Shim is now ready to be the environment's wplaces tool for an RL
round: reward answering from the first search when a quote is present, citation gate accepts evidence handles.
Infra lessons (cost ~1 h): a bare `wait` in a chain script also waits on a backgrounded vLLM/shim server (post_round.sh
gotcha, bit twice) — start servers with `setsid nohup … &` from the launching shell, not inside the script.
Rows: runs/evidence_shim/{shim_a,holdout_a,shim_b,holdout_b,qwen2_*,qwen3_*}/. Shim: EVIDENCE_MODE=append default.

Final (all arms judged 23:50): shim_a 145 / holdout_a 141 strict (append alone did not fix it — the dedupe was the bug). shim_b/holdout_b **default-mode 121/120 vs baseline 114** (+6–7 rows): with the citation gate relaxed the quote-bearing results help a little; under strict they are neutral because the model does not cite the handle.
