"""Reward = mechanical gates (provable, -1) ∧ format gates ∧ judge of record (Gemma holistic, +1/0).
Mirrors rl/rewards.py; the judge is judges/judge_v2.judge_one against [judge_gemma]."""
import asyncio, json, os, re, types
from geogym_blogq import common  # noqa: F401
from geogym_blogq.lab.core.gates import format_flags, unfence, identity_ok
from geogym_blogq.lab.mixture.requalify import primary_name_ok, row_from_trace as _rft
from geogym_blogq.lab.judges import judge_v2 as J

STYLE_ONLY = ("style", "orphan map refs")   # cosmetic-class flags: small penalty, not -1
# round-2 shaping (opt-in via env so round-1 semantics stay reproducible):
REACH_BONUS = float(os.environ.get("GEOGYM_REACH_BONUS", "0"))        # + when a tool result carries the gold ref
PINNED_FAIL = float(os.environ.get("GEOGYM_PINNED_FAIL", "0"))        # judged fail that pinned the gold ref scores this (else 0)
FORMAT_ONLY = float(os.environ.get("GEOGYM_FORMAT_ONLY", "-1"))       # format-gate-only defect (budget/compile/identity stay -1)
HONEST_FAIL = float(os.environ.get("GEOGYM_HONEST_FAIL", "-1"))       # empty map + "couldn't verify": better than a WRONG pin
HONEST_REACH = float(os.environ.get("GEOGYM_HONEST_REACH", "0"))     # round 7c: fraction of REACH_BONUS paid on the honest tier. 0 = none —
# 2026-09-14 (round 10): base score for a JUDGED-WRONG answer that cleared every gate. It was 0.0 from round 3 to round 9, which
# put it ABOVE the honest tier (HONEST_FAIL −0.5) — and with REACH_BONUS a wrong answer that reached the gold scored +0.25 vs an
# honest decline's −0.5, a 0.75 gradient toward confident guessing. Rounds 8 and 9 both drained the decline tier straight into
# wrong answers (ho200 declines 18→3 and 18→6; wrong 43→53 and 43→63) with no net strict gain. Set this negative and raise
# HONEST_FAIL so the ordering is PASS > honest non-resolution > confident wrong. Default 0.0 = pre-round-10 behaviour.
WRONG_FAIL = float(os.environ.get("GEOGYM_WRONG_FAIL", "0"))
                                                                      # reached-and-declined must not outscore unreached-and-declined (that paid the false decline; ROUND7_SPEC)
# Efficiency (Jon 2026-09-09): a bonus on CORRECT answers only, so it can never push toward
# under-searching — a rollout that skips a call and fails collects nothing. 3 calls is the floor a
# grounded answer needs (web_search -> search_places -> place_details); pay only below the median.
EFFICIENCY = float(os.environ.get("GEOGYM_EFFICIENCY", "0"))          # per tool call saved under EFF_BASE
EFF_BASE = int(os.environ.get("GEOGYM_EFF_BASE", "4"))                # calls at or above this earn nothing
EFF_CAP = float(os.environ.get("GEOGYM_EFF_CAP", "0.2"))              # ceiling on the bonus
# Distance pin gate (Jon 2026-09-08): the failure class the reviews actually found is a pin in the WRONG
# PLACE — the wrong Nelson, Corsica for Cinque Terre, a burger joint 205 km away — not a name mismatch.
# On a judged fail, a primary pin farther than PIN_KM from every gold coordinate is -1; a plausible
# alternative near the gold keeps 0 (or +1 if the judge liked it — passes are never gated). Round-2
# probe: 98% of judged passes pin within 50 km of gold, 51% of judged fails pin beyond 100 km.
# 0 disables it and falls back to the name-based identity_by_category gate.
PIN_KM = float(os.environ.get("GEOGYM_PIN_KM", "0"))
# Pre-skip (Jon 2026-09-08, judge throughput): with PIN_PRESKIP=1 a far pin is scored -1 BEFORE the
# judge is called, pass or fail. Round-2 probe: judged passes pin >100 km from gold 1% of the time
# (chain branches, mostly), while far-pin rollouts are 32% of all judge calls. Costs ~1% of positive
# reward, removes a third of the judge load. Jon 2026-09-08: NOT used — the 5090 NVFP4 judge (~60
# verdicts/min) covers the 49/min need without it, and a far alternative the judge likes keeps +1.
PIN_PRESKIP = bool(int(os.environ.get("GEOGYM_PIN_PRESKIP", "0") or 0))
_HONEST_RE = re.compile(r"\b(could\s?n[o']t|was\s+un(?:able|available)|unable to|did\s?n[o']t)\s+(?:\w+\s+){0,3}"
                        r"(find|verify|confirm|locate|resolve|identify)\b|\bno (?:matching|relevant) (?:place|result)", re.I)
_llm = None; _sem = None; _args = None


def _judge_client(section):
    global _llm, _sem, _args
    if _llm is None:
        import httpx
        from openai import AsyncOpenAI
        if os.environ.get("JUDGE_BASE_URL"):
            ep = {"base_url": os.environ["JUDGE_BASE_URL"], "model": os.environ.get("JUDGE_MODEL", "google/gemma-4-31b-it"),
                  "api_key": os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "EMPTY",
                  "concurrency": int(os.environ.get("JUDGE_CONCURRENCY", "16"))}
        else:
            from geogym_blogq.lab.core.config import EP
            if section not in EP:
                raise RuntimeError("geogym: no judge configured. Set JUDGE_BASE_URL, JUDGE_MODEL and JUDGE_API_KEY "
                                   "(any OpenAI-compatible endpoint; google/gemma-4-31b-it on OpenRouter is the reference).")
            ep = EP[section]
        J.set_no_think(ep["base_url"])
        _llm = AsyncOpenAI(base_url=ep["base_url"], api_key=ep.get("api_key") or "EMPTY",
                           http_client=httpx.AsyncClient(limits=httpx.Limits(max_connections=256), timeout=httpx.Timeout(600.0, connect=15.0)))
        _sem = asyncio.Semaphore(min(48, int(ep.get("concurrency", 64))))
        _args = types.SimpleNamespace(judge_model=ep["model"], judge_base_url=ep["base_url"], judge_api_key=ep.get("api_key") or "EMPTY",
                                      max_tokens=1500, votes=1, concurrency=int(ep.get("concurrency", 64)), recheck=False,
                                      select=None, rows=None, out=None, bank=None)
    return _llm, _sem, _args


_NAME_STOP = {"the", "and", "of", "de", "la", "le", "el", "hotel", "restaurant", "park", "beach", "museum", "cafe", "bar", "national"}


def _fold(s):
    """Diacritics stripped, punctuation dropped, trailing plural 's' removed — 2026-09-09 pin-gate
    review: Slaščičarna/Slascicarna, Two Sisters/Two Sister, Café/Cafe all fooled the raw match."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return [w[:-1] if len(w) > 4 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]{2,}", s)]


def _acronym_hit(name, body):
    """LSU ↔ Louisiana State University: an all-caps token in the prose whose letters are the
    initials of the pin's name counts as naming it."""
    ini = "".join(w[0] for w in _fold(name) if w not in _NAME_STOP)
    return len(ini) >= 2 and re.search(rf"\b{re.escape(ini.upper())}\b", body or "") is not None


def _line_name(row, ref):
    tool = "\n".join(str(m.get("content") or "") for m in row["messages"] if m.get("role") == "tool")
    m = re.search(rf"^{re.escape(str(ref))} \| ([^|\n]+?) \|", tool, re.M)
    return m.group(1).strip() if m else None


def identity_by_category(row, q, final):
    """Jon 2026-09-09: a 'wrong venue' is only wrong when the question NAMED the place. For
    attribute/relational questions the primary ref must name the gold place (name match / gold ref /
    400 m). For find/recommend questions an alternative is legitimate — require only that the pin names
    the place the PROSE recommends (its result-line name appears in the answer)."""
    cat = str(q.get("category") or "")
    if cat in ("blog_attribute", "blog_relational"):
        return primary_name_ok(row, q)
    m = re.search(r'\{"map".*\}', final or "", re.S)
    try:
        refs = [e.get("ref") for e in json.loads(m.group(0)).get("map") or [] if isinstance(e, dict)] if m else []
    except Exception:
        refs = []
    if not refs:
        return False, "empty map"
    nm = _line_name(row, refs[0])
    if not nm:
        return False, f"primary ref {refs[0]} not on a result line"
    prose = (final or "").rsplit("\n", 1)[0] if (final or "").rstrip().endswith("}") else (final or "")
    if len(prose.strip()) < 25:
        return True, "no prose to match against"      # empty-prose answers are a format defect, not an identity one
    toks = [t for t in _fold(nm) if t not in _NAME_STOP]
    body = set(_fold(prose))
    if toks and len([t for t in toks if t in body]) / len(toks) < 0.5 and not _acronym_hit(nm, prose):
        # the pin may still be one of several places the prose names — accept any map ref that matches
        for alt in refs[1:4]:
            an = _line_name(row, alt)
            if an:
                at = [t for t in _fold(an) if t not in _NAME_STOP]
                if at and len([t for t in at if t in body]) / len(at) >= 0.5:
                    return True, f"a cited pin ('{an[:30]}') matches the prose"
        return False, f"pin '{nm[:40]}' is not the place the prose recommends"
    return True, "pin matches prose"


def _hav_km(a, b):
    import math
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def pin_distance_km(row, q, final):
    """(km from the primary map ref to the nearest gold coordinate, why). km is None when it cannot be
    measured: empty map, ref not on a coordinate-bearing result line (nearest_places lines carry
    bearing, not coords), or the bank row has no gold_coords. Unmeasurable never penalises."""
    m = re.search(r'\{"map".*\}', final or "", re.S)
    try:
        entries = json.loads(m.group(0)).get("map") or [] if m else []
    except Exception:
        entries = []
    # 2026-09-09 REWARD HACK (round 3, caught at step 9): the policy learned to prepend the journey
    # nodes — {"here": true}, {"travel": "walk"} — so that entry[0] carried no "ref", the primary pin
    # was "unmeasurable", and the -1 never fired. Wrong-place pins went 25.8% -> 0.9% in nine steps
    # WITHOUT the pins improving. Take the first entry that actually names a place, never entry[0].
    refs = [e["ref"] for e in entries if isinstance(e, dict) and isinstance(e.get("ref"), str)]
    if not entries:
        return None, "empty map"
    if not refs:
        # a map that is non-empty but names no place is a defect, not an honest non-answer
        return float("inf"), "map has no place ref"
    gc = []
    for c in (q.get("gold_coords") or {}).values():
        try:
            gc.append((float(c[0]), float(c[1])))
        except Exception:
            pass
    if not gc:
        return None, "no gold coords"
    ev = "\n".join(str(x.get("content") or "") for x in row["messages"] if x.get("role") == "tool")
    lines = {a: (float(la), float(lo)) for a, _, la, lo in J._RES_LINE.findall(ev)}
    p = lines.get(refs[0])
    if not p:
        return None, f"primary ref {refs[0]} has no coordinates in the evidence"
    d = min(_hav_km(p, g) for g in gc)
    return d, f"pin {d:.0f} km from gold"


def row_from_trace(task_id, messages, state):
    """verifiers Message models -> the harness row shape the gates and judge consume."""
    msgs = []; n_tools = 0
    for m in messages:
        role = m.role
        if role == "assistant":
            tcs = []
            for tc in (m.tool_calls or []):
                tcs.append({"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}})
            n_tools += len(tcs)
            msgs.append({"role": "assistant", "content": m.content or "", **({"tool_calls": tcs} if tcs else {})})
        elif role == "tool":
            c = m.content if isinstance(m.content, str) else "\n".join(getattr(p, "text", "") for p in (m.content or []))
            msgs.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": c})
        else:
            c = m.content if isinstance(m.content, str) else "\n".join(getattr(p, "text", "") for p in (m.content or []))
            msgs.append({"role": role, "content": c})
    final = ""
    if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
        final = msgs[-1]["content"] or ""
    over = bool(getattr(state, "over_budget", False))
    protocol = "over_budget" if over else ("ok" if final.strip() else "no_answer")
    capped = False
    if CAP_ANSWER and over and final.strip():
        # §24 fix: the env said "answer now" — score the answer it gave, don't pretend it never answered
        protocol, capped = "ok", True
    # the gates index the handle table BY HANDLE (w1 -> {url, title, site}), as harness rows do
    handles = {p.get("handle"): {"handle": p.get("handle"), "title": p.get("title"), "site": p.get("site"), "url": u}
               for u, p in (getattr(state, "handles", {}) or {}).items() if p.get("handle")}
    return {"id": task_id, "messages": msgs, "final": final, "protocol": protocol, "n_tools": n_tools, "capped": capped,
            "trace": list(getattr(state, "trace", []) or []), "handles": handles, "contract": "v2"}


DROP_INFRA_ERR = os.environ.get("GEOGYM_DROP_INFRA_ERR", "0") == "1"   # round 8+: set to 1 in the launch env
_INFRA_PREFIX = ("search error:", "nearest error:", "route error:", "blog search error:", "page error:", "details error:")


def _infra_err(m):
    c = str(m.get("content") or "")
    return c.lstrip("[]wm0123456789 ").startswith(_INFRA_PREFIX) or c.startswith('{"error": "details error')


# 2026-09-13 (round 9): two search-shaping terms, both OFF unless set in the launch env.
# REPEAT_QUERY: each search_places call whose normalised query repeats an earlier one costs this (cap 3) — round 8 drained the
#   honest tier into 3x-identical-search loops (7 over-budget on ho200). Mechanical; cannot be farmed except by not repeating.
# GROUNDED_SEARCH: paid ONCE when the FIRST search_places query is grounded in text the model had already retrieved (>=2 content
#   tokens, >=60% of them present in earlier tool output) and is not its own web_search query echoed back. Only paid on rollouts
#   that reached a judge verdict (PASS or judged fail), so a grounded query with no answer earns nothing. This is the "grounding,
#   never distance-to-gold" form from REWARD_HACKS §20; still a step-level signal — the guard logs its rate.
REPEAT_QUERY = float(os.environ.get("GEOGYM_REPEAT_QUERY", "0"))
GROUNDED_SEARCH = float(os.environ.get("GEOGYM_GROUNDED_SEARCH", "0"))
_STOP = {"the","of","de","la","le","del","der","die","das","and","a","an","du","des","el","los","las","al","di","da","do","dos","in","at","near","to"}


def _qtoks(s):
    return [t for t in re.findall(r"[^\W\d_]+", (s or "").lower()) if t not in _STOP and len(t) > 1]


def search_shaping(row):
    """(repeat_count, grounded_first) from the message log; cheap and purely mechanical."""
    seen = set(); repeats = 0; first_q = None; prior_txt = []; prior_web_q = []
    for m in row.get("messages") or []:
        if m.get("role") == "tool":
            prior_txt.append(str(m.get("content") or "")); continue
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            if fn.get("name") == "web_search":
                prior_web_q.append(" ".join(_qtoks(args.get("query"))))
            if fn.get("name") != "search_places":
                continue
            toks = _qtoks(args.get("query")); qn = " ".join(toks)
            if not qn:
                continue
            if qn in seen:
                repeats += 1
            seen.add(qn)
            if first_q is None:
                blob = "\n".join(prior_txt).lower()
                echoed = any(w and qn in w for w in prior_web_q)
                first_q = (len(toks) >= 2 and not echoed and sum(t in blob for t in toks) / len(toks) >= 0.6)
    return min(repeats, 3), bool(first_q)


# 2026-09-14 (round 11, Jon): BINARY reward — 1.0 for a strict pass, 0.0 for everything else, no shaping of any kind.
# Rationale: rounds 5-10 each moved a shaping term and none beat v4rl6's .695 (itself a harness constant, not a reward change),
# while the shaping terms produced every entry in this file. Binary trains directly on the ladder metric, so there is no proxy left
# to invert or farm. "PASS" is emitted by _score_base ONLY after judge-correct AND hard format gates clean AND identity_ok, i.e.
# exactly the strict+gates definition the ladder scores — so reading the reason string is the whole implementation.
# Under BINARY every other knob is ignored: reach, pinned-fail, honest tier, format tiers, efficiency, ask cost/bonus, wrong-fail,
# and the round-9 search terms. Infra-error drops and judge-outage raises are KEPT (they are not policy outcomes, REWARD_HACKS 14/17).
# Known consequence (accepted by Jon 2026-09-14): an honest decline and a wrong answer both score 0, and a wrong answer has a
# nonzero chance of being judged correct, so expected value always favours answering — the decline tier should go to ~0.
# All-fail groups give no gradient and that is fine; the zero-advantage curriculum gate already drops them (~2% at a 60% pass rate).
GEOGYM_BINARY = os.environ.get("GEOGYM_BINARY", "0") not in ("", "0")
CAP_ANSWER = os.environ.get("GEOGYM_CAP_ANSWER", "0") not in ("", "0")   # §24 fix, 2026-09-17 — see servers/tools.py
ASK_COST = float(os.environ.get("GEOGYM_ASK_COST", "0.3"))      # round 7: asking on a CLEAR question costs this, whatever follows
ASK_BONUS = float(os.environ.get("GEOGYM_ASK_BONUS", "0.25"))   # round 7: ask → PASS on an AMBIGUOUS question earns this on top


def _is_terminal_ask(final):
    from geogym_blogq.lab.core.harness import looks_like_ask
    body = final.rsplit("\n", 1)[0] if final.rstrip().endswith("}") else final
    return looks_like_ask(body) and ('"map"' not in final or re.search(r'\{"map"\s*:\s*\[\s*\]\s*\}', final) is not None)


FU_STRICT_CALLS = os.environ.get("GEOGYM_FU_STRICT_CALLS", "1") not in ("", "0")
PM_TIER2 = float(os.environ.get("GEOGYM_PM_TIER2", "0"))   # 2026-09-20: partial-match second tier (0 = off)   # two scenarios, two rewards (2026-09-18)


async def score_followup(row, q, judge_section):
    """Turn-2 (follow-up) reward, 2026-09-18. The prompt already holds the completed turn 1; `row` is the whole
    conversation and row["trace"] the turn-2 tool calls only. Two scenarios, each its own reward (Jon):
      in-context  (the key was already in the turn-1 tool results): PASS iff judged correct AND zero turn-2 tool calls
      needs-call                                                   : PASS iff judged correct AND >= 1 turn-2 tool call
    Gates and the judge prompt live in judges/judge_v2.py (followup_gate / judge_user_followup) so eval == training."""
    final = unfence(row.get("final") or "")
    if row["protocol"] != "ok" or not final.strip():
        return 0.0, f"protocol:{row['protocol']} [fu]", None
    if DROP_INFRA_ERR and any(_infra_err(m) for m in row["messages"] if m.get("role") == "tool"):
        raise RuntimeError("infra error in tool results (backend stalled) — rollout dropped")
    row = dict(row)
    m0 = max((i for i, m in enumerate(row["messages"]) if m.get("role") == "user"), default=0)
    row["turn2"] = {"msg0": m0, "family": (q.get("followup") or {}).get("family"), "in_context": q.get("in_context"),
                    "a1": (q.get("turn1") or {}).get("final")}
    os.environ["JUDGE_FU_STRICT_CALLS"] = "1" if FU_STRICT_CALLS else ""
    llm, sem, args = _judge_client(judge_section)
    v = None
    for attempt in range(4):
        v = await J.judge_one(llm, args, row, q, sem)
        if not v.get("judge_error"):
            break
        await asyncio.sleep(3 * (attempt + 1))
    if v is None or v.get("judge_error"):
        raise RuntimeError(f"judge unavailable: {str(v.get('judge_error') if v else 'no verdict')[:120]}")
    scen = "in-context" if q.get("in_context") else "needs-call"
    if v.get("correct"):
        return 1.0, f"PASS [fu {scen} calls={len(row.get('trace') or [])}]", v
    return 0.0, f"{str(v.get('reason') or 'judge fail')[:60]} [fu {scen} calls={len(row.get('trace') or [])}]", v


async def score_partial(row, q, judge_section):
    """Partial-match reward (2026-09-19, Jon: 'the model learned a great piece of content always exists'). The gold is the best
    available answer but fails ONE stated constraint. PASS = names the gold (cited) AND states the mismatch AND no overclaim;
    judge_v2.judge_user_partial carries the rubric; the gold-cited check is mechanical (partial-gate)."""
    final = unfence(row.get("final") or "")
    if row["protocol"] != "ok" or not final.strip():
        return 0.0, f"protocol:{row['protocol']} [pm]", None
    if DROP_INFRA_ERR and any(_infra_err(m) for m in row["messages"] if m.get("role") == "tool"):
        raise RuntimeError("infra error in tool results (backend stalled) — rollout dropped")
    llm, sem, args = _judge_client(judge_section)
    v = None
    for attempt in range(4):
        v = await J.judge_one(llm, args, row, q, sem)
        if not v.get("judge_error"):
            break
        await asyncio.sleep(3 * (attempt + 1))
    if v is None or v.get("judge_error"):
        raise RuntimeError(f"judge unavailable: {str(v.get('judge_error') if v else 'no verdict')[:120]}")
    if v.get("correct"):
        return 1.0, f"PASS [pm calls={row['n_tools']}]", v
    # 2026-09-20 two-tier (GEOGYM_PM_TIER2, default 0.5): the gold is cited and the reply does NOT overclaim, but the mismatch is
    # left unsaid -> partial credit. Round 23 showed a 3-condition pass at a 10% base rate leaves 78% of partial groups all-zero;
    # this tier makes "right place, no false claim" distinguishable from "wrong place / false claim" so the hedge can bootstrap.
    m = re.search(r"overclaim=(\d) mismatch_stated=(\d)", str(v.get("reason") or ""))
    if PM_TIER2 and m and (v.get("mech") or {}).get("cites_gold") and m.group(1) == "0":
        return PM_TIER2, f"pm tier2: gold, no overclaim, mismatch unsaid [pm calls={row['n_tools']}]", v
    return 0.0, f"{str(v.get('reason') or 'judge fail')[:60]} [pm calls={row['n_tools']}]", v


async def score(row, q, judge_section, style_penalty=-0.25, clarify=None):
    """Returns (reward, reason, verdict). Round 7: applies the clarification cost/bonus from the env's record."""
    if q.get("followup"):
        return await score_followup(row, q, judge_section)
    if q.get("partial"):
        return await score_partial(row, q, judge_section)
    rew, reason, v = await _score_base(row, q, judge_section, style_penalty)
    if row.get("capped"):
        reason = f"{reason} [capped]"   # step_stats strips " [..." for the per-step table, keeps it in the top-14 list
    if GEOGYM_BINARY:
        # every gate and the judge already ran inside _score_base; "PASS" is the strict+gates+identity verdict
        return (1.0 if str(reason).startswith("PASS") else 0.0), reason, v
    c = clarify or {}
    if c.get("asked"):
        if str(c.get("label")) == "ambiguous":
            if str(reason).startswith("PASS"):
                rew += ASK_BONUS; reason += f" +ask_bonus({ASK_BONUS})"
            # warranted ask that did not convert: no cost — same as not asking
        else:
            rew -= ASK_COST; reason += f" -ask_cost({ASK_COST})"
    if (REPEAT_QUERY or GROUNDED_SEARCH) and not str(reason).startswith(("protocol", "compile", "memory-first", "zero-tool")):
        _rep, _grounded = search_shaping(row)
        if REPEAT_QUERY and _rep:
            rew += REPEAT_QUERY * _rep; reason += f" repeat_query x{_rep}({REPEAT_QUERY})"
        if GROUNDED_SEARCH and _grounded and v is not None:
            rew += GROUNDED_SEARCH; reason += f" +grounded_search({GROUNDED_SEARCH})"
    return rew, reason, v


async def _score_base(row, q, judge_section, style_penalty=-0.25):
    """Returns (reward, reason, verdict)."""
    final = unfence(row.get("final") or "")
    if row["protocol"] != "ok" or not final.strip():
        return -1.0, f"protocol:{row['protocol']}", None
    if DROP_INFRA_ERR and any(_infra_err(m) for m in row["messages"] if m.get("role") == "tool"):
        # 2026-09-11 (7c step 7): a backend stall is never the policy's outcome — drop the rollout like a judge outage
        # (verifiers discards errored rollouts; never a neutral score, REWARD_HACKS 14/17). Env-gated: off for the live 7c run.
        raise RuntimeError("infra error in tool results (backend stalled) — rollout dropped")
    if row["n_tools"] < 1:
        body = final.rsplit("\n", 1)[0] if final.rstrip().endswith("}") else final
        if "?" in body and len(body) < 700:
            return HONEST_FAIL, "zero-tool clarification (honest tier)", None
        return -1.0, "zero-tool answer", None
    mm = J.mechanical(row, q)
    if mm.get("bad_refs") or mm.get("compile_fail"):
        return -1.0, "compile", None
    if mm.get("memory_first"):
        return -1.0, "memory-first", None
    if any(x.get("hard") for x in mm.get("spatial") or []):
        return -1.0, "spatial contradiction", None
    golds = set(q.get("gold_refs") or [])
    # round 7 (2026-09-11): an honest non-resolution or a (second) clarifying question is the honest tier, full stop —
    # the VOCAB gate ("places database") used to score these −0.5, below honesty, which punished the model's best
    # failure mode (STATE.md round-6 finding). Checked before the format gates so no gate can outrank it.
    _empty_map = re.search(r'\{"map"\s*:\s*\[\s*\]\s*\}', final) is not None
    if _empty_map and (_HONEST_RE.search(final) or _is_terminal_ask(final)):
        _tt = "\n".join(str(m.get("content") or "") for m in row["messages"] if m.get("role") == "tool")
        _reach = REACH_BONUS if (golds and any(re.search(rf"\b{re.escape(g)}\b", _tt) for g in golds)) else 0.0
        return HONEST_FAIL + _reach * HONEST_REACH, ("terminal clarifying question" if _is_terminal_ask(final) else "honest non-resolution"), None
    fl = format_flags(final, row)
    hard_fl = [f for f in fl if not f.startswith(STYLE_ONLY)]
    tool_txt = "\n".join(str(m.get("content") or "") for m in row["messages"] if m.get("role") == "tool")
    reached = any(re.search(rf"\b{re.escape(g)}\b", tool_txt) for g in golds) if golds else False
    reach = REACH_BONUS if reached else 0.0
    if hard_fl:
        return FORMAT_ONLY + reach, "format: " + hard_fl[0][:40], None
    # 2026-09-09 (alt-answer review): the map object is the app's pin. A ref that names something
    # other than the place the prose is about is a defect whatever the judge says — and it is 52% of
    # all judge failures. Honest non-resolution (empty map + "couldn't verify") is scored ABOVE it.
    m_ = re.search(r'\{"map"\s*:\s*\[\s*\]\s*\}', final)
    if m_ and _HONEST_RE.search(final):
        return HONEST_FAIL + reach * HONEST_REACH, "honest non-resolution", None
    if PIN_KM > 0 and PIN_PRESKIP:
        d, why_d = pin_distance_km(row, q, final)
        if d is not None and d > PIN_KM:
            return -1.0 + reach, "pin distance (pre-judge): " + why_d, None
    llm, sem, args = _judge_client(judge_section)
    # A judge outage must never become a training signal: retry with backoff, then ERROR the rollout
    # (verifiers drops errored rollouts) rather than score it 0.  (2026-09-08: a wedged Gemma turned
    # a third of a step into neutral zeros.)
    v = None
    for attempt in range(4):
        v = await J.judge_one(llm, args, row, q, sem)
        if not v.get("judge_error"):
            break
        await asyncio.sleep(20 * (attempt + 1))
    if v is None or v.get("judge_error"):
        raise RuntimeError(f"judge unavailable: {str(v.get('judge_error') if v else 'no verdict')[:120]}")
    if v.get("gradable") is False:
        return 0.0, "ungradable (neutral)", v
    if str(v.get("reason", "")).startswith("incomplete"):
        return -1.0, "incomplete", v
    if v.get("correct"):
        # 2026-09-09 (RL hack review): identity by NAME — the map's first ref must be the place the
        # answer is about (gold ref, name match, or within 400 m); a same-named-ish wrong entity is -1
        if not identity_ok(row, q, final, v):
            return -1.0, "identity: primary ref is not the asked-for place", v
        eff = min(EFF_CAP, EFFICIENCY * max(0, EFF_BASE - row.get("n_tools", EFF_BASE))) if EFFICIENCY else 0.0
        return (1.0 + (style_penalty if fl else 0.0) + reach + eff), \
               "PASS" + (" (style)" if fl else "") + (f" [{row.get('n_tools')} calls +{eff:.2f}]" if eff else ""), v
    # 2026-09-09 (3× pin-gate review): the name gate is only ~18% precise on PASSES — legitimate
    # alternatives and sub-place-vs-container dominate — so it never overturns a pass. On rollouts the
    # judge ALREADY failed, the earlier reviews found wrong pins real and dominant, so it costs -1 there.
    if PIN_KM > 0:
        d, why_d = pin_distance_km(row, q, final)
        if d is not None and d > PIN_KM:
            return -1.0 + reach, "pin distance (on a judged fail): " + why_d, v
        if d is None and why_d == "empty map":
            # no pin at all on a wrong answer sits with the honest tier, whatever the wording — a silent
            # empty map must not score above "couldn't verify"
            return HONEST_FAIL + reach * HONEST_REACH, "empty map on a judged fail", v
    else:
        ok_id, why_id = identity_by_category(row, q, final)
        if not ok_id:
            return -1.0 + reach, "identity (on a judged fail): " + why_id[:50], v
    pinned = any(g in final for g in golds) if golds else False
    return (WRONG_FAIL + (PINNED_FAIL if pinned else 0.0) + reach), "judge fail" + (" (gold pinned)" if pinned else ""), v
