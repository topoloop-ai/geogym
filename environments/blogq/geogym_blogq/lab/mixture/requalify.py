#!/usr/bin/env python3
"""requalify — re-check every candidate SFT trace against the CURRENT gates.

DOCTRINE: deterministic gates are timeless; effective verdict = stored LLM
verdict ∧ mech(latest). This script recomputes the mechanical layer (judge_v2
`mechanical()` under JUDGE_V22: compile defects, memory-first opening, DB
spatial contradictions; `format_flags`; identity by NAME of the primary ref)
over cohort trace files, and separates traces whose stored LLM verdict is
older than the judge of record so they can be re-judged.

  # stage 1: mechanics + list what needs an LLM re-judge
  PYTHONPATH=. JUDGE_V22=1 JUDGE_REWARD_MODE=1 python mixture/requalify.py \
      --traces data/cohorts/pass4/qwen_traces.jsonl --bank data/banks/blogq_all_harvestable.jsonl \
      --judge-of-record v2.2 --out data/cohorts/pass4/qwen_requal
  #   -> <out>.mech_ok.jsonl      traces passing every mechanical gate
  #      <out>.rejudge_rows.jsonl judge_v2-shaped rows for those needing an LLM vote
  #      <out>.report.json        drop counts by reason
  # stage 2 (after judge_v2 over rejudge_rows -> verdicts.jsonl):
  PYTHONPATH=. python mixture/requalify.py --traces ... --bank ... --apply verdicts.jsonl --out ...
  #   -> <out>.kept.jsonl         final traces (mech ok AND v2.2 pass AND identity)
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JUDGE_REWARD_MODE", "1")
os.environ.setdefault("JUDGE_V22", "1")
os.environ.setdefault("BENCH_CONTRACT", "v2")
os.environ.setdefault("BENCH_TWO_TOOL", "1")
os.environ.setdefault("BENCH_BLOG_TOOLS", "1")

from geogym_blogq.lab.core.gates import format_flags  # noqa: E402
from geogym_blogq.lab.judges.judge_v2 import mechanical, _RES_LINE, _toks  # noqa: E402

REF_RE = re.compile(r"\b([NWR]\d{4,})\b")
HELD = "/home/jon/workspace/wanderfull/wplaces/topoloop_model_training/data/blogq_heldout_ids.txt"
LADDER = "data/banks/ho200_ladder.jsonl"


def row_from_trace(t):
    """A judge_v2/harness-shaped row from a cohort trace (some cohorts lack `trace`)."""
    msgs = t["messages"]
    trace = t.get("trace")
    if not trace:
        trace = []
        for m in msgs:
            if m.get("role") == "tool":
                c = str(m.get("content") or "")
                name = "web_search" if c.startswith("WEB RESULTS") else "web_fetch" if c.startswith("[w") else \
                    "search_places" if c.startswith("SEARCH RESULTS") else "nearest_places" if c.startswith("NEAREST") else "place_details"
                trace.append({"tool": name, "args": {}, "refs": REF_RE.findall(c)[:12], "result_head": c[:1200]})
    n_tools = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
    return {"id": t["id"], "final": t["final"], "messages": msgs, "trace": trace, "handles": t.get("handles") or {},
            "protocol": "ok", "n_tools": n_tools, "contract": "v2"}


def primary_name_ok(row, q):
    """Identity by NAME (calibration ruling bq-004914): the first map ref's own
    result line must name the expected place (token overlap >= 0.5), or be the
    gold ref, or sit within 400 m of the gold coords (OSM duplicate)."""
    m = re.search(r'\{"map".*\}', row["final"] or "", re.S)
    if not m:
        return False, "no map object"
    try:
        refs = [e.get("ref") for e in json.loads(m.group(0)).get("map") or [] if isinstance(e, dict)]
    except Exception:
        return False, "malformed map"
    if not refs:
        return False, "empty map"
    first = refs[0]
    if not isinstance(first, str) or not REF_RE.fullmatch(first):
        return False, f"map ref {first!r} is not a place ref (web handle?)"
    if first in (q.get("gold_refs") or []):
        return True, "gold ref"
    ev = "\n".join(str(mm.get("content") or "") for mm in row["messages"] if mm.get("role") == "tool")
    lines = {r_: (name.strip(), float(lat), float(lng)) for r_, name, lat, lng in _RES_LINE.findall(ev)}
    # nearest_places lines carry bearing + fame instead of coordinates
    for r_, name in re.findall(r"^([NWR]\d+) \| ([^|\n]+?) \| [^|\n]+ \| (?:[^|\n]+ \| )?\d+m [NSEW]{0,2} \| fame", ev, re.M):
        lines.setdefault(r_, (name.strip(), None, None))
    if first not in lines:
        return False, "primary ref not on a result line"
    name, lat, lng = lines[first]
    exp = q.get("expected_places") or []
    import difflib
    nt = set(_toks(name))
    for e in exp:
        et = set(_toks(e))
        if et and len(et & nt) / len(et) >= 0.5:
            return True, f"name match '{name}'"
        # spelling/apostrophe variants ("Foo Sackley's" vs "Foosackly's")
        if difflib.SequenceMatcher(None, "".join(_toks(name)), "".join(_toks(e))).ratio() >= 0.8:
            return True, f"fuzzy name match '{name}'"
    gc = q.get("gold_coords") or {}
    for g, c in gc.items():
        try:
            glat, glng = float(c[0]), float(c[1])
        except Exception:
            continue
        if lat is not None and abs(glat - lat) < 0.004 and abs(glng - lng) < 0.005:   # ~400 m
            return True, f"within 400 m of gold {g}"
    return False, f"primary ref names '{name}', expected {exp}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--judge-of-record", default="v2.2")
    ap.add_argument("--apply", help="judge_v2 verdict file(s) for the rejudge rows", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    bank = {}
    for l in open(a.bank):
        r = json.loads(l); bank[r["id"]] = r
    held = set(x.strip() for x in open(HELD)) if os.path.exists(HELD) else set()
    ladder = {json.loads(l)["id"] for l in open(LADDER)} if os.path.exists(LADDER) else set()
    verdicts = {}
    for p in a.apply or []:
        for l in open(p):
            v = json.loads(l); verdicts[v["id"]] = v

    rep = collections.Counter(); reasons = collections.Counter()
    mech_ok = open(f"{a.out}.mech_ok.jsonl", "w")
    rejudge = open(f"{a.out}.rejudge_rows.jsonl", "w")
    kept = open(f"{a.out}.kept.jsonl", "w") if a.apply is not None else None
    seen = set()
    for path in a.traces:
        for l in open(path):
            t = json.loads(l); i = t["id"]
            if i in seen:
                rep["dup"] += 1; continue
            seen.add(i)
            q = bank.get(i)
            if q is None:
                rep["not in bank"] += 1; continue
            if i in held or i in ladder:
                rep["held-out/ladder"] += 1; continue
            row = row_from_trace(t)
            drop = None
            m = mechanical(row, q)
            if m.get("bad_refs") or m.get("compile_fail"):
                drop = "compile"
            elif m.get("memory_first"):
                drop = "memory-first"
            elif any(x.get("hard") for x in m.get("spatial") or []):
                drop = "spatial contradiction"
            else:
                fl = format_flags(row["final"], row)
                if fl:
                    drop = "format: " + fl[0].split(":")[0][:40]
                else:
                    ok, why = primary_name_ok(row, q)
                    if not ok:
                        drop = "identity: " + why[:50]
            if drop:
                rep["drop"] += 1; reasons[drop] += 1
                continue
            t.setdefault("prov", {})["requalified"] = "mech-2026-09-06"
            mech_ok.write(json.dumps(t, ensure_ascii=False) + "\n")
            stored = str((t.get("prov") or {}).get("judge", ""))
            if a.judge_of_record not in stored:
                rep["needs LLM re-judge"] += 1
                rejudge.write(json.dumps(row, ensure_ascii=False) + "\n")
                if kept is not None:
                    v = verdicts.get(i)
                    if v and v.get("correct") and v.get("gradable", True) is not False:
                        t["prov"]["judge"] = v.get("judge_version", a.judge_of_record); kept.write(json.dumps(t, ensure_ascii=False) + "\n"); rep["kept"] += 1
                    else:
                        rep["dropped by re-judge" if v else "no verdict yet"] += 1
            else:
                rep["already " + a.judge_of_record] += 1
                if kept is not None:
                    kept.write(json.dumps(t, ensure_ascii=False) + "\n"); rep["kept"] += 1
    mech_ok.close(); rejudge.close()
    if kept: kept.close()
    report = {"counts": dict(rep), "drop_reasons": dict(reasons.most_common())}
    json.dump(report, open(f"{a.out}.report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
