#!/usr/bin/env python3
"""Hardened agentq judge (task: judge trust program, 2026-08-30).

Design: the audits showed the 35B judge fails at exhaustive checklist work —
negative existence checks (fabricated specifics pass), tool-semantics
(anchor-relative distances accepted as pairwise), provenance in both
directions. So code does what code is good at and the LLM answers ONE
narrow question: is the answer's place the keyed place, and does it answer
the ask, given pre-computed mechanical facts.

Mechanical pre-checks (computed here, injected into the judge prompt):
  refs_ok        every REF cited in the answer appears in the trace
  cites_gold     answer cites one of the key's gold refs
  distances      true anchor->gold + pairwise gold distances (haversine),
                 so the judge never computes geometry
  flagged        specifics in the answer with no support in trace evidence:
                 street-address-like strings and "also known as X" aliases

Usage (rejudge stored bench rows against the v2 bank):
  python judge_v2.py --rows results/agentq_08b_v34_pass{k}.jsonl ... \
      --bank data/agent_questions/questions_gold_v2.jsonl \
      --select data/judge_calibration_labels.jsonl \
      --judge-base-url http://127.0.0.1:8000/v1 \
      --judge-model qwen3.6-35b-a3b-fp8 -o out.jsonl
"""

import argparse
import asyncio
import json
import math
import os
import re

from openai import AsyncOpenAI

try:  # model/ config (endpoints rotate; never hardcode)
    from geogym_blogq.lab.core.config import EP as _EP
    _WPLACES_EP = _EP["wplaces"]["base_url"]
    _WPLACES_KEY = _EP["wplaces"].get("api_key", "EMPTY")
except Exception:  # standalone use
    _WPLACES_EP = os.environ.get("WPLACES_API_ENDPOINT", "http://127.0.0.1:8001")
    _WPLACES_KEY = os.environ.get("WPLACES_API_KEY", "wplaces-local-dev-key")

REF_RE = re.compile(r"\b([NWR]\d{6,})\b")
ADDR_TYPES = (r"Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Boulevard|Blvd\.?|"
              r"Square|Plaza|Lane|Drive|Quay|Platz|Straße|Strasse|Rue|Calle|"
              r"Via|Avenida|Gasse|Gracht|Laan")
ADDR_RE = re.compile(
    r"\b\d+[\w-]*\s+[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+)?\s+(?:%s)\b" % ADDR_TYPES)
ADDR_RE_EU = re.compile(   # "Calle de Alcalá 123" / "Rue de Rivoli, 5"
    r"\b(?:%s)\s+(?:de\s|del\s|la\s|van\s|der\s)?[A-ZÀ-Ý][\w'’À-ÿ.-]+"
    r"(?:\s+[A-ZÀ-Ý][\w'’À-ÿ.-]+)?,?\s+\d+\b" % ADDR_TYPES)
STREET_RE = re.compile(
    r"\b(?:on|off|at|located (?:on|at))\s+((?:[A-Z][\w'’.-]+\s+){0,2}"
    r"(?:%s))\b" % ADDR_TYPES)
ALIAS_RE = re.compile(
    r"(?:also (?:known as|called|spelled)|known locally as)\s+"
    r"[*_\"'«]*([^*_\"'»,.)]{2,40})", re.I)


def hav_km(a, b):
    p = math.pi / 180
    x = (0.5 - math.cos((b[0] - a[0]) * p) / 2
         + math.cos(a[0] * p) * math.cos(b[0] * p)
         * (1 - math.cos((b[1] - a[1]) * p)) / 2)
    return 12742 * math.asin(math.sqrt(x))


def trace_text(trace):
    parts = []
    for t in trace or []:
        parts.append(str(t.get("head") or ""))
        parts.append(" ".join(t.get("refs") or []))
        parts.append(" ".join(t.get("drefs") or []))
        if t.get("query"):
            parts.append(str(t["query"]))
        if t.get("url"):
            parts.append(str(t["url"]))
        parts.append(" ".join(t.get("urls") or []))
    return "\n".join(parts)


_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def set_no_think(base_url):
    """Pick the provider's own way to turn reasoning OFF (or ON with JUDGE_REASONING=low|medium|high)."""
    global _EXTRA_BODY
    eff = os.environ.get("JUDGE_REASONING")
    if "openrouter" in (base_url or ""):
        _EXTRA_BODY = {"reasoning": {"enabled": True, "effort": eff}} if eff else {"reasoning": {"enabled": False}}
    else:
        _EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": bool(eff)}}


def evidence_text(row, cap=None):
    """Evidence the judge sees. Default = compact trace heads (judge_v2 of
    record). JUDGE_FULL_EVIDENCE=1 -> every tool output verbatim from the
    stored messages (judge version 'v2-full'; calibrate before trusting)."""
    # v2.1: FULL evidence by default (DOCTRINE: never truncate between
    # stages). JUDGE_COMPACT_EVIDENCE=1 restores the old head-only view.
    if not os.environ.get("JUDGE_COMPACT_EVIDENCE"):
        msgs = row.get("messages")
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except Exception:
                msgs = None
        if msgs:
            parts = []
            n_ai = sum(1 for m in msgs if m.get("role") == "assistant")
            k_ai = 0
            for m in msgs:
                if m.get("role") == "assistant":
                    k_ai += 1
                    if V22 and (m.get("content") or "").strip() and (m.get("tool_calls") or k_ai < n_ai):
                        parts.append("[ASSISTANT PROSE — the agent's own words at this point, NOT evidence]: "
                                     + str(m.get("content")).strip())
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        parts.append(f"CALL {fn.get('name')} {fn.get('arguments')}")
                elif m.get("role") == "tool":
                    parts.append(str(m.get("content") or ""))
            txt = "\n".join(parts)
            return txt[:cap] if cap else txt
    txt = trace_text(row.get("trace"))
    return txt[:cap] if cap else txt


# v2.1 (2026-09-03): strict mode gained 5g tool-evidence precedence + 5h ungradable
V23 = bool(os.environ.get("JUDGE_V23"))
V22 = bool(os.environ.get("JUDGE_V22")) or V23
# v2.2 (2026-09-05, calibration-set driven): intermediate assistant prose is
# shown to the judge (memory-first is a trajectory property), a deterministic
# LOCATOR pass finds each "unsupported" specific in the evidence and asks the
# judge about entity attachment instead of presence, rule 5i (no primary ref =>
# incomplete) and compile defects (non-ref in the map object, bad refs) are
# mechanical, and the rubric names what is NOT a specific.
# grounding judge (training reward) when JUDGE_REWARD_MODE is set and not "0"; correctness judge otherwise
REWARD_MODE = os.environ.get("JUDGE_REWARD_MODE", "") not in ("", "0")
JUDGE_VERSION = (("v2.3" if V23 else "v2.2" if V22 else "v2.1") if REWARD_MODE else "v2") + ("-compact" if os.environ.get("JUDGE_COMPACT_EVIDENCE") else "")


def mechanical(row, q):
    final = row.get("final") or ""
    meta = q.get("meta") or {}
    ev = (evidence_text(row) + "\n"
          + str(meta.get("chunk") or "") + "\n"
          + str(meta.get("snippet") or ""))
    cited = set(REF_RE.findall(final))
    # 2026-09-07 (bq-092951#9): a ref counts as retrieved only if a TOOL RESULT carries it. The
    # model's own tool-call arguments and prose never do — it invented W6050000000001, asked
    # place_details about it, and the empty {} result plus its own call text made it "seen".
    tool_txt = "\n".join(str(m.get("content") or "") for m in (row.get("messages") or []) if m.get("role") == "tool")
    seen = set(REF_RE.findall(tool_txt)) if row.get("messages") else set(REF_RE.findall(ev))
    seen |= set(REF_RE.findall(str(meta.get("chunk") or "") + str(meta.get("snippet") or "")))
    bad_refs = sorted(cited - seen)
    grefs = [g for g in (q.get("gold_refs") or []) if g]
    gc = q.get("gold_coords") or {}

    dist_lines = []
    anchor = q.get("anchor")
    coords = [gc[g] for g in grefs if gc.get(g)]
    if anchor and coords:
        dist_lines.append(
            f"question anchor -> gold: {hav_km(anchor, coords[0]):.2f} km")
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist_lines.append(
                f"between {grefs[i]} and {grefs[j]}: "
                f"{hav_km(coords[i], coords[j]):.2f} km")

    import difflib
    import unicodedata
    def _norm(s):
        s = unicodedata.normalize("NFKD", s)
        return "".join(c for c in s if not unicodedata.combining(c)).lower()
    ev_norm = _norm(ev)
    ev_names = set(re.findall(r"[A-Za-zÀ-ÿ][\w'’ .-]{3,50}", ev))
    _AB = {"road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd", "drive": "dr", "lane": "ln", "saint": "st", "mount": "mt"}
    def _fold(t):
        return " ".join(_AB.get(w, w) for w in re.sub(r"[^a-z0-9]+", " ", _norm(t)).split())
    ev_fold = _fold(ev)
    flagged = []
    for m in ADDR_RE.findall(final) + ADDR_RE_EU.findall(final):
        if _norm(m) not in ev_norm and _fold(m) not in ev_fold:
            flagged.append(f"address '{m}' not in evidence")
    for m in STREET_RE.findall(final):
        if _norm(m) not in ev_norm and _fold(m) not in ev_fold:
            flagged.append(f"street/location claim '{m}' not in evidence")
    for m in ALIAS_RE.findall(final):
        alias = re.sub(r"^(?:the|a|an|el|la|le|il|der|die|das)\s+", "", m.strip(), flags=re.I).strip()
        if len(alias) < 5 or _norm(alias) in ev_norm:
            continue
        best = max((difflib.SequenceMatcher(None, _norm(alias), _norm(n)).ratio()
                    for n in ev_names), default=0.0)
        if best < 0.62:   # spelling/transliteration variants are fine
            flagged.append(f"claimed alias '{alias}' not in evidence")

    # verify flagged addresses against the places DB (truth beats absence:
    # a real address the model knew is fine; one the DB contradicts is a
    # confirmed fabrication)
    if flagged and cited:
        try:
            import urllib.request as _u
            body=json.dumps({"osm_ids": sorted(cited)[:5]}).encode()
            req=_u.Request(_WPLACES_EP + "/places/details", body,
                {"Content-Type":"application/json","X-API-Key":_WPLACES_KEY})
            det=json.loads(_u.urlopen(req, timeout=10).read())
            db_text=_norm(json.dumps(det, ensure_ascii=False))
            kept=[]
            for fl in flagged:
                m=re.search(r"'(.+)'", fl)
                if fl.startswith("address") and m and _norm(m.group(1)) in db_text:
                    continue          # DB-confirmed real address
                if fl.startswith("address") and m:
                    fl+=" (DB address on file differs)" if any(
                        isinstance(v,dict) and v.get("formatted_address")
                        for v in det.values()) else ""
                kept.append(fl)
            flagged=kept
        except Exception:
            pass

    sm = re.search(r"SOURCES:\s*(\[.*?\])", final, re.S)
    n_sources = 0
    # CONTRACT v2: credit = [text](w<k>) links resolved through the row's handle
    # table; identity = refs in the trailing {"map": [...]} object.
    v2_last = final.rstrip().rsplit("\n", 1)[-1]
    if not sm and v2_last.startswith("{") and '"map"' in v2_last:
        handles = row.get("handles") or {}
        try:
            obj = json.loads(v2_last)
        except Exception:
            obj = None
            flagged.append("malformed trailing map object")
        for e in (obj or {}).get("map") or []:
            rv = str((e or {}).get("ref", "")) if isinstance(e, dict) else ""
            if rv and not re.fullmatch(r"[NWR]\d+", rv):
                flagged.append(f"map ref is not a place ref: {rv[:40]}")
            elif rv and rv not in ev:
                flagged.append(f"cited ref not in evidence: {rv[:20]}")
        for txt, h in re.findall(r"\[([^\]\n]+)\]\((w\d+)\)", final):
            if handles and h not in handles:
                flagged.append(f"credited handle {h} not in evidence")
        sm = None  # skip the v1 marker checks below
    if sm:
        try:
            entries = json.loads(sm.group(1))
            n_sources = len(entries)
            for e in entries:
                if isinstance(e, dict) and e.get("url"):
                    u = str(e["url"]).strip()
                    if not u.startswith("http"):
                        flagged.append(f"url entry is not a URL: {u[:40]}")
                    elif u not in ev:
                        flagged.append(
                            f"cited URL not in evidence: {u[:60]}")
                elif isinstance(e, dict) and e.get("ref"):
                    rv = str(e["ref"]).strip()
                    if not re.fullmatch(r"[NWR]\d+", rv):
                        flagged.append(
                            f"ref entry is not a place ref: {rv[:40]}")
                    elif rv not in ev:
                        flagged.append(
                            f"cited ref not in evidence: {rv[:20]}")
                elif isinstance(e, str):
                    if e.strip() and e.strip() not in ev:
                        flagged.append(f"cited URL not in evidence: {e[:60]}")
        except Exception:
            flagged.append("malformed SOURCES line")
    marker_ns = {int(x) for x in re.findall(r"\[(\d{1,2})\]", final)} if sm else set()
    for n in marker_ns:
        if sm and (n < 1 or n > n_sources):
            flagged.append(f"citation marker [{n}] out of range")
    if sm and n_sources > 0 and not marker_ns:
        flagged.append("SOURCES entries present but no inline [n] markers")
    elif sm and n_sources > 0:
        orphans = set(range(1, n_sources + 1)) - marker_ns
        if orphans:
            flagged.append(
                f"SOURCES entries never cited by a marker: {sorted(orphans)}")
    for m in re.findall(r"[Aa]ccording to ([^,\n]{3,60})", final):
        name = m.strip(' *"“”')
        toks = [w for w in re.split(r"\W+", name) if len(w) > 3][:3]
        if toks and not any(t.lower() in ev.lower() for t in toks):
            flagged.append(f"cited source '{name[:40]}' not in evidence")
    pre = 0
    for m in (row.get("messages") or []):
        if m.get("role") == "assistant":
            pre = len((m.get("content") or "").strip()) if m.get("tool_calls") else 0
            break
    compile_fail = [f for f in flagged if f.startswith(("map ref is not a place ref", "malformed trailing map object"))]
    spatial = spatial_checks(row, q, final) if (V22 and not os.environ.get("JUDGE_NO_SPATIAL")) else []
    pre_txt = ""
    for m in (row.get("messages") or []):
        if m.get("role") == "assistant":
            pre_txt = (m.get("content") or "").strip() if m.get("tool_calls") else ""
            break
    # memory-first (mechanical): a paragraph naming a bolded candidate answer
    # before the first tool call, that is not a clarifying question
    # a genuine clarifying question is short, ends with "?", and names at most one place;
    # a memory brainstorm with rhetorical "?"s and several bolded candidates is not exempt
    _bold = re.findall(r"\*\*[^*\n]{3,80}\*\*", pre_txt)
    _clarify = pre_txt.rstrip().endswith("?") and len(pre_txt) <= 600 and len(_bold) <= 1
    memory_first = bool(pre_txt) and len(pre_txt) > 300 and bool(_bold) and not _clarify
    return {
        "refs_ok": not bad_refs,
        "bad_refs": bad_refs,
        "cites_gold": bool(cited & set(grefs)),
        "distances": dist_lines,
        "flagged": flagged,
        "prose_before_first_call_chars": pre,
        "compile_fail": compile_fail,
        "memory_first": memory_first,
        "spatial": spatial,
    }


STRICT_RULE = ("""
5f. STRICT GROUNDING: any other concrete specific — address, distance, \
time, price, amenity, history, affiliation, count, crowding, quality \
claims — that is absent from the TOOL EVIDENCE is a fabrication and \
FAILS, no matter how plausible or peripheral. SOURCE EVIDENCE settles \
only WHICH place or answer is correct (identity, recommendation \
equivalence); it NEVER supports a specific the agent did not retrieve — \
the agent never saw that passage. The mechanical flags are \
NOT exhaustive — regex cannot catch most invented specifics; absence of \
a flag is NEVER support. YOU must verify: before the verdict, list every \
concrete specific in the answer you searched for and could NOT find in \
TOOL EVIDENCE (paraphrases, translations, name \
variants, transparent arithmetic on evidenced numbers such as unit or \
currency conversion, and distances/bearings shown in search results all \
count as found; generic descriptive language is not a "specific"). \
A non-empty list means the verdict is wrong.
5g. TOOL EVIDENCE HAS PRECEDENCE. A specific that appears in TOOL EVIDENCE \
is supported even if SOURCE EVIDENCE gives a different value — the agent \
can only answer from what it retrieved, and the source passage may be \
older. NEVER list such a specific as unsupported and NEVER fail the answer \
for disagreeing with the source passage.
5i. PRIMARY REF: report as "primary_ref" the places-DB ref cited in the \
answer's SOURCES line (or, in the newer format, its trailing {"map": [...]} \
object) for the MAIN place it recommends or identifies. A ref \
cited only for a secondary mention does not count. If the main place has \
no cited ref (only a url, or SOURCES empty), primary_ref is null — and a \
place answer whose main place was never resolved in the places database \
is INCOMPLETE: "correct": false, reason "main place not resolved in the \
places database".
5n. CREDIT POINTS AT THE RIGHT PAGE: a claim credited to a page ("according to \
[Site](w3), ...") must appear in THAT page's result or fetched text. A claim \
that exists only in a different result (e.g. a number from w14 credited to w3) \
is a misattribution and counts as an unsupported specific.
5h. UNGRADABLE is NARROW (a retrieval outcome, not an answer outcome). \
Set "gradable": false ONLY when the answer's central claim is SUPPORTED BY \
TOOL EVIDENCE and that tool evidence CONFLICTS with SOURCE EVIDENCE (the \
retrieved page says $10, the passage says $6; the retrieved page says \
moderate, the passage says strenuous). Then do not convict and do not \
credit — the trace needs a better retrieval. \
NOT ungradable: an answer that states specifics ABSENT from tool evidence \
(that is a fabrication: "gradable": true, "correct": false, list them); an \
answer naming the wrong place; an answer that ignores a value the tool \
evidence contains. Those are graded, and they fail.""" + ("""
5j. MEMORY-FIRST (trajectory rule). Text marked [ASSISTANT PROSE ...] is the \
agent's own writing at that point in the trace, never evidence. If the agent \
names a candidate answer (a place presented as THE answer) or states a \
concrete specific BEFORE the tool result that supports it exists, the \
trajectory is memory-first: "correct": false, reason starting "memory-first:", \
even when the final answer is right. A short lead-in ("Let me search for \
hotels near X") is fine.
5k. ENTITY ATTACHMENT. For each specific, find where it sits in the TOOL \
EVIDENCE and check it is under the heading / list entry / record of the \
entity the answer asserts it about. Listicle pages put several places in \
sequence: a date, price, hour or feature that belongs to the PREVIOUS or \
NEXT entry, to a neighbouring hotel, or to another branch of a chain, is \
unsupported for this one (state that it is attached to the wrong entity).
5l. WHAT IS NOT A SPECIFIC (never list these as unsupported): locational \
colour ("along the harbour", "in the old town") unless the evidence \
contradicts it or it is itself the answer; complements and arithmetic of \
evidenced facts (Tu-Su hours => closed Monday; 23:00 => 11 PM; currency or \
unit conversion); category framing about a class of places ("state parks \
usually charge") that is not asserted as a fact about this place; general \
knowledge definitions; a date read from the page's URL or post date; \
distances and bearings printed in search results. Search EVERY tool \
result — fetched pages, place_details JSON, search result lines — before \
listing a specific; paraphrases and name variants count as found. A \
distance printed on a search result line is the distance from the agent's \
anchor, never an altitude or a distance from anything else. When the \
agent says a fact could not be verified and the tool results indeed lack \
it, that is correct behaviour — never fail it using SOURCE EVIDENCE.
5m. A specific that the evidence CONTRADICTS (not merely lacks) is the \
stronger failure: name it as contradicted, quoting the evidence line.""" if V22 else "") + ("""

WORKED RULINGS (from the calibration set — apply the same reasoning):
- "closed on Mondays" | place_details opening_hours "Tu-Su 10:00-18:00" => SUPPORTED (complement of the listed days).
- "open 8:30 AM to 11:00 PM" | place_details "Mo-Fr 08:30-23:00" => SUPPORTED (time conversion).
- "Col de la Forcle sits at 2,540 m" | result line "N… | La Forcle | mountain saddle | 3016m | (…)" and w1 says 2,540 m a.s.l. => SUPPORTED; the 3016m on a result line is the distance from the agent's anchor, never an altitude.
- "somewhat unusual for California State Parks, which typically charge a day-use fee" => NOT a specific about this place (category framing); do not list it.
- "right next to the museum along the harbour", results show whale-watching operators' buildings 90 m away => locational colour; do not list it.
- "private waterfall on-site" | w2 (same hotel): "did we mention that this place has a private waterfall on the property?!" => SUPPORTED — search every result, including fetched pages and place_details JSON, before listing anything.
- "reported by a traveler in January 2015" | the page URL is …/2015/01/24/… => SUPPORTED (dating a fact from the page is correct behaviour).
- "the minimum weight requirement could not be verified" | SOURCE EVIDENCE says 60 lbs but no tool result contains it => CORRECT behaviour; never fail an honest "could not verify" from the source passage.
- "free to enter" | w1: "And the best part? The park has free entry!" closes entry 6 (Marsupial Park); the answer is about entry 8 (Regional Playground) => UNSUPPORTED — attached to a different entity.
- "16 spacious and modern rooms with dreamy bathrooms in creamy marble" | the fetched page puts that under "5) A'jia Hotel"; the answer is about "6) Romance Istanbul Hotel" => UNSUPPORTED (adjacent entry), even though the words are on the page.
- "Norway's largest resort hotel" | w1 "one of the largest hotels in Norway", w2 "the largest resort in Trysil" => UNSUPPORTED (a superlative inflated beyond its scope is a misstated specific).
- "known for simple food and clever wine matching" (the user's own words) | the page says the branch reviewed cannot serve wine and the pairings were a guest writer's ideas => UNSUPPORTED and contradicted; a claim that restates the user's premise needs evidence like any other.
- "about 30 minutes away" with no route or travel-time in any result => UNSUPPORTED; durations are always specifics.
- "located just outside the town center" because the result line said "3276m" from the agent's anchor, while the locality column says "Town Centre" => CONTRADICTED (anchor-relative distance is not evidence of where a place is).
- DB record still lists website + phone, a 2023 page gives hours, a page updated Aug 2024 says "now closed"; the answer says closed, names the 2024 source and discloses the DB record => RIGHT (follow the most recent dated source and say so).
- First assistant turn, before any tool call: "…is the **Kalka-Shimla Railway** … 964 curves, 864 tunnels"; the final later names the right railway from w4 => memory-first: "correct": false, whatever the final says.""" if V23 else "")
               if REWARD_MODE else """
5f. TRUE real-world facts absent from the evidence are acceptable when \
peripheral to the ask; facts CENTRAL to what was asked still require \
tool backing per rule 5b.""")

JUDGE_SYSTEM = """You verify a mapping agent's answer against an answer key \
and pre-computed mechanical facts. The mechanical facts are computed by code \
from ground truth — TRUST THEM over your own knowledge or arithmetic.

Hard rules, applied in order:
1. If any cited REF is listed as invalid (bad_refs), verdict is wrong.
2. A flagged address or alias is a fabrication: verdict is wrong. There is \
no "immaterial" fabricated address — inventing an address, alias, transit \
line, statistic, or historical fact the evidence does not contain is the \
failure mode this rubric exists to catch. Examples that MUST fail: an \
invented street address for the place; "over 2,300 visitors daily"; \
invented metro line names. But: a fabrication is a claim UNSUPPORTED by \
evidence AND flagged or material — do NOT declare a claim false from your \
own world knowledge; your knowledge of geography/history/facts is NOT \
evidence. True-seeming peripheral claims, transparent translations, and \
canonical name variants pass. Only the evidence, the key, and the \
mechanical facts can convict.
3. If the answer asserts a distance between the SAME two places as a \
COMPUTED distance, it must agree within ~25%; matching a search result's \
anchor-relative distance instead is wrong. The computed anchor->gold \
distance describes the QUESTION's anchor only — a distance the answer \
gives for some other nearby place is NOT checkable against it; when the \
pair is unclear, do NOT fail on distance alone.
3b. Factual claims about entities in the evidence must match the evidence: \
calling a bridge a park, claiming place A is inside place B, or describing \
what a REF denotes contrary to its name/category in the evidence is wrong \
even when the main place is correctly identified.
4. If the answer states hours/fee/open-status, it must match the tool \
evidence quoted; contradicting it is wrong. Claiming such facts with no \
tool evidence at all is wrong.
5. Apply the key's answer_notes allowances literally — alternatives the \
notes accept ARE correct; things the notes forbid ARE wrong.
5b. If the ASKED field (fee, hours, phone) is absent from the tool \
evidence: correct behavior is stating it is not on file (pointing to the \
place's website is ideal). An unverified numeric estimate for the asked \
field is wrong, even if labeled as general knowledge.
5c. A correct conclusion reached via fabricated intermediate numbers \
(distances/coordinates not supported by evidence) is WRONG — fabricated \
reasoning fails even when the conclusion happens to be right.
5d. Identifying the correct constituent structures at the right \
coordinates satisfies a site-level expected name (naming Pyramid of the \
Sun & Moon at Teotihuacan's coordinates counts as Teotihuacan) — substance \
over umbrella labels.
5e. Transparent translations and canonical name variants (Camii=Mosque, \
Forum Romanum) are always acceptable."""+STRICT_RULE+"""
6. Otherwise judge ONE question: does the answer identify the same \
real-world place(s) the key describes (any legitimate gold ref or the \
key-accepted name suffices — the exact gold REF is not required), and does \
it actually answer what was asked? Judge substance, not style.

Return JSON only: """ + (
    '{"unsupported_specifics": ["<each unfindable concrete specific, '
    'empty if all supported>"], "primary_ref": "<the places-DB ref (W…/N…/R…) '
    'cited in SOURCES or the trailing map object for the MAIN place the answer recommends or identifies; '
    'null if the main place has no cited ref>", "gradable": true|false, '
    '"correct": true|false, "reason": "<≤25 words>"}'
    if REWARD_MODE else
    '{"correct": true|false, "reason": "<≤25 words>"}')


FOLLOWUP_RULES = """
FOLLOW-UP TURN. The user's first question was answered (TURN 1 below, treat that answer as settled); now grade the agent's
reply to the user's FOLLOW-UP only. The follow-up refers to the turn-1 place by pronoun ("it", "there"). Rules:
- The reply must be about the turn-1 place (families detail/attribute/logistics). Answering about a different place is wrong.
- family nearby: the reply must name a place of the asked KIND that the evidence shows near the turn-1 place (a computed
  distance is given when available); it need not be the turn-1 place.
- family detail: the KEY is the ground-truth field value from the places database; the reply must state it (a canonical
  variant is fine). If the reply says the value is unknown/unavailable while the key exists, it is wrong.
- family attribute: the KEY is the source passage's fact; the reply must state it or something the passage equally supports.
- Evidence from turn 1 is legitimate evidence for turn 2 — the agent may answer from what it already retrieved without any
  new tool call. An unsupported specific is still a fabrication.
- Do not penalise the agent for not repeating turn-1 content, and do not re-grade turn 1.
"""


PARTIAL_RULES = """
PARTIAL-MATCH QUESTION. The user's question states ONE constraint that NO place in the area meets; the KEY place is the best
available answer but fails that constraint (the SOURCE PASSAGE establishes this — see MISMATCH and EVIDENCE). Grade the reply
CORRECT only if ALL hold:
 (a) it recommends the KEY place (or an alternative the passage equally supports) with a valid ref;
 (b) it plainly tells the user the place does NOT meet the stated constraint (or that nothing found does) — a hedge like
     "though it isn't X" / "note that it closes at 5" counts; silence about the constraint does NOT;
 (c) it does not claim or imply the constraint IS met. Claiming the place is vegan / open late / budget / wheelchair-friendly
     etc. when the evidence says otherwise is an OVERCLAIM -> wrong, even if the place is right.
A reply that declines to recommend anything ("I couldn't find a place that meets X") is wrong when the KEY place was retrieved
in the evidence — the user still wants the best available option. Set "overclaim": true/false and "mismatch_stated": true/false
in your JSON alongside the verdict.
"""


def judge_user_partial(row, q, mech):
    pm = q.get("partial") or {}; meta = q.get("meta") or {}
    ev = evidence_text(row, cap=int(os.environ.get("JUDGE_EVIDENCE_CAP", "60000")))
    src_ev = meta.get("chunk") or meta.get("snippet")
    return f"""{PARTIAL_RULES}
QUESTION: {q['question']}
KEY: expected={q.get('expected_places') or meta.get('name')} | gold_refs={q.get('gold_refs')}
CONSTRAINT THE KEY FAILS: {pm.get('constraint')}
MISMATCH (true of the key per the passage): {pm.get('mismatch')}
EVIDENCE (verbatim from the passage): {pm.get('evidence_quote')}
SOURCE PASSAGE:
{str(src_ev)[:1800]}
MECHANICAL FACTS (trusted):
- cited refs valid: {mech['refs_ok']}{' bad=' + str(mech['bad_refs']) if mech['bad_refs'] else ''}
- cites a gold ref: {mech['cites_gold']}
- flagged unsupported specifics: {mech['flagged'] or 'none'}
TOOL EVIDENCE:
{ev}
AGENT ANSWER:
{row.get('final')}"""


def judge_user(row, q, mech):
    if q.get("followup"):
        return judge_user_followup(row, q, mech)
    if q.get("partial"):
        return judge_user_partial(row, q, mech)
    ev = evidence_text(row, cap=int(os.environ.get("JUDGE_EVIDENCE_CAP", "60000")))
    meta = q.get("meta") or {}
    src_ev = meta.get("chunk") or meta.get("snippet")
    src_block = (f"\nSOURCE EVIDENCE (the blog passage this question derives "
                 f"from — treat as ground truth; an answer naming a DIFFERENT "
                 f"place that this passage equally supports is CORRECT; "
                 f"factual claims consistent with this passage are "
                 f"supported):\n{str(src_ev)[:1800]}\n"
                 if src_ev else "")
    return f"""QUESTION: {q['question']}
KEY: expected={q.get('expected_places') or meta.get('name')} | gold_refs={q.get('gold_refs') or [q.get('gold')]} | notes={q.get('answer_notes') or ''}{src_block}
MECHANICAL FACTS (trusted):
- cited refs valid: {mech['refs_ok']}{' bad=' + str(mech['bad_refs']) if mech['bad_refs'] else ''}
- cites a gold ref: {mech['cites_gold']}
- computed distances: {mech['distances'] or 'n/a'}
- flagged unsupported specifics: {mech['flagged'] or 'none'}
- agent prose before its first tool call: {mech.get('prose_before_first_call_chars', 0)} chars
TOOL EVIDENCE:
{ev}
AGENT ANSWER:
{row.get('final')}"""


def _turn2_evidence(row, cap):
    """Turn-1 evidence marked as prior context; turn-2 tool results as the new evidence."""
    msgs = row.get("messages") or []
    m0 = (row.get("turn2") or {}).get("msg0", 0)
    def flat(ms):
        parts = []
        for m in ms:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    parts.append(f"CALL {fn.get('name')} {fn.get('arguments')}")
            elif m.get("role") == "tool":
                parts.append(str(m.get("content") or ""))
        return "\n".join(parts)
    t1 = flat(msgs[:m0]); t2 = flat(msgs[m0:])
    t1 = t1[-max(cap - len(t2) - 2000, 4000):]
    return f"=== TURN-1 EVIDENCE (retrieved before the follow-up) ===\n{t1}\n=== TURN-2 EVIDENCE (new tool results after the follow-up; may be empty) ===\n{t2 or '(no tool calls in turn 2)'}"


def followup_gate(row, q):
    """Mechanical checks for a follow-up row. Returns (fail_reason or None, info dict)."""
    fu = q.get("followup") or {}; final = row.get("final") or ""; fam = fu.get("family"); key = str(fu.get("key") or "")
    info = {"family": fam, "in_context": q.get("in_context"), "turn2_calls": len(row.get("trace") or [])}
    # Two scenarios, two rewards (Jon 2026-09-18): JUDGE_FU_STRICT_CALLS=1 makes an in-context follow-up FAIL if the agent
    # made any turn-2 tool call (the answer was already in its context), and a needs-call follow-up FAIL if it made none.
    if os.environ.get("JUDGE_FU_STRICT_CALLS"):
        if q.get("in_context") and info["turn2_calls"] > 0:
            return f"followup-gate: in-context follow-up but {info['turn2_calls']} tool call(s) made", info
        if not q.get("in_context") and info["turn2_calls"] == 0:
            return "followup-gate: needs-call follow-up answered with no tool call", info
    if fam == "detail":
        f = fu.get("key_field"); t = _norm_txt(final)
        if f == "phone":
            d = re.sub(r"\D", "", key)[-7:]; ok = bool(d) and d in re.sub(r"\D", "", final)
        elif f == "website":
            m = re.search(r"https?://(?:www\.)?([^/\s]+)", key); ok = bool(m) and m.group(1).lower() in final.lower()
        elif f == "formatted_address":
            toks = [x for x in _norm_txt(key).split() if len(x) > 2 and not x.isdigit()]
            ok = bool(toks) and sum(x in t for x in toks) >= max(2, int(0.6 * len(toks)))
        else:   # opening_hours: the judge compares; mechanically require SOME time-like token
            ok = bool(re.search(r"\d{1,2}[:.]\d{2}|\d{1,2}\s?(am|pm)|24/7|daily|mon|tue|wed|thu|fri|sat|sun", final, re.I))
        info["key_present"] = ok
        if not ok:
            return f"followup-gate: detail key ({f}) not stated", info
    if fam == "nearby":
        gc = list((q.get("gold_coords") or {}).values()); cited = REF_RE.findall(final)
        tool_txt = "\n".join(str(m.get("content") or "") for m in (row.get("messages") or []) if m.get("role") == "tool")
        coords = {m.group(1): (float(m.group(3)), float(m.group(4))) for m in _RES_LINE.finditer(tool_txt)}
        ds = []
        for c in cited:
            if c in coords and gc:
                ds.append(round(_hav(coords[c], gc[0]), 1))
        info["nearby_km"] = ds
        if not cited:
            return "followup-gate: nearby answer cites no place ref", info
        if ds and min(ds) > 15:
            return f"followup-gate: nearest cited place is {min(ds)} km from the turn-1 place", info
    return None, info


def _hav(a, b):
    import math
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def judge_user_followup(row, q, mech):
    fu = q.get("followup") or {}; meta = q.get("meta") or {}; t2 = row.get("turn2") or {}
    ev = _turn2_evidence(row, cap=int(os.environ.get("JUDGE_EVIDENCE_CAP", "60000")))
    src_ev = meta.get("chunk") or meta.get("snippet")
    src_block = f"\nSOURCE PASSAGE (ground truth for family attribute):\n{str(src_ev)[:1800]}\n" if src_ev else ""
    gate_reason, ginfo = followup_gate(row, q)
    return f"""{FOLLOWUP_RULES}
TURN 1 — user: {q.get('question1')}
TURN 1 — agent (settled): {str(t2.get('a1') or (q.get('turn1') or {}).get('final') or '')[:1200]}
TURN-1 PLACE: expected={q.get('expected_places') or meta.get('name')} | gold_refs={q.get('gold_refs')}
FOLLOW-UP (grade this): {q['question']}
FAMILY: {fu.get('family')} | KEY: {str(fu.get('key'))[:400]}{src_block}
MECHANICAL FACTS (trusted):
- cited refs valid: {mech['refs_ok']}{' bad=' + str(mech['bad_refs']) if mech['bad_refs'] else ''}
- follow-up gate: {gate_reason or 'passed'} {ginfo}
- flagged unsupported specifics: {mech['flagged'] or 'none'}
{ev}
AGENT REPLY TO THE FOLLOW-UP:
{row.get('final')}"""


RECHECK_TRIGGER = re.compile(
    r"fabricat|contradict|incorrect|not supported|hallucinat|invent|falsely|wrong",
    re.I)

RECHECK_SYSTEM = """A verdict of WRONG was issued for the agent answer below, \
for the stated reason. Verify the conviction by QUOTING evidence.
Rules:
- If the reason alleges a fabricated/contradicted/incorrect claim, you must \
quote the exact text from TOOL EVIDENCE, MECHANICAL FACTS, or the KEY that \
proves it. Your own knowledge of the world is not evidence and cannot \
convict.
- A claim merely ABSENT from evidence convicts only if it appears in \
"flagged unsupported specifics" or is central to what was asked.
- If you cannot quote convicting text under these rules, the conviction \
fails: the answer stands as correct.
Return JSON only: {"upheld": true|false, "quote": "<the convicting text or empty>"}"""


async def _one_vote(llm, args, row, q, mech, sem):
    for attempt in range(6):
        try:
            return await _one_vote_inner(llm, args, row, q, mech, sem)
        except Exception as e:
            if attempt == 5:
                # 2026-09-09: this string MUST start with "judge_error:" — judge_one sets its
                # judge_error flag by that prefix, and rl/envs/.../reward.py drops the rollout on that
                # flag. Before this fix the reason read "api error after retries: ...", so a dead judge
                # came back as a perfectly ordinary verdict of correct=False. When the NVFP4 box died
                # mid-round-3b, 1,468 rollouts were trained on fabricated failures (step 2 reward
                # +0.50 -> -0.08, PASS 36.7% -> 0.0%). A judge outage must never be a training signal.
                return False, f"judge_error: api error after retries: {str(e)[:60]}", [], None
            await asyncio.sleep(2 ** attempt + 1)


_HEAD_RE = re.compile(r'^(#{1,6} .*|\[w\d+\] .*|[NWR]\d+ \| .*|\{"[NWR]\d+": .*|SEARCH RESULTS .*|NEAREST PLACES .*|WEB RESULTS:.*)')


def _norm_txt(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


_ABBR = {"road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd", "drive": "dr", "lane": "ln",
         "saint": "st", "mount": "mt", "fort": "ft", "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w",
         "kilometers": "km", "kilometres": "km", "meters": "m", "metres": "m", "minutes": "min", "hours": "hr", "and": ""}
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "from", "is", "are", "was", "were", "this",
         "that", "its", "it", "by", "as", "or", "which", "located", "situated", "features", "featuring", "offers",
         "historic", "historical", "beautiful", "famous", "popular", "known", "well", "very", "also", "just", "approximately", "about"}


def _toks(s):
    out = []
    for w in _norm_txt(s).split():
        w = _ABBR.get(w, w)
        if w and w not in _STOP and (len(w) > 2 or w.isdigit()):
            out.append(w)
    return out


def locate_specific(spec, ev, min_cov=0.6):
    """Deterministic locator: where does a judge-listed specific sit in the
    evidence? Returns (snippet, heading) or None. Verbatim (normalized)
    first; otherwise the 400-char window with the best coverage of the
    specific's content tokens (abbreviations folded, stop/adjective words
    dropped, numbers exact) if coverage >= min_cov."""
    m = re.search(r"(?:claim|address|alias|specific|source)\s+'(.+?)'\s+not in evidence", str(spec))
    if m:
        spec = m.group(1)
    lines = ev.split("\n")
    n = _norm_txt(spec)
    if len(n) < 4:
        return None
    hit = None
    for i, ln in enumerate(lines):
        if n in _norm_txt(ln):
            hit = i; break
    if hit is None:
        toks = _toks(spec)
        if not toks:
            return None
        flat = ev; best = (0.0, None)
        # slide a 400-char window in 200-char steps over the flat evidence
        for k in range(0, max(1, len(flat) - 200), 200):
            win = flat[k:k + 400]
            wt = set(_toks(win))
            cov = sum(1 for t in toks if t in wt) / len(toks)
            if cov > best[0]:
                best = (cov, k)
        if best[0] < min_cov:
            return None
        k = best[1]
        # map char offset back to a line index for the heading walk
        pos = 0
        for i, ln in enumerate(lines):
            pos += len(ln) + 1
            if pos > k:
                hit = i; break
        if hit is None:
            hit = len(lines) - 1
    head = ""
    for j in range(hit, -1, -1):
        m = _HEAD_RE.match(lines[j].strip())
        if m:
            head = lines[j].strip()[:140]; break
    snippet = " ".join(x.strip() for x in lines[max(0, hit - 1):hit + 2])
    if len(snippet) > 600:
        k = _norm_txt(snippet).find(n)
        snippet = snippet[max(0, k - 250):k + 350] if k >= 0 else snippet[:600]
    return snippet, head


_REL = [
    # (regex, supported_within_m, contradicted_beyond_m)
    (r"(?:directly\s+)?(?:across (?:the (?:street|road|square) )?from|opposite|next to|adjacent to|beside|steps (?:away )?from|right (?:in front of|by|beside|next to)|in front of)", 200, 600),
    (r"(?:overlooking|facing|overlooks|faces)", 300, 800),
    (r"(?:near|close to|nearby to|a (?:short|quick|few[- ]minute) (?:walk|stroll) (?:from|to)|(?:a few|\d+) minutes(?:'|s)? walk (?:from|to)|just (?:around the corner|down the (?:street|road|block)) from)", 800, 3000),
    (r"(?:within (?:easy )?walking distance (?:of|from|to)|walkable (?:from|to)|an easy walk (?:from|to))", 2000, 5000),
]
_OBJ = r"\s+(?:the\s+)?((?:[A-ZÀ-Ý][\w'’\-]+|d[eu]|del|de la|la|le|of|the|&)(?:\s+(?:[A-ZÀ-Ý][\w'’\-]+|d[eu]|del|de la|la|le|of|the|&)){0,5})"
# 2026-09-10: the coords are always the LAST column; everything between the distance and them is
# optional and has grown twice (locality on 2026-09-05, "hours ..." after the wplaces-api restart on
# 2026-09-09). The old regex allowed exactly one optional column, so the hours column made every
# result line unparseable -> pin gate, spatial checks and name gate silently inert for 10 RL steps.
_RES_LINE = re.compile(r"^([NWR]\d+) \| ([^|\n]+?) \| [^|\n]+ \| (?:[^|\n]+ \| )*?(?:\d+m|[\d.]+ ?km)[^|\n]*(?: \| [^|\n]+)* \| \(([-\d.]+), ([-\d.]+)\)", re.M)


def spatial_checks(row, q, final):
    """Verify 'A <relation> B' claims against the places DB (truth beats absence):
    subject = the answer's main place (first map ref, coords from its own result
    line in the trace); object resolved by a places search anchored there.
    Returns a list of dicts {claim, object, matched, distance_m, status}."""
    m = re.search(r'\{"map".*\}', final or "", re.S)
    refs = []
    if m:
        try:
            refs = [e.get("ref") for e in json.loads(m.group(0)).get("map") or [] if isinstance(e, dict)]
        except Exception:
            refs = []
    if not refs:
        return []
    ev = evidence_text(row)
    coords = {}
    for r_, name, lat, lng in _RES_LINE.findall(ev):
        coords.setdefault(r_, (name.strip(), float(lat), float(lng)))
    subj = coords.get(refs[0])
    if not subj:
        return []
    out = []
    body = re.sub(r'\{"map".*\}', "", final, flags=re.S)
    body = re.sub(r"\[([^\]]+)\]\(w\d+\)", r"\1", body)
    for rx, ok_m, bad_m in _REL:
        for mm in re.finditer(rx + _OBJ, body):
            obj = mm.group(1).strip()
            if len(obj) < 4 or _norm_txt(obj) in _norm_txt(subj[0]):
                continue
            if _norm_txt(obj) in {"old town", "the old town", "city centre", "city center", "downtown", "town centre", "town center",
                                  "historic centre", "historic center", "main square", "the harbour", "the harbor", "the beach",
                                  "the station", "the airport", "the sea", "the coast", "the river", "the lake", "the park"}:
                continue   # generic locality words: the locality column / colour rule handle these
            try:
                import urllib.request as _u
                req = _u.Request(_WPLACES_EP + "/places/search",
                                 json.dumps({"query": obj, "lat": subj[1], "lng": subj[2], "limit": 5}).encode(),
                                 {"Content-Type": "application/json", "X-API-Key": _WPLACES_KEY})
                res = json.loads(_u.urlopen(req, timeout=10).read()).get("results", [])
            except Exception:
                res = []
            import difflib
            BIG = ("city", "town", "village", "suburb", "locality", "neighbo", "quarter", "hamlet", "administrative",
                   "area", "park", "reserve", "lake", "bay", "sea", "river", "mountain", "peak", "range", "island",
                   "beach", "forest", "wilderness", "valley", "glacier", "region", "county", "district", "coast")
            def _tokmatch(a, bset):
                return any(a == b or (len(a) > 4 and len(b) > 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.75) for b in bset)
            cands = []
            for c in res:
                nm = c.get("name_en") or c.get("name") or ""
                ot = list(dict.fromkeys(_toks(obj))); nt = set(_toks(nm))
                if not ot or not all(_tokmatch(t, nt) for t in ot):
                    continue
                extra = len(nt) - sum(1 for t in nt if any(_tokmatch(o, {t}) for o in ot))
                catc = str(c.get("category") or "").lower()
                isbig = any(k in catc for k in BIG)
                if extra > (2 if isbig else 0):
                    continue
                cands.append((0 if isbig else 1, extra, float(c.get("distance_meters") or 0), c))
            cands.sort(key=lambda x: (x[0], x[1], x[2]))
            best = cands[0][3] if cands else None
            if best is None:
                out.append({"claim": mm.group(0).strip()[:80], "object": obj, "matched": None, "distance_m": None, "status": "unresolved"})
                continue
            cat = str(best.get("category") or "").lower()
            big = any(k in cat for k in ("city", "town", "village", "university", "campus", "stadium", "park", "reserve", "forest", "golf", "zoo", "airport", "resort", "lake", "beach", "suburb", "locality", "neighbo", "quarter", "hamlet", "administrative",
                                        "area", "park", "reserve", "lake", "bay", "sea", "river", "mountain", "peak", "range", "island",
                                        "beach", "forest", "wilderness", "valley", "glacier", "region", "county", "district", "coast"))
            scale = 10 if big else 1
            d = float(best.get("distance_meters") or 0)
            st = "supported" if d <= ok_m * scale else "contradicted" if d > bad_m * scale else "borderline"
            tight = rx.startswith("(?:directly")
            out.append({"claim": mm.group(0).strip()[:80], "object": obj, "matched": best.get("name"), "category": cat[:40],
                        "distance_m": round(d), "status": st, "hard": bool(tight and not big and st == "contradicted")})
    return out


async def _one_vote_inner(llm, args, row, q, mech, sem):
    user_msg = judge_user(row, q, mech)
    async with sem:
        resp = await llm.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM},
                      {"role": "user", "content": user_msg}],
            temperature=0.0, max_tokens=args.max_tokens,
            extra_body=_EXTRA_BODY)
    txt = resp.choices[0].message.content or ""
    m = re.search(r'\{.*\}', txt, re.S)

    def _parses(mm):
        try:
            json.loads(mm.group(0)); return True
        except Exception:
            return False
    if resp.choices[0].finish_reason == "length" or not (m and _parses(m)):
        # 2026-09-05 (bq-000751): the verdict JSON was cut off at max_tokens
        # while listing unsupported specifics and the parse fallback scored it
        # FAIL — 2.4% of pass-4 verdicts were this. One retry with room to
        # finish and a terseness instruction; still broken => judge_error.
        async with sem:
            resp = await llm.chat.completions.create(
                model=args.judge_model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": user_msg
                           + "\n\nOutput the JSON object ONLY, valid JSON, each "
                             "unsupported specific quoted in at most 12 words."}],
                temperature=0.0, max_tokens=max(2000, args.max_tokens),
                extra_body=_EXTRA_BODY)
        txt = resp.choices[0].message.content or ""
        m = re.search(r'\{.*\}', txt, re.S)
    try:
        v = json.loads(m.group(0))
        correct = bool(v.get("correct"))
        reason = v.get("reason")
        gradable = v.get("gradable", True) is not False
        pref = v.get("primary_ref")
        pref = str(pref).strip() if pref and re.match(r"^[WNR]\d+$", str(pref).strip()) else None
        if not gradable:
            correct = False
            reason = "ungradable: " + str(reason or "retrieval conflict")
        if q.get("partial"):
            # 2026-09-19 partial-match tier: the two rubric flags become part of the verdict text so step_stats/reports can count them
            oc = bool(v.get("overclaim")); ms = bool(v.get("mismatch_stated", True))
            if correct and (oc or not ms):
                correct = False
            reason = f"[overclaim={int(oc)} mismatch_stated={int(ms)}] {reason}"
        unsup = []
        if REWARD_MODE:
            unsup = [str(u).strip() for u in (v.get("unsupported_specifics") or [])
                     if str(u).strip()]
            if unsup and correct:
                correct = False
                reason = f"unsupported specifics: {'; '.join(map(str, unsup))[:120]}"
            elif unsup and not correct:
                # surface the named specifics for downstream editors (launder
                # feedback loop); verdict unchanged
                reason = f"{reason} | unsupported specifics: {'; '.join(map(str, unsup))[:300]}"
        if V22 and unsup:
            ev_full = evidence_text(row, cap=int(os.environ.get("JUDGE_EVIDENCE_CAP", "60000")))
            report = []; located = []; missing = []
            for u in unsup:
                loc = locate_specific(u, ev_full)
                if loc:
                    located.append(u)
                    report.append(f"- {u!r}\n    FOUND under: {loc[1] or '(no heading)'}\n    line: {loc[0]}")
                else:
                    missing.append(u)
            if report:
                if missing:
                    report.append("NOT FOUND anywhere in TOOL EVIDENCE (these remain unsupported regardless): "
                                  + "; ".join(repr(u) for u in missing))
                async with sem:
                    r3 = await llm.chat.completions.create(
                        model=args.judge_model,
                        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                                  {"role": "user", "content": user_msg
                                   + "\n\nLOCATOR REPORT (code search, trusted): you listed the specifics below as "
                                     "unsupported, but each was FOUND in TOOL EVIDENCE at the line shown, with the "
                                     "nearest preceding heading / result / record. For each one decide only whether "
                                     "it is attached to the entity the answer asserts it about (rule 5k) — presence is "
                                     "settled. Then re-issue your full verdict JSON (same schema), keeping in "
                                     "unsupported_specifics only items that are absent or attached to a different "
                                     "entity.\n" + "\n".join(report)}],
                        temperature=0.0, max_tokens=max(2000, args.max_tokens),
                        extra_body=_EXTRA_BODY)
                t3 = r3.choices[0].message.content or ""
                m3 = re.search(r'\{.*\}', t3, re.S)
                try:
                    v3 = json.loads(m3.group(0))
                    kept3 = [str(u).strip() for u in (v3.get("unsupported_specifics") or []) if str(u).strip()]
                    # the judge may clear only LOCATED items (by ruling them attached to the
                    # right entity); items the locator never found stay unsupported
                    kept_located = [u for u in located if any(_norm_txt(u) == _norm_txt(k) or _norm_txt(u) in _norm_txt(k) or _norm_txt(k) in _norm_txt(u) for k in kept3)]
                    unsup = missing + kept_located
                    p3 = v3.get("primary_ref")
                    pref = str(p3).strip() if p3 and re.match(r"^[WNR]\d+$", str(p3).strip()) else pref
                    if unsup:
                        correct = False
                        reason = "[locator] unsupported specifics: " + "; ".join(unsup)[:140]
                    else:
                        correct = bool(v3.get("correct"))
                        reason = "[locator] " + str(v3.get("reason") or reason)
                except Exception:
                    pass
        if V22 and gradable:
            # mechanical, timeless: compile defects and unresolved identity are never a pass
            if mech.get("bad_refs") or mech.get("compile_fail"):
                correct = False
                reason = "compile: " + "; ".join((mech.get("compile_fail") or []) + [f"bad ref {b}" for b in mech.get("bad_refs") or []])[:160]
            elif mech.get("memory_first"):
                correct = False
                reason = "memory-first: a candidate answer asserted before the first tool call | " + str(reason)[:120]
            elif not pref and (q.get("gold_refs") or []) and REWARD_MODE:
                # rule 5i lives in the strict prompt only; default mode never reports primary_ref
                correct = False
                reason = "incomplete: main place not resolved in the places database | " + str(reason)[:120]
            elif any(x.get("hard") for x in mech.get("spatial") or []):
                bad = [x for x in mech.get("spatial") if x.get("hard")][0]
                correct = False
                reason = f"spatial claim contradicted by the places DB: '{bad['claim']}' — {bad['matched']} is {bad['distance_m']} m away | " + str(reason)[:100]
            else:
                # a DB-supported relation is evidence: strike it from the unsupported list
                sup = [x for x in mech.get("spatial") or [] if x.get("status") == "supported"]
                if sup and unsup:
                    keep = [u for u in unsup if not any(_norm_txt(x["object"]) in _norm_txt(u) for x in sup)]
                    if len(keep) < len(unsup):
                        unsup = keep
                        if not unsup and str(reason).startswith(("unsupported specifics", "[locator] unsupported")):
                            correct = True
                            reason = "supported after spatial check: " + "; ".join(f"{x['claim']} ({x['distance_m']} m)" for x in sup)[:140]
        if (args.recheck and not correct and reason
                and RECHECK_TRIGGER.search(str(reason))):
            async with sem:
                r2 = await llm.chat.completions.create(
                    model=args.judge_model,
                    messages=[{"role": "system", "content": RECHECK_SYSTEM},
                              {"role": "user", "content":
                               judge_user(row, q, mech)
                               + f"\n\nVERDICT REASON TO VERIFY: {reason}"}],
                    temperature=0.0, max_tokens=500,
                    extra_body=_EXTRA_BODY)
            t2 = r2.choices[0].message.content or ""
            m2 = re.search(r'\{.*\}', t2, re.S)
            try:
                v2 = json.loads(m2.group(0))
                if not v2.get("upheld"):
                    correct = True
                    reason = f"[recheck flipped: no convicting quote] {reason}"
            except Exception:
                pass
        return correct, reason, unsup, pref
    except Exception as e:  # noqa: BLE001
        # NOT a verdict: consumers must treat judge_error rows as unjudged.
        # 2026-09-09 (5090 calibration run): distinguish TRANSPORT failures from parse failures —
        # the broad handler used to label a dropped connection "unparseable", which hides an outage
        # (and, in the RL reward, made an outage look like a model defect).
        _tr = ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout",
               "RemoteProtocolError", "InternalServerError", "RateLimitError", "APIStatusError")
        if type(e).__name__ in _tr or isinstance(e, (TimeoutError, ConnectionError)):
            return False, f"judge_error: transport {type(e).__name__}: {str(e)[:60]}", [], None
        return False, f"judge_error: unparseable after retry: {txt[:80]}", [], None


async def judge_one(llm, args, row, q, sem):
    mech = mechanical(row, q)
    if q.get("followup"):
        gate_reason, ginfo = followup_gate(row, q)
        mech["followup"] = ginfo
        if gate_reason and (row.get("final") or "").strip():
            return {"id": row["id"], "pass": row.get("pass"), "correct": False, "votes": "0/0",
                    "reason": gate_reason, "mech": mech, "judge_version": JUDGE_VERSION + "-followup"}
    if q.get("partial") and (row.get("final") or "").strip() and row.get("protocol") in (None, "ok"):
        if not mech.get("cites_gold"):
            return {"id": row["id"], "pass": row.get("pass"), "correct": False, "votes": "0/0",
                    "reason": "partial-gate: best-available place (gold) not cited", "mech": mech, "judge_version": JUDGE_VERSION + "-partial"}
    if REWARD_MODE and not q.get("followup"):
        n_search = sum(1 for t in (row.get("trace") or [])
                       if t.get("tool") in ("web_search",
                                            "search_travel_blogs",
                                            "search_places"))
        if n_search < 1:
            return {"id": row["id"], "pass": row.get("pass"),
                    "correct": False, "votes": "0/0",
                    "reason": "hard-gate: no search tool used", "mech": mech,
                    "judge_version": JUDGE_VERSION + "-strict"}
    if row.get("protocol") not in (None, "ok") or not (row.get("final") or "").strip():
        return {"id": row["id"], "pass": row.get("pass"), "correct": False,
                "votes": "0/0",
                "reason": f"hard-gate: protocol={row.get('protocol')} "
                          f"empty_final={not (row.get('final') or '').strip()}",
                "mech": mech,
                "judge_version": JUDGE_VERSION + ("-strict" if REWARD_MODE else "")}
    votes = await asyncio.gather(*[
        _one_vote(llm, args, row, q, mech, sem) for _ in range(args.votes)])
    n_yes = sum(1 for c, _, _, _ in votes if c)
    correct = n_yes * 2 > len(votes)
    reason = next((r for c, r, _, _ in votes if c == correct), votes[0][1])
    # union of the specifics any vote could not find — quotes FROM the answer,
    # safe to hand to a rewriter (the free-text reason is NOT: it can carry the answer)
    unsup = sorted({u for _, _, us, _ in votes for u in us})
    n_ungr = sum(1 for _, r, _, _ in votes if str(r).startswith("ungradable:"))
    gradable = not (n_ungr * 2 > len(votes))
    if not gradable:
        correct = False
        reason = next(r for _, r, _, _ in votes if str(r).startswith("ungradable:"))
        unsup = []          # nothing to launder — it is a retrieval outcome
    # identity anchor: the ref the judge names for the MAIN place (majority of non-null)
    prefs = [p for _, _, _, p in votes if p]
    primary_ref = max(set(prefs), key=prefs.count) if prefs and len(prefs) * 2 > len(votes) else None
    judge_error = all(str(r).startswith("judge_error:") for _, r, _, _ in votes)
    return {"id": row["id"], "pass": row.get("pass"), "correct": correct,
            "judge_error": judge_error,
            "gradable": gradable, "primary_ref": primary_ref,
            "votes": f"{n_yes}/{len(votes)}", "reason": reason, "mech": mech,
            "unsupported": unsup,
            "judge_version": JUDGE_VERSION + ("-strict" if REWARD_MODE else "")}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True,
                    help="bench result jsonl files; pass number is taken "
                         "from ..._passK.jsonl filename if present")
    ap.add_argument("--bank", required=True)
    ap.add_argument("--select", help="labels jsonl with id+pass: only "
                                     "rejudge those rows")
    ap.add_argument("--judge-base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge-model", default="qwen3.6-35b-a3b-fp8")
    ap.add_argument("--judge-api-key", default="EMPTY")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--votes", type=int, default=3,
                    help="majority-of-N judging: identical temp-0 calls flip "
                         "~4.5%% of verdicts (serving nondeterminism); voting "
                         "squeezes it out")
    ap.add_argument("--recheck", action="store_true",
                    help="quote-or-flip second stage on fabrication convictions "
                         "(measured nearly ineffective; off by default)")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    set_no_think(args.judge_base_url)
    from geogym_blogq.lab.core.bank import load_bank as _load_bank
    bank = {q["id"]: q for q in _load_bank(args.bank)}
    want = None
    if args.select:
        want = {(json.loads(l)["id"], json.loads(l)["pass"])
                for l in open(args.select)}
    done_ids = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            try:
                done_ids.add((json.loads(l)["id"], json.loads(l).get("pass")))
            except Exception:
                pass
    rows = []
    for path in args.rows:
        m = re.search(r"pass(\d+)\.jsonl$", path)
        k = int(m.group(1)) if m else None
        for line in open(path):
            r = json.loads(line)
            r["pass"] = k
            if (r["id"], k) in done_ids:
                continue
            if r["id"] not in bank:
                continue
            if want is None or (r["id"], k) in want:
                rows.append(r)
                done_ids.add((r["id"], k))
    print(f"{len(rows)} rows to judge "
          f"({len(done_ids) - len(rows)} already done, resumed)")
    import httpx
    llm = AsyncOpenAI(
        base_url=args.judge_base_url, api_key=args.judge_api_key,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=args.concurrency,
                max_keepalive_connections=args.concurrency),
            timeout=httpx.Timeout(600.0, connect=15.0)))
    sem = asyncio.Semaphore(args.concurrency)
    outf = open(args.out, "a")
    # stream: write each verdict the moment it lands (a chunk-wide gather let one
    # slow provider call hold back everything else — 2026-09-05 reasoning run)
    done_n = 0
    tasks = [asyncio.ensure_future(judge_one(llm, args, r, bank[r["id"]], sem)) for r in rows]
    for fut in asyncio.as_completed(tasks):
        r = await fut
        outf.write(json.dumps(r, ensure_ascii=False) + "\n")
        done_n += 1
        if done_n % 25 == 0 or done_n == len(rows):
            outf.flush()
            print(f"{done_n}/{len(rows)}", flush=True)
    outf.flush()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
