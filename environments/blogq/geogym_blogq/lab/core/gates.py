"""Deterministic keep-gates shared by harvest, relaunder, and the sweep.

These are the TIMELESS checks (DOCTRINE: recomputable over every stored
verdict vintage). An LLM verdict is the other half:
    kept = verdict.correct AND all gates pass.

    from geogym_blogq.lab.core.gates import format_flags, identity_ok, keep_gate, SEARCH_TOOLS
"""
import json
import re

SEARCH_TOOLS = ("web_search", "search_travel_blogs", "search_places")
WEB_TOOLS = ("web_fetch", "web_search", "search_travel_blogs", "get_full_page")

META_RE = re.compile(
    # launder-context leaks ONLY: the rewrite talking about the evidence /
    # sources / passage it was handed. "based on the information you
    # provided" (addressing the user) is fine and must NOT match.
    r"(?:based on|according to|from|per) (?:the |this )?(?:provided |available |given |"
    r"gathered |retrieved |source |tool |web |blog |fetched )?"
    r"(?:evidence|source evidence|tool evidence|sources?|passage|search results|tool results)\b"
    r"(?! you)"
    r"|the (?:provided |available |source |tool |web |blog |fetched )?"
    r"(?:evidence|passage|search results|tool results) (?:shows?|indicates?|suggests?|does not|doesn't|states?|"
    r"mentions?|also mentions?|confirms?)",
    re.I)
SRC_RE = re.compile(r'SOURCES:\s*\[\s*(\{|\])')
ADHOC_RE = re.compile(r"\[(?:SOURCE|URL|REF|TOOL|WEB|Source|Url|Ref|Tool|Web)[^\]\n]{0,30}\]")  # [SOURCE 2], [SOURCE EVIDENCE], [TOOL EVIDENCE], [URL 1]
VOCAB_LEAK_RE = re.compile(r"\b(?:SOURCE EVIDENCE|TOOL EVIDENCE|ANSWER TO EDIT|MECHANICAL FACTS)\b")
ACCORDING_RE = re.compile(r"[Aa]ccording to ([^,.\[\n]{3,80}?)\s*\[(\d+(?:\s*,\s*\d+)*)\]")
NUMERIC_RE = re.compile(r"(?:[$€£¥]\s?\d|\d+(?:[.,]\d+)?\s?(?:%|km|mi|m\b|ft|hours?|minutes?|am|pm|a\.m\.|p\.m\.)|\b(?:19|20)\d{2}\b|\b\d{1,2}:\d{2}\b)", re.I)
ASKED_FIELD_RE = re.compile(
    r"opening hours|what time.*(open|close)|when (does|is).*(open|close)|"
    r"entrance fee|admission (fee|price|cost)|phone number|contact number|"
    r"official website", re.I)


MAP_LINE_RE = re.compile(r"^\s*(\{\s*\"map\"\s*:.*\})\s*$")
HANDLE_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((w\d+)\)")


def is_v2(text):
    """CONTRACT v2 final: no SOURCES line, last line is the {"map": [...]} object."""
    if "SOURCES:" in (text or ""):
        return False
    last = unfence(text).rsplit("\n", 1)[-1]
    return bool(MAP_LINE_RE.match(last))


FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```\s*$", re.S)


def unfence(text):
    """Format-only repair: a trailing map object wrapped in a ``` fence loses
    the fence (repairs never delete content)."""
    t = (text or "").rstrip()
    m = FENCE_RE.search(t)
    if m and '"map"' in m.group(1):
        t = t[:m.start()].rstrip() + "\n" + " ".join(m.group(1).split())
    return t


def map_object(text):
    """-> (last-line match, parsed object or None)."""
    last = unfence(text).rsplit("\n", 1)[-1]
    m = MAP_LINE_RE.match(last)
    if not m:
        return None, None
    try:
        return m, json.loads(m.group(1))
    except Exception:
        return m, None


def sources_entries(text):
    """-> (match, entries or None if malformed / absent).

    v1: the SOURCES line. v2: the trailing map object, presented as ref
    entries so identity_ok and the keep gate read both contracts."""
    m = re.search(r"SOURCES:\s*(\[.*\])", text, re.S)
    if not m:
        mm, obj = map_object(text)
        if mm is None:
            return None, None
        if not isinstance(obj, dict):
            return mm, None
        # identity lives in `map` (CONTRACT §2.2); `mentions` never rescues identity (§10.5)
        return mm, [{"ref": e["ref"]} for e in (obj.get("map") or [])
                    if isinstance(e, dict) and e.get("ref")]
    try:
        return m, json.loads(m.group(1))
    except Exception:
        return m, None


BARE_HANDLE_RE = re.compile(r"\[w\d+(?:\]\[w\d+)*\](?!\()")          # [w2] or [w3][w4] not followed by "("
NON_HANDLE_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((?!w\d+\))([^)\s]+)\)")  # [text](W123) / [text](http...)
RAW_REF_RE = re.compile(r"\b[NWR]\d{5,}\b")
VAGUE_CREDIT_RE = re.compile(
    r"(?:according to|based on|as (?:described|noted|reported|mentioned) (?:by|in)|per)\s+"
    r"(?:my (?:research|search(?:es)?|findings)|(?:the )?available information|"
    r"(?:multiple|various|several|many|other|recent|online) (?:sources|reviews|reports|travel(?:l)?ers?|guides|blogs)|"
    r"travel(?:l)?(?:er|ers|ers'|er's)?\s*(?:accounts?|writing|writers|reviews|reports|blogs|guides|resources|experiences|websites|sources)|"
    r"travel(?:l)?ers? who|visitors who|reviewers?|"
    r"(?:travel|visitor|guest|online|customer) (?:reviews|accounts|reports|writing|guides))",
    re.I)
DB_LEAK_RE = re.compile(r"(?:places?[- ]database|\(ref:|\bREF:\s*[NWR]\d|in (?:the|our) database|\btool results?\b|\bsearch results\b|\d+ travel[- ]blog mentions?)", re.I)
QUOTE_RUN_RE = re.compile(r"[\"“]([^\"”\n]{25,})[\"”]")
HEADER_RE = re.compile(r"^#{1,4}\s", re.M)
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+\*\*", re.M)
COORD_RE = re.compile(r"-?\d{1,3}\.\d{3,}\s*°?\s*[NSEW]?,\s*-?\d{1,3}\.\d{3,}")


def _prose_head(text):
    """First line of the answer prose, for opener checks."""
    for ln in (text or "").splitlines():
        if ln.strip():
            return ln.strip()[:80]
    return ""


def format_flags_v2(text, row):
    """CONTRACT v2 §3 mechanical checks that need no judge. Empty = clean."""
    probs = []
    text = unfence(text)
    if META_RE.search(text):
        probs.append("meta-leak")
    mm, obj = map_object(text)
    body = text.rsplit("\n", 1)[0].rstrip() if mm else text
    if mm is None:
        return ["no trailing map object"]
    if obj is None or not isinstance(obj, dict):
        return ["map object not valid JSON"]
    if set(obj) - {"map", "mentions", "title"}:
        probs.append(f"unknown keys in object: {sorted(set(obj) - {'map', 'mentions', 'title'})}")
    handles = row.get("handles") or {}
    # 2026-09-07: retrieved = present in a TOOL RESULT. The trace's per-call `refs` include
    # place_details ARGUMENTS, which let an invented ref pass as retrieved (bq-092951#9).
    _tool_txt = "\n".join(str(m.get("content") or "") for m in (row.get("messages") or []) if m.get("role") == "tool")
    trace_refs = set(re.findall(r"\b[NWR]\d{4,}\b", _tool_txt)) if row.get("messages") else \
        {x for t in (row.get("trace") or []) for x in (t.get("refs") or []) + list(t.get("drefs") or [])}
    for e in obj.get("map") or []:
        if isinstance(e, dict) and e.get("ref"):
            if not re.fullmatch(r"[NWR]\d+", str(e["ref"])):
                probs.append(f"map ref not a place ref: {str(e['ref'])[:20]}")
            elif trace_refs and e["ref"] not in trace_refs:
                probs.append(f"map ref not retrieved: {e['ref']}")
        elif isinstance(e, dict) and (e.get("travel") or e.get("here")):
            continue
        else:
            probs.append(f"bad map entry {str(e)[:40]}")
    # 2026-09-09 (RL hack review, `map_ref_stuffing`): every map ref after the first must be a place
    # the prose names — its result line's name (≥50% of its tokens) must appear in the body
    try:
        _tool = "\n".join(str(m.get("content") or "") for m in (row.get("messages") or []) if m.get("role") == "tool")
        _names = {m.group(1): m.group(2) for m in re.finditer(r"^([NWR]\d+) \| ([^|\n]+?) \|", _tool, re.M)}
        _body_toks = set(re.findall(r"[a-z0-9]{3,}", body.lower()))
        _orphans = []
        for e in (obj.get("map") or [])[1:]:
            ref = e.get("ref") if isinstance(e, dict) else None
            nm = _names.get(str(ref)) if ref else None
            if nm:
                toks = [t for t in re.findall(r"[a-z0-9]{3,}", nm.lower()) if t not in ("the", "and", "hotel", "restaurant", "park", "beach", "museum")]
                if toks and len([t for t in toks if t in _body_toks]) / len(toks) < 0.5:
                    _orphans.append(ref)
        if len(_orphans) >= 2:
            probs.append(f"orphan map refs not named in prose: {', '.join(map(str, _orphans[:3]))}")
    except Exception:
        pass
    if re.search(r"\[\d+(?:\s*,\s*\d+)*\]|SOURCES:", body):
        probs.append("v1 residue ([n] marker or SOURCES line)")
    if BARE_HANDLE_RE.search(body):
        probs.append("bare handle marker in prose")
    for txt, tgt in NON_HANDLE_LINK_RE.findall(body):
        probs.append(f"link target is not a handle: ({tgt[:30]})")
        break
    if RAW_REF_RE.search(body):
        probs.append("raw place ref in prose")
    if DB_LEAK_RE.search(body):
        probs.append("prompt/tool vocabulary leak (database/ref/results)")
    if VAGUE_CREDIT_RE.search(body) and not HANDLE_LINK_RE.search(body):
        probs.append("vague credit phrase with no page linked")
    if len(HEADER_RE.findall(body)) >= 3 or len(BULLET_RE.findall(body)) >= 6:
        probs.append("style: spec-sheet formatting (headers/bold bullets)")
    # 2026-09-10 (Jon): the RL-learned tic. After rounds 1-3, 97-98% of ho200 answers opened
    # "According to [Site](w1), ..." — the citation as the subject of the sentence. The style SFT
    # halved it; this keeps RL from sharpening it back. Cosmetic class: -0.25 on a pass, never a fail.
    if re.match(r"\s*(according to|based on)\b", _prose_head(text), re.I):
        probs.append("style: 'According to' opener (citation as sentence subject)")
    if re.search(r"https?://", body):
        probs.append("raw url in prose")
    if COORD_RE.search(body):
        probs.append("coordinates in prose")
    if ADHOC_RE.search(body) or VOCAB_LEAK_RE.search(body):
        probs.append("prompt-vocabulary leak")
    linked = set()
    # 2026-09-08 (mechanical audit): when the row carries no handle table, derive the handles the
    # trace actually assigned from the tool results — a link to a handle no result ever produced
    # ("[Wikipedia](w4)" with no w4 anywhere) is a fabricated citation
    _tool_txt = "\n".join(str(m.get("content") or "") for m in (row.get("messages") or []) if m.get("role") == "tool")
    seen_handles = set(re.findall(r"\[(w\d+)\]", _tool_txt)) | set(re.findall(r"^(w\d+):\s", _tool_txt, re.M))
    for txt, h in HANDLE_LINK_RE.findall(body):
        linked.add(h)
        if handles and h not in handles:
            probs.append(f"handle {h} not in trace")
        elif not handles and h not in seen_handles:
            # 2026-09-09 (RL hack review): "[Wikipedia](w1)" in a trace that never ran web_search —
            # no handle table at all means NO handle is valid
            probs.append(f"handle {h} not in trace")
        elif handles:
            page = handles[h]
            t = {w for w in re.findall(r"[a-z0-9]{3,}", txt.lower())}
            title = page.get("title") or ""
            head = re.split(r" — | – | - | \| |│| » | // |: ", title, maxsplit=1)[0]
            tail = title[len(head):]
            # a site name in the title's tail ("... │Made to Travel") is the site, not a place
            site_from_tail = tail.strip(" —–-|│»/:") if 3 < len(tail.strip(" —–-|│»/:")) < 40 else ""
            ref = {w for w in re.findall(r"[a-z0-9]{3,}", (head + " " + site_from_tail + " " + (page.get("site") or "") + " " + (page.get("url") or "")).lower())}
            if re.sub(r"[^a-z0-9]", "", txt.lower()) == re.sub(r"[^a-z0-9]", "", title.lower()):
                ref |= {w for w in re.findall(r"[a-z0-9]{3,}", txt.lower())}   # the full title is always fine
            sq = re.sub(r"[^a-z0-9]", "", txt.lower())
            dom = re.sub(r"^https?://(www\.)?", "", page.get("url") or "").split("/")[0]
            dom_base = re.sub(r"[^a-z0-9]", "", dom.split(".")[0].lower())
            import difflib
            names_ok = (len(t & ref) / len(t) >= 0.5 if t else False) \
                or (len(sq) > 4 and sq in re.sub(r"[^a-z0-9]", "", (head + (page.get("site") or "") + (page.get("url") or "")).lower())) \
                or (len(sq) >= 8 and sq in re.sub(r"[^a-z0-9]", "", title.lower())) \
                or (len(dom_base) >= 6 and (sq.startswith(dom_base) or dom_base.startswith(sq) and len(sq) >= 6
                                            or difflib.SequenceMatcher(None, sq, dom_base).ratio() >= 0.8))
            if t and not names_ok:
                probs.append(f"link text '{txt[:30]}' does not name {h}")
    # fetched ⇒ credited
    for t in row.get("trace") or []:
        if t.get("tool") in ("web_fetch", "get_full_page") and t.get("url") and handles:
            # 2026-09-08 (mechanical audit): a fetch that returned no page (budget refusal, error,
            # not in corpus) has nothing to credit
            head = str(t.get("result_head") or "")
            if not head:   # merged/probe rows drop result_head: find the tool message that answered this fetch
                msgs = row.get("messages") or []
                for k, m in enumerate(msgs):
                    if m.get("role") == "assistant" and any(str(t["url"]) in str((tc.get("function") or {}).get("arguments") or "")
                                                            for tc in (m.get("tool_calls") or [])):
                        nxt = [x for x in msgs[k + 1:k + 6] if x.get("role") == "tool"]
                        head = str((nxt[0].get("content") if nxt else "") or "")[:200]
                        break
            if re.search(r"^\s*(\[w\d+\][^\n]*\n)?\s*(Fetch budget exhausted|page error|\(page not found|fetch error|blog search error)", head):
                continue
            h = next((k for k, v in handles.items() if v.get("url") == t["url"]), None)
            if h and h not in linked:
                probs.append(f"fetched page {h} never credited")
    if used_web(row) and len(body) > 250 and not linked and (NUMERIC_RE.search(body) or QUOTE_RUN_RE.search(body)):
        probs.append("web used and numbers/quotes stated but no page credited")
    # 2026-09-09 (RL hack review, `uncredited_specifics`): a token link on a vague closing sentence
    # must not launder specifics stated elsewhere. A paragraph whose numbers come from WEB results
    # (not from the places DB) needs a handle link in THAT paragraph.
    if used_web(row) and linked:
        try:
            _msgs = row.get("messages") or []
            _db = "\n".join(str(m.get("content") or "") for m in _msgs if m.get("role") == "tool"
                            and str(m.get("content") or "").startswith(("SEARCH RESULTS", "NEAREST", "{")))
            _web = "\n".join(str(m.get("content") or "") for m in _msgs if m.get("role") == "tool"
                             and str(m.get("content") or "").startswith(("WEB RESULTS", "[w")))
            _bad = 0
            for para in re.split(r"\n\s*\n", body):
                if HANDLE_LINK_RE.search(para) or len(para) < 40:
                    continue
                nums = [n for n in re.findall(r"(?<![\w.])\d[\d,.:]*\d(?![\w.])|(?<![\w.])\d(?![\w.])", para) if len(n) >= 2]
                web_nums = [n for n in nums if n in _web and n not in _db and n not in (row.get("question") or "")]
                if web_nums:
                    _bad += 1
            if _bad >= 1 and _web:
                probs.append(f"specifics from web pages in {_bad} uncredited paragraph(s)")
        except Exception:
            pass
    return probs


def used_web(row):
    return any(t.get("tool") in WEB_TOOLS for t in (row.get("trace") or []))


def format_flags(text, row):
    """Contract defects in a final. Empty list = clean. Each string names
    the defect (fed back to the rewriter as its reason)."""
    from geogym_blogq.lab.core.citations import compile_final
    if is_v2(text):
        return format_flags_v2(text, row)
    probs = []
    if META_RE.search(text):
        probs.append("meta-leak")
    if not SRC_RE.search(text):
        probs.append("no SOURCES line")
    _, cerrs = compile_final(text)
    if cerrs:
        probs.append("compile: " + "; ".join(cerrs[:3])[:200])
    empty_src = bool(re.search(r"SOURCES:\s*\[\s*\]", text))
    if used_web(row) and len(text) > 250 and empty_src:
        probs.append("web used but SOURCES empty")
    if empty_src and NUMERIC_RE.search(text):
        probs.append("states a number/price/time with SOURCES empty")
    if ADHOC_RE.search(text):
        probs.append("ad-hoc marker like [SOURCE 2] / [SOURCE EVIDENCE] (must be [n])")
    if VOCAB_LEAK_RE.search(text):
        probs.append("prompt-vocabulary leak (SOURCE EVIDENCE / TOOL EVIDENCE in the answer)")
    m, entries = sources_entries(text)
    body = text[:m.start()] if m else text
    for src, marks in ACCORDING_RE.findall(body):
        for n in re.findall(r"\d+", marks):
            i = int(n) - 1
            if entries and 0 <= i < len(entries) and "url" not in entries[i]:
                probs.append(f"'According to {src.strip()[:30]}' cites [{n}] which is a ref, not a url")
                break
    return probs


def gold_in_evidence(row, q):
    """Did the gold place appear in anything the agent saw? (ref in a tool
    result, or the gold name in a result head)."""
    gold = {g for g in (q.get("gold_refs") or []) if g}
    trace = row.get("trace") or []
    ev_refs = {x for t in trace for x in (t.get("refs") or [])}
    if gold & ev_refs:
        return True
    gname = ((q.get("meta") or {}).get("name") or "").lower()
    if gname:
        heads = " ".join(str(t.get("head") or "") for t in trace).lower()
        if gname in heads:
            return True
    return False


def identity_ok(row, q, text=None, verdict=None):
    """Identity rule (DOCTRINE): a place answer is anchored by a places-DB
    ref the agent retrieved AND cites in SOURCES. Refs are the identity
    claim and the app-join key; "the gold was in the search results" is
    not a citation. Questions with no gold ref (pure web questions) are
    exempt."""
    gold = {g for g in (q.get("gold_refs") or []) if g}
    if not gold:
        return True
    text = text if text is not None else (row.get("final") or "")
    _, entries = sources_entries(text)
    cited = {e["ref"] for e in (entries or []) if isinstance(e, dict) and "ref" in e}
    if not cited:
        return False
    ev_refs = {x for t in (row.get("trace") or []) for x in (t.get("refs") or [])}
    if verdict and "primary_ref" in verdict:
        # the judge named the anchor for the MAIN place: it must be cited and retrieved
        p = verdict.get("primary_ref")
        return bool(p) and p in cited and (p in ev_refs if ev_refs else True)
    # older verdicts: the main place's ref is unknown — require a GOLD ref among
    # the citations (a ref for a side mention must not rescue the row)
    return bool(cited & gold) and (bool(cited & gold & ev_refs) if ev_refs else True)


def keep_gate(row, verdict, q, *, held=(), harvestable=None, kept_ids=(),
              text=None, require_identity=True):
    """The full deterministic keep decision. `verdict` is a judge_v2 row."""
    i = row["id"]
    if i in held or i in kept_ids:
        return False, "held/kept"
    if harvestable is not None and i not in harvestable:
        return False, "not harvestable"
    if verdict and verdict.get("gradable") is False:
        return False, "ungradable"   # retrieval outcome: re-roll, never launder
    if not verdict or not verdict.get("correct"):
        return False, "verdict"
    if row.get("protocol") != "ok":
        return False, "protocol"
    if sum(1 for t in row.get("trace") or [] if t.get("tool") in SEARCH_TOOLS) < 1 \
            and not (q or {}).get("clarify_expected"):
        # routing variant 3 (REWARD_HACKS 6): the right answer is a question,
        # and it may legitimately involve zero searches
        return False, "no search"
    if not (verdict.get("mech") or {}).get("refs_ok", True):
        return False, "refs_ok"
    text = text if text is not None else (row.get("final") or "")
    fl = format_flags(text, row)
    if fl:
        return False, "format: " + fl[0]
    if q and ASKED_FIELD_RE.search(q.get("question") or ""):
        tools = {t.get("tool") for t in row.get("trace") or []}
        if "place_details" not in tools:
            return False, "details-flow"
    if require_identity and q and not identity_ok(row, q, text, verdict):
        return False, "identity"
    return True, "ok"


# ---------------------------------------------------------------------------
# routing oracle (geogym/routing) — gold is COMPUTED, so the check is
# mechanical: right endpoints, right mode, a real calculate_route call, and
# the stated numbers inside tolerance of that call / the spec. See
# geogym/routing/README.md for the harm classes each WRONG reason maps to.

ROUTE_MODE_CLASS = {"walk": "pedestrian", "bike": "bicycle", "car": "auto", "drive": "auto",
                    "taxi": "auto", "motorcycle": "auto", "bus": "bus"}
_NUM = r"(\d+(?:[.,]\d+)?)"
_DIST_UNIT = r"(km|kilomet(?:er|re)s?|mi|miles?|m|met(?:er|re)s?|ft|feet)"
DIST_RANGE_RE = re.compile(_NUM + r"\s*(?:-|–|to)\s*" + _NUM + r"[\s-]*" + _DIST_UNIT + r"\b", re.I)
DIST_RE = re.compile(_NUM + r"[\s-]*" + _DIST_UNIT + r"\b", re.I)
DUR_RANGE_RE = re.compile(_NUM + r"\s*(?:-|–|to)\s*" + _NUM + r"\s*(min(?:ute)?s?|h(?:ou)?rs?)\b", re.I)
DUR_HM_RE = re.compile(r"(\d+(?:[.,]\d+)?)[\s-]*(?:h|hr|hrs|hours?)\b(?:\s*(?:and\s*)?(\d+)[\s-]*(?:m|min|mins|minutes?)\b)?", re.I)
DUR_MIN_RE = re.compile(_NUM + r"[\s-]*(?:min|mins|minutes?)\b", re.I)
DUR_WORDS = [(re.compile(r"\b(?:an |one )?hour and a half\b", re.I), 5400),
             (re.compile(r"\bhalf an hour\b", re.I), 1800), (re.compile(r"\bquarter of an hour\b", re.I), 900),
             (re.compile(r"\btwo hours\b", re.I), 7200), (re.compile(r"\b(?:an|one) hour\b", re.I), 3600)]
CLARIFY_RE = re.compile(
    # an ask that names what is missing: the city/area, or the user's location.
    # Adverbs may sit between the modal and the verb ("could you PLEASE clarify"),
    # which the first version missed and scored correct asks as failures (2026-09-08).
    r"which (?:city|town|one|location|area|country|region|branch)|what city|"
    r"where (?:are you|is this|exactly|in the world)|"
    r"(?:your|the user's) (?:current )?location|share your location|use your location|"
    r"(?:could|can|would|will)\s+you\s+(?:\w+\s+){0,3}"
    r"(?:tell|clarify|specify|confirm|provide|share|let me know|give)|"
    r"(?:please|kindly)\s+(?:\w+\s+){0,2}(?:tell|clarify|specify|confirm|provide|share|let me know)|"
    r"(?:clarify|specify|confirm|know|identify)\s+(?:the\s+|which\s+|what\s+)?"
    r"(?:specific\s+|exact\s+)?(?:city|town|location|locations|area|region|branch|place)|"
    r"need (?:to know|more)\b[^.?!]{0,40}(?:city|town|location|area|where)|"
    r"do you mean|let me know (?:which|where)|"
    r"there are (?:many|several|multiple)|more than one|several places (?:named|called)|"
    r"which .{0,40}\bdo you mean|need to know (?:the |which |where )?(?:specific |exact |precise )?"
    r"(?:locations?|city|cities|address|place)|(?:you|you're|you are) referring to|"
    r"please (?:specify|provide|tell me)|common names? for|"
    r"many (?:places|businesses|locations|restaurants|cafes|hotels)", re.I)
UNROUTABLE_RE = re.compile(
    r"(?:no|couldn'?t|could not|unable to|can'?t|cannot|not able to|didn'?t|did not|wasn'?t able to|"
    r"failed to) (?:\w+ ){0,3}(?:find|compute|calculate|get|locate|generate|route|return|produce|determine)"
    r"|no (?:walking|driving|cycling|bike|bus|direct)? ?route|route(?:r|ing)? (?:tool )?(?:failed|returned an error|"
    r"could not)|not routable", re.I)
WALK_YES_RE = re.compile(r"\b(?:yes\b|walkable|easy walk|comfortable walk|you can (?:easily |comfortably )?walk|"
                         r"within walking distance|a (?:short|quick|pleasant|easy|brisk|manageable) walk|"
                         r"definitely (?:doable|walkable)|very doable|totally doable|doable on foot|"
                         r"reasonable walk|easily walkable)", re.I)
WALK_NO_RE = re.compile(r"\b(?:not (?:really |very |particularly )?(?:walkable|practical|ideal|convenient)|too far|"
                        r"not within walking|wouldn'?t (?:recommend )?walk|i'?d (?:not|n'?t) walk|"
                        r"not (?:a )?(?:short|easy|practical|reasonable|quick) walk|(?:a |quite a |pretty )?long walk|"
                        r"isn'?t walkable|better to (?:take|drive|bike|cycle|grab|use)|rather (?:long|far)|"
                        r"a (?:bit|little) (?:far|long)|on the long(?:er)? side|consider (?:a |taking )?(?:taxi|bus|bike|car)|"
                        r"not a walk i'?d|no,)", re.I)

_MI, _FT = 1609.344, 0.3048


def _num(v):
    """'1,497' is a thousands separator; '1,5' is a European decimal."""
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", v):
        return float(v.replace(",", ""))
    return float(v.replace(",", "."))


def _to_m(v, unit):
    v = _num(v)
    u = unit.lower()
    if u.startswith("k"):
        return v * 1000
    if u.startswith("mi"):
        return v * _MI
    if u in ("m",) or u.startswith("met"):
        return v
    if u.startswith("f"):
        return v * _FT
    return v


def stated_numbers(body):
    """-> (distances_m: [(lo, hi)], durations_s: [(lo, hi)]) stated in the
    prose. Ranges keep both ends; points are (x, x). Durations are parsed
    (and consumed) first so "1h 10m" never yields a 10-metre distance."""
    dists, durs = [], []
    text = body
    for rx, secs in DUR_WORDS:           # longest phrases first ("hour and a half" before "an hour")
        if rx.search(text):
            durs.append((secs, secs))
            text = rx.sub(" ", text)
    for m in DUR_RANGE_RE.finditer(text):
        k = 60 if m.group(3).lower().startswith("m") else 3600
        durs.append((_num(m.group(1)) * k, _num(m.group(2)) * k))
    text = DUR_RANGE_RE.sub(" ", text)
    for m in DUR_HM_RE.finditer(text):
        x = _num(m.group(1)) * 3600 + (int(m.group(2)) * 60 if m.group(2) else 0)
        durs.append((x, x))
    text = DUR_HM_RE.sub(" ", text)
    for m in DUR_MIN_RE.finditer(text):
        x = _num(m.group(1)) * 60
        durs.append((x, x))
    text = DUR_MIN_RE.sub(" ", text)
    for m in DIST_RANGE_RE.finditer(text):
        dists.append((_to_m(m.group(1), m.group(3)), _to_m(m.group(2), m.group(3))))
    text = DIST_RANGE_RE.sub(" ", text)
    for m in DIST_RE.finditer(text):
        x = _to_m(m.group(1), m.group(2))
        dists.append((x, x))
    return dists, durs


def _within(rng, gold, tol, floor):
    """A stated point/range agrees with gold: midpoint inside tolerance
    (relative tol with an absolute floor for rounding) and the range no
    wider than half the gold — a wide range is tolerance gaming."""
    lo, hi = min(rng), max(rng)
    mid = (lo + hi) / 2
    allow = max(tol * gold, floor)
    return abs(mid - gold) <= allow and (hi - lo) <= 0.5 * gold + floor


def _hav_m(a, b):
    import math
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


RESULT_LINE_RE = re.compile(r"^([NWR]\d+) \| .*?\| \(([-\d.]+), ([-\d.]+)\)\s*$", re.M)


def trace_calls(row):
    """Walk the stored messages: -> (calls, ref_coords, tool_text).
    calls = [{"name", "args", "result"}] in order; ref_coords from
    search-result lines "REF | name | cat | dist | (lat, lng)"."""
    msgs = row.get("messages") or []
    results = {}
    for m in msgs:
        if m.get("role") == "tool":
            results[m.get("tool_call_id")] = m.get("content") or ""
    calls, coords, texts = [], {}, []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = (tc.get("function") or {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            res = results.get(tc.get("id"), "")
            calls.append({"name": fn.get("name"), "args": args, "result": res})
            texts.append(res)
            for ref, lat, lng in RESULT_LINE_RE.findall(res):
                coords[ref] = (float(lat), float(lng))
    return calls, coords, "\n".join(texts)


def route_ok(row, spec, text=None):
    """Oracle verdict for a routing task.
    -> {"bucket": RIGHT|WRONG|MAYBE, "correct": bool, "reason": str,
        "wrong_class": str|None, "primary_ref": str|None, "detail": {...}}"""
    from geogym_blogq.lab.core.citations import compile_final
    final = text if text is not None else (row.get("final") or "")
    v2 = is_v2(final)
    if v2:
        mm, obj = map_object(final)
        body = unfence(final).rsplit("\n", 1)[0].rstrip() if mm else final   # prose = everything above the object line
        _, ents = sources_entries(final)          # map refs (identity); mentions never rescue identity
        cited = [e["ref"] for e in (ents or []) if isinstance(e, dict) and e.get("ref")]
        # the travel node commits to a mode (and, once the harness prints handles, a route)
        mentions = [e["ref"] for e in ((obj or {}).get("mentions") or []) if isinstance(e, dict) and e.get("ref")]
        grammar = map_grammar_errors(obj)
        travel = [e for e in ((obj or {}).get("map") or []) if isinstance(e, dict) and e.get("travel")]
        obj_mode = ROUTE_MODE_CLASS.get(str(travel[0]["travel"]).lower()) if travel else None
        obj_route = travel[0].get("route") if travel else None
    else:
        comp, cerrs = compile_final(final)
        body = comp["body"]
        cited = [s["value"] for s in comp["sources"] if s["type"] == "ref"]
        mentions, obj_mode, obj_route, grammar = [], None, None, []
    dists, durs = stated_numbers(body)
    calls, coords, tool_text = trace_calls(row)
    routes = [c for c in calls if c["name"] == "calculate_route"]
    tol = spec.get("tolerance") or {"distance": 0.15, "duration": 0.20}
    detail = {"stated_dist": dists, "stated_dur": durs, "cited": cited, "contract": "v2" if v2 else "v1",
              "map_grammar": grammar or None,
              "object_mode": obj_mode, "object_route": obj_route,
              "n_route_calls": len(routes), "n_search": sum(1 for c in calls if c["name"] == "search_places")}

    def out(bucket, reason, wrong_class=None, primary=None):
        return {"bucket": bucket, "correct": bucket == "RIGHT", "reason": reason,
                "wrong_class": wrong_class, "primary_ref": primary, "detail": detail}

    # ---- variant 3: the only right answer is a question
    if spec.get("clarify_expected"):
        asks = "?" in body and bool(CLARIFY_RE.search(body))
        if cited:
            return out("WRONG", "ambiguous names resolved to arbitrary refs and cited (clarify expected)", "answered_ambiguous")
        if dists or durs:
            return out("WRONG", "ambiguous question answered with a distance/time", "answered_ambiguous")
        if routes:
            return out("WRONG", "routed an ambiguous pair instead of asking", "answered_ambiguous")
        if asks:
            return out("RIGHT", "asked which city / for the user's location; no numbers, no refs")
        if "?" in body:
            return out("MAYBE", "asks a question but not recognisably about the location/city")
        return out("WRONG", "neither answered nor asked for clarification", "no_clarify")

    # ---- variants 1/2/4: clarifying instead of answering is the counterweight failure
    if not dists and not durs and "?" in body and CLARIFY_RE.search(body) and not routes:
        return out("WRONG", "asked for clarification on an unambiguous task", "unneeded_clarify")

    # endpoints (spec) -> coordinates
    if spec["shape"] == "closer_of_two":
        endpoints = [spec["anchor_place"]] + list(spec["candidates"])
    else:
        endpoints = [spec["from"], spec["to"]]
    need_refs = [e for e in endpoints if e.get("ref")]
    primary = spec["to"].get("ref") if spec["shape"] != "closer_of_two" else spec["anchor_place"].get("ref")

    # identity: every endpoint with a ref must be cited, exactly or by a cited ref within 150 m
    def cited_for(e):
        if e["ref"] in cited:
            return e["ref"]
        for c in cited:
            if c in coords and _hav_m(coords[c], (e["lat"], e["lng"])) <= 250:
                return c
        return None

    # a travel node may commit to a computation handle ([r1] in the tool output);
    # when it does, that call is the one to verify (CONTRACT v2.1 pt 1)
    def call_by_handle(h):
        """A travel node identifies its computation by the short handle the tool
        printed ([r1]) or by the prod route_id the result carries — both name one
        call, so both are legal (2026-09-08: the teacher used route_id 15/18)."""
        if not h:
            return None
        h = str(h)
        for c in routes:
            res = str(c.get("result") or "")
            if res.startswith(f"[{h}]") or f'"route_id": "{h}"' in res:
                return c
        return None

    # the route call whose endpoints match the spec's (either direction), in the right mode
    def match_call(a, b):
        best = None
        for c in routes:
            args = c["args"]
            try:
                f = (float(args["from_location"][1]), float(args["from_location"][0]))
                t = (float(args["to_location"][1]), float(args["to_location"][0]))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            fwd = _hav_m(f, (a["lat"], a["lng"])) <= 250 and _hav_m(t, (b["lat"], b["lng"])) <= 250
            rev = _hav_m(f, (b["lat"], b["lng"])) <= 250 and _hav_m(t, (a["lat"], a["lng"])) <= 250
            if not (fwd or rev):
                continue
            # how_far questions state no mode: the model's choice is legitimate and
            # its numbers are checked against the call it actually made
            mode_ok = (spec["shape"] == "how_far"
                       or ROUTE_MODE_CLASS.get(str(args.get("mode", "")).lower()) == spec["mode"])
            try:
                res = json.loads(str(c["result"]).split("] ", 1)[-1])   # v2 results start with "[r<k>] " (mined100b: 3 false invented_route)
            except (json.JSONDecodeError, TypeError):
                res = {"error": str(c["result"])[:80]}
            cand = {"mode_ok": mode_ok, "result": res, "reverse": rev and not fwd,
                    "mode": ROUTE_MODE_CLASS.get(str(args.get("mode", "")).lower())}
            if mode_ok and "error" not in res:
                return cand          # a successful call in the asked mode wins outright
            if best is None or (mode_ok and not best["mode_ok"]):
                best = cand          # else: prefer right-mode-but-failed over wrong-mode
        return best

    if spec["shape"] == "closer_of_two":
        C = spec["anchor_place"]
        # v2.1: the losing candidate is a mention, not part of the leg — accept it from `mentions`
        cited_all = cited + mentions
        def cited_for_cand(e):
            if e["ref"] in cited_all:
                return e["ref"]
            for c in cited_all:
                if c in coords and _hav_m(coords[c], (e["lat"], e["lng"])) <= 250:
                    return c
            return None
        miss = [e["name"] for e in need_refs if not (cited_for(e) or (e in spec["candidates"] and cited_for_cand(e)))]
        if miss:
            return out("WRONG", f"endpoint(s) not cited/resolved: {miss}", "wrong_entity", primary)
        mc = [match_call(C, cand) for cand in spec["candidates"]]
        if any(m is None for m in mc):
            return out("WRONG", "no calculate_route call to one or both candidates", "no_route_call", primary)
        if any(not m["mode_ok"] for m in mc):
            return out("WRONG", f"route computed in the wrong mode (asked {spec['mode']})", "mode_gaming", primary)
        winner = next(c for c in spec["candidates"] if c["ref"] == spec["closer"])
        loser = next(c for c in spec["candidates"] if c["ref"] != spec["closer"])
        sents = re.split(r"(?<=[.!?])\s+", body)
        key = re.compile(r"\b(?:closer|nearer|shorter|quicker|faster|less far|closest|nearest)\b", re.I)
        win_s = [s for s in sents if key.search(s) and winner["name"].lower() in s.lower()]
        lose_s = [s for s in sents if key.search(s) and loser["name"].lower() in s.lower()
                  and winner["name"].lower() not in s.lower()]
        if win_s and not lose_s:
            return out("RIGHT", f"names {winner['name']} as closer; both legs routed in {spec['mode']}", None, primary)
        if lose_s and not win_s:
            return out("WRONG", f"names {loser['name']} as closer", "wrong_comparison", primary)
        return out("MAYBE", "comparison sentence not decidable mechanically", None, primary)

    a, b = spec["from"], spec["to"]
    miss = [e["name"] for e in need_refs if not cited_for(e)]
    m = None
    if v2 and obj_route:
        c = call_by_handle(obj_route)
        if c is None:
            return out("WRONG", f"travel node cites route handle {obj_route}, which no call produced", "bad_handle", primary)
        try:
            res_h = json.loads(str(c["result"]).split("] ", 1)[-1])
        except (json.JSONDecodeError, TypeError):
            res_h = {"error": "unparsable"}
        args = c["args"]
        try:
            f_ = (float(args["from_location"][1]), float(args["from_location"][0]))
            t_ = (float(args["to_location"][1]), float(args["to_location"][0]))
        except (KeyError, TypeError, ValueError, IndexError):
            f_ = t_ = (0.0, 0.0)
        fwd = _hav_m(f_, (a["lat"], a["lng"])) <= 250 and _hav_m(t_, (b["lat"], b["lng"])) <= 250
        rev = _hav_m(f_, (b["lat"], b["lng"])) <= 250 and _hav_m(t_, (a["lat"], a["lng"])) <= 250
        if not (fwd or rev):
            return out("WRONG", f"the cited route {obj_route} is not between the asked places", "wrong_endpoint", primary)
        m = {"mode_ok": ROUTE_MODE_CLASS.get(str(args.get("mode", "")).lower()) == spec["mode"],
             "result": res_h, "reverse": rev and not fwd,
             "mode": ROUTE_MODE_CLASS.get(str(args.get("mode", "")).lower())}
    if m is None:
        m = match_call(a, b)
    uncited = False
    if miss and cited:
        # the route call itself proves which places were used: if its endpoints
        # match the asked places, the missing ref is a citation-completeness
        # defect (repairable from the trace), not a wrong entity
        if m is not None and m["mode_ok"]:
            uncited = True
        else:
            return out("WRONG", f"endpoint(s) not cited/resolved: {miss}", "wrong_entity", primary)
    if m is None:
        if routes:
            return out("WRONG", "calculate_route called with endpoints > 250 m from the asked places", "wrong_endpoint", primary)
        if dists or durs:
            return out("WRONG", "distance/time stated with no calculate_route call (anchor-relative or invented)", "no_route_call", primary)
        return out("WRONG", "no calculate_route call and no number", "no_route_call", primary)
    if not m["mode_ok"]:
        return out("WRONG", f"route computed in the wrong mode (asked {spec['mode']})", "mode_gaming", primary)
    if v2 and obj_mode and spec["shape"] != "how_far" and obj_mode != spec["mode"]:
        return out("WRONG", f"object commits to {obj_mode}, asked {spec['mode']}", "mode_gaming", primary)
    res = m["result"]

    # unroutable pair: the honest answer is "no route", with no leg numbers
    if not spec.get("routable", True) or "error" in res:
        if dists or durs:
            return out("WRONG", "tool found no route but the answer states a distance/time", "invented_route", primary)
        if UNROUTABLE_RE.search(body):
            return out("RIGHT", "reported that no route could be found; no numbers", None, primary)
        return out("MAYBE", "no route found by the tool; answer neither states numbers nor says so plainly", None, primary)

    tool_d, tool_s = res.get("distance_meters"), res.get("duration_seconds")
    # the same shim on rollouts: a nonsensical Valhalla route (walk/bike > 2.5x crow,
    # auto/bus > 3.5x) is the tool's fault, not the model's — MAYBE, never trained,
    # and listed for debugging (verify.py writes suspect_routes.jsonl)
    crow = _hav_m((a["lat"], a["lng"]), (b["lat"], b["lng"]))
    lim = 2.5 if m.get("mode") in ("pedestrian", "bicycle") else 3.5
    if tool_d and crow > 100 and tool_d > lim * crow:
        detail["suspect_route"] = {"crow_m": round(crow), "distance_m": tool_d,
                                   "ratio": round(tool_d / crow, 2), "mode": m.get("mode"),
                                   "from": [a["lat"], a["lng"]], "to": [b["lat"], b["lng"]]}
        return out("MAYBE", f"suspect Valhalla route: {tool_d/crow:.1f}x crow-flies ({m.get('mode')}); excluded from training", None, primary)
    same_mode = m.get("mode") == spec["mode"]
    # the spec's numbers only count when the model routed in the spec's mode
    golds_d = [g for g in ((spec.get("distance_m") if same_mode else None), tool_d) if g]
    golds_s = [g for g in ((spec.get("duration_s") if same_mode else None), tool_s) if g]
    d_ok = [r for r in dists if any(_within(r, g, tol["distance"], 100) for g in golds_d)]
    s_ok = [r for r in durs if any(_within(r, g, tol["duration"], 120) for g in golds_s)]
    # every other stated number must be supported somewhere in the tool evidence
    other_calls = [c for c in routes if c is not m]
    other_d, other_s = [], []
    for c in other_calls:
        try:
            r = json.loads(str(c["result"]).split("] ", 1)[-1])
            other_d.append(r.get("distance_meters") or 0)
            other_s.append(r.get("duration_seconds") or 0)
        except (json.JSONDecodeError, TypeError):
            pass
    bad = []
    # elevation gain/loss quoted from the route result (in metres, or converted from feet) is a
    # supported number, not an off-by (mined100 review: "236 feet (72 meters)" from elevation_gain_m 72)
    elev = [float(x) for x in (res.get("elevation_gain_m"), res.get("elevation_loss_m")) if isinstance(x, (int, float))]
    for r in dists:
        if r in d_ok or any(_within(r, g, tol["distance"], 100) for g in other_d if g):
            continue
        tok = f"{r[0]:g}"
        if tok in tool_text:
            continue
        if any(abs(r[0] - e) <= max(2.0, 0.05 * e) for e in elev):
            continue
        bad.append(f"{r[0]:.0f}m")
    for r in durs:
        if r in s_ok or any(_within(r, g, tol["duration"], 120) for g in other_s if g):
            continue
        if f"{r[0]/60:g}" in tool_text or f"{r[0]:g}" in tool_text:
            continue
        bad.append(f"{r[0]/60:.0f}min")
    detail["tool_result"] = {"distance_m": tool_d, "duration_s": tool_s}
    if bad:
        return out("WRONG", f"stated numbers outside tolerance / unsupported: {bad[:4]}", "off_by", primary)

    if miss:
        # numbers and route are right but refs are missing: a format defect
        # (DOCTRINE: compile-fail blocks RIGHT, does not establish WRONG)
        detail["uncited_endpoints"] = miss
        return out("MAYBE", f"{'endpoint ref(s) not cited' if uncited else 'no SOURCES/refs cited'} for {miss} (format; route and numbers ok)", None, primary)
    if grammar:
        return out("MAYBE", "trip card malformed: " + "; ".join(grammar[:2]))
    shape = spec["shape"]
    if shape == "walkable":
        yes, no = bool(WALK_YES_RE.search(body)), bool(WALK_NO_RE.search(body))
        want = spec.get("walkable")
        if want is True and yes and not no:
            return out("RIGHT", "says walkable; numbers within tolerance", None, primary)
        if want is False and no and not yes:
            return out("RIGHT", "says not walkable; numbers within tolerance", None, primary)
        if (want is True and no and not yes) or (want is False and yes and not no):
            return out("WRONG", f"walkability verdict contradicts the route ({(tool_s or 0)/60:.0f} min)", "wrong_walkable", primary)
        if s_ok or d_ok:
            return out("MAYBE", "walkability not stated plainly; numbers are right", None, primary)
        return out("WRONG", "no walkability verdict and no number", "no_number", primary)
    if shape == "how_far":
        if d_ok:
            return out("RIGHT", "distance within tolerance", None, primary)
        if s_ok:
            return out("MAYBE", "asked distance, gave only a time (within tolerance)", None, primary)
        return out("WRONG", "no distance stated", "no_number", primary)
    # how_long / from_here
    if s_ok:
        return out("RIGHT", "duration within tolerance", None, primary)
    if d_ok:
        return out("MAYBE", "asked time, gave only a distance (within tolerance)", None, primary)
    return out("WRONG", "no duration stated", "no_number", primary)


# ---------------------------------------------------------------------------
# routing SET oracle (scenarios D/E: nearby_within, along_route). Gold is the
# exhaustive set from the same wplaces endpoint the tool calls
# (wplaces/ROUTING_COMPOSITIONS.md); the model's answer is graded on
# membership, minutes and honesty about an empty set.

SET_TOOL_ARGS = {"nearby_within": ("search_places", "within_minutes"), "along_route": ("search_along_route", "route_id")}
def map_grammar_errors(obj):
    """CONTRACT §2.2 level 1: the line alternates place, travel, place. A travel
    node first or last, or two in a row, is a malformed trip card (the client
    derives from/to from the neighbours, so it cannot render one)."""
    if not isinstance(obj, dict):
        return ["object is not a JSON object"]
    m = obj.get("map")
    if m is None or not isinstance(m, list):
        return [] if m is None else ["map is not a list"]
    errs = []
    kinds = ["travel" if isinstance(e, dict) and "travel" in e else "place" for e in m]
    if kinds and kinds[0] == "travel":
        errs.append("map starts with a travel node (no origin)")
    if kinds and kinds[-1] == "travel":
        errs.append("map ends with a travel node (no destination)")
    for i in range(1, len(kinds)):
        if kinds[i] == kinds[i - 1] == "travel":
            errs.append("two travel nodes in a row")
            break
    for e in m:
        if isinstance(e, dict) and not ({"ref", "travel", "here"} & set(e)):
            errs.append(f"node with no ref/travel/here: {list(e)[:3]}")
            break
    return errs


# 2026-09-09 (mined100 review): the result line starts with the compute handle "[m1] ORIGIN: …" under
# contract v2, and the line-start anchor made every server-resolved origin look like a guessed coordinate.
ORIGIN_LINE_RE = re.compile(r"^(?:\[m\d+\] )?ORIGIN: ([NWR]\d+|coordinates) \| [^|\n]* \| \((-?[\d.]+), (-?[\d.]+)\)", re.M)
NOTHING_RE = re.compile(r"(?:nothing|no (?:places?|caf[eé]s?|options?|results?|\w+))\b.{0,60}\b(?:within|reachable|on the way|along|in (?:that|the) (?:budget|range|time))"
                        r"|(?:couldn'?t|could not|didn'?t|did not|unable to) find (?:any|anything|a)\b|none (?:are |is )?(?:within|reachable|on the way)", re.I)


def set_ok(row, spec, text=None):
    """-> same verdict dict as route_ok. RIGHT = the right tool with the asked
    budget/profile (or the asked leg), every named place a gold member, stated
    minutes within tolerance (eta 15%/2 min; added_time 25%/2 min), empty gold
    answered as empty; the recommendation among the notable members."""
    final = text if text is not None else (row.get("final") or "")
    v2 = is_v2(final)
    if v2:
        mm, obj = map_object(final)
        body = unfence(final).rsplit("\n", 1)[0].rstrip() if mm else final
        _, ents = sources_entries(final)
        cited = [e["ref"] for e in (ents or []) if isinstance(e, dict) and e.get("ref")]
        cited += [e["ref"] for e in ((obj or {}).get("mentions") or []) if isinstance(e, dict) and e.get("ref")]
    else:
        from geogym_blogq.lab.core.citations import compile_final
        comp, _ = compile_final(final)
        body = comp["body"]
        cited = [s["value"] for s in comp["sources"] if s["type"] == "ref"]
    gold = spec["gold"]
    members = {m["ref"]: m for m in gold["members"]}
    tol = spec.get("tolerance") or {"eta": 0.15, "eta_floor_min": 2, "added": 0.25, "added_floor_min": 2}
    calls, coords, tool_text = trace_calls(row)
    tool_name, key = SET_TOOL_ARGS[spec["shape"]]
    detail = {"cited": cited, "gold_size": len(members), "empty": spec["empty"]}

    def out(bucket, reason, wrong_class=None):
        return {"bucket": bucket, "correct": bucket == "RIGHT", "reason": reason, "wrong_class": wrong_class,
                "primary_ref": cited[0] if cited else None, "detail": detail}

    # ---- clarify-expected set tasks (2026-09-09: set_ok had no clarify branch; every clarify-tagged
    # set task was scored WRONG whatever the model did). Same semantics as route_ok's variant 3.
    if spec.get("clarify_expected"):
        dists_c, durs_c = stated_numbers(body)
        set_calls = [c for c in calls if c["name"] in ("search_places", "search_along_route") and key in c["args"]]
        asks = "?" in body and bool(CLARIFY_RE.search(body))
        if cited:
            return out("WRONG", "ambiguous origin resolved to an arbitrary ref and cited (clarify expected)", "answered_ambiguous")
        if dists_c or durs_c or set_calls:
            return out("WRONG", "ambiguous question answered with a computed set / numbers", "answered_ambiguous")
        if asks:
            return out("RIGHT", "asked which city / for the user's location; no numbers, no refs")
        if "?" in body:
            return out("MAYBE", "asks a question but not recognisably about the location/city")
        return out("WRONG", "neither answered nor asked for clarification", "no_clarify")

    # the set call, with the asked parameters; the closest near-miss is classified
    call = None
    miss = None   # (wrong_class, reason) of the best near-miss call
    PROF = {"walking": "pedestrian", "cycling": "bicycle", "driving": "auto"}
    for c in calls:
        if c["name"] != tool_name or key not in c["args"]:
            continue
        if spec["shape"] == "nearby_within":
            prof = PROF.get(str(c["args"].get("profile") or "walking").lower())
            budget_ok = abs(float(c["args"].get("within_minutes") or 0) - spec["within_minutes"]) <= 0.01
            # the anchor must be the asked origin (or the user's position), not a guessed
            # coordinate. The tool's ORIGIN line (server-side resolution by ref/name) is the
            # authority when present; raw lat/lng otherwise.
            om = ORIGIN_LINE_RE.search(c.get("result") or "")
            try:
                if om and om.group(1) != "coordinates":
                    o_ref, o_lat, o_lng = om.group(1), float(om.group(2)), float(om.group(3))
                    d_anchor = 0.0 if (spec["from"].get("ref") and o_ref == spec["from"]["ref"]) else \
                        _hav_m((o_lat, o_lng), (spec["from"]["lat"], spec["from"]["lng"]))
                else:
                    d_anchor = _hav_m((float(c["args"].get("lat")), float(c["args"].get("lng"))), (spec["from"]["lat"], spec["from"]["lng"]))
            except (TypeError, ValueError):
                d_anchor = 1e9
            anchor_ok = d_anchor <= 250
            if budget_ok and prof == spec["mode"] and anchor_ok:
                call = c; break
            miss = miss or ((("unresolved_origin", f"reachability anchored {d_anchor:.0f} m from the asked origin (guessed coordinates instead of resolving it)") if not anchor_ok
                            else ("mode_gaming", f"reachability asked {spec['mode']}, called {prof}") if prof != spec["mode"]
                            else ("wrong_budget", f"within_minutes {c['args'].get('within_minutes')} vs asked {spec['within_minutes']}")))
        else:
            from geogym_blogq.lab.tools.route import parse_route_id
            p = parse_route_id(c["args"].get("route_id"))
            if not p:
                miss = miss or ("no_set_call", "search_along_route with an unknown route_id")
                continue
            frm, to, mode = p
            # origin: the asked place, or the user's own position when one was given
            d_from = _hav_m((frm[1], frm[0]), (spec["from"]["lat"], spec["from"]["lng"]))
            if spec.get("user_location"):
                d_from = min(d_from, _hav_m((frm[1], frm[0]), tuple(spec["user_location"])))
            d_to = _hav_m((to[1], to[0]), (spec["to"]["lat"], spec["to"]["lng"]))
            ok_ends = d_from <= 250 and d_to <= 250   # 2026-09-09: same tolerance as citation (duplicate OSM entries of one place)
            ok_mode = ROUTE_MODE_CLASS.get(mode) == spec["mode"]
            ok_dev = float(c["args"].get("max_detour_min") or 5) <= spec["max_detour_min"] + 0.01
            if ok_ends and ok_mode and ok_dev:
                call = c; break
            miss = miss or ((("wrong_endpoint", f"route endpoints {d_from:.0f} m / {d_to:.0f} m from the asked places") if not ok_ends
                            else ("mode_gaming", f"route in {mode}, asked {spec['mode']}") if not ok_mode
                            else ("wrong_budget", f"detour {c['args'].get('max_detour_min')} > asked {spec['max_detour_min']}")))
    named = [r for r in cited if r != (spec["from"].get("ref"))]
    named = [r for r in named if r != ((spec.get("to") or {}).get("ref"))]
    if call is None:
        if miss:
            return out("WRONG", miss[1], miss[0])
        if named:
            return out("WRONG", "named places with no reachability/along-route call (from a plain search or memory)", "no_set_call")
        if spec["empty"] and NOTHING_RE.search(body):
            return out("MAYBE", "said nothing is in range but never called the tool")
        return out("WRONG", f"no {tool_name} call", "no_set_call")
    # membership
    non = [r for r in named if r not in members]
    if non:
        # a ref the tool itself returned in this call is a member of the tool's set; anything else is not
        if all(r in (call.get("result") or "") for r in non):
            detail["tool_set_extra"] = non   # gold was capped at 100; treat as member
        else:
            return out("WRONG", f"named place(s) outside the reachable/along-route set: {non[:3]}", "non_member")
    if spec["empty"]:
        # naming the tool's own "nearest outside the budget" place is the honest answer, not a member claim
        nearest = re.findall(r"nearest: ([NWR]\d+)", call.get("result") or "")
        others = [r for r in named if r not in nearest]
        if others:
            return out("WRONG", "gold set is empty but the answer names places as within budget", "non_member")
        if NOTHING_RE.search(body):
            return out("RIGHT", "empty set reported honestly")
        return out("MAYBE", "empty set; answer does not say so plainly")
    if not named:
        if NOTHING_RE.search(body):
            return out("WRONG", "gold set non-empty but the answer says nothing is in range", "missed_set")
        return out("MAYBE", "no place cited by ref; membership not checkable")
    # minutes: every stated duration must match a named member's eta / added time (or the leg for along_route)
    _, durs = stated_numbers(body)
    golds = []
    for r in named:
        m = members.get(r)
        if not m:
            continue
        if m.get("eta_min") is not None:
            golds.append((m["eta_min"] * 60, tol["eta"], tol["eta_floor_min"] * 60))
        if m.get("added_time_min") is not None:
            golds.append((m["added_time_min"] * 60, tol["added"], tol["added_floor_min"] * 60))
    if spec["shape"] == "along_route" and spec.get("leg"):
        golds.append((spec["leg"]["duration_s"], 0.2, 120))
    if spec["shape"] == "nearby_within":
        golds.append((spec["within_minutes"] * 60, 0.0, 1))       # restating the budget is fine
    else:
        golds.append((spec["max_detour_min"] * 60, 0.0, 1))
    bad = [d for d in durs if not any(_within(d, g, t, f) for g, t, f in golds)]
    # a number that appears verbatim in tool output is supported even if it is not a member's
    bad = [d for d in bad if f"{d[0]/60:g}" not in tool_text]
    if bad:
        return out("WRONG", f"stated minutes not matching any named member: {[round(b[0]/60) for b in bad[:3]]}", "off_by")
    # a good recommendation is either notable (top-5 by fame) or close (top-3 by minutes)
    by_eta = sorted(members.values(), key=lambda m: (m.get("eta_min") if m.get("eta_min") is not None else 1e9))
    close = {m["ref"] for m in by_eta[:3]}
    detail["top_refs"] = gold.get("top_refs")
    # the tool returned its fame-ranked top-N: anything it returned is a legitimate pick;
    # a member the model dug up some other way must at least be notable or close
    if named[0] not in (call.get("result") or "") and gold.get("top_refs") \
            and named[0] not in gold["top_refs"] and named[0] not in close:
        return out("MAYBE", "members and minutes right; first recommendation was not returned by the call and is neither notable nor close")
    return out("RIGHT", f"{len(named)} member(s) named with matching minutes")
