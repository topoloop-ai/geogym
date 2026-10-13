#!/usr/bin/env python3
"""Run the 200-question mapping-agent eval (data/agent_questions/) against an
OpenAI-compatible LLM and score it with an LLM judge.

Unlike bench_agent.py (fixed anchor + pre-baked turn-1 candidates), this is the
freeform contract: the model supplies its OWN anchor coordinates to
search_places, can fetch place_details, and answers in plain text. Scoring is
an LLM-judge verdict against each question's expected_places + answer_notes.

Resumable: one JSONL row per question id; rerun skips completed ids.

Usage:
  python bench_questions.py --base-url https://openrouter.ai/api/v1 \
      --api-key $KEY --model google/gemini-3.5-flash-lite \
      --out results/agentq_flashlite.jsonl
  python bench_questions.py --summarize results/agentq_flashlite.jsonl
"""

import argparse
import contextlib
import copy
import asyncio
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
try:
    from geogym_blogq.lab.core.config import EP as _EP, legacy as _legacy
except Exception:  # standalone
    _EP, _legacy = {}, None
_QDEFAULT = str((_legacy("agentq_questions") if _legacy else None) or HERE / "data/agent_questions/questions_all.jsonl")
_GDEFAULT = str((_legacy("agentq_gold") if _legacy else None) or HERE / "data/agent_questions/questions_gold.jsonl")
QFILE = Path(os.environ.get("AGENTQ_FILE", _QDEFAULT))
# gold refs (backfilled 2026-08-24) power the declined-findable override;
# gold_coords (backfilled 2026-08-28) power the mechanical funnel
GOLD_FILE = Path(os.environ.get("BENCH_QUESTIONS", _GDEFAULT))
GOLD_REFS, GOLD_COORDS = {}, {}
if GOLD_FILE.exists():
    from geogym_blogq.lab.core.bank import load_bank as _load_bank
    for _q in _load_bank(GOLD_FILE):
        GOLD_REFS[_q["id"]] = [r for r in _q.get("gold_refs") or [] if r]
        GOLD_COORDS[_q["id"]] = _q.get("gold_coords") or {}
DECLINE_RE = re.compile(
    r"couldn|could not|not find|unable|not in the|no match|doesn't appear", re.I)

WPLACES = os.environ.get("WPLACES_API_ENDPOINT", (_EP.get("wplaces") or {}).get("base_url", "http://127.0.0.1:8001"))
WPLACES_KEY = os.environ.get("WPLACES_API_KEY", (_EP.get("wplaces") or {}).get("api_key", "wplaces-local-dev-key"))

# 2026-09-14: on 222 probe questions where the model never surfaced the gold name, replaying its own web query found it 0% of the
# time, pasting the RAW QUESTION found it 28.8%, and querying the gold name itself (oracle, = what the corpus holds) found 68.9%.
# The model over-compresses its web queries — plausibly bleed from the "keep query SHORT" rule written for search_places.
# BENCH_VERBATIM_WEB_QUERY=1 adds one prompt line telling it to keep the user's wording on the web.
# **REJECTED as a training aid (Jon, 2026-09-14): "the model should learn eventually to make up its own queries"** — a prompt
# crutch does not transfer to the on-device product and would mask the behaviour we want trained. Left off by default and kept only
# as a diagnostic for re-measuring the 28.8% recoverable bucket. Do not enable it for a training round or a ladder row.
VERBATIM_WEB_QUERY = os.environ.get("BENCH_VERBATIM_WEB_QUERY", "") not in ("", "0")

SYSTEM = """You are a mapping assistant backed by a places database (~49M OSM places).

Tools:
- search_places(query?, lat, lng): semantic + lexical search near an anchor you supply from your own knowledge of geography. Use 3-4+ decimal places of precision; one decimal (~11 km) is too coarse and will search the wrong part of a city. Keep query SHORT — a place name or category, no "near X" context. OMIT query entirely to just see what is around the anchor: the closest places, ranked so famous landmarks stay visible, each with distance and compass bearing. The radius is automatic (tight in cities, wide in wilderness). Misspellings, translations and native scripts all work. Each result shows its neighbourhood or street and town: when a name has several branches, pick the one whose area matches what the user or the page said.
- place_details(refs): address, website, phone, opening hours, fee, wheelchair access, wikipedia link and travel-blog mention count for up to 5 places you already found.

Rules:
- ALWAYS ground your answer in tool results — never answer a place question from memory alone.
- If the user states their location, use it as the search anchor.
- If searches come up empty or the premise is wrong (place is fictional, demolished, or in a different city than claimed), SAY SO plainly — never present a similarly-named different place as the answer without flagging the difference.
- You have ONE call each of search_places and web_search — make each \
count with your best-formed query. Then fetch a promising page, check \
details of found refs, or conclude.
- If your attempts fail, say plainly what you could and could not verify. \
An honest "I couldn't confirm this" is always better than an invented \
specific.
- Every place you recommend or identify must be RESOLVED in the places database: when a web page names the place, run search_places on that name (a short query, anchored on the city) so you can cite its ref. Search the web first when the question is about vibe, experience or "which place", then resolve the named place. A place answer that cites only a web page is incomplete.
- Finish with a concise plain-text answer naming the specific place(s) you found."""

if os.environ.get("BENCH_STRICT_PROMPT"):
    SYSTEM += """
- STRICT GROUNDING: state ONLY specifics (addresses, prices, times, \
distances, amenities, history) that appear in your tool results. Do NOT \
add background facts, figures or color from memory — an answer with one \
invented specific is a failed answer. If you know something but could \
not verify it in the tools, either omit it or say it is unverified."""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_places",
        "description": "Search the places database near an anchor coordinate "
                       "you supply. Radius is automatic.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "Short place-name or category query. "
                                     "Omit to list what is nearby instead "
                                     "(closest places, fame-ranked, with "
                                     "distance and bearing)."},
            "lat": {"type": "number", "description": "Anchor latitude."},
            "lng": {"type": "number", "description": "Anchor longitude."}},
            "required": ["lat", "lng"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "place_details",
        "description": "Fetch contact/hours/fee/accessibility/wikipedia/blog "
                       "details for up to 5 refs from earlier search results.",
        "parameters": {"type": "object", "properties": {
            "refs": {"type": "array", "items": {"type": "string"},
                     "description": "REFs copied from your search results (W…/N…/R… ids), max 5."}},
            "required": ["refs"], "additionalProperties": False}}},
]

if os.environ.get("BENCH_NO_NEAREST"):
    TOOLS = [t for t in TOOLS
             if t["function"]["name"] != "nearest_places"]

if os.environ.get("BENCH_BLOG_TOOLS"):
    TOOLS.append({"type": "function", "function": {
        "name": "web_search",
        "description": "Curated web search over high-quality travel "
                       "writing worldwide. Returns titles, URLs and "
                       "snippets of articles with real traveler "
                       "experiences, tips, prices and opinions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"}},
            "required": ["query"], "additionalProperties": False}}})
    TOOLS.append({"type": "function", "function": {
        "name": "web_fetch",
        "description": "Fetch the readable text of a web page returned by "
                       "web_search.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}},
            "required": ["url"], "additionalProperties": False}}})
    SYSTEM = SYSTEM.replace(
        "\n\nRules:",
        "\n- web_search(query): curated web search over travel writing — "
        "real traveler experiences, tips, prices and opinions the places "
        "database cannot hold. Use it when the question hinges on what "
        "travelers experienced (fees paid, timing, vibe, comparisons)."
        + (" Search the USER'S OWN WORDS: copy the question nearly verbatim, keeping every "
           "distinguishing detail (the street, the dish, the view, the adjective). Do NOT "
           "compress it to keywords — the short-query rule is for search_places only, and on "
           "the web it throws away the words that match the page."
           if VERBATIM_WEB_QUERY else "")
        + "\n"
        "- web_fetch(url): read the full text of a page returned by "
        "web_search.\n\nRules:")
    SYSTEM += ("\n- CITATIONS: mark each sourced claim inline with [1], "
               "[2], ... and end the answer with exactly one line: "
               "SOURCES: [{\"ref\": \"<ref from your results>\"}, "
               "{\"url\": \"<url from your results>\"}] in marker order — a ref "
               "entry for facts from the places database, a url entry "
               "(copied verbatim from your tool results, never invented) "
               "for facts from a web page. You may still name the "
               "article/site naturally in the text. SOURCES: [] if nothing "
               "is cited.")

# CONTRACT v2.2 (CONTRACT.md §1-2): handles on web results + fetches, credit as
# markdown links on handles, one trailing {"map": [...]} object. Switched by
# BENCH_CONTRACT=v2 so v1 rollouts/evals and the routing session are untouched.
CONTRACT = os.environ.get("BENCH_CONTRACT", "v1")
# 2026-09-08 (Jon): skip the per-rollout inline judge. The harvest pipelines (style/) re-judge every
# row with judges/judge_v2.py afterwards, so the inline verdict is thrown away — and at 1 call per
# rollout it doubles the load on a judge box the RL loop is already saturating. Rows still carry
# every field except a real `correct`/`judge_reason`.
NO_INLINE_JUDGE = os.environ.get("BENCH_NO_JUDGE", "") not in ("", "0")
V2_RULE = (
    "\n- CREDIT: every web result and fetched page carries a handle like [w2]. "
    "Credit a page at the point of use with a markdown link whose target is its "
    "handle and whose text is the site or title as shown, e.g. "
    "\"According to [Full Suitcase](w2), the village is known for ...\". "
    "Write in your own words; never copy a URL, a coordinate or a run of more "
    "than ~25 words from a page. Name places naturally in the prose.\n"
    "- END with exactly one line: a JSON object {\"map\": [{\"ref\": \"<ref>\"}, ...]} "
    "listing the place the answer is about FIRST (its ref from your results), "
    "then other places the answer names. When the answer is a JOURNEY, write it as "
    "the line it is: place, travel, place — {\"map\": [{\"ref\": \"<from>\"}, "
    "{\"travel\": \"walk\", \"route\": \"r1\"}, {\"ref\": \"<to>\"}]}, where the travel "
    "node names the mode and the handle of the calculate_route result your numbers "
    "came from ([r1], [r2] ... in the tool output). A travel node always sits "
    "BETWEEN two places — never first, never last. When the journey starts where "
    "the user is, that first node is {\"here\": true}, e.g. {\"map\": [{\"here\": true}, "
    "{\"travel\": \"walk\", \"route\": \"r1\"}, {\"ref\": \"<to>\"}]}. Put places you merely "
    "mention in \"mentions\": [{\"ref\": ...}] instead of the line. {\"map\": []} if no place is cited. "
    "That object is the last line of your answer, on its own, with no code fence "
    "and nothing after it. Never write bare handles like [w1] in the prose; the "
    "handle only ever appears as a link target."
    + ("" if os.environ.get("BENCH_V2_EXAMPLE", "1") == "0" else
       " Example ending:\n"
       "  ...open daily until 5 pm according to [Travel Lemming](w3).\n"
       "  {\"map\": [{\"ref\": \"<ref of that place from your search results>\"}]}"))
if CONTRACT == "v2" and os.environ.get("BENCH_BLOG_TOOLS"):
    SYSTEM = re.sub(r"\n- CITATIONS:.*$", V2_RULE, SYSTEM, flags=re.S)
    assert "CREDIT:" in SYSTEM
    for _t in TOOLS:
        if _t["function"]["name"] == "web_search":
            _t["function"]["description"] = (
                "Curated web search over high-quality travel writing worldwide. "
                "Each result carries a handle like [w2], its title, a one-line "
                "summary and the matching passage from the page.")
        if _t["function"]["name"] == "web_fetch":
            _t["function"]["description"] = (
                "Read the full text of a page from web_search when its passage "
                "is not enough. Pass the page's URL exactly as given by the "
                "handle table in the result.")
CHUNK_CHARS = int(os.environ.get("BENCH_CHUNK_CHARS", "1800"))


class ComputeHandles:
    """Per-conversation handles for COMPUTED results, the counterpart of the web
    handle table (CONTRACT v2.1 pt 1): `r<k>` for a route, `m<k>` for a set
    (reachable / along-route). A travel node commits to the computation its
    numbers came from, so the oracle checks the call the model actually used
    instead of guessing which of several calls it meant."""

    def __init__(self):
        self.n = {"r": 0, "m": 0}
        self.calls = {}      # handle -> {"tool", "args"}

    def add(self, kind, tool, targs):
        self.n[kind] += 1
        h = f"{kind}{self.n[kind]}"
        self.calls[h] = {"tool": tool, "args": targs}
        return h

    def table(self):
        return dict(self.calls)


class Handles:
    """Handles assigned once per conversation, in order of first appearance,
    keyed by URL (CONTRACT §1). Shared by search rendering, fetch labelling,
    the row's `handles` table and the v2 gates."""

    def __init__(self):
        self.by_url, self.n = {}, 0

    @staticmethod
    def site_of(url, title=""):
        dom = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        if dom.endswith("wikipedia.org"):
            return "Wikipedia"
        for sep in (" — ", " – ", " - ", " | ", " » ", " // "):
            if sep in title:
                site = title.rsplit(sep, 1)[1].strip().rstrip(",.;:!")
                for sep2 in (" — ", " – ", " - ", " | ", " » ", " // "):
                    if sep2 in site:
                        site = site.split(sep2, 1)[0].strip()
                # a date, a year, a number or a generic word is not a site name
                if 2 < len(site) < 60 and not re.search(r"\b(?:19|20)\d{2}\b", site) \
                        and not re.fullmatch(r"[\d\s.,/-]+", site) \
                        and site.lower() not in ("blog", "travel blog", "home", "travel", "guide", "review", "reviews"):
                    return site
        return dom

    def get(self, url, title=""):
        url = url.strip().rstrip(".,")
        if url not in self.by_url:
            self.n += 1
            self.by_url[url] = {"handle": f"w{self.n}", "url": url, "title": title or "",
                                "site": self.site_of(url, title)}
        elif title and not self.by_url[url]["title"]:
            self.by_url[url]["title"] = title
            self.by_url[url]["site"] = self.site_of(url, title)
        return self.by_url[url]

    def table(self):
        return {p["handle"]: {"url": u, "title": p["title"], "site": p["site"]} for u, p in self.by_url.items()}

if os.environ.get("BENCH_ROUTE_TOOL"):
    # Mirrors tokyo's prod calculate_route exactly (name, signature, result
    # shape) so the trained skill transfers verbatim; same metadata/geometry
    # split as Gemini's Maps grounding (duration+distance to the model,
    # polyline stays server-side behind route_id).
    TOOLS.append({"type": "function", "function": {
        "name": "calculate_route",
        "description": "Calculate detailed routing with turn-by-turn "
                       "directions for ground transportation. Returns "
                       "accurate duration, distance, and route_id for "
                       "rendering the actual path. Only works for: walk, "
                       "bike, car, bus, motorcycle, taxi. For other modes "
                       "(train, ferry, plane), estimate travel time "
                       "yourself without calling this tool.",
        "parameters": {"type": "object", "properties": {
            "from_location": {"type": "array", "items": {"type": "number"},
                              "description": "[longitude, latitude] coordinate pair for starting location"},
            "to_location": {"type": "array", "items": {"type": "number"},
                            "description": "[longitude, latitude] coordinate pair for destination"},
            "mode": {"type": "string",
                     "enum": ["walk", "bike", "car", "bus", "motorcycle", "taxi"],
                     "description": "Travel mode for routing."}},
            "required": ["from_location", "to_location", "mode"],
            "additionalProperties": False}}})
    SYSTEM = SYSTEM.replace(
        "\n\nRules:",
        "\n- calculate_route(from_location [lng,lat], to_location [lng,lat], "
        "mode): real travel duration and distance over the road/path "
        "network (walk, bike, car, bus, motorcycle, taxi). ALWAYS use this "
        "— never estimate from coordinates — when answering how far apart "
        "places are, how long travel takes, or whether something is "
        "walkable. For train/ferry/plane, estimate yourself and say so."
        "\n\nRules:")

if os.environ.get("BENCH_CLARIFY_RULE"):
    # routing variant-3 A/B (2026-09-05): a short, useful ask instead of a
    # re-search loop or a memory-padded question
    SYSTEM += ("\n- If you cannot tell WHICH place is meant (a chain or a common name "
               "with no city or location given), do not guess and do not keep "
               "searching: ask in one or two sentences for the city, or offer to "
               "use the user's location. Name no candidate places from memory.")

if os.environ.get("BENCH_ASK_USER"):
    # 2026-09-10 (Jon): clarification as a TURN, not an exit. The model may ask the user one
    # question; a simulated traveller (who knows only their own situation, from the source
    # article) answers; the model must still finish with a grounded answer. Asking is meant to
    # carry a small reward cost in RL; here it is just recorded (row["asks"]).
    TOOLS.append({"type": "function", "function": {
        "name": "ask_user",
        "description": "Ask the traveller ONE short clarifying question when you genuinely "
                       "cannot tell which place or area they mean (a common name with no "
                       "city, no locatable anchor, or several equally plausible readings). "
                       "They will answer about their own situation only. You may ask once.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string", "description": "One or two sentences. Name no candidate places from memory."}},
            "required": ["question"], "additionalProperties": False}}})
    if os.environ.get("BENCH_ASK_USER") == "2":
        # teacher/seeding variant: ask FIRST whenever the anchor is missing or shared
        SYSTEM += ("\n- ask_user(question): BEFORE any search, check whether the question names a place "
                   "you can locate (a town, region, or a place name that is not shared by many places). If "
                   "it does not — no city given, a chain or common name, 'the island', 'this hotel', a "
                   "description only — your FIRST call must be ask_user: one short question for the "
                   "town/region or which one they mean. Name no candidates from memory. Then continue "
                   "with their answer. Ask at most once, and never ask for the answer itself.")
    else:
        SYSTEM += ("\n- ask_user(question): if you cannot tell WHICH place or area is meant, ask the "
                   "traveller once, briefly, then continue with their answer. Asking costs you; do not "
                   "ask when the question already names a locatable place, and never ask for the answer itself.")

_ask_llm, _ask_model = None, None
SCENARIOS = {}   # id -> scenario line (data/banks/scenarios.jsonl via BENCH_SCENARIO_FILE)
if os.environ.get("BENCH_SCENARIO_FILE") and os.path.exists(os.environ["BENCH_SCENARIO_FILE"]):
    for _l in open(os.environ["BENCH_SCENARIO_FILE"]):
        if _l.strip():
            _d = json.loads(_l); SCENARIOS[_d["id"]] = _d.get("scenario") or ""
ASK_SIM_SYSTEM = """You are role-playing the TRAVELLER who asked a mapping assistant a question. You know only your own situation (given below as an excerpt or a situation line): roughly where you are or are going, which country/region/town, what kind of place you meant, your dates and constraints.
The assistant has asked you a clarifying question. Answer it in one or two plain sentences, in first person, from the excerpt and the original question only.
Hard rules:
- Never name, spell, hint at or describe the specific place that answers your original question. If asked for it, say you don't know it — that is why you asked.
- Give only what a real traveller would know: town/region/country, rough position, the kind of place, your constraints.
- If the excerpt does not settle what the assistant asks, say you are not sure.
- No preamble, no quotes from the article, no lists."""


_ASK_RE = re.compile(r"(which (city|town|area|region|island|country|one|location)|where are you|could you (tell|let|clarify|specify)|"
                     r"can you (tell|let|clarify|specify)|please (tell|let|specify|clarify)|more (information|details)|"
                     r"what city|which .{0,40}\?)", re.I)


def looks_like_ask(text):
    """A plain-text final that is a clarifying question (no map object, short, asks the user something)."""
    t = (text or "").strip()
    return bool(t) and '"map"' not in t and len(t) < 700 and "?" in t and bool(_ASK_RE.search(t))


async def ask_user_sim(sim_llm, sim_model, q, question, sem):
    """Simulated traveller reply for ask_user. Blocks the gold name/refs mechanically."""
    meta = q.get("meta") or {}
    scen = SCENARIOS.get(q["id"]) or q.get("scenario")
    if scen:
        # 2026-09-10 (Jon): the simulated traveller must not know the answer. With a scenario line
        # (written offline, gold-name-checked) the simulator never sees the article at all.
        ctx = (f"ORIGINAL QUESTION (yours): {q['question']}\n"
               f"YOUR SITUATION: {scen}\n\n"
               f"ASSISTANT'S QUESTION TO YOU: {question}")
    elif os.environ.get("BENCH_ASK_REQUIRE_SCENARIO"):
        return "I'm not sure — I don't have more detail than what I asked. Please go with what you have.", False
    else:
        chunk = re.sub(r"\s+", " ", str(meta.get("chunk") or meta.get("snippet") or ""))[:2500]
        ctx = (f"ORIGINAL QUESTION (yours): {q['question']}\n"
               f"ARTICLE: {meta.get('title') or ''}\nEXCERPT: {chunk}\n\n"
               f"ASSISTANT'S QUESTION TO YOU: {question}")
    reply = ""
    for attempt in range(2):
        try:
            async with sem:
                r = await sim_llm.chat.completions.create(
                    model=sim_model, temperature=0.3, max_tokens=120,
                    messages=[{"role": "system", "content": ASK_SIM_SYSTEM}, {"role": "user", "content": ctx}],
                    extra_body=({"reasoning": {"enabled": False}} if "openrouter" in str(sim_llm.base_url) else {"chat_template_kwargs": {"enable_thinking": False}}))
            reply = (r.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            reply = ""
        if not reply:
            continue
        low = reply.lower()
        leak = [n for n in (q.get("expected_places") or []) if n and n.lower() in low]
        leak += [ref for ref in (q.get("gold_refs") or []) if ref and ref in reply]
        if not leak:
            return reply, False
    return "I'm not sure — I don't know the exact place, that's why I'm asking. Please go with what you have.", True

if os.environ.get("BENCH_ROUTE_TOOL") == "2":
    # v2 toolset (wplaces routing compositions, 2026-09-06): reachability on
    # search_places, and search along a computed route. Gold for these shapes
    # comes from the same endpoints (never from /places/search).
    _sp2 = TOOLS[0]["function"]["parameters"]["properties"]
    _sp2["within_minutes"] = {"type": "number", "description":
        "Only places reachable within this many minutes of travel from (lat, lng) "
        "by `profile`. Results carry real travel minutes; an empty result names the nearest place outside the budget."}
    _sp2["profile"] = {"type": "string", "enum": ["walking", "cycling", "driving"],
                       "description": "Travel profile for within_minutes (default walking)."}
    _sp2["origin_ref"] = {"type": "string", "description":
        "With within_minutes: start from this place (a ref from an earlier result) instead of lat/lng."}
    _sp2["origin_name"] = {"type": "string", "description":
        "With within_minutes: start from the place with this name, resolved near lat/lng (the city or area). "
        "Use this when the user names their starting place — never guess its coordinates."}
    # attribute constraints (2026-09-10): filtered server-side over the WHOLE candidate set, tri-state
    _sp2["open_at"] = {"type": "string", "description":
        "Keep places open at this LOCAL date-time (ISO, e.g. 2026-09-15T05:30). Each result then shows open/closed/"
        "hours unknown with that day's hours; closed places are removed, unknown ones listed after the open ones."}
    _sp2["wheelchair"] = {"type": "boolean", "description": "Keep wheelchair-accessible places (tagged yes/limited); 'no' removed; untagged shown as unknown."}
    _sp2["free"] = {"type": "boolean", "description": "Keep places with no entry fee; fee-charging removed; untagged shown as unknown."}
    TOOLS.append({"type": "function", "function": {
        "name": "search_along_route",
        "description": "Places on the way along a route you already computed with "
                       "calculate_route, whose detour adds at most max_detour_min minutes. "
                       "Returned in travel order with minutes from the origin and the detour cost.",
        "parameters": {"type": "object", "properties": {
            "route_id": {"type": "string", "description": "route_id from a calculate_route result."},
            "query": {"type": "string", "description": "What kind of place (\"coffee\", \"a viewpoint\", \"lunch\")."},
            "max_detour_min": {"type": "number", "description": "Max added minutes (default 5)."},
            "open_at": {"type": "string", "description": "Keep stops open at this LOCAL date-time (ISO); results show open/closed/unknown."},
            "wheelchair": {"type": "boolean", "description": "Keep wheelchair-accessible stops."},
            "free": {"type": "boolean", "description": "Keep stops with no entry fee."}},
            "required": ["route_id", "query"], "additionalProperties": False}}})
    SYSTEM = SYSTEM.replace(
        "\n\nRules:",
        "\n- search_places(..., within_minutes, profile): the same search limited to places "
        "reachable within a travel-time budget from the anchor; each result carries real "
        "travel minutes. Use it for \"X within a 10-minute walk\", \"what can I reach from "
        "here\". Pass the user's budget as within_minutes and their travel mode as profile "
        "(walking / cycling / driving). Anchor on the user's stated location when given; when the "
        "question names the starting place instead, pass it as origin_name (with lat/lng = the city or "
        "area) or origin_ref — the result's ORIGIN line shows what it resolved to. Never anchor on "
        "coordinates from memory. If the ORIGIN line says AMBIGUOUS, ask which one. If it returns "
        "nothing, say so and name the nearest option it reports.\n"
        "- CONSTRAINTS: when the user needs a place open at a time, wheelchair accessible or free, pass "
        "open_at / wheelchair / free on search_places (with or without within_minutes). The filter runs over "
        "every candidate, not just the ones shown, and each result says open / closed (opens HH:MM) / hours "
        "unknown. Report known-open places as open, unknown ones only with that caveat, and when none "
        "satisfies say so and give the nearest alternative or the earliest opening the results show.\n"
        "- FIRST SEARCH: be sure of the CITY before you anchor. If the question names a city or "
        "region, anchor there. If you are not certain which city a named place is in, web_search "
        "its name first (results show the town) and anchor on that — a first search anchored on a "
        "guessed city wastes the whole budget (mined100: Seattle for Spokane, Aruba for Curaçao).\n"
        "- search_along_route(route_id, query, max_detour_min): places on the way along a "
        "route from calculate_route, with the minutes each adds. Use it for \"somewhere for "
        "coffee on the way\". Recommend from what it returns; if it returns nothing, say "
        "nothing is on the way within that detour. Every stop or place you recommend from a "
        "reachability or along-route result must appear in the final object by ref — in the "
        "journey line if it is a stop, otherwise in \"mentions\" — never only in the prose.\n\nRules:")

if os.environ.get("BENCH_TWO_TOOL"):
    _sp = TOOLS[0]["function"]["parameters"]
    _sp["required"] = ["query", "lat", "lng"]
    _sp["properties"]["query"]["description"] = \
        "Short place-name or category query."
    TOOLS.insert(1, {"type": "function", "function": {
        "name": "nearest_places",
        "description": "List the closest places around a coordinate, "
                       "fame-blended so notable landmarks surface among "
                       "nearby minor spots. Results carry distance and "
                       "compass bearing from the query point.",
        "parameters": {"type": "object", "properties": {
            "lat": {"type": "number"}, "lng": {"type": "number"}},
            "required": ["lat", "lng"], "additionalProperties": False}}})
    SYSTEM = SYSTEM.replace(
        " OMIT query entirely to just see what is around the anchor: the "
        "closest places, ranked so famous landmarks stay visible, each with "
        "distance and compass bearing.", "")
    SYSTEM = SYSTEM.replace(
        "- place_details(refs):",
        "- nearest_places(lat, lng): the closest places around a point, "
        "fame-ranked with distance and compass bearing. Use for what's-near "
        "and direction questions.\n- place_details(refs):")

# ---- flow variant (BENCH_FLOW=web_first): web tools listed FIRST + an explicit
# research order. Default keeps the historical places-first listing.
if os.environ.get("BENCH_FLOW") == "web_first" and "- web_search(" in SYSTEM:
    _lines = SYSTEM.split("\n")
    _web = [l for l in _lines if l.startswith("- web_search(") or l.startswith("- web_fetch(")]
    _rest = [l for l in _lines if l not in _web]
    _ti = next(i for i, l in enumerate(_rest) if l.strip() == "Tools:")
    _lines = _rest[:_ti + 1] + _web + _rest[_ti + 1:]
    SYSTEM = "\n".join(_lines)
    SYSTEM = SYSTEM.replace("Rules:\n", "Rules:\n- RESEARCH ORDER: for questions about vibe, experience, "
        "recommendations, comparisons or \"which place is known for…\", run web_search FIRST — travel "
        "writing is where such places are named. When a page names the place, run search_places on that "
        "exact name (short query, anchored on the city) to resolve it and get its ref; then place_details "
        "if a field (hours, fee, phone) is asked. Use search_places first only when the user already names "
        "the place or asks what is near a point.\n", 1)
    SYSTEM_FLOW = "web_first"
    # the API tool list in the same order as the prompt
    TOOLS[:] = [t for t in TOOLS if t["function"]["name"] in ("web_search", "web_fetch")] + \
               [t for t in TOOLS if t["function"]["name"] not in ("web_search", "web_fetch")]
else:
    SYSTEM_FLOW = "places_first"

JUDGE_SYSTEM = """You judge a mapping agent's answer to a places question.

You get: the question (with region and, when relevant, the user's location), the
gold expectation (expected_places + answer_notes written by the question author),
and the agent's tool activity and final answer.

Judge CORRECT when the final answer satisfies the answer_notes: it identifies the
expected place(s) (name variants/translations of the same real-world entity count),
and for wrong-premise questions it declines or corrects rather than confidently
naming a different entity. An honest decline is ONLY correct when the place is
genuinely absent — if the gold expectation names a real findable place, a
decline (however well-worded) is INCORRECT. Judge INCORRECT when the answer names the wrong entity,
dodges a findable answer, invents unverified specifics, or answers from memory in
direct contradiction of its own tool results. Ignore style; judge substance.

Reply with ONLY a JSON object: {"correct": true/false, "reason": "<one sentence>"}"""


def haversine_km(lat1, lng1, lat2, lng2):
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def mech_stats(qid, trace, final):
    """Mechanical funnel vs backfilled gold: anchor error (km) on each search,
    gold surfaced in results, gold passed to details, gold named in final,
    query churn. None where not applicable (no gold / no searches)."""
    gcoords = GOLD_COORDS.get(qid) or {}
    grefs = GOLD_REFS.get(qid, [])
    searches = [t for t in trace if t["tool"] == "search_places"]
    anchor_kms = []
    for t in searches:
        try:
            lat, lng = float(t["args"].get("lat")), float(t["args"].get("lng"))
        except (TypeError, ValueError):
            continue
        errs = [haversine_km(lat, lng, c[0], c[1]) for c in gcoords.values()]
        if errs:
            t["anchor_km"] = round(min(errs), 1)
            anchor_kms.append(min(errs))
    qcounts = Counter((t["args"].get("query") or "").strip().lower()
                      for t in searches)
    surfaced = None
    if grefs and searches:
        seen = {r for t in searches for r in t.get("refs", [])}
        surfaced = any(g in seen for g in grefs)
    detailed = None
    if grefs and any(t["tool"] == "place_details" for t in trace):
        dref = {str(r) for t in trace if t["tool"] == "place_details"
                for r in (t["args"].get("refs") or [])}
        detailed = any(g in dref for g in grefs)
    return {
        "anchor_first_km": round(anchor_kms[0], 1) if anchor_kms else None,
        "anchor_min_km": round(min(anchor_kms), 1) if anchor_kms else None,
        "gold_surfaced": surfaced,
        "gold_detailed": detailed,
        "gold_named": any(g in (final or "") for g in grefs) if grefs else None,
        "q_repeat": max(qcounts.values()) if qcounts else 0,
    }


async def replay_at_gold(http, qid, trace):
    """True if ANY of the episode's search queries surfaces a gold ref when
    anchored at the gold coords; False if none do; None if nothing to replay."""
    gcoords = GOLD_COORDS.get(qid) or {}
    grefs = set(GOLD_REFS.get(qid, []))
    queries = list(dict.fromkeys(
        (t["args"].get("query") or "").strip()
        for t in trace if t["tool"] == "search_places"))
    queries = [x for x in queries if x]
    if not gcoords or not grefs or not queries:
        return None
    for query in queries[:4]:
        for lat, lng in gcoords.values():
            try:
                res = await wp_search(http, query, lat, lng)
            except Exception:  # noqa: BLE001
                continue
            if any(c["ref"] in grefs for c in res):
                return True
    return False


# Search/nearest results carry a LOCALITY column (neighbourhood or street,
# then city) so two branches of one name can be told apart — bq-102446
# (2026-09-05): three "Cuban Coffee Queen" refs, the passage said "old
# seaport", nothing in the evidence let the model or the judge pick the
# branch. Off with BENCH_LOCALITY=0 to reproduce pass 4/5 rendering.
SHOW_LOCALITY = os.environ.get("BENCH_LOCALITY", "1") != "0"


def locality_of(c):
    parts = [c.get("neighbourhood") or c.get("street") or "", c.get("locality") or ""]
    parts = [p for p in parts if p]
    if len(parts) == 2 and parts[0] == parts[1]:
        parts = parts[:1]
    return ", ".join(parts)


# 2026-09-10 (Jon): PLACE CONTEXT. Every ho200 gold place has a travel-blog snippet naming it in
# wplaces' blog_references sidecar, and 21% have an LLM-written description of their website — and the
# harness showed the model NONE of it (place_details rendered blog_references as a bare count). The
# model was choosing between "Aloha Table | restaurant" and "Aloha Roastery | cafe" with no evidence
# either way, and took the top hit: wrong-entity pins were 25/79 unreached ho200 questions. With the
# flag on, each search result gets an indented context line beneath it (kept OFF the result line so
# every `(lat, lng)`-anchored parser is untouched), and place_details carries the snippets verbatim.
# Off by default so the ladder stays comparable until it is re-run with it on.
PLACE_CONTEXT = os.environ.get("BENCH_PLACE_CONTEXT", "") not in ("", "0")
CONTEXT_SNIPPET_CHARS = int(os.environ.get("BENCH_CONTEXT_CHARS", "150"))
# 2026-09-13: a result with no blog/site evidence used to get a filler line ("no travel-blog mentions on file") — pure noise for a
# 0.8B (5 of 6 results in a typical set). Off by default now; BENCH_CONTEXT_EMPTY_LINE=1 restores the round-8 observation.
CONTEXT_EMPTY_LINE = os.environ.get("BENCH_CONTEXT_EMPTY_LINE", "0") not in ("", "0")
# Wikipedia entries in WEB RESULTS. Round-8 separation (LADDER.md 2026-09-13): 3 entries x 400-char extracts cost v4rl6 −4.5 strict
# zero-shot (crowds the blog snippets; the losses are misattribution, not wiki citations). Defaults now 1 entry, 200 chars, always
# shown; BENCH_WIKI_MAX=3 BENCH_WIKI_CHARS=400 restores the round-8 observation, BENCH_WIKI_MAX=0 removes wiki entirely,
# BENCH_WIKI_ONLY_IF_THIN=N shows wiki only when fewer than N blog hits came back.
WIKI_MAX = int(os.environ.get("BENCH_WIKI_MAX", "1"))
WIKI_CHARS = int(os.environ.get("BENCH_WIKI_CHARS", "200"))
WIKI_ONLY_IF_THIN = int(os.environ.get("BENCH_WIKI_ONLY_IF_THIN", "0"))


def _ctx_line(d):
    """One line of identity evidence for a place, from a /places/details record."""
    parts = []
    br = (d.get("blog_references") or {}).get("references") or []
    for r in br[:2]:
        sn = " ".join(str(r.get("snippet") or "").split())
        if sn:
            parts.append(f'blog "{sn[:CONTEXT_SNIPPET_CHARS]}" ({r.get("domain") or "?"})')
    web = d.get("web") or {}
    if web.get("description"):
        parts.append("site: " + " ".join(str(web["description"]).split())[:CONTEXT_SNIPPET_CHARS])
    if not parts:
        return "no travel-blog mentions on file"
    return " | ".join(parts)



TOPOLOOP = bool(os.environ.get("TOPOLOOP_API_KEY"))
TOPOLOOP_BASE = os.environ.get("TOPOLOOP_API_BASE", "https://api.topoloop.com").rstrip("/")


async def _post_retry(http, url, json, headers=None, timeout=30, tries=6):
    """POST with backoff on 429/5xx (the topoloop free tier is 2 r/s, burst 10)."""
    delay = 1.0
    r = None
    for i in range(tries):
        r = await http.post(url, json=json, headers=headers or {}, timeout=timeout)
        if r.status_code not in (429, 502, 503, 504) or i == tries - 1:
            return r
        await asyncio.sleep(delay)
        delay = min(delay * 2, 8.0)
    return r

async def wp_context(http, results):
    """ref -> context line for a search result set (details endpoint takes 5 refs per call)."""
    refs = [c["ref"] for c in results if isinstance(c, dict) and c.get("ref")]
    out = {}
    for i in range(0, len(refs), 5):
        try:
            r = await _post_retry(http, f"{WPLACES}/places/details", json={"osm_ids": refs[i:i + 5]},
                                  headers={"X-API-Key": WPLACES_KEY}, timeout=30)
            r.raise_for_status()
            for ref, d in r.json().items():
                if isinstance(d, dict):
                    out[ref] = _ctx_line(d)
        except Exception as e:  # context is best-effort; never break a search over it
            for ref in refs[i:i + 5]:
                out.setdefault(ref, f"(context unavailable: {str(e)[:40]})")
    return out


def fmt_results(results, context=None, handles=None):
    if not results:
        return "(no results)"
    lines = []
    stats = None
    for c in results:
        if "_stats" in c:
            stats = c; continue
        loc = f"{locality_of(c) or '—'} | " if SHOW_LOCALITY else ""
        cx = ""
        if "open" in c:
            cx = (f" | open {c.get('opens_at')}-{c.get('closes_at')}" if c.get("open") is True
                  else (f" | closed (opens {c.get('opens_at')})" if c.get("open") is False and c.get("opens_at") else (" | closed that day" if c.get("open") is False else " | hours unknown")))
        if "wheelchair_ok" in c:
            cx += f" | wheelchair {c.get('wheelchair') or 'unknown'}"
        if "free_ok" in c:
            cx += f" | fee {c.get('fee') or 'unknown'}"
        lines.append(f"{c['ref']} | {c['name']} | {c['category']} | "
                     f"{loc}{c['distance_m']}m{cx} | ({c['lat']}, {c['lng']})")
        if c.get("blog_quote"):   # 2026-09-30 evidence shim: the vouching sentence + its source, as a context line
            if handles is not None and c.get("blog_source") and os.environ.get("BENCH_EVIDENCE_HANDLE", "") not in ("", "0"):
                hp = handles.get(c["blog_source"], c.get("blog_title") or "")
                lines.append("    ↳ blog [" + hp["handle"] + "] " + hp["site"] + ": \"" + c["blog_quote"] + "\"")
            else:
                lines.append('    ↳ blog: "' + c["blog_quote"] + '" — ' + (c.get("blog_source") or ""))
        if context is not None:
            cx_line = context.get(c['ref'])
            if cx_line and not cx_line.startswith("(context unavailable"):
                lines.append(f"    ↳ {cx_line}")
            elif CONTEXT_EMPTY_LINE:
                lines.append(f"    ↳ {cx_line or 'no travel-blog mentions on file'}")
    if stats and stats.get("_stats"):
        st = stats["_stats"]
        lines.insert(0, f"CONSTRAINT {json.dumps(stats.get('_constraints'))}: {st.get('matched', 0)} satisfy, {st.get('unknown', 0)} unknown (shown after), {st.get('excluded', 0)} excluded")
    return "\n".join(lines)


WSEARCH_EP = os.environ.get("WSEARCH_EP", (_EP.get("wsearch") or {}).get("base_url", "http://127.0.0.1:5000"))
WIKI_EP = os.environ.get("WIKI_EP", (_EP.get("wiki") or {}).get("base_url", "http://127.0.0.1:8003"))
# Routing (route/matrix/isochrone/expansion/locate): prefer the LOCAL planet
# Valhalla on ripper ([valhalla_local], docker valhalla-local :8005 — 5-15x
# faster than Helsinki from here, 2026-09-06) when configured; env overrides.
# /height stays on the REMOTE ([valhalla]) — the local box has no elevation
# tiles and answers null heights.
_VL_LOCAL = _EP.get("valhalla_local") or {}
_VL_REMOTE = _EP.get("valhalla") or {}
VALHALLA_EP = os.environ.get("VALHALLA_SERVER_URL") or _VL_LOCAL.get("base_url") \
    or _VL_REMOTE.get("base_url", "http://65.21.229.102")
VALHALLA_KEY = os.environ.get("VALHALLA_API_KEY") if os.environ.get("VALHALLA_SERVER_URL") else None
if VALHALLA_KEY is None:
    VALHALLA_KEY = (_VL_LOCAL.get("api_key", "") if VALHALLA_EP == _VL_LOCAL.get("base_url")
                    else _VL_REMOTE.get("api_key", ""))
VALHALLA_HEIGHT_EP = os.environ.get("VALHALLA_HEIGHT_URL") or _VL_REMOTE.get("base_url", VALHALLA_EP)
VALHALLA_HEIGHT_KEY = os.environ.get("VALHALLA_HEIGHT_KEY") or _VL_REMOTE.get("api_key", VALHALLA_KEY)
# prime_server rejects an empty header VALUE with 400 "Malformed HTTP request", so a
# keyless (local) Valhalla must be called with no X-API-Key header at all — sending
# an empty one made every route/locate call fail as an infra error (2026-09-08).
VALHALLA_HEADERS = {"X-API-Key": VALHALLA_KEY} if VALHALLA_KEY else {}
VALHALLA_HEIGHT_HEADERS = {"X-API-Key": VALHALLA_HEIGHT_KEY} if VALHALLA_HEIGHT_KEY else {}
# kill-and-resample budgets (defaults = the blogq harvest contract; pair
# questions — routing/identity — need BENCH_SEARCH_BUDGET=2 to look up both
# entities)
# tool results that mean "the service did not answer" — never the model's fault
INFRA_ERR = ("search error:", "nearest error:", "route error:",
             "blog search error:", "page error:", "details error:")
SEARCH_BUDGET = int(os.environ.get("BENCH_SEARCH_BUDGET", "1"))
WEB_BUDGET = int(os.environ.get("BENCH_WEB_BUDGET", "1"))
if SEARCH_BUDGET != 1 or WEB_BUDGET != 1:
    # keep the prompt's stated budget in sync with enforcement — a model
    # told "ONE search" while two are allowed resorts to nearest-scanning
    # flail instead of searching the second entity (measured, 2026-09-02)
    _N = {1: "ONE", 2: "TWO", 3: "THREE"}
    SYSTEM = SYSTEM.replace(
        "You have ONE call each of search_places and web_search",
        f"You have {_N.get(SEARCH_BUDGET, SEARCH_BUDGET)} calls of "
        f"search_places and {_N.get(WEB_BUDGET, WEB_BUDGET)} of web_search")
# BENCH_FETCH_BUDGET (default 0 = unlimited, the historical behaviour): the
# n+1th web_fetch is refused with the budget message and tools close, so the
# model must answer — a soft stop (protocol stays ok), unlike search overruns.
FETCH_BUDGET = int(os.environ.get("BENCH_FETCH_BUDGET", "0"))
if FETCH_BUDGET:
    _Nf = {1: "ONE", 2: "TWO", 3: "THREE"}.get(FETCH_BUDGET, str(FETCH_BUDGET))
    for _s in ("You have ONE call each of search_places and web_search",
               f"of web_search"):
        if _s in SYSTEM:
            SYSTEM = SYSTEM.replace(_s, _s + f" and {_Nf} of web_fetch", 1)
            break
_pages_cons = None

_VL_COSTING = {"walk": "pedestrian", "bike": "bicycle", "car": "auto",
               "drive": "auto", "bus": "bus", "motorcycle": "motorcycle",
               "taxi": "taxi"}


async def _vl_call(http, payload):
    r = await http.post(f"{VALHALLA_EP}/route", json=payload,
                        headers=VALHALLA_HEADERS, timeout=30)
    routes = r.json().get("routes") or []
    return routes[0] if routes else None


async def _vl_gain_loss(http, rt):
    """elevation_gain_m / elevation_loss_m for a routed shape via /height.
    Returns {} (keys omitted, per the elevation contract) when the server
    has no coverage or anything fails."""
    try:
        coords = (rt.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            return {}
        interval = max(30, int(rt.get("distance", 0) / 400))
        step = max(1, len(coords) // 500)
        shape = [{"lat": c[1], "lon": c[0]} for c in coords[::step]]
        if coords[-1] != coords[::step][-1]:
            shape.append({"lat": coords[-1][1], "lon": coords[-1][0]})
        r = await http.post(f"{VALHALLA_HEIGHT_EP}/height",
                            json={"shape": shape,
                                  "resample_distance": interval},
                            headers=VALHALLA_HEIGHT_HEADERS, timeout=30)
        heights = [h for h in (r.json().get("height") or [])]
        gain = loss = 0.0
        prev = None
        seen = False
        for h in heights:
            if h is None:
                prev = None
                continue
            seen = True
            if prev is not None:
                d = h - prev
                if d > 0:
                    gain += d
                else:
                    loss -= d
            prev = h
        if not seen:
            return {}
        return {"elevation_gain_m": round(gain),
                "elevation_loss_m": round(loss)}
    except Exception:  # noqa: BLE001
        return {}


async def vl_route(http, targs):
    """Prod-shaped calculate_route: same payload, result dict and route_id
    format as tokyo.py's calculate_route (geometry never enters context).

    Midpoint shim (invisible to the model): the current tileset has
    graph-level not_thru artifacts (Venice repro, RESULTS 2026-09-02) that
    make direct queries detour 2-3x while the path provably exists
    piecewise. When route/crow-flies ratio looks pathological, retry with a
    pass-through via at the geometric midpoint and keep the faster result.
    Remove once the tileset rebuild lands (canary: Al Squero -> da Lele
    ~1.5 km walk)."""
    frm, to = targs["from_location"], targs["to_location"]
    mode = str(targs.get("mode") or "walk").lower()
    costing = _VL_COSTING.get(mode, "pedestrian")
    loc_f = {"lon": float(frm[0]), "lat": float(frm[1])}
    loc_t = {"lon": float(to[0]), "lat": float(to[1])}
    payload = {"locations": [loc_f, loc_t],
               "costing": costing, "format": "osrm",
               "shape_format": "geojson"}
    if costing == "pedestrian":
        # prod's profile minus its 3 km/h hiking walking_speed: that speed
        # makes urban durations ~60% slow, and calibrated models contradict
        # the tool (measured: Gemini corrects 8min->5min, judge fails it).
        # 5.1 km/h = Valhalla default; trail tolerance kept so hikes route.
        payload["costing_options"] = {"pedestrian": {
            "use_hills": 1.0, "max_hiking_difficulty": 6,
            "step_penalty": 0, "max_grade": 30,
            "use_trails": 1.0, "shortest": False}}
    rt = await _vl_call(http, payload)
    if not rt:
        return json.dumps({"error": "No route found"})
    crow = 2 * 6371000 * math.asin(math.sqrt(
        math.sin(math.radians(loc_t["lat"] - loc_f["lat"]) / 2) ** 2
        + math.cos(math.radians(loc_f["lat"]))
        * math.cos(math.radians(loc_t["lat"]))
        * math.sin(math.radians(loc_t["lon"] - loc_f["lon"]) / 2) ** 2))
    if crow > 100 and rt.get("distance", 0) > 2.5 * crow:
        via = {"lon": (loc_f["lon"] + loc_t["lon"]) / 2,
               "lat": (loc_f["lat"] + loc_t["lat"]) / 2,
               "type": "through", "radius": 300}
        alt = await _vl_call(http, {**payload,
                                    "locations": [loc_f, via, loc_t]})
        # compare by distance: the pathological direct route can still win
        # on duration due to per-edge speed quirks in the hiking profile
        if alt and alt.get("distance", 1e18) < rt.get("distance", 0):
            rt = alt
    secs, meters = rt.get("duration", 0), rt.get("distance", 0)
    if secs < 60:
        dur = f"{int(secs)} sec"
    elif secs < 3600:
        dur = f"{int(secs / 60)} min"
    else:
        h, m = int(secs / 3600), int((secs % 3600) / 60)
        dur = f"{h}h {m}m" if m else f"{h}h"
    result = {
        "duration_seconds": secs, "distance_meters": meters,
        "duration_formatted": dur, "mode": mode,
        "route_id": f"route_{frm[0]}_{frm[1]}_{to[0]}_{to[1]}_{mode}"}
    result.update(await _vl_gain_loss(http, rt))
    return json.dumps(result)


async def ws_search(http, query, limit=5, handles=None):
    out = []
    wiki_results = []
    try:
        if TOPOLOOP:
            r = await _post_retry(http, f"{TOPOLOOP_BASE}/web/search", json={"query": query, "limit": 5},
                                  headers={"X-API-Key": os.environ["TOPOLOOP_API_KEY"]}, timeout=30)
            w = RuntimeError("no wiki service behind the topoloop API")
        else:
            r, w = await asyncio.gather(
                http.post(f"{WSEARCH_EP}/api/search",
                          json={"query": query, "limit": 5}, timeout=30),
                http.post(f"{WIKI_EP}/wiki/serp",
                          json={"query": query}, timeout=30),
                return_exceptions=True)
        if not isinstance(r, Exception):
            for res in r.json().get("results", [])[:5]:
                url = res.get("page_url") or res.get("url") or ""
                title = (res.get("page_title") or res.get("title") or "")[:100]
                ctx = (res.get("context") or res.get("content") or "")[:200]
                if handles is not None:
                    p = handles.get(url, title)
                    chunk = re.sub(r"\s+", " ", res.get("content") or "").strip()
                    if len(chunk) > CHUNK_CHARS:
                        chunk = chunk[:CHUNK_CHARS].rsplit(" ", 1)[0] + " …"
                    head = f"[{p['handle']}] {title}" + ("" if p["site"].lower() in title.lower() else f" — {p['site']}")
                    out.append(f"{head}\n  {ctx}" + (f"\n  {chunk}" if chunk and chunk != ctx else ""))
                else:
                    out.append(f"{url}\n  {title}\n  {ctx}")
        if not isinstance(w, Exception):
            wiki_results = w.json().get("results", []) or []
    except Exception:
        pass
    en_qids = {x.get("qid") for x in wiki_results if x.get("lang") == "en"}
    seen, n = set(), 0
    n_blog = len(out)
    if WIKI_ONLY_IF_THIN and n_blog >= WIKI_ONLY_IF_THIN:
        wiki_results = []
    for res in wiki_results:
        qid = res.get("qid")
        if qid in seen or n >= WIKI_MAX:
            continue
        if res.get("lang") != "en" and qid in en_qids:
            continue
        seen.add(qid)
        n += 1
        if handles is not None:
            p = handles.get(res.get("url") or "", f"{(res.get('title') or '')[:100]} (Wikipedia)")
            out.append(f"[{p['handle']}] {(res.get('title') or '')[:100]} (Wikipedia)\n  "
                       f"{(res.get('extract') or '')[:WIKI_CHARS]}")
        else:
            out.append(f"{res.get('url')}\n  "
                       f"{(res.get('title') or '')[:100]} (Wikipedia)\n  "
                       f"{(res.get('extract') or '')[:200]}")
    if handles is not None and out:
        # the handle table: the only place URLs appear, so web_fetch can be called
        table = "\n".join(f"{p['handle']}: {u}" for u, p in handles.by_url.items()
                          if p["handle"] in {ln.split("]")[0][1:] for ln in out})
        return "WEB RESULTS:\n" + "\n".join(out) + "\nHANDLES:\n" + table
    return "WEB RESULTS:\n" + "\n".join(out) if out else "(no web results)"


def ws_page(url):
    if TOPOLOOP:
        import urllib.request as _u
        import time as _t
        body = json.dumps({"url": url}).encode()
        for i in range(5):
            try:
                req = _u.Request(f"{TOPOLOOP_BASE}/web/page", body,
                                 {"Content-Type": "application/json", "X-API-Key": os.environ["TOPOLOOP_API_KEY"]})
                d = json.loads(_u.urlopen(req, timeout=30).read())
                return (d.get("content") or "")[:12000] or "(page not found in corpus)"
            except _u.HTTPError as e:
                if e.code == 404:
                    return "(page not found in corpus)"
                if e.code in (429, 502, 503, 504) and i < 4:
                    _t.sleep(1.5 * (i + 1))
                    continue
                return f"(page error: {e.code})"
            except Exception as e:
                return f"(page error: {e})"
    if "wikipedia.org/wiki/" in url:
        import urllib.request as _u
        import urllib.parse as _p
        try:
            lang = url.split("//")[1].split(".")[0]
            title = _p.unquote(url.split("/wiki/")[-1]).replace("_", " ")
            body = json.dumps({"title": f"{lang}:{title}"}).encode()
            req = _u.Request(f"{WIKI_EP}/wiki/article", body,
                             {"Content-Type": "application/json"})
            d = json.loads(_u.urlopen(req, timeout=30).read())
            txt = d.get("text") or d.get("extract") or ""
            return txt[:12000] if txt else "(article not found)"
        except Exception as e:
            return f"(wiki fetch error: {e})"
    global _pages_cons
    import sqlite3
    if _pages_cons is None:
        _pages_cons = []
        for db in ((_EP.get("wsearch") or {}).get("pages_db", "/home/jon/workspace/wsearch/data/pages_valhalla.db"),
                   "/home/jon/workspace/wsearch/data/pages.db"):
            try:
                _pages_cons.append(sqlite3.connect(
                    f"file:{db}?mode=ro", uri=True, check_same_thread=False))
            except Exception:
                pass
    for con in _pages_cons:
        rows = con.execute(
            "SELECT content FROM chunks WHERE page_url=? "
            "ORDER BY chunk_index LIMIT 30", (url,)).fetchall()
        if rows:
            return "\n\n".join(r[0] for r in rows)[:12000]
    return "(page not found in corpus)"


TOOLCALL_TEXT_RE = re.compile(
    r"<tool_call>\s*<function=([\w.\-]+)>(.*?)</function>\s*</tool_call>",
    re.S)
_PARAM_RE = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.S)


class _TextFn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _TextToolCall:
    """Duck-typed stand-in for openai's tool-call object, built from qwen
    <tool_call> text when the server has no tool-call parser."""

    def __init__(self, i, name, args):
        self.id = f"texttc-{i}"
        self.type = "function"
        self.function = _TextFn(name, json.dumps(args, ensure_ascii=False))

    def model_dump(self):
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


def parse_text_tool_calls(content):
    calls = []
    for i, m in enumerate(TOOLCALL_TEXT_RE.finditer(content or "")):
        args = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            v = pm.group(2)
            try:
                args[pm.group(1)] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                args[pm.group(1)] = v
        calls.append(_TextToolCall(i, m.group(1), args))
    return calls


def _bearing(lat1, lng1, lat2, lng2):
    import math as _m
    dl = _m.radians(lng2 - lng1)
    y = _m.sin(dl) * _m.cos(_m.radians(lat2))
    x = (_m.cos(_m.radians(lat1)) * _m.sin(_m.radians(lat2))
         - _m.sin(_m.radians(lat1)) * _m.cos(_m.radians(lat2)) * _m.cos(dl))
    b = (_m.degrees(_m.atan2(y, x)) + 360) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((b + 22.5) // 45) % 8]


async def wp_nearest(http, lat, lng, limit=10):
    r = await _post_retry(http, f"{WPLACES}/places/nearest",
                          json={"lat": lat, "lng": lng, "limit": limit},
                          headers={"X-API-Key": WPLACES_KEY}, timeout=30)
    r.raise_for_status()
    out = []
    for c in r.json().get("results", [])[:limit]:
        t = str(c.get("osm_type", ""))[:1].upper()
        ref = f"{t}{c.get('osm_id')}"
        d = c.get("distance_meters")
        brg = _bearing(lat, lng, c.get("lat"), c.get("lng"))
        dtxt = f"{d:.0f}m" if d is not None else "?m"
        loc = f"{locality_of(c) or '—'} | " if SHOW_LOCALITY else ""
        out.append(f"{ref} | {c.get('name_en') or c.get('name')} | "
                   f"{c.get('category')} | {loc}{dtxt} {brg} | "
                   f"fame {c.get('fame') or 0:.2f}")
    return out


# 2026-09-08 (Jon): explicit search radius in km; unset/0 = the server's dynamic radius (5-100 km by
# density, median 30 on real model searches). Replay on the round-2 probe: the 0.8B's anchors are
# routinely 10-300 km off, and a 100 km radius surfaces the gold in 65% of the no-reach searches whose
# anchor was 10-100 km off while keeping gold in the top 6 on 98.7% of searches that already had it.
WPLACES_SEARCH_RADIUS_KM = float(os.environ.get("WPLACES_SEARCH_RADIUS_KM", "0") or 0)


async def wp_search(http, query, lat, lng, constraints=None):
    body = {"query": query, "lat": lat, "lng": lng, "limit": 6}
    if WPLACES_SEARCH_RADIUS_KM > 0:
        body["radius"] = WPLACES_SEARCH_RADIUS_KM
    if constraints:
        body["constraints"] = constraints
    r = await _post_retry(http, f"{WPLACES}/places/search",
                          json=body,
                          headers={"X-API-Key": WPLACES_KEY}, timeout=30)
    r.raise_for_status()
    out = []
    for c in r.json().get("results", []):
        oid = str(c["osm_id"])
        ref = oid if oid[:1].isalpha() else \
            {"node": "N", "way": "W", "relation": "R"}.get(c.get("osm_type", ""), "?") + oid
        out.append({"ref": ref, "name": c.get("name_en") or c.get("name"),
                    "category": c.get("category"),
                    "distance_m": round(c.get("distance_meters") or 0),
                    "lat": round(c.get("lat", 0), 5), "lng": round(c.get("lng", 0), 5),
                    "locality": c.get("locality"), "neighbourhood": c.get("neighbourhood"),
                    "street": c.get("street"), **{k: c.get(k) for k in ("open", "opens_at", "closes_at", "opening_hours", "wheelchair", "fee", "wheelchair_ok", "free_ok") if k in c and (c.get(k) is not None or k in ("open",))},
                    # 2026-09-30 evidence shim: a blog sentence that vouches for the place + its source.
                    # Only present when the endpoint is the evidence shim; passed through untouched so the
                    # model sees web evidence inside the search result (BENCH_EVIDENCE=0 strips it for A/B).
                    **({k: c[k] for k in ("blog_quote", "blog_source", "blog_title") if c.get(k)}
                       if os.environ.get("BENCH_EVIDENCE", "1") != "0" else {})})
    if constraints:
        out.append({"_stats": r.json().get("constraint_stats"), "_constraints": r.json().get("constraints")})
    return out


DETAIL_FIELDS = ["name", "formatted_address", "website", "phone", "opening_hours", "fee",
                 "wheelchair", "wikipedia", "wikidata"]   # formatted_address added 2026-09-06 (Jon)


async def wp_details(http, refs):
    r = await _post_retry(http, f"{WPLACES}/places/details",
                          json={"osm_ids": refs[:5]},
                          headers={"X-API-Key": WPLACES_KEY}, timeout=30)
    r.raise_for_status()
    out = {}
    for ref, d in r.json().items():
        if not isinstance(d, dict):
            out[ref] = d
            continue
        slim = {("address" if k == "formatted_address" else k): d[k] for k in DETAIL_FIELDS if d.get(k) is not None}
        br = d.get("blog_references") or {}
        if br.get("blog_count"):
            slim["travel_blog_mentions"] = br["blog_count"]
        if PLACE_CONTEXT:
            sn = [{"blog": r.get("domain"), "says": " ".join(str(r.get("snippet") or "").split())[:CONTEXT_SNIPPET_CHARS]}
                  for r in (br.get("references") or [])[:3] if r.get("snippet")]
            if sn:
                slim["travel_blog_snippets"] = sn
            web = d.get("web") or {}
            if web.get("description"):
                slim["website_summary"] = " ".join(str(web["description"]).split())[:CONTEXT_SNIPPET_CHARS * 2]
        out[ref] = slim or "(no details on file)"
    # 2026-09-10 (card12): a 6-ref call silently dropped the 6th (Tamoka, the one accessible restaurant).
    # Never truncate silently — say which refs were not looked up.
    for ref in refs[5:]:
        out[ref] = "(not looked up: place_details takes at most 5 refs per call — call again for this one)"
    return out


# vLLM takes chat_template_kwargs to disable thinking; OpenRouter ignores it;
# Google's OpenAI-compatible endpoint REJECTS unknown fields (400). Set
# BENCH_NO_TEMPLATE_KWARGS=1 for Google direct.
ROLLOUT_EXTRA = {} if os.environ.get("BENCH_NO_TEMPLATE_KWARGS") else {"chat_template_kwargs": {"enable_thinking": False}}


class _StoredMsg:
    """Duck-typed assistant message rebuilt from a stored row (resume)."""

    def __init__(self, content, tool_calls):
        self.content, self.tool_calls = content, tool_calls


TERMINATED = "(rollout terminated:"


def resume_state(row):
    """Rebuild the live rollout state from a stored harness row so the loop can
    continue where the row stopped: messages (system prompt swapped for the
    CURRENT one, so the continuation decodes under the budget now enforced),
    trace, usage, handles, and the tool calls the stored assistant turn issued
    that never got a result (the killed over-budget call, plus any siblings in
    the same turn). Nothing the model said is altered."""
    msgs = copy.deepcopy(row["messages"])
    # drop a trailing final (rows with a final are complete; nothing to resume)
    if msgs and msgs[-1]["role"] == "assistant" and not msgs[-1].get("tool_calls"):
        msgs.pop()
    swapped = False
    if msgs and msgs[0]["role"] == "system" and msgs[0]["content"] != SYSTEM:
        msgs[0]["content"] = SYSTEM
        swapped = True
    answered = {m.get("tool_call_id") for m in msgs if m["role"] == "tool"}
    last_ai = next((m for m in reversed(msgs) if m["role"] == "assistant" and m.get("tool_calls")), None)
    pending = []
    for tc in (last_ai or {}).get("tool_calls") or []:
        if tc.get("id") in answered:
            continue
        fn = tc.get("function") or {}
        a = fn.get("arguments")
        c = _TextToolCall(0, fn.get("name"), a if isinstance(a, dict) else {})
        if isinstance(a, str):
            c.function.arguments = a
        c.id = tc.get("id") or c.id
        pending.append(c)
    trace = []
    for t in row.get("trace") or []:
        head = t.get("head") or ""
        if head.startswith(TERMINATED):
            continue
        a = {k: t.get(kk) for k, kk in (("query", "query"), ("url", "url"), ("lat", "lat"),
                                        ("lng", "lng"), ("refs", "drefs")) if t.get(kk) is not None}
        trace.append({"tool": t["tool"], "args": a, "refs": t.get("refs") or [],
                      "result_head": head, "anchor_km": t.get("anchor_km")})
    handles = None
    if CONTRACT == "v2":
        handles = Handles()
        for h, p in sorted((row.get("handles") or {}).items(), key=lambda kv: int(kv[0][1:])):
            handles.get(p["url"], p.get("title") or "")
    usage = dict(row.get("usage") or {"prompt": 0, "completion": 0})
    info = {"from_protocol": row.get("protocol"), "n_tools_before": len(trace),
            "pending": [c.function.name for c in pending], "system_swapped": swapped,
            "usage_before": dict(usage)}
    return msgs, trace, usage, handles, _StoredMsg(last_ai.get("content") if last_ai else None, pending), info


_COORD_RE = re.compile(r"\((-?\d{1,3}\.\d+), (-?\d{1,3}\.\d+)\)")


def _endpoints_seen(targs, trace, q, tol_m=150):
    """Both route endpoints within tol_m of a coordinate the conversation has
    seen: any "(lat, lng)" in a tool result so far, or the user's location."""
    seen = []
    if q.get("anchor"):
        seen.append((float(q["anchor"][0]), float(q["anchor"][1])))
    for t in trace:
        for la, lo in _COORD_RE.findall(str(t.get("result_head") or "")):
            seen.append((float(la), float(lo)))
    try:
        pts = [(float(targs["from_location"][1]), float(targs["from_location"][0])),
               (float(targs["to_location"][1]), float(targs["to_location"][0]))]
    except (KeyError, TypeError, ValueError, IndexError):
        return True   # malformed args: let the tool report its own error
    for p in pts:
        if not any(haversine_km(p[0], p[1], s[0], s[1]) * 1000 <= tol_m for s in seen):
            return False
    return True


async def run_question(llm, judge_llm, http, q, args, sem, resume=None):
    user = q["question"]
    if q.get("anchor"):
        user += f"\n(My location: {q['anchor'][0]}, {q['anchor'][1]})"
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    trace, usage = [], {"prompt": 0, "completion": 0}
    if q.get("prefix"):
        # 2026-09-11: seeded continuation — the row carries earlier turns (the model's own clarifying ask and the
        # traveller's reply) and the rollout continues from there. Recorded like a live text ask so rows["asks"]
        # and the seed filters see it. Used to harvest ask -> reply -> answer traces for the round-7 SFT.
        for _pm in q["prefix"]:
            msgs.append({"role": _pm["role"], "content": _pm["content"]})
        _asks = [m["content"] for m in q["prefix"] if m["role"] == "assistant"]
        _reps = [m["content"] for m in q["prefix"] if m["role"] == "user"]
        for _a, _r in zip(_asks, _reps):
            trace.append({"tool": "ask_user", "args": {"question": _a.strip()[:400], "text_ask": True, "seeded": True,
                                                        "leak_blocked": False}, "refs": [], "result_head": f"TRAVELLER: {_r}"})
    turn2 = None
    if q.get("turn1"):
        # 2026-09-18 FOLLOW-UP mode: replay the checkpoint's own completed turn-1 rollout verbatim (what the iOS shell
        # sends on a follow-up: the full prior transcript incl. tool calls/results), then append the follow-up as the
        # next user turn. `trace` stays turn-2-only so budgets are fresh; handles are rebuilt so [wN] refs resolve.
        msgs = copy.deepcopy(q["turn1"]["messages"])
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = SYSTEM
        for m in msgs:
            m.pop("tool_calls_raw", None)
        msgs.append({"role": "user", "content": user})
        turn2 = {"qid": q.get("qid"), "family": (q.get("followup") or {}).get("family"), "msg0": len(msgs) - 1,
                 "q1": q.get("question1"), "a1": q["turn1"].get("final"), "in_context": q.get("in_context")}
    final, protocol, nudged = None, "ok", False
    tools_open = True
    capped = False
    handles = Handles() if CONTRACT == "v2" else None
    chandles = ComputeHandles()
    if turn2 and handles is not None:   # follow-up mode: turn-1 [wN] handles must resolve in turn 2
        for h, pmeta in sorted((q["turn1"].get("handles") or {}).items(), key=lambda kv: int(kv[0][1:])):
            handles.get(pmeta["url"], pmeta.get("title") or "")
    pending, resumed = None, None
    if resume is not None:
        msgs, trace, usage, handles, pending, resumed = resume_state(resume)
        if not pending.tool_calls:
            pending = None   # nothing to execute: just ask the model to continue

    for _ in range(2 * args.max_tools + 2):
        stored_turn = pending is not None
        if stored_turn:
            msg, pending = pending, None
        else:
            async with sem:
                resp = await llm.chat.completions.create(
                    model=args.model, messages=msgs,
                    tools=TOOLS if tools_open else None,
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    extra_body=ROLLOUT_EXTRA)
            # servers launched without --enable-auto-tool-choice return qwen
            # tool calls as raw <tool_call> text in content (2026-09-03 pass-2
            # incident: 89,926 zero-tool rollouts). Parse them client-side so
            # bench works against any endpoint config.
            _m0 = resp.choices[0].message
            if not _m0.tool_calls and _m0.content and "<tool_call>" in _m0.content:
                _parsed = parse_text_tool_calls(_m0.content)
                if _parsed:
                    _m0.tool_calls = _parsed
                    _m0.content = TOOLCALL_TEXT_RE.sub("", _m0.content).strip()
            u = resp.usage
            usage["prompt"] += u.prompt_tokens or 0
            usage["completion"] += u.completion_tokens or 0
            msg = resp.choices[0].message

        if not msg.tool_calls:
            final = msg.content or ""
            if resp.choices[0].finish_reason == "length":
                protocol = "truncated"
                break
            # 2026-09-12: the model appends an empty {"map": []} to its plain-text asks; looks_like_ask rejects any
            # text containing "map", so the reply turn never fired (1/400 ho200 rows). Strip it first, as ClarifyEnv does.
            _ask_body = re.sub(r'\s*\{"map"\s*:\s*\[\s*\]\s*\}\s*$', "", final or "").strip()
            if os.environ.get("BENCH_ASK_USER") and looks_like_ask(_ask_body) \
                    and not any(t["tool"] == "ask_user" for t in trace):
                # 2026-09-10: a text-form clarifying question (no tool call) is an ask too — the model
                # on the FP8 rental never calls ask_user, it just asks. Feed the traveller's reply as
                # the next user turn and let the model continue. Recorded in trace as ask_user
                # with text_ask=True so rows["asks"] and the seed filter see it.
                _reply, _blocked = await ask_user_sim(_ask_llm, _ask_model, q, _ask_body, sem)
                trace.append({"tool": "ask_user", "args": {"question": _ask_body[:400], "text_ask": True,
                                                            "leak_blocked": _blocked},
                              "refs": [], "result_head": f"TRAVELLER: {_reply}"})
                msgs.append({"role": "assistant", "content": final})
                msgs.append({"role": "user", "content": _reply})
                final = None
                continue
            if not final.strip() and not nudged:
                # Gemini-via-OpenRouter sometimes stops with empty content
                # after tool turns — nudge once for the actual answer.
                nudged = True
                msgs.append({"role": "assistant", "content": ""})
                msgs.append({"role": "user", "content":
                             "Give your final answer now in plain text."})
                continue
            break

        if not stored_turn:   # a resumed turn is already in msgs verbatim
            msgs.append({"role": "assistant", "content": msg.content,
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        # A stored turn executes its tools without an LLM call first, so the
        # semaphore that normally paces tool traffic never applied: 22,927
        # resumed rollouts put ~3K requests into httpcore's pool at once and
        # its per-tick assignment scan (quadratic) pinned the CPU (2026-09-05).
        async with (sem if stored_turn else contextlib.nullcontext()):
            for tc in msg.tool_calls:
                try:
                    targs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    targs = {}
                res_refs = []
                # An unreachable service is OUR fault, not the model's: a call that
                # never got an answer must not consume the search budget. (2026-09-04:
                # a harness pointed at a dead wplaces port produced 840 "over_budget"
                # rollouts that were really connection failures.)
                def _billable(t):
                    return not str(t.get("result_head") or "").startswith(INFRA_ERR)
                n_ws = sum(1 for t in trace if t["tool"] in ("web_search", "search_travel_blogs")
                           and _billable(t))
                n_ps = sum(1 for t in trace if t["tool"] == "search_places" and _billable(t))
                n_wf = sum(1 for t in trace if t["tool"] in ("web_fetch", "get_full_page") and _billable(t))
                if len(trace) >= args.max_tools:
                    result = "Tool budget exhausted — answer now from what you have."
                    tools_open = False   # next completion gets no tools: it must answer
                    capped = True        # 2026-09-18: counted like a search-cap hit (ladder_line "capped")
                                         # (2026-09-04 v2 smoke: 10 identical re-fetches
                                         # after this message, 235K prompt tokens)
                elif (tc.function.name in ("web_search", "search_travel_blogs")
                      and n_ws >= WEB_BUDGET) or (tc.function.name == "search_places"
                                                  and n_ps >= SEARCH_BUDGET):
                    if os.environ.get("BENCH_CAP_ANSWER"):
                        # 2026-09-17 REWARD_HACKS §24: mirror the RL env (GEOGYM_CAP_ANSWER) — refuse the call, close the
                        # tools, let the model answer from what it has; the row is judged normally with capped=True.
                        result = "Tool budget exhausted — answer now from what you have."
                        tools_open = False
                        capped = True
                    else:
                        # kill-and-resample: a third search attempt invalidates the
                        # rollout (harvest discards protocol != "ok"; retries resample)
                        protocol = "over_budget"
                        result = "(rollout terminated: search budget exceeded)"
                        trace.append({"tool": tc.function.name, "args": targs,
                                      "refs": [], "result_head": result})
                        break
                elif FETCH_BUDGET and tc.function.name in ("web_fetch", "get_full_page") \
                        and n_wf >= FETCH_BUDGET:
                    result = "Fetch budget exhausted — answer now from what you have."
                    tools_open = False
                elif tc.function.name == "search_places" and targs.get("within_minutes") \
                        and os.environ.get("BENCH_ROUTE_TOOL") == "2":
                    if not q.get("anchor") and not (targs.get("origin_ref") or targs.get("origin_name")):
                        # environment rule, not a hint: with no user location in the conversation a
                        # raw-coordinate origin can only be a guess (dry runs 2-4: the teacher guessed
                        # in 52 of 62 calls even when told not to). The app knows the same thing.
                        result = ("Tool refused: no user location was given in this conversation, so "
                                  "lat/lng here would be a guess. Pass origin_name (the starting place as "
                                  "the user named it, with lat/lng = the city or area) or origin_ref "
                                  "(a ref from an earlier result), and retry.")
                    else:
                        try:
                            from geogym_blogq.lab.tools.route import reachable
                            result, res_refs = await reachable(http, targs)
                            result = f"[{chandles.add('m', 'search_places', targs)}] {result}"
                        except Exception as e:  # noqa: BLE001
                            result = f"search error: {e}"
                elif tc.function.name == "search_along_route":
                    try:
                        from geogym_blogq.lab.tools.route import along_route
                        result, res_refs = await along_route(http, targs)
                        result = f"[{chandles.add('m', 'search_along_route', targs)}] {result}"
                    except Exception as e:  # noqa: BLE001
                        result = f"route error: {e}"
                elif tc.function.name == "search_places":
                    try:
                        q_ = (targs.get("query") or "").strip()
                        if not q_:
                            lat_, lng_ = (float(targs.get("lat", 0)),
                                          float(targs.get("lng", 0)))
                            lines = await wp_nearest(http, lat_, lng_)
                            res_refs = [ln.split(" | ")[0] for ln in lines]
                            result = (f"NEAREST PLACES (around {lat_}, {lng_}):\n"
                                      + "\n".join(lines) if lines
                                      else "(no places found)")
                        else:
                            from geogym_blogq.lab.tools.route import constraints_from as _cxf
                            res = await wp_search(http, q_,
                                                  float(targs.get("lat", 0)),
                                                  float(targs.get("lng", 0)), constraints=_cxf(targs))
                            res_refs = [c["ref"] for c in res if "ref" in c]
                            result = (f"SEARCH RESULTS (query: \"{q_}\", "
                                      f"anchor: {targs.get('lat')}, "
                                      f"{targs.get('lng')}):\n{fmt_results(res, await wp_context(http, res) if PLACE_CONTEXT else None, handles=handles)}")
                    except Exception as e:  # noqa: BLE001
                        result = f"search error: {e}"
                elif tc.function.name == "nearest_places":
                    try:
                        lines = await wp_nearest(http, float(targs.get("lat", 0)),
                                                 float(targs.get("lng", 0)))
                        res_refs = [ln.split(" | ")[0] for ln in lines]
                        result = (f"NEAREST PLACES (around {targs.get('lat')}, "
                                  f"{targs.get('lng')}):\n" + "\n".join(lines)
                                  if lines else "(no places found)")
                    except Exception as e:  # noqa: BLE001
                        result = f"nearest error: {e}"
                elif tc.function.name == "calculate_route" and os.environ.get("BENCH_ROUTE_TOOL") == "2" \
                        and not _endpoints_seen(targs, trace, q):
                    # endpoint provenance (2026-09-06): a route endpoint must be a coordinate this
                    # conversation has actually seen — a search/nearest result, an ORIGIN line, or the
                    # user's location. Anything else is a coordinate from memory (dry runs 2-5: 12 of
                    # 25 along-route tasks routed to a remembered destination 1-12 km off).
                    result = ("Tool refused: one or both endpoints are not coordinates from this "
                              "conversation (a search_places / nearest_places result or the user's "
                              "location). Resolve the place with search_places first and use the "
                              "coordinates it returns, then retry.")
                elif tc.function.name == "calculate_route":
                    try:
                        result = await vl_route(http, targs)
                        if not str(result).startswith(("route error", "Tool refused")):
                            result = f"[{chandles.add('r', 'calculate_route', targs)}] {result}"
                    except Exception as e:  # noqa: BLE001
                        result = f"route error: {e}"
                elif tc.function.name in ("web_search", "search_travel_blogs"):
                    try:
                        result = await ws_search(http, targs.get("query", ""),
                                                 targs.get("limit", 5), handles=handles)
                    except Exception as e:  # noqa: BLE001
                        result = f"blog search error: {e}"
                elif tc.function.name in ("web_fetch", "get_full_page"):
                    try:
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, ws_page, str(targs.get("url", "")))
                        if handles is not None and str(targs.get("url", "")).startswith("http"):
                            p = handles.get(str(targs.get("url", "")))
                            result = f"[{p['handle']}] {p['title'] or p['url']}\n{result}"
                    except Exception as e:  # noqa: BLE001
                        result = f"page error: {e}"
                elif tc.function.name == "place_details":
                    try:
                        # 2026-09-07 (bq-092951#9): refs come only from tool results. An invented
                        # ref used to get a silent {} back; now it gets a named error and no lookup.
                        _seen = set(re.findall(r"\b[NWR]\d{4,}\b", "\n".join(
                            str(m.get("content") or "") for m in msgs if m.get("role") == "tool")))
                        _want = [str(x) for x in targs.get("refs", [])]
                        _known = [x for x in _want if x in _seen]
                        _det = await wp_details(http, _known) if _known else {}
                        for x in _want:
                            if x not in _seen:
                                _det[x] = "ERROR: unknown ref — not in any of your search results"
                        result = json.dumps(_det, ensure_ascii=False)
                    except Exception as e:  # noqa: BLE001
                        result = f"details error: {e}"
                elif tc.function.name == "ask_user" and os.environ.get("BENCH_ASK_USER"):
                    n_ask = sum(1 for t in trace if t["tool"] == "ask_user")
                    if n_ask >= 1:
                        result = "The traveller already answered once and is waiting — answer now with what you have."
                    else:
                        _reply, _blocked = await ask_user_sim(_ask_llm, _ask_model, q, str(targs.get("question", "")), sem)
                        result = f"TRAVELLER: {_reply}"
                        if _blocked:
                            targs = dict(targs, leak_blocked=True)
                else:
                    result = f"Unknown tool {tc.function.name}."
                trace.append({"tool": tc.function.name, "args": targs,
                              "refs": res_refs, "result_head": result[:1200]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        if protocol == "over_budget":
            final = final or ""
            break
    else:
        protocol = "loop_cap"

    if final is None:
        final, protocol = "", "no_answer"

    # CONTRACT v2 format repair (format only, never facts): a final without the
    # trailing map object gets ONE tool-less turn that asks for the object alone.
    # The object is appended to the final; the nudge turn is NOT kept in
    # messages. Recorded as map_nudged so yield can be measured per prompt rev.
    map_nudged = False
    if handles is not None and final.strip() and protocol == "ok":
        from geogym_blogq.lab.core.gates import is_v2, unfence
        final = unfence(final)
        if not is_v2(final):
            async with sem:
                nresp = await llm.chat.completions.create(
                    model=args.model,
                    messages=msgs + [{"role": "assistant", "content": final},
                                     {"role": "user", "content":
                                      "Your answer is missing its last line. Output ONLY the JSON object "
                                      "{\"map\": [{\"ref\": \"<ref>\"}, ...]} for the answer above: the ref of the "
                                      "place it is about first (a ref from your search results), then other "
                                      "places it names. No prose, no code fence."}],
                    temperature=0.0, max_tokens=200,
                    extra_body=ROLLOUT_EXTRA)
            ntext = (nresp.choices[0].message.content or "").strip()
            m_ = re.search(r"\{.*\}", ntext, re.S)
            if m_:
                try:
                    obj = json.loads(m_.group(0))
                    if isinstance(obj, dict) and "map" in obj:
                        final = final.rstrip() + "\n" + json.dumps(obj, ensure_ascii=False)
                        map_nudged = True
                except json.JSONDecodeError:
                    pass
        elif os.environ.get("BENCH_TRAVEL_NUDGE", "1") != "0":
            # 2026-09-09 (mined100: 13 of 100 answers dropped the travel node): a journey answer whose
            # object has no travel node, while the trace holds a successful calculate_route, gets ONE
            # tool-less repair turn for the object alone (format repair, appended; the turn is not kept).
            from geogym_blogq.lab.core.gates import map_object
            mm_, obj_ = map_object(final)
            has_travel = any(isinstance(e, dict) and e.get("travel") for e in ((obj_ or {}).get("map") or []))
            route_ok_calls = [m_ for m_ in msgs if m_.get("role") == "tool" and '"duration_seconds"' in str(m_.get("content") or "")]
            if mm_ and not has_travel and route_ok_calls and len(((obj_ or {}).get("map") or [])) >= 1:
                async with sem:
                    nresp = await llm.chat.completions.create(
                        model=args.model,
                        messages=msgs + [{"role": "assistant", "content": final},
                                         {"role": "user", "content":
                                          "The JSON object on your last line is missing the travel node for the route you "
                                          "computed. Output ONLY the corrected object as a journey line — "
                                          "{\"map\": [{\"ref\": \"<from>\"}, {\"travel\": \"<walk|bike|car|bus>\", \"route\": \"r<k>\"}, "
                                          "{\"ref\": \"<to>\"}]} (or {\"here\": true} for your own position as the first node), using the "
                                          "handle [r<k>] of the calculate_route result your numbers came from. No prose, no code fence."}],
                        temperature=0.0, max_tokens=200, extra_body=ROLLOUT_EXTRA)
                ntext = (nresp.choices[0].message.content or "").strip()
                m_ = re.search(r"\{.*\}", ntext, re.S)
                if m_:
                    try:
                        obj = json.loads(m_.group(0))
                        if isinstance(obj, dict) and any(isinstance(e, dict) and e.get("travel") for e in (obj.get("map") or [])):
                            final = unfence(final).rsplit("\n", 1)[0].rstrip() + "\n" + json.dumps(obj, ensure_ascii=False)
                            map_nudged = True
                    except json.JSONDecodeError:
                        pass

    mech = mech_stats(q["id"], trace, final)
    # gold-anchored replay: when the model never surfaced gold, rerun its own
    # query text FROM the gold coords — separates anchor-placement failures
    # (query fine, wrong spot / outside the dynamic radius) from query failures
    # (can't retrieve gold even from the perfect anchor = phrasing or a wplaces
    # recall gap)
    if mech.get("gold_surfaced") is False:
        mech["query_at_gold"] = await replay_at_gold(http, q["id"], trace)

    # ---- judge
    gold = {"expected_places": q["expected_places"],
            "answer_notes": q["answer_notes"]}
    jmsgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content":
            f"QUESTION ({q['category']}, {q['region']}): {user}\n\n"
            f"GOLD: {json.dumps(gold, ensure_ascii=False)}\n\n"
            f"AGENT TOOL ACTIVITY:\n"
            + "\n".join(f"- {t['tool']}({json.dumps(t['args'], ensure_ascii=False)}) "
                        f"-> {t['result_head'][:200]}" for t in trace)
            + f"\n\nAGENT FINAL ANSWER:\n{final or '(empty)'}"},
    ]
    if NO_INLINE_JUDGE:
        jtext = '{"correct": false, "reason": "inline judge skipped (BENCH_NO_JUDGE)"}'
    else:
        async with sem:
            jresp = await judge_llm.chat.completions.create(
                model=args.judge_model, messages=jmsgs,
                temperature=0.0, max_tokens=800,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        jtext = jresp.choices[0].message.content or ""
    try:
        jstart = jtext.index("{")
        verdict = json.loads(jtext[jstart:jtext.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        m = re.search(r'"correct"\s*:\s*(true|false)', jtext)
        if m:
            verdict = {"correct": m.group(1) == "true",
                       "reason": "regex-recovered: " + jtext[:150]}
        else:
            verdict = {"correct": False,
                       "reason": f"unparseable judge: {jtext[:120]}"}

    # deterministic override (Neuschwanstein rule): a decline on a question
    # whose gold place is findable is wrong, whatever the judge said
    grefs = GOLD_REFS.get(q["id"], [])
    if verdict.get("correct") and q.get("category") != "negative" and grefs \
            and DECLINE_RE.search(final or "") \
            and not any(g in (final or "") for g in grefs):
        verdict = {"correct": False,
                   "reason": "OVERRIDE declined-findable: " + str(verdict.get("reason", ""))[:150]}
    return {"id": q["id"], "category": q["category"],
            "difficulty": q["difficulty"],
            "correct": bool(verdict.get("correct")),
            "judge_reason": verdict.get("reason"),
            "final": final[:4000], "n_tools": len(trace),
            "asks": [{"q": t["args"].get("question"), "reply": t["result_head"][:400],
                      "leak_blocked": bool(t["args"].get("leak_blocked"))}
                     for t in trace if t["tool"] == "ask_user"],
            "messages": msgs + ([{"role": "assistant", "content": final}]
                                if final else []),
            "tools": [f"{t['tool']}:{t['args'].get('query', t['args'].get('refs'))}"
                      for t in trace],
            "mech": mech,
            "trace": [{"tool": t["tool"],
                       "query": t["args"].get("query"),
                       "url": t["args"].get("url"),
                       "lat": t["args"].get("lat"), "lng": t["args"].get("lng"),
                       "drefs": t["args"].get("refs"),
                       "anchor_km": t.get("anchor_km"),
                       "refs": t.get("refs", [])[:6],
                       "urls": re.findall(r"https?://\S+",
                                          t["result_head"])[:8],
                       "head": t["result_head"][:12000
                               if t["tool"] in ("web_fetch", "get_full_page")
                               else 1200]} for t in trace],
            "protocol": protocol, "capped": capped, "usage": usage, "turn2": turn2,
            "contract": CONTRACT, "map_nudged": map_nudged,
            "resumed": resumed,
            "handles": handles.table() if handles is not None else None,
            "compute_handles": chandles.table() or None,
            "infra_errors": sum(1 for t in trace
                                if str(t.get("result_head") or "").startswith(INFRA_ERR))}


async def main_async(args):
    import httpx
    from openai import AsyncOpenAI

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                done.add(json.loads(line)["id"])
            except json.JSONDecodeError:
                pass
    from geogym_blogq.lab.core.bank import load_bank as _load_bank
    qs = _load_bank(QFILE)
    qs = [q for q in qs if q["id"] not in done]
    if args.resume_from:
        # continue stored rollouts that were killed on a budget overrun of the
        # named tool: the stored prefix is replayed verbatim (system prompt
        # swapped for the current one), the killed call is executed under the
        # budget now enforced, and the model finishes the rollout.
        kill = {"web_search": ("web_search", "search_travel_blogs"),
                "search_places": ("search_places",)}[args.resume_kill]
        stored = {}
        for p in args.resume_from:
            for line in open(p):
                r = json.loads(line)
                tr = r.get("trace") or []
                if r.get("protocol") == "over_budget" and tr and tr[-1]["tool"] in kill \
                        and str(tr[-1].get("head") or "").startswith(TERMINATED):
                    stored[r["id"]] = r
        qs = [q for q in qs if q["id"] in stored]
        for q in qs:
            q["_resume"] = stored[q["id"]]
        print(f"resume: {len(stored)} stored rollouts killed on {args.resume_kill}; "
              f"{len(qs)} of them in the bank and not yet done")
    if args.limit:
        qs = qs[:args.limit]
    print(f"{len(qs)} questions to run ({len(done)} already done)")

    llm = AsyncOpenAI(
        base_url=args.base_url, api_key=args.api_key,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=args.concurrency,
                max_keepalive_connections=args.concurrency),
            timeout=httpx.Timeout(600.0, connect=15.0)))
    judge_llm = AsyncOpenAI(base_url=args.judge_base_url,
                            api_key=args.judge_api_key or args.api_key)
    global _ask_llm, _ask_model, ROLLOUT_EXTRA
    if "openrouter" in args.base_url:
        # OpenRouter ignores chat_template_kwargs. Default: let the model think (the teacher only uses ask_user
        # reliably with thinking on — runs/ask_smoke/README.md); BENCH_OR_NO_REASONING=1 turns it off.
        ROLLOUT_EXTRA = {"reasoning": {"enabled": False}} if os.environ.get("BENCH_OR_NO_REASONING") else {}
    _ask_llm, _ask_model = judge_llm, args.judge_model
    if os.environ.get("BENCH_ASK_SIM_SECTION"):
        import tomllib as _toml
        _sec = _EP.get(os.environ["BENCH_ASK_SIM_SECTION"]) or \
            _toml.load(open(HERE.parent / "config/endpoints.local.toml", "rb"))[os.environ["BENCH_ASK_SIM_SECTION"]]
        _ask_llm = AsyncOpenAI(base_url=_sec["base_url"],
                               api_key=_sec.get("api_key") or os.environ.get(_sec.get("api_key_env", ""), "") or "EMPTY")
        _ask_model = _sec["model"]
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=15.0),
            limits=httpx.Limits(max_connections=256,
                                max_keepalive_connections=64)) as http:
        with open(out, "a") as sink:
            # Stream: write each rollout the moment it finishes. The old
            # gather-per-batch waited for all 768 in a batch, so a live run
            # showed nothing for ~16 min and one slow rollout stalled 767
            # others. The semaphore already bounds concurrency.
            done_n = [0]

            async def one(q):
                try:
                    row = await run_question(llm, judge_llm, http, q, args, sem,
                                             resume=q.get("_resume"))
                except Exception as e:  # noqa: BLE001
                    print(f"  ERROR {q['id']}: {e}", file=sys.stderr)
                    return
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                done_n[0] += 1
                if done_n[0] % 50 == 0:
                    sink.flush()
                if done_n[0] % 200 == 0:
                    print(f"{done_n[0]}/{len(qs)}", flush=True)

            CH = max(args.concurrency * 8, 400)
            for i in range(0, len(qs), CH):
                await asyncio.gather(*[one(q) for q in qs[i:i + CH]])
                sink.flush()
            sink.flush()
            print(f"{done_n[0]}/{len(qs)}", flush=True)


def summarize(path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    if not n:
        sys.exit("no rows")
    acc = sum(r["correct"] for r in rows) / n
    print(f"{path}: n={n}  accuracy={acc:.3f}")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        print(f"  {cat:18} n={len(sub):3}  acc={sum(r['correct'] for r in sub)/len(sub):.3f}")
    for d in ["easy", "medium", "hard"]:
        sub = [r for r in rows if r["difficulty"] == d]
        if sub:
            print(f"  [{d:6}] n={len(sub):3}  acc={sum(r['correct'] for r in sub)/len(sub):.3f}")
    tools = sum(r["n_tools"] for r in rows) / n
    proto = Counter(r["protocol"] for r in rows)
    print(f"  tools/q={tools:.2f}  protocol={dict(proto)}")
    # negatives excluded: anchoring at the wrong-premise location is correct
    mechs = [r["mech"] for r in rows
             if r.get("mech") and r.get("category") != "negative"]
    if mechs:
        anch = [m["anchor_first_km"] for m in mechs
                if m.get("anchor_first_km") is not None]
        surf = [m["gold_surfaced"] for m in mechs
                if m.get("gold_surfaced") is not None]
        named = [m["gold_named"] for m in mechs
                 if m.get("gold_named") is not None]
        churn = sum(1 for m in mechs if m.get("q_repeat", 0) >= 3)
        anch.sort()
        if anch:
            print(f"  FUNNEL anchor: median {anch[len(anch)//2]:.1f} km, "
                  f"≤40km {sum(1 for a in anch if a <= 40)/len(anch):.0%} "
                  f"(n={len(anch)})")
        if surf:
            print(f"  FUNNEL surfaced {sum(surf)/len(surf):.0%} (n={len(surf)}) "
                  f"→ named {sum(named)/max(1,len(named)):.0%} (n={len(named)}) "
                  f"· churn(≥3 same query) {churn}")
        qg = [m.get("query_at_gold") for m in mechs
              if m.get("query_at_gold") is not None]
        if qg:
            print(f"  FUNNEL unsurfaced split: {sum(qg)}/{len(qg)} anchor-"
                  f"placement (query works @gold), {len(qg)-sum(qg)} "
                  f"query/recall failures")


if __name__ == "__main__":
    import faulthandler, signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)   # kill -USR1 <pid> dumps every stack to stderr
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summarize", metavar="RESULTS_JSONL")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default=None)
    # default judge: cherry's free local Qwen3.6-35B (was google/gemini-3.6-flash
    # via OpenRouter — pass those explicitly to reproduce the older judged runs)
    ap.add_argument("--judge-model", default="Qwen/Qwen3.6-35B-A3B-FP8")
    ap.add_argument("--judge-base-url", default="http://100.105.93.91:8000/v1")
    ap.add_argument("--judge-api-key", default=os.environ.get("OPENROUTER_API_KEY") or "EMPTY")
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tools", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume-from", nargs="+", metavar="ROWS_JSONL",
                    help="continue stored over_budget rollouts from their killed call "
                         "(bank = AGENTQ_FILE as usual; only ids present in both run)")
    ap.add_argument("--resume-kill", choices=["web_search", "search_places"], default="web_search",
                    help="which budget overrun the stored rows died on")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        if not (args.model and args.out):
            ap.error("--model and --out required")
        asyncio.run(main_async(args))
