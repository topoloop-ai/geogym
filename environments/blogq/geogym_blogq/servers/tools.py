"""The five Wanderfugl tools as a SHARED verifiers toolset (one process per worker; per-rollout
state = budgets + handle table + refs seen). Result strings are byte-identical to core/harness.py's
so the policy sees exactly what it was trained on."""
import json, re
import asyncio
import os
import httpx
import verifiers.v1 as vf
from geogym_blogq import common  # noqa: F401  (sys.path + env)
from geogym_blogq.lab.core import harness as H


# 2026-09-17 REWARD_HACKS §24 fix (GEOGYM_CAP_ANSWER=1): once a search budget is hit, EVERY further tool call is refused with
# the same "answer now" message and reward.py scores the answer that follows normally (protocol ok, row["capped"]=True).
# Off (default) = the pre-round-18 behaviour: the answer after the cap message is scored 0 regardless of content.
CAP_ANSWER = os.environ.get("GEOGYM_CAP_ANSWER", "0") not in ("", "0")
# 2026-09-18 round 20: a TOTAL tool-call ceiling (0 = off). Round 19 showed that with CAP_ANSWER the per-tool cap is free and gets
# farmed (cap rate 8% → 24% in one round); the ceiling bounds it by construction. Same refusal message, so no observation shift.
# Mirrors core/harness.py --max-tools (len(trace) >= max_tools → refuse, close tools, answer).
MAX_TOOLS = int(os.environ.get("GEOGYM_MAX_TOOLS", "0") or 0)


class GeogymState(vf.State):
    n_search: int = 0
    n_web: int = 0
    n_fetch: int = 0
    over_budget: bool = False
    # round 7d (2026-09-12): a fresh allowance granted by ClarifyEnv when the traveller reply turn happens — the
    # continuation gets room without raising the base budget for every episode (5/3/2 taught a persistence tail).
    extra_search: int = 0
    extra_web: int = 0
    extra_fetch: int = 0
    handles: dict[str, dict] = {}          # url -> {handle, title, site}
    n_handles: int = 0
    refs_seen: list[str] = []
    trace: list[dict] = []                 # harness-shaped trace entries (tool, args, refs, result_head)


def _handles(state):
    h = H.Handles(); h.by_url = dict(state.handles); h.n = state.n_handles
    return h


def _store(state, h):
    state.handles = dict(h.by_url); state.n_handles = h.n


class _BoundedClient(httpx.AsyncClient):
    """httpx client whose in-flight request count is capped by an asyncio.Semaphore (FIFO), not by the pool."""

    def __init__(self, concurrency: int, **kw):
        kw.setdefault("limits", httpx.Limits(max_connections=max(concurrency, 256),
                                             max_keepalive_connections=max(concurrency, 256)))
        super().__init__(**kw)
        self._sem = asyncio.Semaphore(concurrency)

    async def send(self, *a, **kw):
        async with self._sem:
            return await super().send(*a, **kw)


class GeogymToolset(vf.Toolset[vf.SharedToolsetConfig, GeogymState]):
    TOOL_PREFIX = None   # bare names: the policy was trained on search_places, web_search, ...

    async def setup(self) -> None:
        # 2026-09-13: httpx's pool degrades badly once thousands of requests queue for a bounded pool (measured: 2,048
        # requests through 256 connections = 225 s; through 2,048 connections = 47 s; the servers were idle either way).
        # A step's 1,024 rollouts share this client, so bound in-flight requests with a FIFO semaphore INSTEAD of the pool
        # (GEOGYM_HTTP_CONCURRENCY, default 192 — wplaces @300 km serves ~60-90 qps on 8 workers, wsearch/wiki ~50-100),
        # and give the pool at least that many connections so nothing ever waits inside httpcore. Keep-alive on the
        # servers is 300 s so pooled connections survive model-generation gaps.
        self.http = _BoundedClient(int(os.environ.get("GEOGYM_HTTP_CONCURRENCY", "192")),
                                   timeout=httpx.Timeout(60.0, connect=10.0))

    def _capped(self, tool, args):
        s = self.state
        if MAX_TOOLS and len(s.trace) >= MAX_TOOLS and not s.over_budget:
            s.over_budget = True
            return self._log(tool, args, "Tool budget exhausted — answer now from what you have.")
        if CAP_ANSWER and s.over_budget:
            return self._log(tool, args, "Tool budget exhausted — answer now from what you have.")
        return None

    def _log(self, tool, args, result, refs=None):
        s = self.state
        s.trace = s.trace + [{"tool": tool, "args": args, "refs": list(refs or []), "result_head": result[:1200]}]
        if refs:
            s.refs_seen = s.refs_seen + [r for r in refs if r not in s.refs_seen]
        return result

    @vf.tool
    async def search_places(self, query: str, lat: float, lng: float) -> str:
        """Search the places database near an anchor coordinate you supply. Radius is automatic."""
        if (_c := self._capped("search_places", {k: v for k, v in locals().items() if k != "self"})) is not None:
            return _c
        s = self.state
        if s.n_search >= common.BUDGETS["search_places"] + s.extra_search:
            s.over_budget = True
            return self._log("search_places", {"query": query, "lat": lat, "lng": lng}, "Tool budget exhausted — answer now from what you have.")
        s.n_search += 1
        try:
            if not (query or "").strip():
                lines = await H.wp_nearest(self.http, float(lat), float(lng))
                refs = [ln.split(" | ")[0] for ln in lines]
                result = f"NEAREST PLACES (around {lat}, {lng}):\n" + "\n".join(lines) if lines else "(no places found)"
            else:
                res = await H.wp_search(self.http, query, float(lat), float(lng))
                refs = [c["ref"] for c in res]
                ctx = await H.wp_context(self.http, res) if H.PLACE_CONTEXT else None
                result = f"SEARCH RESULTS (query: \"{query}\", anchor: {lat}, {lng}):\n{H.fmt_results(res, ctx)}"
        except Exception as e:  # noqa: BLE001
            result, refs = f"search error: {e}", []
        return self._log("search_places", {"query": query, "lat": lat, "lng": lng}, result, refs)

    @vf.tool
    async def nearest_places(self, lat: float, lng: float) -> str:
        """List the closest places around a coordinate, fame-blended so notable landmarks surface among nearby minor spots. Results carry distance and compass bearing from the query point."""
        if (_c := self._capped("nearest_places", {k: v for k, v in locals().items() if k != "self"})) is not None:
            return _c
        s = self.state
        if s.n_search >= common.BUDGETS["search_places"] + s.extra_search:
            s.over_budget = True
            return self._log("nearest_places", {"lat": lat, "lng": lng}, "Tool budget exhausted — answer now from what you have.")
        s.n_search += 1
        try:
            lines = await H.wp_nearest(self.http, float(lat), float(lng))
            refs = [ln.split(" | ")[0] for ln in lines]
            result = f"NEAREST PLACES (around {lat}, {lng}):\n" + "\n".join(lines) if lines else "(no places found)"
        except Exception as e:  # noqa: BLE001
            result, refs = f"nearest error: {e}", []
        return self._log("nearest_places", {"lat": lat, "lng": lng}, result, refs)

    @vf.tool
    async def place_details(self, refs: list[str]) -> str:
        """Fetch contact/hours/fee/accessibility/wikipedia/blog details for up to 5 refs from earlier search results."""
        if (_c := self._capped("place_details", {k: v for k, v in locals().items() if k != "self"})) is not None:
            return _c
        s = self.state
        want = [str(x) for x in (refs or [])][:5]
        seen = set(s.refs_seen) | set(re.findall(r"\b[NWR]\d{4,}\b", "\n".join(t.get("result_head", "") for t in s.trace)))
        known = [x for x in want if x in seen]
        try:
            det = await H.wp_details(self.http, known) if known else {}
        except Exception as e:  # noqa: BLE001
            det = {"error": f"details error: {e}"}
        for x in want:
            if x not in seen:
                det[x] = "ERROR: unknown ref — not in any of your search results"
        return self._log("place_details", {"refs": want}, json.dumps(det, ensure_ascii=False), known)

    @vf.tool
    async def web_search(self, query: str, limit: int = 5) -> str:
        """Curated web search over high-quality travel writing worldwide. Each result carries a handle like [w2], its title, a one-line summary and the matching passage from the page."""
        if (_c := self._capped("web_search", {k: v for k, v in locals().items() if k != "self"})) is not None:
            return _c
        s = self.state
        if s.n_web >= common.BUDGETS["web_search"] + s.extra_web:
            s.over_budget = True
            return self._log("web_search", {"query": query}, "Tool budget exhausted — answer now from what you have.")
        s.n_web += 1
        h = _handles(s)
        try:
            result = await H.ws_search(self.http, query, int(limit or 5), handles=h)
        except Exception as e:  # noqa: BLE001
            result = f"blog search error: {e}"
        _store(s, h)
        return self._log("web_search", {"query": query, "limit": limit}, result)

    @vf.tool
    async def web_fetch(self, url: str) -> str:
        """Read the full text of a page from web_search when its passage is not enough. Pass the page's URL exactly as given by the handle table in the result."""
        if (_c := self._capped("web_fetch", {k: v for k, v in locals().items() if k != "self"})) is not None:
            return _c
        s = self.state
        if s.n_fetch >= common.BUDGETS["web_fetch"] + s.extra_fetch:
            return self._log("web_fetch", {"url": url}, "Fetch budget exhausted — answer now from what you have.")
        s.n_fetch += 1
        h = _handles(s)
        try:
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(None, H.ws_page, str(url))
            p = h.by_url.get(str(url))
            if p and str(url).startswith("http"):
                result = f"[{p['handle']}] {p.get('title') or url}\n{result}"
        except Exception as e:  # noqa: BLE001
            result = f"page error: {e}"
        _store(s, h)
        entry = {"tool": "web_fetch", "args": {"url": url}, "url": str(url), "refs": [], "result_head": result[:1200]}
        s.trace = s.trace + [entry]
        return result


if __name__ == "__main__":
    GeogymToolset.run()
