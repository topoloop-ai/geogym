"""Shared routing tool adapters (v2 toolset) — thin clients over wplaces'
shape/routing compositions (wplaces/ROUTING_COMPOSITIONS.md, 2026-09-06) and
Valhalla via the harness's prod-shaped route call.

    reachable(http, args)      search_places(..., within_minutes, profile) -> /places/reachable
    along_route(http, args)    search_along_route(route_id, query, max_detour_min) -> /places/along_route
    parse_route_id(route_id)   the harness's route_id encodes from/to/mode, so no server-side store

Never import harness state here; the harness passes its client and reads
config the same way (`core.config.EP["wplaces"]`). Environments compose these
and never fork them (README: contribution contract).
"""
import json
import re

from geogym_blogq.lab.core.config import EP

WPLACES = EP.get("wplaces", {}).get("base_url", "http://127.0.0.1:8001")
WPLACES_KEY = EP.get("wplaces", {}).get("api_key", "wplaces-local-dev-key")
PROFILE = {"walk": "walking", "walking": "walking", "pedestrian": "walking",
           "bike": "cycling", "cycling": "cycling", "bicycle": "cycling",
           "car": "driving", "drive": "driving", "driving": "driving", "auto": "driving",
           "taxi": "driving", "motorcycle": "driving", "bus": "driving"}
ROUTE_ID_RE = re.compile(r"^route_(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)_(-?[\d.]+)_([a-z]+)$")
MIN_SIM = 0.45          # relevance floor: an honest empty set beats an optician for "viewpoint"
# Strict place TYPES go through the canonical taxonomy (a free-text "beach" at the
# 0.45 floor admitted "Big Blue Spa" and "Yogurt Beach"); everything else is free text.
STRICT = {
    "beach": ["beach"], "beaches": ["beach"], "museum": ["museum", "art_gallery"], "museums": ["museum", "art_gallery"],
    "gallery": ["art_gallery"], "art gallery": ["art_gallery"], "park": ["park", "garden"], "parks": ["park", "garden"],
    "garden": ["garden"], "viewpoint": ["viewpoint"], "viewpoints": ["viewpoint"], "lookout": ["viewpoint"],
    "church": ["place_of_worship"], "temple": ["place_of_worship"], "mosque": ["place_of_worship"], "shrine": ["place_of_worship"],
    "church or temple": ["place_of_worship"], "market": ["market"], "markets": ["market"], "supermarket": ["supermarket"],
    "grocery store": ["supermarket"], "grocery": ["supermarket"], "bakery": ["bakery"], "bakeries": ["bakery"],
    "ice cream": ["ice_cream"], "gelato": ["ice_cream"], "bar": ["bar", "pub"], "bars": ["bar", "pub"], "pub": ["pub", "bar"],
    "brewery": ["winery_brewery"], "winery": ["winery_brewery"],
}


def resolve_query(q):
    """-> (free_text or None, categories or None). Strips a leading article."""
    t = re.sub(r"^(a|an|the|some)\s+", "", (q or "").strip().lower())
    if t in STRICT:
        return None, STRICT[t]
    return (q or "").strip() or None, None


def parse_route_id(route_id):
    """route_<from_lng>_<from_lat>_<to_lng>_<to_lat>_<mode> -> (from[lng,lat], to[lng,lat], mode)."""
    m = ROUTE_ID_RE.match(str(route_id or ""))
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2))], [float(m.group(3)), float(m.group(4))], m.group(5)


def _ref(c):
    oid = str(c.get("osm_id"))
    return oid if oid[:1].isalpha() else {"node": "N", "way": "W", "relation": "R"}.get(c.get("osm_type"), "?") + oid


def _fmt(c, extra):
    return (f"{_ref(c)} | {c.get('name_en') or c.get('name')} | {c.get('primary_category') or c.get('category')} | "
            + extra + _cx(c) + f" | ({round(c.get('lat', 0), 5)}, {round(c.get('lng', 0), 5)})")


# ---- attribute constraints (2026-09-10, RL_DESIGN §13): the filter lives in wplaces (search/constraints.py);
# the adapter forwards open_at / wheelchair / free and prints the tri-state evidence per result line.
def constraints_from(args):
    cx = {}
    if args.get("open_at"):
        cx["open_at"] = str(args["open_at"]).strip()
    if args.get("wheelchair"):
        cx["wheelchair"] = True
    if args.get("free"):
        cx["free"] = True
    return cx or None


def _cx(c):
    """' | open 05:00-22:00' / ' | closed (opens 07:30)' / ' | hours unknown' (+ wheelchair / fee) when a constraint ran."""
    bits = []
    if "open" in c and (c.get("open") is not None or c.get("opening_hours") is not None or "opens_at" in c):
        if c.get("open") is True:
            bits.append(f"open {c.get('opens_at')}-{c.get('closes_at')}")
        elif c.get("open") is False:
            bits.append(f"closed (opens {c.get('opens_at')})" if c.get("opens_at") else "closed that day")
        else:
            bits.append("hours unknown")
    if "wheelchair_ok" in c:
        bits.append(f"wheelchair {c.get('wheelchair') or 'unknown'}")
    if "free_ok" in c:
        bits.append(f"fee {c.get('fee') or 'unknown'}")
    return (" | " + " · ".join(bits)) if bits else ""


def origin_text(o):
    """One line the model reads and the verifier parses:
    ORIGIN: <ref or coordinates> | <name> | (<lat>, <lng>) [| routed from a road point N m away] [| AMBIGUOUS: alt1; alt2]"""
    if not o:
        return "ORIGIN: (unresolved)"
    ref = o.get("ref") or "coordinates"
    name = o.get("name") or ""
    loc = o.get("locality") or ""
    line = f"ORIGIN: {ref} | {name}{(' (' + loc + ')') if loc else ''} | ({round(o.get('lat', 0), 5)}, {round(o.get('lng', 0), 5)})"
    rf = o.get("routed_from") or {}
    sd = rf.get("snap_distance_m", rf.get("distance_m"))
    if sd is not None:
        line += f" | routed from a road point {sd:.0f} m away"
    if o.get("ambiguous"):
        alts = "; ".join(f"{a_.get('name')} ({a_.get('ref')})" for a_ in (o.get("alternatives") or [])[:3])
        line += f" | AMBIGUOUS — could also be: {alts}"
    return line


async def reachable(http, args):
    """search_places with within_minutes: places reachable within a travel-time
    budget. Renders like a search result plus travel minutes; the empty case
    names the nearest unreachable place with its real minutes."""
    body = {"within_minutes": float(args["within_minutes"]),
            "navigation_profile": PROFILE.get(str(args.get("profile") or "walking").lower(), "walking"),
            "limit": int(args.get("limit") or 8), "rank": "popularity"}
    # origin: a ref already seen, a name resolved near (lat, lng), or raw coordinates
    if args.get("origin_ref"):
        body["origin_ref"] = str(args["origin_ref"]).strip()
    elif args.get("origin_name"):
        body["origin_name"] = str(args["origin_name"]).strip()
        body["origin_near"] = [float(args["lng"]), float(args["lat"])]
    else:
        body["lat"], body["lng"] = float(args["lat"]), float(args["lng"])
    q = (args.get("query") or "").strip()
    qt, cats = resolve_query(q)
    if qt:
        body["query"], body["min_similarity"] = qt, MIN_SIM
    if cats:
        body["categories"] = cats
    cx = constraints_from(args)
    if cx:
        body["constraints"] = cx
    r = await http.post(f"{WPLACES}/places/reachable", json=body,
                        headers={"X-API-Key": WPLACES_KEY}, timeout=90)
    r.raise_for_status()
    d = r.json()
    lines = [_fmt(c, f"{c.get('eta_min', 0):.0f} min {body['navigation_profile']} · {c.get('route_distance_m', 0):.0f} m")
             for c in d.get("results", [])]
    forq = f' for "{q}"' if q else ""
    o = d.get("origin") or {}
    origin_line = origin_text(o)
    head = (f"{origin_line}\nREACHABLE within {body['within_minutes']:g} min {body['navigation_profile']} of that origin{forq}: "
            f"{d.get('count', 0)} shown of {d.get('candidates_total', d.get('candidates', 0))} in range")
    if d.get("constraint_stats"):
        st = d["constraint_stats"]
        head += (f"\nCONSTRAINT {json.dumps(d.get('constraints'))}: {st.get('matched', 0)} satisfy, {st.get('unknown', 0)} unknown "
                 f"(shown after), {st.get('excluded', 0)} excluded — of {st.get('evaluated', 0)} in range")
    if not lines:
        nu = d.get("nearest_unreachable")
        if nu:
            eta = nu.get("eta_min", 0)
            lines = ["(none within budget) nearest: " + _fmt(nu, f"{eta:.0f} min")]
        else:
            lines = ["(none within budget)"]
    refs = [_ref(c) for c in d.get("results", [])]
    if o.get("ref"):
        refs = [o["ref"]] + refs
    return head + "\n" + "\n".join(lines), refs


async def along_route(http, args):
    """search_along_route(route_id, query, max_detour_min): places on the way
    whose detour adds at most max_detour_min, in travel order."""
    parsed = parse_route_id(args.get("route_id"))
    if not parsed:
        return "route error: unknown route_id (use the route_id returned by calculate_route)", []
    frm, to, mode = parsed
    body = {"waypoints": [frm, to], "navigation_profile": PROFILE.get(mode, "walking"),
            "time_deviation": float(args.get("max_detour_min") or 5), "limit": int(args.get("limit") or 8)}
    q = (args.get("query") or "").strip()
    qt, cats = resolve_query(q)
    if qt:
        body["query"], body["min_similarity"] = qt, MIN_SIM
    if cats:
        body["categories"] = cats
    cx = constraints_from(args)
    if cx:
        body["constraints"] = cx
    r = await http.post(f"{WPLACES}/places/along_route", json=body,
                        headers={"X-API-Key": WPLACES_KEY}, timeout=90)
    r.raise_for_status()
    d = r.json()
    lines = [_fmt(c, f"at {c.get('eta_min', 0):.0f} min · +{c.get('added_time_min', 0):.0f} min detour")
             for c in d.get("results", [])]
    rt = d.get("route") or {}
    forq = f' for "{q}"' if q else ""
    head = (f"ALONG ROUTE {args.get('route_id')} ({body['navigation_profile']}, "
            f"{rt.get('duration_s', 0) / 60:.0f} min direct){forq}, "
            f"detour ≤ {body['time_deviation']:g} min: {d.get('count', 0)} shown of {d.get('candidates', 0)} candidates, in travel order")
    if d.get("constraint_stats"):
        st = d["constraint_stats"]
        head += (f"\nCONSTRAINT {json.dumps(d.get('constraints'))}: {st.get('matched', 0)} satisfy, {st.get('unknown', 0)} unknown "
                 f"(shown after), {st.get('excluded', 0)} excluded — of {st.get('evaluated', 0)} on the way")
    if not lines:
        lines = ["(nothing on the way within the detour budget)"]
    refs = [_ref(c) for c in d.get("results", [])]
    return head + "\n" + "\n".join(lines), refs


# ---- gold-side helpers (generation / verification use the raw responses)
async def reachable_raw(http, lat, lng, minutes, profile, query=None, limit=100, mode="isochrone", constraints=None):
    body = {"lat": lat, "lng": lng, "within_minutes": minutes, "navigation_profile": profile,
            "limit": limit, "mode": mode, "rank": "popularity"}
    if constraints:
        body["constraints"] = constraints
    qt, cats = resolve_query(query)
    if qt:
        body["query"], body["min_similarity"] = qt, MIN_SIM
    if cats:
        body["categories"] = cats
    r = await http.post(f"{WPLACES}/places/reachable", json=body, headers={"X-API-Key": WPLACES_KEY}, timeout=120)
    r.raise_for_status()
    return r.json()


async def along_route_raw(http, frm, to, profile, deviation, query=None, limit=100, constraints=None):
    body = {"waypoints": [frm, to], "navigation_profile": profile, "time_deviation": deviation, "limit": limit}
    if constraints:
        body["constraints"] = constraints
    qt, cats = resolve_query(query)
    if qt:
        body["query"], body["min_similarity"] = qt, MIN_SIM
    if cats:
        body["categories"] = cats
    r = await http.post(f"{WPLACES}/places/along_route", json=body, headers={"X-API-Key": WPLACES_KEY}, timeout=120)
    r.raise_for_status()
    return r.json()
