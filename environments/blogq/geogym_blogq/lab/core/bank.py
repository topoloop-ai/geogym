"""Question banks: the lab format (expected_places, gold_refs, gold_coords, meta) and the
published Ask Maps Basic format (gold, source). Everything loads through `load_bank`."""
import json


def normalize(q):
    """Return a question row in the lab format, converting the published format if needed."""
    if "expected_places" in q or "gold" not in q:
        return q
    g = q.get("gold") or {}
    src = q.get("source") or {}
    refs = [r for r in (g.get("refs") or []) if r]
    coords = {r: [g["lat"], g["lng"]] for r in refs if g.get("lat") is not None and g.get("lng") is not None}
    out = dict(q)
    out.update({
        "difficulty": q.get("difficulty", "medium"),
        "region": q.get("region"),
        "anchor": q.get("anchor"),
        "expected_places": [g["name"]] if g.get("name") else [],
        "expected_tools": None,
        "answer_notes": q.get("answer_notes") or (f"Grounded in blog '{src.get('title', '')}'" if src else ""),
        "gold_refs": refs,
        "gold_fields": q.get("gold_fields") or {},
        "gold_coords": coords,
        "meta": {"name": g.get("name"), "type": g.get("type"), "gold_lat": g.get("lat"), "gold_lng": g.get("lng"),
                 "src_url": src.get("url"), "url": src.get("url"), "title": src.get("title"),
                 "snippet": src.get("snippet"), "chunk": src.get("chunk") or src.get("snippet")},
    })
    return out


def load_bank(path):
    with open(path) as f:
        return [normalize(json.loads(l)) for l in f if l.strip()]
