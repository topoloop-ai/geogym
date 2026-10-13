#!/usr/bin/env python3
"""The citation compiler — ONE canonical parser/validator for the citation
contract (CITATIONS.md). Every consumer (judge mechanical layer, launder
enforcement gates, keep gate, eval validity metrics) imports THIS instead
of hand-rolling regexes; scattered checks are how type gaps happen
(2026-09-03: url-in-ref-field passed refs_ok for weeks of machinery).

Grammar:
  markers  := [n] | [n, m, ...]        (grouped markers are VALID — blessed
                                        2026-09-03, they read better)
  SOURCES  := last line: SOURCES: [ {"ref": "<NWR id>"} | {"url": "<http…>"} , ... ]
  ref      := ^[NWR]\\d+$   (places DB)   url := ^https?://
  every marker resolves 1-based into SOURCES; every entry referenced by
  >=1 marker; no duplicate entries; entries appear verbatim in the trace
  (when trace text provided).

compile_final(text, trace_text=None) -> (Compiled, errors: list[str])
Compiled: body, markers=[(number, char_pos)], sources=[{"type","value"}].
errors empty == compiles clean. normalize(text) fixes the MECHANICAL
repairs that need no judgment: url-in-ref key, duplicate entries.
"""
import json
import re

MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# any bracket whose contents include an NWR id or a url (lone or mixed)
PSEUDO_RE = re.compile(
    r"\[\s*((?:[NWR]\d{4,}|https?://[^\s\],]+)"
    r"(?:\s*,\s*(?:[NWR]\d{4,}|https?://[^\s\],]+))*)\s*\]")
REF_RE = re.compile(r"^[NWR]\d+$")
URL_RE = re.compile(r"^https?://\S+$")
SOURCES_RE = re.compile(r"^SOURCES:\s*(\[.*\])\s*$", re.M | re.S)


def compile_final(text, trace_text=None):
    errors = []
    m = SOURCES_RE.search(text or "")
    sources = []
    if not m:
        errors.append("missing SOURCES line")
        body = text or ""
    else:
        body = (text[:m.start()]).rstrip()
        raw = m.group(1)
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"SOURCES not valid JSON: {e}")
            arr = []
        seen = set()
        for k, e in enumerate(arr, 1):
            if not isinstance(e, dict) or len(e) != 1:
                errors.append(f"entry {k}: must be exactly one of "
                              f"{{'ref':…}} or {{'url':…}} (got {e!r:.60})")
                continue
            (key, val), = e.items()
            val = str(val)
            if key == "ref" and not REF_RE.match(val):
                errors.append(f"entry {k}: ref {val!r:.60} is not an NWR id"
                              + (" (looks like a url — wrong key)"
                                 if val.startswith("http") else ""))
            elif key == "url" and not URL_RE.match(val):
                errors.append(f"entry {k}: url {val!r:.60} is not http(s)")
            elif key not in ("ref", "url"):
                errors.append(f"entry {k}: unknown key {key!r}")
            sig = (key, val)
            if sig in seen:
                errors.append(f"entry {k}: duplicate of an earlier entry")
            seen.add(sig)
            sources.append({"type": key, "value": val})
    # raw refs/urls used as pseudo-markers in the body — [W12345],
    # [https://…], or mixed [N123, https://…] — a defect family found
    # 2026-09-03: invisible to the numeric marker grammar, so check
    # explicitly for ANY bracket containing an NWR id or a url
    for rm in PSEUDO_RE.finditer(body):
        errors.append(f"raw ref/url in inline bracket "
                      f"{rm.group(0)[:70]!r} (cite via numbered [n] "
                      "+ SOURCES entries)")
    markers = []
    for mm in MARKER_RE.finditer(body):
        for num in re.split(r"\s*,\s*", mm.group(1)):
            markers.append((int(num), mm.start()))
    used = set()
    for num, pos in markers:
        if not (1 <= num <= len(sources)):
            errors.append(f"marker [{num}] out of range (1..{len(sources)})")
        else:
            used.add(num)
    if sources and not markers:
        # the writer emitted a correct SOURCES list but no inline markers.
        # ONE defect (repairable by inserting markers), not N orphans —
        # and never a reason to delete the citations. See normalize().
        errors.append(f"{len(sources)} SOURCES entries but no inline [n] "
                      "markers in the body")
    else:
        for k in range(1, len(sources) + 1):
            if k not in used:
                errors.append(f"entry {k} never referenced by a marker")
    if trace_text is not None:
        for k, s in enumerate(sources, 1):
            if s["value"] not in trace_text:
                errors.append(f"entry {k}: {s['type']} {s['value']!r:.60} "
                              "not found verbatim in trace")
    return {"body": body, "markers": markers, "sources": sources}, errors


def normalize(text):
    """Deterministic repairs needing no judgment: url stored under 'ref'
    key -> 'url'; duplicate SOURCES entries dropped; entries never
    referenced by any marker dropped (contract-mandated: "drop entries
    nothing refers to"); markers renumbered. Grouped [n, m] markers are
    VALID and left alone. Non-repairable defects (out-of-range markers to
    nonexistent sources, non-JSON) are left for the compiler to reject."""
    m = SOURCES_RE.search(text or "")
    if not m:
        return text, False
    try:
        arr = json.loads(m.group(1))
    except json.JSONDecodeError:
        return text, False
    # repair raw ref/url pseudo-markers — [W12345], [https://…], mixed
    # [N123, https://…]: map each component to a numbered marker,
    # appending SOURCES entries where missing
    body0 = text[:m.start()]
    pseudo_repaired = False
    if PSEUDO_RE.search(body0):
        pseudo_repaired = True
        vals = [str(next(iter(e.values())))
                for e in arr if isinstance(e, dict) and len(e) >= 1]

        def fix_bracket(mo):
            nums = []
            for part in re.split(r"\s*,\s*", mo.group(1)):
                part = part.strip()
                if part not in vals:
                    arr.append({"ref": part} if REF_RE.match(part)
                               else {"url": part})
                    vals.append(part)
                nums.append(vals.index(part) + 1)
            return "[" + ", ".join(str(n) for n in
                                   sorted(dict.fromkeys(nums))) + "]"

        body0 = PSEUDO_RE.sub(fix_bracket, body0)
        text = body0 + "SOURCES: " + json.dumps(arr, ensure_ascii=False)
        m = SOURCES_RE.search(text)
    body = text[:m.start()].rstrip()
    referenced = set()
    for mo in MARKER_RE.finditer(body):
        for n in re.split(r"\s*,\s*", mo.group(1)):
            referenced.add(int(n))
    fixed, remap, seen = [], {}, {}
    changed = False
    # Orphan-drop applies ONLY when the body actually has markers. With no
    # markers at all, every entry looks "orphaned" and dropping them
    # deletes a valid citation list — measured 2026-09-03: this silently
    # emptied SOURCES on 5,477 judged-correct rollouts, 5,477 of which
    # carried the gold ref. Unmarked SOURCES is a marker defect, not an
    # orphan defect; compile_final reports it and the text is preserved.
    drop_orphans = bool(referenced)
    for k, e in enumerate(arr, 1):
        if not isinstance(e, dict):
            return text, False
        if drop_orphans and k not in referenced:
            changed = True          # orphaned entry: contract says drop
            continue
        if "ref" in e and str(e["ref"]).startswith("http"):
            e = {"url": e["ref"]}
            changed = True
        if len(e) > 1 and "url" in e:  # e.g. {"ref": x, "url": x}
            e = {"url": e["url"]}
            changed = True
        (key, val), = e.items() if len(e) == 1 else (("url", ""),)
        sig = (key, str(val))
        if sig in seen:
            remap[k] = seen[sig]
            changed = True
            continue
        fixed.append({key: val})
        seen[sig] = len(fixed)
        remap[k] = len(fixed)
    if not changed and not pseudo_repaired:
        return text, False

    def renum(mo):
        nums = [remap.get(int(n), int(n))
                for n in re.split(r"\s*,\s*", mo.group(1))]
        uniq = sorted(dict.fromkeys(nums))
        return "[" + ", ".join(str(n) for n in uniq) + "]"

    body = MARKER_RE.sub(renum, body)
    line = "SOURCES: " + json.dumps(fixed, ensure_ascii=False)
    return body + "\n\n" + line, True


if __name__ == "__main__":
    import sys
    rows = [json.loads(l) for l in open(sys.argv[1])]
    bad = 0
    for r in rows:
        _, errs = compile_final(r.get("final") or "")
        if errs:
            bad += 1
            print(f"{r.get('id')}: {'; '.join(errs[:3])}")
    print(f"{bad}/{len(rows)} rows with compile errors")
