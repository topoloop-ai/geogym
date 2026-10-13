"""Config loader: endpoints (rotating, local-only) + frozen paths.

    from geogym_blogq.lab.core.config import EP, PATHS, legacy, local
    EP["judge"]["base_url"]          # today's judge endpoint
    legacy("blogq_bank")             # absolute path of a frozen legacy input
    local("data") / "banks"          # local layout

endpoints.local.toml is gitignored; endpoints.example.toml documents shape.
Keys marked *_env are resolved from the environment at load time.
"""
import os
try:
    import tomllib
except ImportError:  # python < 3.11
    tomllib = None
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"


def _load(name):
    p = CONFIG / name
    if tomllib is None or not p.exists():
        return {}
    with open(p, "rb") as f:
        return tomllib.load(f)


def _resolve_env(d):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k.endswith("_env") and isinstance(v, str):
                out[k[:-4]] = os.environ.get(v, "")
            else:
                out[k] = _resolve_env(v)
        return out
    return d


PATHS = _load("paths.toml")
EP = _resolve_env(_load("endpoints.local.toml") or _load("endpoints.example.toml"))


def legacy(key):
    """Absolute path of a frozen legacy artifact by its [legacy.files] key; None if no paths.toml."""
    try:
        return Path(PATHS["legacy"]["root"]) / PATHS["legacy"]["files"][key]
    except KeyError:
        return None


def local(key):
    try:
        return Path(PATHS["local"][key])
    except KeyError:
        return ROOT / key


def endpoint(section):
    """EP['a']['b'] via dotted name, e.g. endpoint('launder.gemma')."""
    d = EP
    for part in section.split("."):
        d = d[part]
    return d
