"""Shared bootstrap: endpoints from the environment, and the prompt flags the model was trained with
(the SAME flags v4 was trained with: places 3 / web 2 / fetch 1 / locality / route tool)."""
import os, sys

# Two inputs: a topoloop API key (places + web tools) and an OpenAI-compatible judge endpoint.
# verifiers' subprocess runtime strips any variable whose name contains API_KEY before launching tool
# servers, so the key is read as TOPOLOOP_KEY (the name the topoloop docs use); TOPOLOOP_API_KEY also
# works in-process. Same for the judge: JUDGE_KEY or JUDGE_API_KEY.
if os.environ.get("TOPOLOOP_KEY") and not os.environ.get("TOPOLOOP_API_KEY"):
    os.environ["TOPOLOOP_API_KEY"] = os.environ["TOPOLOOP_KEY"]
if os.environ.get("JUDGE_KEY") and not os.environ.get("JUDGE_API_KEY"):
    os.environ["JUDGE_API_KEY"] = os.environ["JUDGE_KEY"]
if not os.environ.get("TOPOLOOP_API_KEY") and not os.environ.get("WPLACES_API_ENDPOINT"):
    sys.stderr.write("geogym: set TOPOLOOP_KEY (free key at https://dash.topoloop.com/signup) so the tools have an endpoint\n")
if os.environ.get("TOPOLOOP_API_KEY"):
    base = os.environ.get("TOPOLOOP_API_BASE", "https://api.topoloop.com").rstrip("/")
    os.environ.setdefault("WPLACES_API_ENDPOINT", base)
    os.environ.setdefault("WPLACES_API_KEY", os.environ["TOPOLOOP_API_KEY"])
    os.environ.setdefault("WSEARCH_EP", base)
    os.environ.setdefault("WSEARCH_KEY", os.environ["TOPOLOOP_API_KEY"])

for k, v in {"BENCH_TWO_TOOL": "1", "BENCH_BLOG_TOOLS": "1", "BENCH_CONTRACT": "v2", "BENCH_SEARCH_BUDGET": "3",
             "BENCH_WEB_BUDGET": "2", "BENCH_FETCH_BUDGET": "1", "BENCH_LOCALITY": "1", "BENCH_ROUTE_TOOL": "1",
             "BENCH_CHUNK_CHARS": "2600", "JUDGE_REWARD_MODE": "1", "JUDGE_V22": "1"}.items():
    os.environ.setdefault(k, v)
os.environ.pop("BENCH_STRICT_PROMPT", None)

BUDGETS = {"search_places": int(os.environ["BENCH_SEARCH_BUDGET"]), "web_search": int(os.environ["BENCH_WEB_BUDGET"]),
           "web_fetch": int(os.environ["BENCH_FETCH_BUDGET"])}
