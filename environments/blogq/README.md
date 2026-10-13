# blogq

A verifiers environment where a model answers a traveller's question by searching OpenStreetMap places and a travel-blog corpus, then answers with citations a verifier can check. It is the environment behind [topoloop Finch 0.1](https://huggingface.co/topoloop/finch-0.1-0.8B), a 0.8B model that ties Gemini 3.8 Flash on [Ask Maps Basic](https://huggingface.co/datasets/topoloop/ask-maps-basic) and runs offline on a phone. Story: [blog post](https://jonready.com/blog/posts/rl-on-a-0_8b-travel-agent.html).

- **Two inputs**: a topoloop API key for the tools and an OpenAI-compatible endpoint for the judge
- **Multi-turn tool use**: four V1 tools over real geo data, with a tool budget the model has to respect
- **Two-layer verifier**: mechanical gates (every citation appears verbatim in the trace, the pinned place was retrieved, the map object parses) and then an LLM judge
- **Binary reward**: 1 if the answer is the right place, cited by a ref the model retrieved, with no unsupported specifics; 0 otherwise
- **Difficulty-matched bank**: the zero-advantage curriculum gate keeps groups with gradient

> Finch was trained on two RTX 3090s (1 inference, 1 trainer) with a third card serving the judge. `rl.toml` is that configuration.

## Setup

Inside a [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) checkout (main from October 2026 or later; the flattened `eval.toml` schema is required), install the environment into prime-rl's own venv and check it imports. `--no-deps` keeps prime-rl's pinned `verifiers`:

```bash
uv pip install -e /path/to/geogym/environments/blogq --no-deps
uv run python -c "import geogym_blogq"
```

prime-rl resolves `env.taskset.id = "geogym-blogq"` by importing the `geogym_blogq` module, so nothing else needs registering. Relative bank paths in the tomls resolve against `environments/blogq`, so the eval works from any directory.

Without prime-rl, for the direct harness path below, a plain venv is enough:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e environments/blogq
```

Set the two inputs:

```bash
export TOPOLOOP_KEY=tl_live_...              # free at https://dash.topoloop.com/signup; places + web tools
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=google/gemma-4-31b-it      # what the published numbers were made with
export JUDGE_KEY=...
```

Any OpenAI-compatible judge works. A local Gemma 4 26B-A4B on one 3090 agrees with the 31B within a point or two.

## Task

The model has to take a traveller's question, find the place, and say something true about it. It has four tools:

- `search_places(query, lat, lng)`: semantic search over an OpenStreetMap-derived places index, 300 km around the anchor
- `place_details(ref)`: everything the index knows about one place
- `web_search(query)`: search over a travel-blog corpus, returning snippets with handles the model must cite
- `web_fetch(url)`: the page behind a handle, chunked

The budget is 3 places searches, 2 web searches and 1 fetch. Exceeding it is a fail before the judge looks.

The output contract is strict: prose cites the pages it used by handles the harness minted during the episode, and a trailing JSON object carries the main place as a ref the model actually retrieved. No URLs, no coordinates, no `[1]` citations.

## Scoring

Mechanical gates first. Then the judge, run two ways:

- **grounding** (`JUDGE_REWARD_MODE=1`, the default and the training reward): a specific claim is only credited if the tool output the model received supports it
- **correctness** (`JUDGE_REWARD_MODE=0`): right place, sensible answer, the fair comparison for models that never saw the citation contract

Both verdicts for nine models on Ask Maps Basic are browsable in the [trace explorer](https://jonready.com/blog/posts/ho100/).

## Questions

A frontier model reads a travel-blog passage and writes the question a traveller would ask to find that place, with the place's OpenStreetMap ref as gold. `env.taskset.bank` points at a JSONL of them. `eval/ask_maps_basic.jsonl` is the hundred from [Ask Maps Basic](https://huggingface.co/datasets/topoloop/ask-maps-basic), which `eval.toml` uses.

## Baseline evaluation

```bash
uv run inference --vllm.model Qwen/Qwen3.5-0.8B --vllm.tool-call-parser qwen3_coder
uv run eval @ environments/blogq/eval.toml
```

Override the model with `-m` to evaluate a checkpoint, for example `-m topoloop/finch-0.1-0.8B`.

## Training

```bash
uv run rl @ environments/blogq/rl.toml
```

`rl.toml` ships pointing at the eval bank so it runs out of the box; replace `env.taskset.bank` with your own training questions in the same JSONL format before a real run, and never train on the questions you evaluate on. The question generator is the next release.

## Running the harness directly

This is how the Ask Maps Basic numbers were produced, and it works for any OpenAI-compatible endpoint, including hosted models through OpenRouter, with no trainer installed. Run it from `environments/blogq` (the relative paths below depend on it), with the two inputs set. For a smoke test first, `head -3 eval/ask_maps_basic.jsonl > q3.jsonl` and point the two bank variables at it; expect `3/3` from the harness and `wrote grounding.jsonl` from the judge.

```bash
export PYTHONPATH=$PWD
export BENCH_CONTRACT=v2 BENCH_TWO_TOOL=1 BENCH_BLOG_TOOLS=1 BENCH_SEARCH_BUDGET=3 BENCH_WEB_BUDGET=2 BENCH_FETCH_BUDGET=1 \
       BENCH_LOCALITY=1 BENCH_CHUNK_CHARS=2600 WPLACES_SEARCH_RADIUS_KM=300 BENCH_NO_JUDGE=1 \
       AGENTQ_FILE=eval/ask_maps_basic.jsonl BENCH_QUESTIONS=eval/ask_maps_basic.jsonl

# 1. rollouts: any OpenAI-compatible endpoint (add BENCH_NO_TEMPLATE_KWARGS=1 for OpenRouter)
python -m geogym_blogq.lab.core.harness --base-url http://localhost:8000/v1 --model finch \
       --concurrency 4 --max-tools 8 --temperature 0 --out rows.jsonl

# 2. judge, grounding (the training reward)
JUDGE_REWARD_MODE=1 JUDGE_V22=1 python -m geogym_blogq.lab.judges.judge_v2 --rows rows.jsonl --bank eval/ask_maps_basic.jsonl \
       --judge-base-url $JUDGE_BASE_URL --judge-model $JUDGE_MODEL --judge-api-key $JUDGE_KEY \
       --max-tokens 1500 --votes 1 --concurrency 4 -o grounding.jsonl

# 3. judge, correctness (the headline chart)
JUDGE_REWARD_MODE=0 JUDGE_V22=1 python -m geogym_blogq.lab.judges.judge_v2 --rows rows.jsonl --bank eval/ask_maps_basic.jsonl \
       --judge-base-url $JUDGE_BASE_URL --judge-model $JUDGE_MODEL --judge-api-key $JUDGE_KEY \
       --max-tokens 1500 --votes 1 --concurrency 4 -o correctness.jsonl
```

Each verdict row has `id`, `correct` (bool) and `reason`. The `correct` field inside `rows.jsonl` is a placeholder under `BENCH_NO_JUDGE=1`; the verdicts are the judge outputs. Judge runs resume: an existing output file skips rows already judged, so delete it to re-judge. A local judge works the same way, for example `JUDGE_BASE_URL=http://localhost:8006/v1 JUDGE_MODEL=gemma-4-26b-it JUDGE_API_KEY=EMPTY`. The free topoloop tier is 2 requests a second, so keep `--concurrency` low; the client backs off on 429.

## What the policy found

`docs/REWARD_HACKS.md` is 24 numbered ways a 0.8B policy got reward it had not earned against this verifier, and what closed each one. The best took nine steps. `docs/LADDER.md` is every round with its number.

## License

Apache 2.0.
