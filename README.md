# geogym

RL environments for map assistants, from [topoloop](https://github.com/topoloop-ai). Each environment is a [verifiers](https://github.com/PrimeIntellect-ai/verifiers) taskset that trains with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), with real tools over open geo data and a verifier that only credits what the model actually retrieved.

Every environment needs the same two inputs: a topoloop API key for the tools and an OpenAI-compatible endpoint for the judge.

```bash
export TOPOLOOP_KEY=tl_live_...       # free at https://dash.topoloop.com/signup
export JUDGE_BASE_URL=https://openrouter.ai/api/v1
export JUDGE_MODEL=google/gemma-4-31b-it
export JUDGE_KEY=...
```

## Environments

| Environment | Task | Eval | Status |
|---|---|---|---|
| [`blogq`](environments/blogq) | Answer a traveller's question by searching OpenStreetMap places and a travel-blog corpus, with citations a verifier can check | [Ask Maps Basic](https://huggingface.co/datasets/topoloop/ask-maps-basic), 100 questions, nine models scored | Released. Trained [topoloop Finch 0.1](https://huggingface.co/topoloop/finch-0.1-0.8B), which ties Gemini 3.8 Flash on the eval and runs offline on a phone |
| `routing` | Plan a route between places under a mode and time constraint, over a routing engine | | Next |
| `constraints` | Find a place that satisfies several constraints at once: open now, within walking distance, serves X | | Planned |
| `hiking` | Trails, elevation, water and daylight, over OpenStreetMap and elevation data | | Planned |

Each environment directory follows the prime-rl example layout: a package, an `eval.toml`, an `rl.toml`, and a README with the task, the tools, the scoring and the commands. Install one into a prime-rl checkout with `uv pip install -e environments/<name> --no-deps`; prime-rl finds the taskset by module name.

## Why

A small model knows almost no geography, so everything it says has to come from what it looked up, and it has to say so. These environments reward exactly that. The story of the first one, including the 24 ways a 0.8B policy found to hack its verifier, is in the [blog post](https://jonready.com/blog/posts/rl-on-a-0_8b-travel-agent.html).

## License

Apache 2.0.
