"""geogym blogq taskset: one task per bank question; system prompt = the canonical harness prompt
(v4's training prompt); reward from geogym_blogq.reward."""
import json
import os, os, random
from pydantic import Field
import verifiers.v1 as vf
from geogym_blogq import common  # noqa: F401
from geogym_blogq.servers.tools import GeogymToolset, GeogymState
from geogym_blogq import reward as R
from geogym_blogq.lab.core.harness import SYSTEM


class BlogqData(vf.TaskData):
    qid: str
    question: str
    gold: dict = Field(default_factory=dict)


class BlogqTaskConfig(vf.TaskConfig):
    judge_section: str = "judge_gemma"
    style_penalty: float = -0.25


class BlogqTask(vf.Task[BlogqData, GeogymState, BlogqTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.qid

    @vf.reward
    async def geogym(self, trace: vf.Trace) -> float:
        state = trace.state if hasattr(trace, "state") else None
        row = R.row_from_trace(self.data.qid, trace.messages, state or GeogymState())
        q = dict(self.data.gold); q.setdefault("id", self.data.qid); q.setdefault("question", self.data.question)
        clarify = dict((trace.info or {}).get("clarify") or {}) if hasattr(trace, "info") else {}
        rew, reason, v = await R.score(row, q, self.config.judge_section, self.config.style_penalty, clarify=clarify)
        try:
            trace.info["geogym"] = {"reason": reason, "verdict": (v or {}).get("reason") if v else None,
                                    "protocol": row["protocol"], "n_tools": row["n_tools"],
                                    "asked": bool(clarify.get("asked")), "ask_label": clarify.get("label")}
            trace.record_metric("asked", float(bool(clarify.get("asked"))))
            trace.record_metric("ambiguous", float(clarify.get("label") == "ambiguous"))
            if clarify.get("asked"):
                trace.record_metric("pass_given_asked", float(str(reason).startswith("PASS")))
        except Exception:
            pass
        return float(rew)

    @vf.metric(priority=-1)
    async def passed(self, trace: vf.Trace) -> float:
        return float(str((trace.info.get("geogym") or {}).get("reason", "")).startswith("PASS"))

    @vf.metric(priority=-1)
    async def mechanical_fail(self, trace: vf.Trace) -> float:
        r = str((trace.info.get("geogym") or {}).get("reason", ""))
        return float(r.startswith(("protocol", "compile", "memory-first", "spatial", "format", "incomplete", "zero-tool answer")))


class BlogqConfig(vf.TasksetConfig):
    bank: str = os.environ.get("GEOGYM_BANK", "")   # empty = the bundled Ask Maps Basic hundred
    exclude_ids: str = ""          # file of ids to skip (held-out / already-trained)
    limit: int = 0
    seed: int = 41
    shuffle: bool = True
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: BlogqTaskConfig = BlogqTaskConfig()


class GeogymBlogqTaskset(vf.Taskset[BlogqTask, BlogqConfig]):
    @classmethod
    def toolsets(cls, config: BlogqConfig) -> list[vf.Toolset]:
        return [GeogymToolset(config.tools)]

    def load(self) -> list[BlogqTask]:
        skip = set()
        if self.config.exclude_ids and os.path.exists(self.config.exclude_ids):
            skip = {l.strip() for l in open(self.config.exclude_ids) if l.strip()}
        from geogym_blogq.lab.core.bank import load_bank as _load_bank
        bank = self.config.bank or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ask_maps_basic.jsonl")
        if not os.path.isabs(bank) and not os.path.exists(bank):
            here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), bank)
            if os.path.exists(here):
                bank = here
        rows = _load_bank(bank)
        rows = [r for r in rows if r["id"] not in skip]
        if self.config.shuffle:
            random.Random(self.config.seed).shuffle(rows)
        if self.config.limit:
            rows = rows[: self.config.limit]
        gold_keys = ("expected_places", "gold_refs", "gold_coords", "gold_fields", "answer_notes", "meta", "category", "region", "anchor",
                     "ask_label", "scenario",   # round 7: clarification label + traveller scenario (rl_r7_bank_labelled.jsonl)
                     "followup", "in_context", "question1", "qid",   # 2026-09-18: follow-up (turn-2) rows
                     "partial")   # 2026-09-19: partial-match twins
        tasks = []
        for i, r in enumerate(rows):
            gold = {k: r.get(k) for k in gold_keys if r.get(k) is not None}
            prompt = r["question"]
            if r.get("turn1"):
                # FOLLOW-UP task: the prompt is the completed turn-1 conversation (the checkpoint's own passing rollout,
                # minus its system prompt) + the follow-up as the new user turn. Tool calls/results are replayed verbatim
                # so the policy sees exactly what the app shows it. Handles/final go to gold (env seeds the tool state).
                prompt = _turn1_messages(r["turn1"]["messages"]) + [{"role": "user", "content": r["question"]}]
                gold["turn1"] = {"handles": r["turn1"].get("handles"), "final": r["turn1"].get("final")}
            tasks.append(BlogqTask(BlogqData(idx=i, qid=r["id"], question=r["question"], system_prompt=SYSTEM, prompt=prompt, gold=gold),
                                   self.config.task))
        return tasks


def _turn1_messages(msgs):
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        if role == "assistant":
            tcs = [{"id": tc["id"], "type": "function", "name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                   for tc in (m.get("tool_calls") or [])]
            out.append({"role": "assistant", "content": m.get("content") or "", **({"tool_calls": tcs} if tcs else {})})
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id") or "", "content": str(m.get("content") or "")})
        else:
            out.append({"role": role, "content": str(m.get("content") or "")})
    return out
