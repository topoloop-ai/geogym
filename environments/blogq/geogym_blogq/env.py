"""ClarifyEnv (round 7, 2026-09-11): clarification as a TURN, not an exit.

The policy plays the task as before (tools via GeogymToolset). If its reply is a plain-text clarifying
question (no map object, question phrasing — core.harness.looks_like_ask), the env answers it ONCE and
lets the policy continue:
  - ask_label == "ambiguous": a simulated traveller (the judge-section Gemma) replies from the row's
    scenario line only (never the article), gold names/refs mechanically blocked; on any failure the
    reply is the scenario line verbatim (never the gold name), else a plain "not sure" (unknown never becomes a free pass).
  - otherwise: a fixed template — the question already gives the place; search again and answer.
A second ask ends the episode (scored as the honest tier by reward.py). The exchange is recorded on
trace.info["clarify"] so the reward can apply the ask cost / ambiguous bonus and step_stats can count it.
No new tool, no prompt change: v4rl6 already asks in exactly this form (16% of ho200).
"""
import asyncio, os, re
import verifiers.v1 as vf
from verifiers.v1.envs.single_agent.env import SingleAgentEnvConfig
from geogym_blogq import common  # noqa: F401  (prompt env + sys.path)
from geogym_blogq import reward as R
from geogym_blogq.lab.core.harness import looks_like_ask, ASK_SIM_SYSTEM

TEMPLATE = os.environ.get("GEOGYM_ASK_TEMPLATE",
    "I've told you everything I know — the question already gives the place or area. Please search again using that and answer.")
NOT_SURE = "I'm not sure of the exact place — that's why I'm asking. Please go with what you have."


async def simulate_reply(question, scenario, ask_text, gold_names, gold_refs, judge_section):
    """Gemma plays the traveller from the scenario line only. Returns (reply, source)."""
    llm, sem, args = R._judge_client(judge_section)
    ctx = (f"ORIGINAL QUESTION (yours): {question}\nYOUR SITUATION: {scenario}\n\nASSISTANT'S QUESTION TO YOU: {ask_text}")
    qlow = question.lower()
    names = [n for n in (gold_names or []) if n and n.lower() not in qlow]
    # 2026-09-11 (7b abandoned at step 3): a 60 s timeout starved the simulator behind the judge queue (300+ waiting) —
    # every ambiguous ask fell back to "not sure" from step 2 on. The judge queue clears in ~5 min; wait for it. The
    # simulator does NOT take the judge semaphore (that only throttles client-side; the server is FIFO anyway).
    for attempt in range(3):
        try:
            if True:
                r = await asyncio.wait_for(llm.chat.completions.create(
                    model=args.judge_model, temperature=0.3, max_tokens=120,
                    messages=[{"role": "system", "content": ASK_SIM_SYSTEM}, {"role": "user", "content": ctx}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}), timeout=float(os.environ.get("GEOGYM_SIM_TIMEOUT", "600")))
            reply = (r.choices[0].message.content or "").strip()
        except Exception:
            reply = ""
        if not reply:
            continue
        low = reply.lower()
        if any(n.lower() in low for n in names) or any(g in reply for g in (gold_refs or [])):
            continue   # leaked the answer — retry once, then fall back
        return reply, "simulator"
    # 2026-09-11 (7c prep): on simulator failure the traveller says the SCENARIO LINE verbatim (it carries the locality and
    # never the gold name — checked again here), so a timeout is never a "not sure" that starves the ask branch.
    sc = (scenario or "").strip()
    if sc and not any(n.lower() in sc.lower() for n in names) and not any(g in sc for g in (gold_refs or [])):
        return sc, "scenario_fallback"
    return NOT_SURE, "simulator_fallback"


class ClarifyEnv(vf.Env[SingleAgentEnvConfig]):
    async def run(self, task, agents):
        gold = dict(getattr(task.data, "gold", {}) or {})
        label = str(gold.get("ask_label") or "clear").lower()
        scenario = gold.get("scenario") or ""
        async with agents.agent.interaction(task) as it:
            if gold.get("followup"):
                # 2026-09-18 FOLLOW-UP task: seed the tool state with turn-1's web handles (so [wN] fetches resolve) and the
                # refs turn 1 retrieved (so place_details on a turn-1 ref is allowed); budgets start fresh. One turn, no ask branch.
                from geogym_blogq.servers.tools import GeogymState
                from geogym_blogq.lab.core.harness import Handles
                h = Handles()
                for hd, pmeta in sorted(((gold.get("turn1") or {}).get("handles") or {}).items(), key=lambda kv: int(kv[0][1:])):
                    h.get(pmeta["url"], pmeta.get("title") or "")
                seen = []
                for m in (task.data.prompt or []):
                    if getattr(m, "role", None) == "tool" or (isinstance(m, dict) and m.get("role") == "tool"):
                        c = m.content if not isinstance(m, dict) else m.get("content")
                        seen += [x for x in re.findall(r"\b[NWR]\d{3,}\b", str(c or "")) if x not in seen]
                cur = it.trace.state
                base = cur.model_dump() if hasattr(cur, "model_dump") else {}
                base = {k: v for k, v in base.items() if k in GeogymState.model_fields}
                base.update({"handles": dict(h.by_url), "n_handles": h.n, "refs_seen": seen})
                it.trace.state = GeogymState.model_validate(base)
                await it.turn()
                it.trace.info["clarify"] = {"asked": False, "label": "followup"}
                return
            seg = await it.turn()                     # prompted task: its opening reply
            rec = {"asked": False, "label": label}
            reply_text = seg.last_reply or ""
            body = re.sub(r'\s*\{"map"\s*:\s*\[\s*\]\s*\}\s*$', "", reply_text).strip()   # strip a trailing EMPTY map object
            has_pins = '"map"' in body                                                   # a non-empty map = an answer, not an ask
            if not seg.terminated and body and not has_pins and looks_like_ask(body):
                ask_text = body
                if label == "ambiguous" and scenario.strip():
                    reply, src = await simulate_reply(task.data.question, scenario, ask_text,
                                                      gold.get("expected_places"), gold.get("gold_refs"),
                                                      task.config.judge_section)
                else:
                    reply, src = TEMPLATE, "template"
                rec.update({"asked": True, "ask_text": ask_text[:400], "reply": reply[:400], "reply_source": src})
                # round 7d (2026-09-12): the continuation gets a fresh tool allowance (GEOGYM_REPLY_EXTRA="search,web,fetch";
                # "0,0,0" = round 7c behaviour). The tool server pulls trace.state before every call, so a write here lands.
                extra = [int(x) for x in os.environ.get("GEOGYM_REPLY_EXTRA", "2,1,1").split(",")]
                if any(extra):
                    from geogym_blogq.servers.tools import GeogymState
                    cur = it.trace.state
                    base = cur.model_dump() if hasattr(cur, "model_dump") else {}
                    base = {k: v for k, v in base.items() if k in GeogymState.model_fields}
                    base.update({"extra_search": extra[0], "extra_web": extra[1], "extra_fetch": extra[2]})
                    it.trace.state = GeogymState.model_validate(base)
                    rec["extra_budget"] = extra
                it.trace.info["clarify"] = dict(rec)
                seg = await it.turn(reply)            # one continuation; a further ask is terminal
            it.trace.info["clarify"] = dict(rec)


__all__ = ["ClarifyEnv"]
