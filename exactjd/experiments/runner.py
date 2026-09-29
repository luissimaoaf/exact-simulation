"""Resumable, censoring-aware experiment runner (Paper 2 App. B / Paper 3 App. E.3 protocol).

* One immutable JSONL row per REQUESTED run: config, run index, seed/stream id, status, wall time, primitive-call
  counters, peak RSS, interruption reason.  Rows are appended and flushed immediately.
* Statuses: `completed`, `censored` (explicit Budget exhausted -> BudgetExhausted), `interrupted` (KeyboardInterrupt /
  external kill is detected on resume as a run with a `started` row and no final row).
* Censored/interrupted runs are NEVER discarded and the surviving outputs are NOT presented as an exact sample:
  `summarize` reports completion counts, runtime quantiles over all requested runs (censored times are lower bounds).
* Long single runs resume from pickled checkpoints (state = current time/state/events/rng state) and reproduce the
  original stochastic realisation.
"""
from __future__ import annotations

import json
import os
import pickle
import resource
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..rng import Budget, BudgetExhausted, StreamFactory


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


class ExperimentLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(_jsonable(row)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in open(self.path) if l.strip()]

    def final_rows(self, config: str) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for r in self.rows():
            if r["config"] == config and r["status"] != "started":
                out[r["run"]] = r
        return out

    def started_without_final(self, config: str) -> list[int]:
        started = {r["run"] for r in self.rows() if r["config"] == config and r["status"] == "started"}
        return sorted(started - set(self.final_rows(config)))


def run_experiment(config: str, n_runs: int, run_fn: Callable[[np.random.Generator, Budget, dict], dict],
                   log: ExperimentLog, seed: int = 0, max_seconds: float | None = None, max_calls: int | None = None,
                   ckpt_dir: str | Path | None = None, verbose: bool = True) -> list[dict]:
    """Run indices 0..n_runs-1 that do not yet have a final row.  `run_fn(rng, budget, ctx)` returns a result dict;
    ctx = {'run': i, 'ckpt_path': Path|None} lets run_fn checkpoint/resume itself."""
    streams = StreamFactory(seed)
    done = log.final_rows(config)
    for i in log.started_without_final(config):        # previous interruption: record it, then redo from checkpoint
        log.append(dict(config=config, run=i, status="interrupted", wall=float("nan"),
                        reason="process ended before a final row was written (resumed)"))
        done.pop(i, None)
    for i in range(n_runs):
        if i in done and done[i]["status"] in ("completed", "censored"):
            continue
        rng = streams.generator(config, i)
        budget = Budget(max_calls=max_calls, max_seconds=max_seconds)
        ckpt = None if ckpt_dir is None else Path(ckpt_dir) / f"{config}_{i}.pkl"
        log.append(dict(config=config, run=i, status="started", seed=seed))
        t0 = time.perf_counter()
        row = dict(config=config, run=i, seed=seed)
        try:
            res = run_fn(rng, budget, dict(run=i, ckpt_path=ckpt))
            row.update(status="completed", result=res)
        except BudgetExhausted as e:
            row.update(status="censored", reason=str(e))
        except KeyboardInterrupt:
            row.update(status="interrupted", reason="KeyboardInterrupt")
            row.update(wall=time.perf_counter() - t0, calls=budget.calls)
            log.append(row)
            raise
        row.update(wall=time.perf_counter() - t0, calls=dict(budget.calls),
                   peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        log.append(row)
        if ckpt is not None and ckpt.exists() and row["status"] == "completed":
            ckpt.unlink()
        if verbose:
            print(f"[{config}] run {i}: {row['status']} in {row['wall']:.2f}s", flush=True)
    return list(log.final_rows(config).values())


def summarize(rows: list[dict]) -> dict:
    """Completion counts and runtime quantiles over ALL requested runs (censored wall times are lower bounds)."""
    walls = np.array([r["wall"] for r in rows if r.get("wall") == r.get("wall")])
    n = len(rows)
    comp = [r for r in rows if r["status"] == "completed"]
    out = dict(requested=n, completed=len(comp), censored=sum(r["status"] == "censored" for r in rows),
               interrupted=sum(r["status"] == "interrupted" for r in rows),
               completion_fraction=len(comp) / max(n, 1))
    if len(walls):
        q = np.quantile(walls, [0.5, 0.75, 0.9, 0.99, 1.0])
        out.update(wall_q50=q[0], wall_q75=q[1], wall_q90=q[2], wall_q99=q[3], wall_max=q[4],
                   note="quantiles over all requested runs; censored runs contribute their time-at-limit (lower bound)")
    if rows:
        out["peak_rss_mb_max"] = max(r.get("peak_rss_mb", 0.0) for r in rows)
    calls: dict[str, int] = {}
    for r in rows:
        for k, v in (r.get("calls") or {}).items():
            calls[k] = calls.get(k, 0) + v
    out["primitive_calls_total"] = calls
    return out


# ------------------------------------------------------------------------------------------ skeleton run helper
def skeleton_run(diffusion, jump, bridge_factory_fn, x0, T, rng, budget, ctx, use_filter=True, refine_eps=None):
    """Run Algorithm D with pickled checkpoints at every event boundary; resumes an interrupted realisation."""
    from ..jump.skeleton import simulate_skeleton
    resume = None
    ck = ctx.get("ckpt_path")
    if ck is not None and Path(ck).exists():
        with open(ck, "rb") as f:
            resume = pickle.load(f)

    def save(state):
        if ck is not None:
            Path(ck).parent.mkdir(parents=True, exist_ok=True)
            tmp = str(ck) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(state, f)
            os.replace(tmp, ck)

    sk = simulate_skeleton(diffusion, jump, bridge_factory_fn(budget), x0, T, rng, budget=budget,
                           use_filter=use_filter, checkpoint=save if ck else None, resume=resume)
    info = [dict(theta=sg.info.theta, label=sg.info.label, ell=sg.info.ell, cands=sg.info.n_candidates,
                 trials=sg.info.skeleton_trials, bridges=sg.info.bridges_built, label_refine=sg.info.label_refinements)
            for sg in sk.segments]
    out = dict(n_events=sk.n_events, taus=[e.tau for e in sk.events], terminal=sk.terminal.tolist(),
               events=[dict(tau=e.tau, pre=e.pre.tolist(), post=e.post.tolist()) for e in sk.events], segments=info)
    if refine_eps is not None:
        out["refined_radius"] = sk.refine(refine_eps).radius
    return out
