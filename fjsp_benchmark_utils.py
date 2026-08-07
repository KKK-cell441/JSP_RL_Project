"""Parsers and dispatch baselines for standard FJSP benchmark files."""
import numpy as np
from fjsp_env import FJSPEnv


def parse_fjs(path):
    with open(path, encoding="utf-8") as f:
        tokens = f.read().split()
    it = iter(tokens)
    n_jobs = int(next(it))
    n_machines = int(next(it))
    _flex = next(it)
    alternatives = []
    for _ in range(n_jobs):
        n_ops = int(next(it))
        ops = []
        for _ in range(n_ops):
            n_opts = int(next(it))
            opts = []
            for _ in range(n_opts):
                mach = int(next(it)) - 1
                dur = int(next(it))
                opts.append((mach, dur))
            ops.append(opts)
        alternatives.append(ops)
    return n_jobs, n_machines, alternatives


def build_env(n_jobs, n_machines, alternatives):
    return FJSPEnv(n_jobs, n_machines, np.ones((n_jobs, n_machines)), alternatives)


def run_spt(env):
    env.reset()
    while not env.done:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        best_j, best_oi, best_d = None, None, float("inf")
        for j in avail:
            opts = env.alternatives[j][env.job_progress[j]]
            d = min(d for _, d in opts)
            if d < best_d:
                best_j, best_oi, best_d = j, 0, d
        if best_j is None:
            break
        # Choose the option with the shortest processing time.
        opts = env.alternatives[best_j][env.job_progress[best_j]]
        best_oi = min(range(len(opts)), key=lambda oi: opts[oi][1])
        env.step_option(best_j, best_oi)
    return env.get_makespan(), sum(env.job_done_time) / env.n_jobs
