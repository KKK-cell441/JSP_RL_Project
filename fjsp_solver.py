"""CP-SAT solver for flexible job shop instances with machine alternatives."""
import numpy as np
from ortools.sat.python import cp_model


def solve_fjsp(alternatives, time_limit=5.0):
    """Solve FJSP with explicit machine options.

    alternatives[j][k] = list of (machine, duration) options for operation k.
    Returns (makespan, mean_flow_time, schedule).
    """
    n_jobs = len(alternatives)
    max_m = max(m for job in alternatives for opts in job for m, _ in opts) + 1
    model = cp_model.CpModel()
    total = sum(min(d for _, d in opts) for job in alternatives for opts in job)
    horizon = total * 2

    starts, ends, presence = {}, {}, {}
    machine_ops = [[] for _ in range(max_m)]
    job_finish = [None] * n_jobs

    for j in range(n_jobs):
        for k, opts in enumerate(alternatives[j]):
            for oi, (m, dur) in enumerate(opts):
                s = model.NewIntVar(0, horizon, f"s_{j}_{k}_{oi}")
                e = model.NewIntVar(0, horizon, f"e_{j}_{k}_{oi}")
                active = model.NewBoolVar(f"a_{j}_{k}_{oi}")
                iv = model.NewOptionalIntervalVar(s, dur, e, active, f"i_{j}_{k}_{oi}")
                machine_ops[m].append(iv)
                starts[(j, k, oi)] = s
                ends[(j, k, oi)] = e
                presence[(j, k, oi)] = active
                model.Add(e == 0).OnlyEnforceIf(active.Not())
            model.AddExactlyOne([presence[(j, k, oi)] for oi in range(len(opts))])
        for k in range(len(alternatives[j])):
            if k > 0:
                for oi_prev in range(len(alternatives[j][k - 1])):
                    for oi_cur in range(len(alternatives[j][k])):
                        model.Add(starts[(j, k, oi_cur)] >= ends[(j, k - 1, oi_prev)]).OnlyEnforceIf(
                            [presence[(j, k - 1, oi_prev)], presence[(j, k, oi_cur)]]
                        )
        last_k = len(alternatives[j]) - 1
        last_opts = alternatives[j][last_k]
        job_end = model.NewIntVar(0, horizon, f"jend_{j}")
        model.Add(job_end == sum(ends[(j, last_k, oi)] for oi in range(len(last_opts))))
        job_finish[j] = job_end

    for m in range(max_m):
        if machine_ops[m]:
            model.AddNoOverlap(machine_ops[m])

    makespan = model.NewIntVar(0, horizon, "makespan")
    for jf in job_finish:
        model.Add(makespan >= jf)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, None

    schedule = {}
    for j in range(n_jobs):
        for k, opts in enumerate(alternatives[j]):
            for oi in range(len(opts)):
                if solver.Value(presence[(j, k, oi)]):
                    schedule[(j, k)] = (opts[oi][0], solver.Value(starts[(j, k, oi)]), solver.Value(ends[(j, k, oi)]))
                    break
    mks = solver.Value(makespan)
    ft = sum(solver.Value(jf) for jf in job_finish) / n_jobs
    return mks, ft, schedule
