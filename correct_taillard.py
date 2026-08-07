"""Correct JSPLIB Taillard parsing and CP-SAT validation.

JSPLIB Taillard format: first line 'n m', then one line per job containing
n*m pairs (machine, processing_time). Operations preserve line order.
"""
import numpy as np
from ortools.sat.python import cp_model


def parse_jsplib_taillard(path):
    """Return (n_jobs, n_machines, operations) where operations[j] = [(machine,dur),...]."""
    with open(path) as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith("#")]
    parts = lines[0].split()
    n_jobs, n_mach = int(parts[0]), int(parts[1])
    operations = []
    for j in range(n_jobs):
        tokens = lines[1 + j].split()
        ops = []
        for k in range(n_mach):
            machine = int(tokens[2 * k])
            duration = int(tokens[2 * k + 1])
            ops.append((machine, duration))
        operations.append(ops)
    return n_jobs, n_mach, operations


def solve_jsplib_cp(operations, n_mach, time_limit=5.0):
    """Solve JSP with JSPLIB operation order using CP-SAT. Returns makespan and schedule."""
    n_jobs = len(operations)
    model = cp_model.CpModel()
    total = sum(d for job in operations for _, d in job)
    horizon = total * 2
    starts = {}
    ends = {}
    machine_ops = [[] for _ in range(n_mach)]
    job_finish = []
    for j in range(n_jobs):
        prev = None
        for k, (m, dur) in enumerate(operations[j]):
            s = model.NewIntVar(0, horizon, f"s_{j}_{k}")
            e = model.NewIntVar(0, horizon, f"e_{j}_{k}")
            iv = model.NewIntervalVar(s, dur, e, f"i_{j}_{k}")
            starts[(j, k)] = s
            ends[(j, k)] = e
            machine_ops[m].append(iv)
            if prev is not None:
                model.Add(s >= prev)
            prev = e
        job_finish.append(prev)
    for m in range(n_mach):
        if machine_ops[m]:
            model.AddNoOverlap(machine_ops[m])
    makespan = model.NewIntVar(0, horizon, "mk")
    for jf in job_finish:
        model.Add(makespan >= jf)
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sched = {}
        for j in range(n_jobs):
            for k, (m, d) in enumerate(operations[j]):
                sched[(j, k)] = (m, solver.Value(starts[(j, k)]), solver.Value(ends[(j, k)]))
        return solver.ObjectiveValue(), solver.StatusName(status), sched
    return None, solver.StatusName(status), None


if __name__ == "__main__":
    import os
    path = os.path.join(r"C:\Users\17302\Desktop\JSP_RL_Project", "taillard", "ta01")
    n, m, ops = parse_jsplib_taillard(path)
    print(f"ta01: {n}x{m}, first job ops={ops[0][:5]}")
    mk, status, sched = solve_jsplib_cp(ops, m, time_limit=10.0)
    print(f"CP-SAT 10s: makespan={mk}, status={status}")
    print("Expected optimal (feedback): 1231")
