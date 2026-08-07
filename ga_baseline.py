"""Genetic Algorithm baseline for job shop scheduling comparison.

Standard operation-sequence encoding + active schedule decoding (Giffler-Thompson).
"""
import numpy as np
import random


def decode_operations(seq, n_jobs, n_machines, pt):
    """Decode an operation sequence (each job appears n_machines times) into schedule.

    Returns makespan. seq is permutation of job ids with repetitions.
    """
    # Build operation lists per job (in machine order)
    ops = {}
    for j in range(n_jobs):
        ops[j] = [(m, pt[j][m]) for m in range(n_machines)]
    prog = {j: 0 for j in range(n_jobs)}
    machine_free = [0.0] * n_machines
    job_avail = {j: 0.0 for j in range(n_jobs)}
    for job in seq:
        k = prog[job]
        if k >= n_machines:
            continue
        m, dur = ops[job][k]
        start = max(job_avail[job], machine_free[m])
        end = start + dur
        machine_free[m] = end
        job_avail[job] = end
        prog[job] += 1
    return max(job_avail.values())


def random_seq(n_jobs, n_machines):
    s = []
    for j in range(n_jobs):
        s.extend([j] * n_machines)
    random.shuffle(s)
    return s


def jox_crossover(p1, p2, n_jobs, n_machines):
    """Job-order crossover: preserve a subset of jobs' positions from p1, fill rest from p2."""
    keep = set(random.sample(range(n_jobs), n_jobs // 2))
    child = [None] * len(p1)
    positions = {}
    for i, j in enumerate(p1):
        if j in keep:
            child[i] = j
    # Fill remaining from p2 preserving order
    rem = []
    count = {j: 0 for j in range(n_jobs)}
    for j in child:
        if j is not None:
            count[j] += 1
    rem_counts = {j: n_machines - count[j] for j in range(n_jobs)}
    filler = []
    for j in p2:
        if rem_counts[j] > 0:
            filler.append(j)
            rem_counts[j] -= 1
    fi = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = filler[fi]
            fi += 1
    return child


def ga_solve(pt, n_jobs, n_machines, pop_size=40, generations=100, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    pop = [random_seq(n_jobs, n_machines) for _ in range(pop_size)]
    fits = [decode_operations(s, n_jobs, n_machines, pt) for s in pop]
    best = min(fits)
    for gen in range(generations):
        new_pop = []
        while len(new_pop) < pop_size:
            # tournament select two parents
            def tournament():
                idx = random.sample(range(pop_size), 3)
                return min(idx, key=lambda i: fits[i])
            p1 = pop[tournament()]
            p2 = pop[tournament()]
            c = jox_crossover(p1, p2, n_jobs, n_machines)
            # mutation: swap two random positions
            if random.random() < 0.1:
                a, b = random.sample(range(len(c)), 2)
                c[a], c[b] = c[b], c[a]
            new_pop.append(c)
        pop = new_pop
        fits = [decode_operations(s, n_jobs, n_machines, pt) for s in pop]
        gen_best = min(fits)
        if gen_best < best:
            best = gen_best
    return best
