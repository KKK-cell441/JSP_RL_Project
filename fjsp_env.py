"""Flexible Job Shop environment with explicit machine alternatives."""
import numpy as np
from jsp_env import JSPEnv, Operation


class FJSPEnv(JSPEnv):
    def __init__(self, n_jobs, n_machines, pt, alternatives, seed=None):
        self.alternatives = alternatives
        super().__init__(n_jobs, n_machines, pt, seed)

    def reset(self):
        state = super().reset()
        self.operations = []
        for j in range(self.n_jobs):
            ops = []
            for k, opts in enumerate(self.alternatives[j]):
                m, dur = opts[0]
                ops.append(Operation(job_id=j, op_idx=k, machine=m, duration=dur))
            self.operations.append(ops)
        self.job_progress = [0] * self.n_jobs
        self.job_available = [0.0] * self.n_jobs
        self.machine_free = [0.0] * self.n_machines
        self.job_done_time = [0.0] * self.n_jobs
        self.time = 0.0
        self.schedule = []
        self.done = False
        return state

    def _get_available_operations(self):
        avail = []
        for j in range(self.n_jobs):
            if self.job_progress[j] < len(self.operations[j]):
                opts = self.alternatives[j][self.job_progress[j]]
                if any(self.machine_free[m] <= self.time + 1e-9 for m, _ in opts):
                    avail.append(j)
        return avail

    def apply_option(self, job, option_id):
        opts = self.alternatives[job][self.job_progress[job]]
        m, dur = opts[option_id]
        op = self.operations[job][self.job_progress[job]]
        op.machine = m
        op.duration = dur

    def step_option(self, job, option_id):
        self.apply_option(job, option_id)
        return self.step(job)
