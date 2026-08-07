"""
Job Shop Scheduling Environment with Disjunctive Graph Representation.

State: disjunctive graph of operations. The environment advances time only
when no operation can start at the current time, allowing concurrent
scheduling across machines.

Action space: choose which pending job's next operation to schedule.
Reward: negative makespan at episode end.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Operation:
    job_id: int
    op_idx: int
    machine: int
    duration: int


class JSPEnv:
    """Deterministic job shop scheduling environment."""

    def __init__(self, n_jobs: int, n_machines: int, processing_times: np.ndarray,
                 seed: Optional[int] = None,
                 operations: Optional[List[List[Tuple[int, float]]]] = None):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.pt = processing_times
        self.operation_list = operations
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        # Build operations. For generated instances the machine order is fixed
        # to [0, ..., m-1]; JSPLIB files preserve their own operation order.
        self.operations = []
        if self.operation_list is not None:
            for j, seq in enumerate(self.operation_list):
                ops = []
                for idx, (m, d) in enumerate(seq):
                    ops.append(Operation(job_id=j, op_idx=idx, machine=int(m), duration=float(d)))
                self.operations.append(ops)
        else:
            for j in range(self.n_jobs):
                machine_order = list(range(self.n_machines))
                ops = []
                for idx, m in enumerate(machine_order):
                    t = float(self.pt[j, m])
                    if t > 0:
                        ops.append(Operation(job_id=j, op_idx=idx, machine=m, duration=t))
                self.operations.append(ops)

        self.job_progress = [0] * self.n_jobs
        self.job_available = [0.0] * self.n_jobs  # time when next op of job can start
        self.machine_free = [0.0] * self.n_machines
        self.job_done_time = [0.0] * self.n_jobs
        self.time = 0.0
        self.schedule = []
        self.done = False
        self.reward = 0.0
        self.step_count = 0
        return self._get_state()

    @classmethod
    def from_operations(cls, n_jobs: int, n_machines: int,
                        operation_sequences: List[List[Tuple[int, float]]],
                        seed: Optional[int] = None) -> "JSPEnv":
        """Build an environment directly from JSPLIB operation sequences.

        operation_sequences[j] is the ordered list [(machine, duration), ...]
        for job j, exactly as written in the benchmark file.
        """
        dummy = np.zeros((n_jobs, n_machines), dtype=float)
        return cls(n_jobs, n_machines, dummy, seed=seed, operations=operation_sequences)

    def _get_available_operations(self) -> List[int]:
        """Job indices whose next operation machine is free at current time."""
        avail = []
        for j in range(self.n_jobs):
            if self.job_progress[j] < len(self.operations[j]):
                op = self.operations[j][self.job_progress[j]]
                if self.machine_free[op.machine] <= self.time + 1e-9:
                    avail.append(j)
        return avail

    def _get_state(self) -> dict:
        return {
            "progress": list(self.job_progress),
            "machine_free": list(self.machine_free),
            "time": float(self.time),
            "done": self.done,
            "available": self._get_available_operations(),
        }

    def _advance_time(self):
        """Advance to the next machine-free event if no operations available."""
        while not self._get_available_operations() and not self.done:
            future = [m for m in self.machine_free if m > self.time + 1e-9]
            if not future:
                # All machines free but no available ops means all jobs done
                if all(p == len(self.operations[j]) for j, p in enumerate(self.job_progress)):
                    self.done = True
                    return
                break
            self.time = min(future)

    def step(self, action_job: int) -> Tuple[dict, float, bool, dict]:
        """Schedule the next operation of action_job."""
        assert not self.done, "Episode already done"
        avail = self._get_available_operations()
        assert action_job in avail, f"Job {action_job} not available"

        op = self.operations[action_job][self.job_progress[action_job]]
        start = max(self.time, self.machine_free[op.machine], self.job_available[action_job])
        end = start + op.duration
        self.schedule.append((action_job, op.machine, start, end))
        self.machine_free[op.machine] = end
        self.job_available[action_job] = end
        self.job_progress[action_job] += 1

        if self.job_progress[action_job] == len(self.operations[action_job]):
            self.job_done_time[action_job] = end

        self.step_count += 1

        # Check if all jobs complete
        if all(p == len(self.operations[j]) for j, p in enumerate(self.job_progress)):
            self.done = True
            makespan = max(self.job_done_time)
            self.reward = -makespan
            return self._get_state(), self.reward, self.done, {"makespan": makespan}

        # Advance time if no more operations can start now
        self._advance_time()

        return self._get_state(), 0.0, False, {}

    def get_makespan(self) -> float:
        return max(self.job_done_time) if self.done else None

    @staticmethod
    def generate_random_instance(n_jobs: int, n_machines: int,
                                 min_dur: int = 1, max_dur: int = 20,
                                 seed: int = 0) -> np.ndarray:
        """Generate random job shop processing time matrix."""
        rng = np.random.default_rng(seed)
        pt = rng.integers(min_dur, max_dur + 1, size=(n_jobs, n_machines)).astype(float)
        return pt


# ============== Heuristic baselines ==============
def schedule_with_heuristic(env: JSPEnv, rule: str) -> float:
    """Run the environment with a dispatching rule and return makespan."""
    env.reset()
    while not env.done:
        avail = env._get_available_operations()
        if not avail:
            env._advance_time()
            continue
        if rule == "FIFO":
            job = min(avail)
        elif rule == "SPT":
            job = min(avail, key=lambda j: env.operations[j][env.job_progress[j]].duration)
        elif rule == "LPT":
            job = max(avail, key=lambda j: env.operations[j][env.job_progress[j]].duration)
        elif rule == "MWKR":
            def rem(j):
                return sum(o.duration for o in env.operations[j][env.job_progress[j]:])
            job = max(avail, key=rem)
        elif rule == "LWKR":
            def rem(j):
                return sum(o.duration for o in env.operations[j][env.job_progress[j]:])
            job = min(avail, key=rem)
        elif rule == "Random":
            job = int(env.rng.choice(avail))
        else:
            raise ValueError(f"Unknown rule: {rule}")
        env.step(job)
    return env.get_makespan()


if __name__ == "__main__":
    pt = JSPEnv.generate_random_instance(6, 5, seed=42)
    env = JSPEnv(6, 5, pt)
    print("Operations per job:")
    for j, ops in enumerate(env.operations):
        print(f"  Job {j}: {[(o.machine, o.duration) for o in ops]}")
    for rule in ["FIFO", "SPT", "LPT", "MWKR", "LWKR"]:
        mk = schedule_with_heuristic(env, rule)
        print(f"{rule}: makespan={mk:.1f}")

class DynamicJSPEnv(JSPEnv):
    """Job shop with jobs arriving over time."""
    def __init__(self, n_jobs, n_machines, processing_times, seed=None, arrival_scale=50.0):
        self.arrival_scale = arrival_scale
        self.arrived = [False] * n_jobs
        super().__init__(n_jobs, n_machines, processing_times, seed)

    def reset(self):
        state = super().reset()
        self.arrivals = [float(self.rng.exponential(self.arrival_scale)) for _ in range(self.n_jobs)]
        self.arrived = [False] * self.n_jobs
        return state

    def _get_available_operations(self):
        if not hasattr(self, 'arrivals'):
            return super()._get_available_operations()
        for j in range(self.n_jobs):
            if not self.arrived[j] and self.time >= self.arrivals[j]:
                self.arrived[j] = True
        avail = super()._get_available_operations()
        return [j for j in avail if self.arrived[j]]

    def _advance_time(self):
        while not self._get_available_operations() and not self.done:
            future_m = [m for m in self.machine_free if m > self.time + 1e-9]
            future_a = [a for j, a in enumerate(self.arrivals) if not self.arrived[j]]
            candidates = future_m + future_a
            if not candidates:
                if all(p == len(self.operations[j]) for j, p in enumerate(self.job_progress)):
                    self.done = True
                    return
                break
            self.time = min(candidates)
