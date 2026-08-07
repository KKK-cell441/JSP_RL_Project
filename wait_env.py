"""Add wait action to job shop env: policy can choose to advance time to next event.

This enables learning WHEN to schedule, not just WHAT to schedule.
"""
import numpy as np
from jsp_env import JSPEnv


class WaitableJSPEnv(JSPEnv):
    """JSP environment where action -1 means 'wait until next event'."""

    def __init__(self, n_jobs, n_machines, processing_times, seed=None, max_waits=20):
        super().__init__(n_jobs, n_machines, processing_times, seed)
        self.max_waits = max_waits
        self.wait_count = 0

    def reset(self):
        state = super().reset()
        self.wait_count = 0
        return state

    def _get_wait_available(self):
        """Whether a wait action is possible (there is a future event)."""
        future = [m for m in self.machine_free if m > self.time + 1e-9]
        return len(future) > 0 and self.wait_count < self.max_waits

    def do_wait(self):
        """Advance time to the next machine-free event. Returns new state."""
        future = [m for m in self.machine_free if m > self.time + 1e-9]
        if future:
            self.time = min(future)
            self.wait_count += 1
        # Check if any jobs complete
        return self._get_state()

    def get_available_actions(self):
        """Return list of job indices plus -1 for wait if available."""
        avail = self._get_available_operations()
        if self._get_wait_available():
            return avail + [-1]
        return avail


def run_with_wait(env, policy_fn, max_steps=200):
    """Run env with policy that can choose wait (-1)."""
    env.reset()
    steps = 0
    while not env.done and steps < max_steps:
        avail = env._get_available_operations()
        if not avail:
            if env._get_wait_available():
                env.do_wait()
            else:
                env._advance_time()
            steps += 1
            continue
        can_wait = hasattr(env, "_get_wait_available") and env._get_wait_available()
        action = policy_fn(env, avail, can_wait)
        if action == -1:
            env.do_wait()
        else:
            env.step(action)
        steps += 1
    return env.get_makespan(), sum(env.job_done_time) / env.n_jobs
