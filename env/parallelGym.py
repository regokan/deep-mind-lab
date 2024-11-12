from multiprocessing import Pipe, Process
from typing import Any, List, Tuple

import numpy as np

from .openai import Gym


class CloudpickleWrapper:
    """Wrapper to enable multiprocessing with cloudpickle serialization."""

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    """Worker process to handle environment interactions asynchronously."""
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info, _ = env.step(data)
                if done:
                    ob = env.reset()[0]
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset()[0]
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
        except EOFError:
            print("EOFError: Worker encountered an issue and will shut down.")
            break


class ParallelGym(Gym):
    """Parallelized version of the Gym environment."""

    def __init__(self, name: str, n_envs: int, render_mode: str = "rgb_array"):
        super().__init__(name, render_mode)
        self.n_envs = n_envs
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.processes = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(self.env))
            )
            for work_remote, remote in zip(self.work_remotes, self.remotes)
        ]
        for process in self.processes:
            process.daemon = True
            process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step_async(self, actions: List[Any]):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> np.ndarray:
        """Reset all environments and return initial observations."""
        try:
            # Send reset signal to all remotes and wait for all environments to confirm reset
            for remote in self.remotes:
                remote.send(("reset", None))

            # Capture the initial observations returned by each environment
            initial_obs = np.stack([remote.recv() for remote in self.remotes])
            return initial_obs

        except EOFError as e:
            print(
                "EOFError during reset. One or more environments failed to reset properly."
            )
            # Optionally add a retry mechanism or a fallback here if EOFError persists.
            raise e

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True