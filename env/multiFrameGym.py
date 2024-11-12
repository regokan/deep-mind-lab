from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from policy.base import BasePolicy

from .openai import Gym


class MultiFrameGym(Gym):
    """Concrete implementation for OpenAI Gym environments."""

    def __init__(self, name: str, render_mode: str = "rgb_array"):
        super().__init__(name, render_mode)

    def watch(
        self,
        policy: BasePolicy,
        display: Any,
        action1: int,
        action2: int,
        steps: int = 1000,
        nrand: int = 1,
        preprocess_fn: Callable = None,
        device: str = "cpu",
    ) -> None:
        self.reset()

        # star game
        self.step(1)

        # perform nrand random steps in the beginning
        for _ in range(nrand):
            frame1, _, is_done, _, _ = self.step(np.random.choice([action1, action2]))
            frame2, _, is_done, _, _ = self.step(0)

        img = plt.imshow(self.render())
        for _ in range(steps):

            frame_input = (
                preprocess_fn([frame1, frame2])
                if preprocess_fn
                else np.array([frame1, frame2])
            ).to(device)

            prob = policy.select_action(frame_input)

            # RIGHT = 4, LEFT = 5
            action = action1 if np.random.rand() < prob else action2
            frame1, _, is_done, _, _ = self.step(action)
            frame2, _, is_done, _, _ = self.step(0)

            img.set_data(self.render())
            plt.axis("off")
            display.display(plt.gcf())
            display.clear_output(wait=True)

            if is_done:
                break

        self.close()

        return
