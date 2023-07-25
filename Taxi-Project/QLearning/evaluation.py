from __future__ import annotations
import numpy as np

import gym
from gym.wrappers import RecordVideo

from utils import get_hyper_params
import dill

class Evaluation:
    def __init__(
        self
    ):
        self.hps = get_hyper_params()
        self.q_table = self._prepare()
        self.env = gym.make(self.hps["environment"], render_mode="rgb_array")

    def _prepare(self):
        with open(f"q_tables/{self.hps['model_name']}", 'rb') as file:
            q_table = dill.load(file)
        return q_table
    
    def _get_action(self, obs):
        return int(np.argmax(self.q_table[obs]))
    
    def visual_inspection(self):
        self.env = RecordVideo(self.env, './video', name_prefix=self.hps["prefix_video"])
        obs = self.env.reset()
        obs = obs[0]
        done = False

        while not done:
            action = self._get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
        self.env.close()
    
    def success_rate(self, n_episodes=100):
        # Evaluate the agent for 100 episodes with epsilon=0 (SUCCESS RATE)
        success_count = 0
        for _ in range(n_episodes):
            obs = self.env.reset()
            obs = obs[0]
            done = False
            while not done:
                action = self._get_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

            # Check if the agent successfully completed the task
            if reward == 20:
                success_count += 1

        success_rate = success_count / n_episodes
        print("Success rate: {:.2f}%".format(success_rate * 100))
            

if __name__ == '__main__':
    eval = Evaluation()
    eval.success_rate()
    eval.visual_inspection()

