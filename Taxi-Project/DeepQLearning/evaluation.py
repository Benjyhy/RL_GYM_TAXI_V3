from __future__ import annotations

import gym
from gym.wrappers import RecordVideo
import torch as T

from main import DeepQNetwork

from utils import get_hyper_params

class Evaluation:
    def __init__(
        self
    ):
        self.hps = get_hyper_params()
        self.env = gym.make(self.hps["environment"], render_mode="rgb_array")
        self.model = self._prepare()

    def _prepare(self):
        model = DeepQNetwork(
            0.001,
            n_actions=self.env.action_space.n, 
            input_dims=len(list(self.env.decode(self.env.reset()[0]))), 
            fc1_dims=self.hps["layer_one"], 
            fc2_dims=self.hps["layer_two"]
        )
        model.load_state_dict(T.load(f"model/{self.hps['model_name']}.pth"))
        model.eval()
        return model
    
    def _get_action(self, obs):
        state = T.tensor(list(self.env.decode(obs))).view(1,-1)
        actions = self.model.forward(state)
        action = T.argmax(actions).item()
        return  action
    
    def visual_inspection(self):
        # Wrap the environment with a monitor
        self.env = RecordVideo(self.env, './video', name_prefix=self.hps["prefix_video"])

        # Run the agent for one episode
        obs = self.env.reset()
        obs = obs[0]
        done = False

        while not done:
            action = self._get_action(obs)
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

        # Close the environment and the monitor
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

