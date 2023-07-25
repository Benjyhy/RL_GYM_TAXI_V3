from agent import TaxiAgent
from IPython.display import clear_output
from tqdm import tqdm
import gym
from utils import learning_curves, get_hyper_params

class Trainer():
    def __init__(self):
        self.hps = get_hyper_params()
        self.env = gym.make(self.hps["environment"], render_mode="rgb_array")
        self.agent = TaxiAgent(
            self.env,
            learning_rate=self.hps["learning_rate"],
            initial_epsilon=self.hps["epsilon_start"],
            epsilon_decay=self._get_epsilon_decay(),
            final_epsilon=self.hps["epsilon_end"],
            discount_factor=self.hps["gamma"]
        )

    def _get_epsilon_decay(self):
        return self.hps["epsilon_start"] / (self.hps["n_episodes"] / 2)

    def train(self):
        done = False
        observation, info = self.env.reset()
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, deque_size=self.hps["n_episodes"])
        for episode in tqdm(range(self.hps["n_episodes"])):
            obs, info = self.env.reset()
            done = False
            clear_output()
            
            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(obs, action, reward, terminated, next_obs)
                done = terminated or truncated
                obs = next_obs

            self.agent.decay_epsilon()
        self.agent.save_q_table(self.hps["model_name"])
        learning_curves(self.env, self.agent, "learning_curves_q_table")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()