from agent import DQNAgent
import gym
import numpy as np
from utils import get_hyper_params, learning_curves

class Trainer():
    def __init__(self):
        self.hps = get_hyper_params()
        self.env = gym.make(self.hps["environment"])
        self.agent = DQNAgent(
            gamma=self.hps["gamma"], 
            epsilon=self.hps["epsilon_start"], 
            batch_size=self.hps["batch_size"], 
            n_actions=self.env.action_space.n, 
            eps_end=self.hps["epsilon_end"], 
            eps_dec=self.hps["epsilon_decay"],
            input_dims=len(list(self.env.decode(self.env.reset()[0]))), 
            lr=self.hps["learning_rate"],
            fc1_dims=self.hps["layer_one"],
            fc2_dims=self.hps["layer_two"]
        )
        self.scores = []
        self.eps_history = []
        self.n_games = self.hps["n_episodes"]

    def train(self):
        for i in range(self.n_games):
            score = 0
            done = False
            observation = self.env.reset()[0]
            state = np.array(list(self.env.decode(observation)))

            while not done:
                if isinstance(observation, int):
                    observation = np.array(list(self.env.decode(observation)))
                action = self.agent.choose_action(self.env, observation)
                observation_, reward, terminated, trundated, info = self.env.step(action)
                done = terminated or trundated
                observation_ = np.array(list(self.env.decode(observation_)))
                score += reward
                self.agent.store_transition(observation, action, reward, observation_, done, self.env)
                self.agent.learn()
                observation = observation_

            self.scores.append(score)
            self.eps_history.append(self.agent.epsilon)

            avg_score = np.mean(self.scores[-100:])

            if i % 100 == 0:
                print('episode', i, 'score %.2f' % score,
                    'average score %.2f' % avg_score,
                    'epsilon %.2f' %self.agent.epsilon)

        self.agent.Q_eval.save(f'{self.hps["model_name"]}.pth')

        x = [i + 1 for i in range(self.n_games)]
        learning_curves(x, self.scores, self.eps_history, "taxi_v3_plot_learning")

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()