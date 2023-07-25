import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def get_hyper_params():
    parser = argparse.ArgumentParser(description="Hyperparams for training q learning agent",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-env", "--environment", type=str, default="Taxi-v3", help="name of the gym environment")
    parser.add_argument("-model", "--model-name", type=str, default="model", help="name of the trained model")
    parser.add_argument("-video", "--prefix-video", type=str, default="visual_inspect", help="name of video for visual inspection")
    parser.add_argument("-g", "--gamma", type=float, default=0.7, help="discount rate")
    parser.add_argument("-es", "--epsilon-start", type=float, default=1.0, help="start of epsilon")
    parser.add_argument("-ee", "--epsilon-end", type=float, default=0.3, help="end of epsilon")
    parser.add_argument("-ed", "--epsilon-decay", type=float, default=5e-6, help="decay rate for epsilon, floating-point value")
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-n", "--n-episodes", type=int, default=50_000, help="number of episodes")
    args = parser.parse_args()
    config = vars(args)
    return config

def learning_curves(env, agent, filename, rolling_length=500):
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    model_folder_path = './plots'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    plt.savefig(f'plots/{filename}.png')