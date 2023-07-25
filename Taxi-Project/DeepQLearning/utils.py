 
import argparse
import matplotlib.pyplot as plt
import numpy as np

def get_hyper_params():
    parser = argparse.ArgumentParser(description="Hyperparams for training deep q agent",
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
    parser.add_argument("-bs", "--batch-size", type=int, default=512, help="batch size for the replay memory")
    parser.add_argument("-l1", "--layer-one", type=int, default=512, help="layer one size for dqn")
    parser.add_argument("-l2", "--layer-two", type=int, default=256, help="layer two size for dqn")
    args = parser.parse_args()
    config = vars(args)
    return config

def learning_curves(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(f'plots/{filename}.png')