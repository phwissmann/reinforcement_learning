import numpy as np
from matplotlib import pyplot
from collections import namedtuple

NormDist = namedtuple('NormDist', 'mean std')

NUM_STEPS = 500


def init_actions_stationary(num: int):
    """ Create Gaussian distributions with random means and stds"""
    actions = list()
    for i in range(num):
        mean = np.random.normal(0, 1)
        std = 1
        actions.append(NormDist(mean, std))

    return actions


def init_actions_non_stationary(num: int):
    """ Create Gaussian distributions with same random mean and std"""
    mean = np.random.normal(0, 1)
    std = 1
    actions = list()
    for i in range(num):
        actions.append(NormDist(mean, std))

    return actions


def update_actions_non_stationary(actions: list, perturbation: float):
    increment = np.random.normal(0, perturbation)

    updated_actions = list()
    for action in actions:
        updated_actions.append(NormDist(action.mean + increment, action.std))

    return updated_actions


def select_action(sample_averages: np.array, eps: float):
    if np.random.uniform() < eps:
        return np.random.choice(range(0, len(sample_averages)))
    else:
        return np.random.choice(np.where(sample_averages == sample_averages.max())[0])


def draw(actions: list, action: int):
    return np.random.normal(actions[action].mean, actions[action].std)


def eps_greedy(actions: list, eps: float, steps: int, non_stationary: bool = False):

    # Estimated rewards Q_t(a)
    sample_averages = np.zeros(len(actions))

    reward_per_step = np.zeros(steps)
    chosen_action = np.zeros(len(actions), dtype=int)

    for s in range(steps):

        if non_stationary:
            actions = update_actions_non_stationary(actions, 0.01)

        action = select_action(sample_averages, eps)
        chosen_action[action] = chosen_action[action]+1

        reward = draw(actions, action)
        reward_per_step[s] = reward

        #sample_averages[action] = sample_averages[action] + 1 / \
        #    (chosen_action[action])*(reward - sample_averages[action])

        alpha = 0.1
        sample_averages[action] = sample_averages[action] + alpha*(reward - sample_averages[action])



    return reward_per_step


def main(non_stationary: bool):

    eps = [0.0, 0.1, 0.01]

    num_runs = 1000
    num_steps = 10000
    num_actions = 10

    actions = init_actions_stationary(num_actions) if non_stationary else init_actions_stationary(num_actions)

    average_rewards = np.zeros((len(eps), num_steps))

    for e in range(len(eps)):
        for r in range(num_runs):
            # rewards of current run
            rewards = eps_greedy(actions, eps[e], num_steps, non_stationary)
            # incremental average
            average_rewards[e, :] = average_rewards[e][:] + \
                1/(r+1)*(rewards - average_rewards[e][:])

    steps = np.arange(start=0, stop=num_steps)
    pyplot.plot(steps, average_rewards[0], '-',
                average_rewards[1], '-', average_rewards[2], '-')
    pyplot.xlabel('Steps')
    pyplot.ylabel('Reward Average')
    pyplot.legend(['eps=0.0', 'eps=0.1,', 'eps=0.01'])

    title = "non_stationary" if non_stationary else "stationary"
    pyplot.savefig(title + '.png')
    pyplot.show()


if __name__ == "__main__":
    non_stationary = True
    main(non_stationary)
