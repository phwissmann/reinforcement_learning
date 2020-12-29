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

    updated_actions = list()
    for action in actions:
        increment = np.random.normal(0, perturbation)
        updated_actions.append(NormDist(action.mean + increment, action.std))

    return updated_actions


def select_action(sample_averages: np.array, eps: float):
    if np.random.uniform() < eps:
        return np.random.choice(range(0, len(sample_averages)))
    else:
        return np.random.choice(np.where(sample_averages == sample_averages.max())[0])


def draw(actions: list, action: int):
    return np.random.normal(actions[action].mean, actions[action].std)


def find_optimal_action(actions: list):
    max_mean = 0.0

    for a in actions:
        if a.mean > max_mean:
            max_mean = a.mean

    optimal_actions = [actions.index(x) for x in actions if x.mean == max_mean]
    return optimal_actions


def eps_greedy(actions: list, eps: float, steps: int, non_stationary: bool = False):

    # Estimated rewards Q_t(a)
    sample_averages = np.zeros(len(actions))

    reward_per_step = np.zeros(steps)
    optimal_action_per_step = np.zeros(steps)
    action_chosen_count = np.zeros(len(actions), dtype=int)

    for s in range(steps):

        if non_stationary:
            actions = update_actions_non_stationary(actions, 0.01)

        optimal_action = find_optimal_action(actions)

        selected_action = select_action(sample_averages, eps)

        optimal_action_per_step[s] = 1 if selected_action in optimal_action else 0

        action_chosen_count[selected_action] = action_chosen_count[selected_action]+1

        reward = draw(actions, selected_action)
        reward_per_step[s] = reward

        # Sample average
        sample_averages[selected_action] = sample_averages[selected_action] + 1 / \
            (action_chosen_count[selected_action]) * \
            (reward - sample_averages[selected_action])

        # Constant step size parameter
        # alpha = 0.1
        # sample_averages[selected_action] = sample_averages[selected_action] + \
        #    alpha*(reward - sample_averages[selected_action])

    return reward_per_step, optimal_action_per_step


def main(non_stationary: bool):

    eps = [0.0, 0.1, 0.01]

    num_runs = 1000
    num_steps = 2000
    num_actions = 10

    actions = init_actions_non_stationary(
        num_actions) if non_stationary else init_actions_stationary(num_actions)

    average_rewards = np.zeros((len(eps), num_steps))
    optimal_actions = np.zeros((len(eps), num_steps))

    for e in range(len(eps)):
        for r in range(num_runs):
            # rewards of current run
            rewards, optimal_action_selected = eps_greedy(
                actions, eps[e], num_steps, non_stationary)
            # incremental average of rewards
            average_rewards[e, :] = average_rewards[e][:] + \
                1/(r+1)*(rewards - average_rewards[e][:])
            # incremental average of optimal actions taken
            optimal_actions[e, :] = optimal_actions[e][:] + 1 / \
                (r+1)*(optimal_action_selected - optimal_actions[e][:])

    steps = np.arange(start=0, stop=num_steps)

    f, (pl1, pl2) = pyplot.subplots(2, 1, sharex=True)

    pl1.plot(steps, average_rewards[0], '-',
             average_rewards[1], '-', average_rewards[2], '-')
    pl1.set_xlabel('Steps')
    pl1.set_ylabel('Reward Average')
    pl1.legend(['eps=0.0', 'eps=0.1,', 'eps=0.01'])

    title = "non_stationary" if non_stationary else "stationary"

    pl2.plot(steps, optimal_actions[0], '-')
    pl2.plot(steps, optimal_actions[1], '-')
    pl2.plot(steps, optimal_actions[2], '-')

    pl2.set_xlabel('Steps')
    pl2.set_ylabel('Optimal Action')
    pl2.legend(['eps=0.0', 'eps=0.1,', 'eps=0.01'])

    title = "non_stationary" if non_stationary else "stationary"
    f.savefig(title + '.png')
    f.show()


if __name__ == "__main__":
    non_stationary = True
    main(non_stationary)
