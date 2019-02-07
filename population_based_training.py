import numpy as np
from scipy import stats
import torch
import agent

def population_based_training(agents, env, brain_name,
                              max_episode=20,
                              max_t=1100,
                              episodes_between_exploit=20,
                              p_threshold = 0.05,
                              checkpoint_episodes = None):
    checkpoint_episodes = checkpoint_episodes if checkpoint_episodes else episodes_between_exploit
    episode_i = len(agents[0].history) + 1
    while episode_i <= max_episode:
        episode_i += episodes_between_exploit
        randomized_agent_idx = np.random.permutation(len(agents))
        for agent_i in range(0, len(agents), 2):
            agent_pair = [agents[randomized_agent_idx[agent_i]], agents[randomized_agent_idx[agent_i+1]]]
            print("Paired {} with {}".format(agent_pair[0].name, agent_pair[1].name))
            prev_t = len(agent_pair[0].history)
            agent.ddpg_collab(
                agent_pair, env, brain_name,
                max_episode=prev_t + episodes_between_exploit,
                checkpoint_episodes=checkpoint_episodes
            )
            for i in range(0,2):
                print("\r\nAgent: {}, mean return: {:.2f}".format(agent_pair[i].name, agent_pair[i].get_running_mean_return(episodes_between_exploit)))
        for target_i, target in enumerate(agents):
            # uniformly select a candidate that is not the same as target
            candidate_i = np.random.randint(0, len(agents) - 1)
            if candidate_i >= target_i:
                candidate_i += 1
            candidate = agents[candidate_i]
            # exploit
            candidate_scores = candidate.history.tail(episodes_between_exploit)['return']
            target_scores = target.history.tail(episodes_between_exploit)['return']
            exploited = False
            if candidate_scores.mean() > target_scores.mean():
                statistic, p = stats.ttest_ind(candidate_scores, target_scores, equal_var=False)
                if p < p_threshold:
                    print("Overwriting {} with {}, p = {:.2f}".format(target.name, candidate.name, p))
                    agent_name = target.name
                    torch.save(target.full_save_dict(),
                              "retire_{}_episode_{}.pth".format(agent_name, len(target.history)))
                    exploited = True
                    target.load_state_dict(candidate.state_dict())
                else:
                    print("{} performed worse than {} but it wasn't significant, p = {:.2f}".format(target.name, candidate.name, p))
            if exploited:
                # explore
                source_hyperparameters = candidate.hyperparameter_dict()
                target.load_mutated_hyperparameter_dict(source_hyperparameters)
