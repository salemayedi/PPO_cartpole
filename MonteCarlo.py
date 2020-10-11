import torch

#### Returns based on Monte Carlo
def discounted_rewards(rewards, masks, device, gamma=0.99):
    new_rewards = []
    discounted_reward = 0
    for reward, is_terminal in zip(reversed(rewards), reversed(masks)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        new_rewards.insert(0, discounted_reward)

    new_rewards = torch.tensor(new_rewards).to(device)
    #new_rewards = (new_rewards - new_rewards.mean()) / (new_rewards.std() + 1e-4)
    return new_rewards

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]