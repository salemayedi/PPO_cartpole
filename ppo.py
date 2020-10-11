from torch.distributions import Normal
import torch
import torch.optim as optim
import numpy as np
from model import ActorCritic
import torch.nn as nn
from MonteCarlo import discounted_rewards
from torch.utils.data import DataLoader, TensorDataset

class PPO():
    def __init__(self, ActorCritic, state_dim, action_dim,  action_std, lr, betas, gamma, K_epochs, batch_size,eps_clip, device, env):
        self.lr = lr # lr ADAM
        self.betas = betas # momentum in ADAM
        self.gamma = gamma # formonte carlo reward
        self.eps_clip = eps_clip # clipping
        self.K_epochs = K_epochs # k epochs to update the SGD
        self.batch_size = batch_size
        self.device = device
        self.action_std = action_std
        self.env = env

        self.old_policy = ActorCritic(state_dim, action_dim).to(device)
        self.new_policy = ActorCritic(state_dim, action_dim).to(device)


        self.optimizer = optim.Adam(self.new_policy.parameters(), lr=lr, betas=betas)
        #### make sure to initialize them the same
        self.old_policy.load_state_dict(self.new_policy.state_dict())

        #self.MseLoss = nn.SELoss()

    def sample_action_old_policy(self, state, sample_batch):
        state = torch.Tensor(state.reshape(1, -1)).squeeze()
        action_mu = self.old_policy.actor(state)
        action_var = self.action_std
        dist = Normal(action_mu, action_var ** 2)
        action = dist.sample()
        action = torch.clamp(action, self.env.action_space.low[0], self.env.action_space.high[0])
        logprob = dist.log_prob(action)

        #### Update the sample batch with thee result of the observation
        sample_batch.states.append(state)
        sample_batch.actions.append(action)
        sample_batch.logprobs.append(logprob)
        return action

    def update (self, sample_batch):

        ## Monte Carlo Estimate of rewards
        rewards = discounted_rewards(sample_batch.rewards, sample_batch.is_terminals, self.device).unsqueeze(1).detach()
        ### get old_states, old_actions, old_probs from sample_batch
        old_states = torch.stack(sample_batch.states).to(self.device).detach()
        old_actions = torch.stack(sample_batch.actions).to(self.device).detach()
        old_logprobs = torch.stack(sample_batch.logprobs).to(self.device).detach()

        dataset = TensorDataset (rewards, old_states, old_actions, old_logprobs)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.K_epochs):
            # add mini batch
            for idx, batch_data in enumerate(dataloader):
                #print('epoch: ', epoch,  ' batch_idx ', idx)
                (batch_rewards, batch_old_states, batch_old_actions, batch_old_logprobs) = batch_data
                #### evaluate old actions and values in the new policy
                new_action_mu = self.new_policy.actor(batch_old_states)
                new_action_var = self.action_std
                new_dist = Normal(new_action_mu, new_action_var**2)

                new_logprobs = new_dist.log_prob(batch_old_actions)
                new_dist_entropy = new_dist.entropy()#.reshape(-1)
                new_state_values = self.new_policy.critic(batch_old_states)#.squeeze()

                # ratio of new_policy/ old_policy
                ratios = torch.exp(new_logprobs - batch_old_logprobs)#.reshape(-1)
                # Losses

                advantages = batch_rewards - new_state_values


                surr1 = ratios * advantages

                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * (new_state_values- batch_rewards)**2 - 0.01 * new_dist_entropy

                # gradient step
                self.optimizer.zero_grad()
                #loss.register_hook(lambda grad: print(grad))
                #loss.mean().backward(retain_graph=True )
                loss.mean().backward()
                # check if you want to clip the weights
                #torch.nn.utils.clip_grad_norm_(self.new_policy.parameters(), 10)
                self.optimizer.step()
        #import pdb; pdb.set_trace()
        # update the old policy with the wights of the new policy
        self.old_policy.load_state_dict(self.new_policy.state_dict())

