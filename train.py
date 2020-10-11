from continuous_cartpole import ContinuousCartPoleEnv
from reward import reward
from ppo import PPO
from model import ActorCritic
from MonteCarlo import Memory
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

######## Hyperparameters #########
max_nb_episodes = 1000
T = 1024 #
N = 1
update_time = N*T
K_epochs = 25
batch_size = 32
eps_clip = 0.1  # to encourage policy change
gamma = 0.99
lr = 0.00025
betas = (0.9, 0.99)
action_std = 0.25
max_length_episode = 650
render = False
######## environment #########
env = ContinuousCartPoleEnv(reward)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
# torch.seed()
# env.seed()
# np.random.seed()

######## Cuda ##########
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

####### intialization########


running_reward = 0
avg_length = 0
avg_running_reward = 0
time_step = 0
nb_episode = 0
update_step = 0
length_episode = 0
curr_nb_episode = 0

sample_batch = Memory()
#old_policy = ActorCritic(state_dim, action_dim ).to(device)
ppo = PPO(ActorCritic, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, batch_size,eps_clip, device, env)

state = env.reset()
i_episode = 0
length_episode = 0
length_episodes = []
rewards = []
print_length_episodes = []
print_rewards = []

####### Start Trining ######
while (i_episode < max_nb_episodes):

    for t in range(T):
        if i_episode == max_nb_episodes - 20:
            render = True
        if render:
            env.render()
        # Old Policy

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = ppo.sample_action_old_policy(state, sample_batch)
        # print(action.grad)
        state, reward, done, _ = env.step(action.cpu().numpy())
        # Update sample batch with reward and is_terminal
        sample_batch.rewards.append(reward)
        sample_batch.is_terminals.append(done)
        running_reward += reward
        length_episode += 1
        if done or length_episode == max_length_episode:
            state = env.reset()
            done = True
            length_episodes.append(length_episode)
            rewards.append(running_reward)
            print_length_episodes.append(length_episode)
            print_rewards.append(running_reward)
            running_reward = 0
            length_episode = 0
            i_episode += 1
            if (i_episode % 20 == 0):
                # print(same_or_not)
                print('episode= {} \t avg length= {} \t avg reward= {}'.format(i_episode, np.mean(print_length_episodes),
                                                                                np.mean(print_rewards) ))

                torch.save(ppo.new_policy.state_dict(), './weights/PPO_{}.dat'.format(i_episode))
                print_rewards = []
                print_length_episodes = []
                running_reward = 0
    
    # print('Updating')
    #same_or_not = ppo.update(sample_batch)
    ppo.update(sample_batch)
    sample_batch.clear_memory()
    #print('Updated')
    
    
rewards = pd.DataFrame(rewards)
length_episodes = pd.DataFrame(length_episodes)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(rewards, label='rewards')
ax1.plot(rewards.rolling(50).mean(), label='avg rewards')
ax1.legend()
ax1.set_xlabel('episodes')
ax1.set_ylabel('Rewards')
ax2.plot(length_episodes, label='length episodes')
ax2.plot(length_episodes.rolling(50).mean(), label='avg length episodes')
ax2.legend()
ax2.set_xlabel('episodes')
ax2.set_ylabel('Length Episodes')
plt.show()

