import torch
import torch.nn as nn
#
###### initialization
# def init_params(m):
#     if isinstance(m, nn.Linear):
#         nn.init.normal_(m.weight, mean=0., std=0.5)
#         nn.init.constant_(m.bias, 0.1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # self.apply(init_params)

    def forward(self, state):
        raise NotImplementedError


### shared weights
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#
#         # action mean range -1 to 1
#         self.base = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#         )
#         self.actor = nn.Sequential(
#             self.base,
#             nn.Linear(64, action_dim),
#             nn.Tanh()
#         )
#
#         self.critic = nn.Sequential(
#             self.base,
#             nn.Linear(64, 1)
#         )
#         # self.apply(init_params)
#
#     def forward(self, state):
#         raise NotImplementedError


##### define by hand
# class actor(nn.Module):
#
#     def __init__(self, state_dim, action_dim):
#         super(actor, self).__init__()
#
#         self.linear1 = nn.Linear(state_dim, 64)
#         self.linear2 = nn.Linear(64, 64)
#         self.linear3 = nn.Linear(64, 32)
#         self.linear4 = nn.Linear(32, action_dim)
#         self.tanh = nn.Tanh()
#     def forward(self, x):
#         x = self.tanh(self.linear1(x))
#         x = self.tanh(self.linear2(x))
#         x = self.tanh(self.linear3(x))
#         x = self.tanh(self.linear4(x))
#         return x
#
#
# class critic(nn.Module):
#
#     def __init__(self, state_dim):
#         super(critic, self).__init__()
#
#         self.linear1 = nn.Linear(state_dim, 64)
#         self.linear2 = nn.Linear(64, 64)
#         self.linear3 = nn.Linear(64, 32)
#         self.linear4 = nn.Linear(32, 1)
#         self.tanh = nn.Tanh()
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.tanh(x)
#         x = self.tanh(self.linear2(x))
#         x = self.tanh(self.linear3(x))
#         x = self.tanh(self.linear4(x))
#         return x
#
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
#         self.actor = actor(state_dim, action_dim)
#         self.critic = critic(state_dim)