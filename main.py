import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExperienceMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def store(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size=32):
        random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)    

class DQNN(nn.Module):
    #Try stride = 1
    def __init__(self, h, w, outputs):
        super(DQNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

   
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.shape[0], -1))


class Agent():
    def __init__(self, h, w, num_actions, epsilon=0.5, eps_min=0.05):
        self.policy_net = DQNN(h, w, 1)
        self.tgt_net = DQNN(h, w, 1)
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.batch_size = 32
        self.memory = ExperienceMemory(10000, self.batch_size)
    def select_act(self, state):
        sample = random.random()
        if sample < self.epsilon:
            return torch.tensor(random.randrange(self.num_actions))
        else:
            with torch.no_grad():
                return self.policy_net(state).max(0)[1]
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        else:    
            next_states, rewards, actions, states = list(zip(*self.memory.sample(self.batch_size)))
            next_states_stack = torch.stack(next_states, dim=0)
            rewards_stack = torch.stack(rewards, dim=0)
            actions_stack = torch.stack(actions, dim=0)
            states_stack = torch.stack(states, dim=0)
            outputs = self.policy_net(states_stack).gather(1, actions_stack.view(-1,1))
            qvalues = 

def get_screen():
    return 0



episodes = 1000
agnt = Agent(1,1,env.action_space.n)

for e in range(episodes):
    env.reset()
    current_screen = get_screen()
    last_screen = get_screen()
    state = last_screen - current_screen
    for t in count():
        action = agnt.select_act(state)
        _, reward, done, _ = env.step(action)
        if done:
            break
        last_screen = current_screen
        current_screen = get_screen()
        next_state = last_screen - current_screen
        agnt.memory.store((next_state, reward, action, state))
        state = next_state
        agnt.learn()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()        