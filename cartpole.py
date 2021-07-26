import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import namedtuple, deque
from itertools import count
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = gym.make('CartPole-v1').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




episode_durations = []
duration_means = []
fig, ax = plt.subplots()

def plot_animate(i):
    ax.clear()
    ax.set_ylabel('duration')
    ax.set_xlabel('episode')
    ax.set_title('Training')

    ax.plot(episode_durations, label='episode duration')
    ax.plot(duration_means, label='duration 100 MA')
    

ani = animation.FuncAnimation(fig, plot_animate, interval=1000)

class ExperienceMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def store(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size=32):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)    

class DQNN(nn.Module):
    #Try stride = 1
    def __init__(self, input_dim, output_dim):
        super(DQNN, self).__init__()
        self.lin1 = nn.Linear(input_dim, 32)
        self.norm1 = nn.LayerNorm(32)
        self.lin2 = nn.Linear(32,32)
        self.norm2 = nn.LayerNorm(32)
        self.lin3 = nn.Linear(32, output_dim)
        #self.bn1 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.norm1(self.lin1(x)))
        x = F.relu(self.norm2(self.lin2(x)))
        x = F.relu(self.lin3(x))
        return x



class Agent():
    def __init__(self, state_space, action_space, gamma, memory_size, learning_rate, epsilon, eps_decay, eps_min):
        self.policy_net = DQNN(state_space, action_space)
        self.tgt_net = DQNN(state_space, action_space)
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.num_actions = action_space
        self.batch_size = 32
        self.memory = ExperienceMemory(memory_size)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters(), learning_rate)
        self.loss_func = nn.MSELoss()
        self.last_loss = 999
    def select_act(self, state):
        sample = random.random()
        self.epsilon = max(self.eps_min, self.epsilon * eps_decay)
        if sample < self.epsilon:
            return torch.tensor(random.randrange(self.num_actions))
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].squeeze()
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        else:    
            next_states, rewards, actions, states, done = list(zip(*self.memory.sample(self.batch_size)))
            next_states_stack = torch.cat(next_states, dim=0)
            rewards_stack = torch.stack(rewards, dim=0)
            actions_stack = torch.stack(actions, dim=0)
            states_stack = torch.cat(states, dim=0)
            q_values = self.policy_net(states_stack).gather(1, actions_stack.view(-1,1))
            next_state_maxes = self.tgt_net(next_states_stack).max(1)[0].unsqueeze(1).detach()
            target_q_values = rewards_stack.unsqueeze(1) + self.gamma * next_state_maxes
            self.optimizer.zero_grad()
            loss = self.loss_func(q_values, target_q_values)
            #loss = F.mse_loss(q_values, target_q_values)
            self.last_loss = loss.item()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            
    def update_tgt_net(self):
        self.tgt_net.load_state_dict(self.policy_net.state_dict())        
    def get_last_loss(self):
        return self.last_loss    



env.reset()

episodes = 1000
steps_done = 0

action_space = env.action_space.n
state_space = 4
tgt_update_interval = 10
epsilon = 0.99
eps_decay = 0.995
eps_min = 0.05
gamma = 0.95
memory_size = 10000
lr = 0.0001
agnt = Agent(state_space,action_space, gamma, memory_size, lr, epsilon, eps_decay, eps_min)
done_reward = 0

with mlflow.start_run():
    mlflow.log_param("tgt_update_interval", tgt_update_interval)
    mlflow.log_param("memory_size", memory_size)
    mlflow.log_param("epsilon", epsilon)
    mlflow.log_param("eps_decay", eps_decay)
    mlflow.log_param("eps_min", eps_min)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("model", str(agnt.policy_net))
    mlflow.log_param("loss", str(agnt.loss_func))
    mlflow.log_param("optmizer", str(agnt.optimizer))
    mlflow.log_param("lr", lr)
    mlflow.log_param("done_reward", done_reward)

    for e in range(episodes):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        for t in count():
            env.render()
            action = agnt.select_act(state)
            #Test adding done in memory and checking it in learn
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            if done:
                reward = torch.tensor(done_reward)
                agnt.memory.store((next_state, reward, action, state, done))
                print(f"Episode {e} finished after {t} timesteps")
                episode_durations.append(t+1)
                duration_means.append(sum(episode_durations[-100:]) / 100)
                break
            reward = torch.tensor(reward)
            agnt.memory.store((next_state, reward, action, state, done))
            state = next_state
            agnt.learn()
            steps_done += 1
            if t % tgt_update_interval:
                agnt.update_tgt_net()
    
        mlflow.log_metric("steps_done", steps_done)
        mlflow.log_metric("episodes done", e)
        mlflow.log_metric("last_avg_score", duration_means[-1])
        mlflow.log_metric("last_score", episode_durations[-1])
            

env.close()        