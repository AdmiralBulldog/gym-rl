
import gym
import math
import random
import numpy as np
import copy
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


env = gym.make('Pendulum-v0').unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

episode_scores = []
score_means = []
fig, ax = plt.subplots()


def plot_animate(i):
    ax.clear()
    ax.set_ylabel('score')
    ax.set_xlabel('episode')
    ax.set_title('Training')

    ax.plot(episode_scores, label='episode score')
    ax.plot(score_means, label='score 100 MA')
    

ani = animation.FuncAnimation(fig, plot_animate, interval=1000)


class ExperienceMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def store(self, transition):
        self.memory.append(transition)
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)    


class DQNN_actor(nn.Module):
    def __init__(self, state_space, action_space):
        #TODO Try batch norm to state input
        #TODO Try wiout batchnorm
        super(DQNN_actor, self).__init__()
        self.lin1 = nn.Linear(state_space, 400)
        self.lin2 = nn.Linear(400,300)
        self.lin3 = nn.Linear(300, action_space)
        nn.init.uniform_(self.lin1.weight, a=-0.003, b=0.003)
        nn.init.uniform_(self.lin2.weight, a=-0.003, b=0.003)
        nn.init.uniform_(self.lin3.weight, a=-0.003, b=0.003)
        
    def forward(self, x):
        #x = self.norm0(x.to(device))
        x = F.relu(self.lin1(x.to(device)))
        x = F.relu(self.lin2(x))
        return torch.tanh(self.lin3(x))
        #return x*2


class DQNN_critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQNN_critic, self).__init__()
        self.lin1_s = nn.Linear(state_space, 400)
        self.lin2 = nn.Linear(400+action_space,300)
        self.lin3 = nn.Linear(300, 1)
        nn.init.uniform_(self.lin1_s.weight, a=-0.003, b=0.003)
        nn.init.uniform_(self.lin2.weight, a=-0.003, b=0.003)
        nn.init.uniform_(self.lin3.weight, a=-0.003, b=0.003)
        
    def forward(self, states, actions):
        #TODO TRY CONCAT
        #TODO Try more batchnorm
        xs = states.to(device)
        xa = actions.to(device)
        xs = F.relu(self.lin1_s(xs))
        x = torch.cat((xs, xa), dim=1)
        x = F.relu(self.lin2(x))
        return self.lin3(x)


class OUNoise(object):
    def __init__(self, action_space=env.action_space, mu=0.0, theta=0.15, sigma=0.20):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = sigma
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action):
        #self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        ou_state = self.evolve_state()
        return np.clip(action + ou_state, self.low, self.high)


class Agent():
    def __init__(self, state_space, action_space, gamma, memory_size, learning_rate_actor, learning_rate_critic, tau, batch_size):
        self.actor_policy = DQNN_actor(state_space, action_space).to(device)
        self.critic_policy = DQNN_critic(state_space, action_space).to(device)
        self.actor_tgt = DQNN_actor(state_space, action_space).to(device)
        self.critic_tgt = DQNN_critic(state_space, action_space).to(device)
        self.noise = OUNoise()
        self.tau = tau
        self.gamma = gamma
        self.num_actions = action_space         
        self.batch_size = batch_size
        self.memory = ExperienceMemory(memory_size)
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer_actor = optim.Adam(self.actor_policy.parameters(), lr=learning_rate_actor)
        self.optimizer_critic = optim.Adam(self.critic_policy.parameters(), lr=learning_rate_critic, weight_decay=0.001)
        self.loss_func = nn.MSELoss()
        self.last_loss_critic = 999
        self.last_loss_actor = 999

    def select_act(self, state):
        self.actor_policy.eval()
        with torch.no_grad():
            return self.actor_policy(state).squeeze()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        else:    
            self.actor_policy.train()
            next_states, rewards, actions, states, done = list(zip(*self.memory.sample(self.batch_size)))
            next_states_stack = torch.cat(next_states, dim=0)
            rewards_stack = torch.stack(rewards, dim=0)
            actions_stack = torch.stack(actions, dim=0)
            states_stack = torch.cat(states, dim=0)
            done_stack = torch.stack(done, dim=0)

            #print(f"{states_stack=}, \n {actions_stack=},\n {rewards_stack=},\n {next_states_stack=},\n {done=}")
            next_actions = self.actor_tgt(next_states_stack)
            q_values = self.critic_policy(states_stack, actions_stack)
            target_q_values = rewards_stack + self.gamma * self.critic_tgt(next_states_stack, next_actions) * (1-done_stack)
                        
            self.optimizer_critic.zero_grad()
            loss_critic = self.loss_func(q_values, target_q_values)
            #self.last_loss_critic = loss_critic.item()
            loss_critic.backward()
            #for param in self.critic_policy.parameters():
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer_critic.step()

            policy_actions = self.actor_policy(states_stack)
            actions_qvalues = self.critic_policy(states_stack, policy_actions)
            self.optimizer_actor.zero_grad()

            #TODO CHECK IF MEAN WORKS AS EXPECTED
            loss_actor = -torch.mean(actions_qvalues)
            
            #self.last_loss_actor = loss_actor.item()
            loss_actor.backward()
            #for param in self.actor_policy.parameters():
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer_actor.step()    
            self._update_tgt_nets(self.actor_policy, self.actor_tgt)
            self._update_tgt_nets(self.critic_policy, self.critic_tgt)
            
    def _update_tgt_nets(self, policy, target):
        for policy_param, target_param in zip(policy.parameters(), target.parameters()):
            target_param.data.copy_(self.tau*policy_param.data + (1.0-self.tau)*target_param.data)      
    def get_last_losses(self):
        return self.last_loss_critic, self.last_loss_actor


env.reset()

episodes = 100000
steps_done = 0

action_space = env.action_space.shape[0]
state_space = 3
batch_size = 128
tau = 0.001
gamma = 0.99
memory_size = 100000
lr_actor = 0.0001
lr_critic = 0.001
agnt = Agent(state_space,action_space, gamma, memory_size, lr_actor, lr_critic, tau, batch_size)
done_reward = -100

action_array = np.zeros((4,10))
wiggling_tracker = [0]*4
            

# Seen mistakes: wrong shape in replay buffer
#                should use policy critic when learning actor?
with mlflow.start_run():
    mlflow.log_param("memory_size", memory_size)
    mlflow.log_param("tau", tau)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("actor_model", str(agnt.actor_policy))
    mlflow.log_param("critic_model", str(agnt.critic_policy))
    mlflow.log_param("loss", str(agnt.loss_func))
    mlflow.log_param("optmizer", str(agnt.optimizer_actor))
    mlflow.log_param("lr_actor", lr_actor)
    mlflow.log_param("lr_critic", lr_critic)

    mlflow.log_param("done_reward", done_reward)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("theta", agnt.noise.theta)
    mlflow.log_param("sigma", agnt.noise.sigma)
    
    for e in range(episodes):
        agnt.noise.reset()
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        episode_reward = 0
        
        for t in range(300):
            env.render()
            action = agnt.select_act(state).cpu()#.data.numpy()
            #print(f"{np.array(action)=}")   
            action = agnt.noise.get_action(np.array(action))     
            next_state, reward, done, _ = env.step(np.array(action))
            episode_reward += reward
            
            next_state = torch.tensor(next_state, dtype=torch.float, device=device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float, device=device).unsqueeze(0)
            done = torch.tensor(done, dtype=torch.float, device=device).unsqueeze(0)
                
            agnt.memory.store((next_state, reward, torch.tensor(action, dtype=torch.float, device=device), state, done))
            
            state = next_state
            agnt.learn()
            steps_done += 1

            if done:
                break

        episode_scores.append(episode_reward)
        score_means.append(sum(episode_scores[-100:]) / 100)
        last_loss_critic, last_loss_actor = agnt.get_last_losses()
        print(f"Episode {e} finished after {t} timesteps, loss critic: {last_loss_critic}, {last_loss_actor}")
        
    mlflow.log_metric("steps_done", steps_done)
    mlflow.log_metric("episodes done", e)
    mlflow.log_metric("last_avg_score", score_means[-1])
    mlflow.log_metric("last_score", episode_scores[-1])
    
env.close()        