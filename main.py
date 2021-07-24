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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

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
    def __init__(self, h, w, num_actions, eps_max=0.9, eps_min=0.05, eps_decay=200, gamma=0.9):
        self.policy_net = DQNN(h, w, num_actions)
        self.tgt_net = DQNN(h, w, num_actions)
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.num_actions = num_actions
        self.batch_size = 32
        self.memory = ExperienceMemory(10000)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.loss_func = nn.SmoothL1Loss()
        self.last_loss = 999
    def select_act(self, state):
        sample = random.random()
        epsilon = self.eps_min + (self.eps_max-self.eps_min) * math.exp(-1. * steps_done / self.eps_decay)
        if sample < epsilon:
            return torch.tensor(random.randrange(self.num_actions))
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].squeeze()
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        else:    
            next_states, rewards, actions, states = list(zip(*self.memory.sample(self.batch_size)))
            next_states_stack = torch.cat(next_states, dim=0)
            rewards_stack = torch.stack(rewards, dim=0)
            actions_stack = torch.stack(actions, dim=0)
            states_stack = torch.cat(states, dim=0)
            q_values = self.policy_net(states_stack).gather(1, actions_stack.view(-1,1))
            next_state_maxes = self.tgt_net(next_states_stack).max(1)[0].unsqueeze(1).detach()
            target_q_values = rewards_stack.unsqueeze(1) + self.gamma * next_state_maxes

            self.optimizer.zero_grad()
            #loss_func = nn.SmoothL1Loss()
            loss = F.smooth_l1_loss(q_values, target_q_values)
            #self.last_loss = loss.item()
            loss.backward()
            #for param in self.policy_net.parameters():
            #    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            
    def update_tgt_net(self):
        self.tgt_net.load_state_dict(self.policy_net.state_dict())        
    def get_last_loss(self):
        return self.last_loss    



episodes = 1000
env.reset()
init_screen = get_screen()
_, _, h, w = init_screen.shape
agnt = Agent(h,w,env.action_space.n)
tgt_update_interval = 10
steps_done = 0


for e in range(episodes):
    env.reset()
    current_screen = get_screen()
    last_screen = get_screen()
    state = last_screen - current_screen
    for t in count():
        action = agnt.select_act(state)
        _, reward, done, _ = env.step(action.item())
        last_screen = current_screen
        current_screen = get_screen()
        next_state = last_screen - current_screen
        if done:
            reward = torch.tensor(-reward)
            agnt.memory.store((next_state, reward, action, state))
            print(f"Episode {e} finished after {t} timesteps")
            print(f"Loss: {agnt.last_loss}, reward: {reward}")
            episode_durations.append(t+1)
            plot_durations()
            break
        reward = torch.tensor(reward)
        agnt.memory.store((next_state, reward, action, state))
        state = next_state
        agnt.learn()
        steps_done += 1
        if t % tgt_update_interval:
            agnt.update_tgt_net()
        

env.close()        