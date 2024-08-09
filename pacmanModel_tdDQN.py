import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.notebook import tqdm
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import random
from collections import deque




ORIENTATIONS = ["up", "right", "down", "left"]

RELATIVE_ACTIONS = {
    "forward": 0,
    "right": 1,
    "backward": 2,
    "left": 3
}

def get_absolute_action(orientation, relative_action):
    orientation_index = ORIENTATIONS.index(orientation)
    absolute_action = (orientation_index + RELATIVE_ACTIONS[relative_action]) % 4
    return absolute_action

def update_orientation(orientation, relative_action):
    orientation_index = ORIENTATIONS.index(orientation)
    new_orientation_index = (orientation_index + RELATIVE_ACTIONS[relative_action]) % 4
    return ORIENTATIONS[new_orientation_index]

# Add experience buffer
# Make sure you get new state by env, not NN

#Adds short memory of expeirences to sample from 
class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)
    

class Agent(nn.Module):

    def __init__(self, env: gym.Env, device, lr: float = 0.1, eps: float = 0.80, 
                 gamma: float = 0.995, buffer_size: int = 10000, 
                 batch_size: int = 32, optimizer = None, loss = None):
        
        super(Agent,self).__init__()
        self.env = env
        self.device = device
        self.lr = lr   # Learning rate
        self.eps = eps # Exploration/Exploitation policy
        self.gamma = gamma # Reward Decay
        self.n_actions = env.action_space.n
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.orientation = "up"

        # Q function approximator -> CNN to FC Linear NN
        self.q_net = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=8, stride=4, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_actions)
            ).to(device)
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(self.q_net.parameters(), lr=self.lr)

        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.MSELoss()


    def forward(self, x):
        x = self.q_net(x)
        return x

    """
        Implementing greedy epsilon eploration
    """
    def get_action(self, state):

        action_type = None

        if (random.random() < self.eps):
            relative_action = random.choice(list(RELATIVE_ACTIONS.keys()))
            action_type = "exploration"
            if (self.eps > .01):
                self.eps *= 0.99995 # Decay eps upon exploration
        else: 
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32).permute(0, 1, 2).unsqueeze(0).to(self.device) # Configure tensor dimensions
            q_actions = self.forward(state)           
            relative_action_index = torch.argmax(q_actions).item()
            relative_action = list(RELATIVE_ACTIONS.keys())[relative_action_index % 4]
            action_type = "exploitation"

        absolute_action = get_absolute_action(self.orientation, relative_action)
        self.orientation = update_orientation(self.orientation, relative_action)

        return absolute_action, action_type
    
    def update_network(self):

        if len(self.replay_buffer) < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        states = torch.tensor(states, dtype=torch.float32).permute(0, 1, 2, 3).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 1, 2, 3).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.forward(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Compute value update for q value
            next_q_values = self.forward(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            # We don't use bellman as explicitly as it is defined
            # We compute the target q instead, compute loss, and back propagate
            # Target q = R + ymax(a)Q(s',a')
            expected_state_action_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = self.loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def main():
    env_name = "ALE/Boxing-v5"
    episodes = 1000
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    env = gym.make(env_name)
    env = AtariPreprocessing(env, frame_skip=1, scale_obs=True ,screen_size=25)
    env = FrameStack(env, num_stack=16)
    model = Agent(env, device)
    obs, info = env.reset(seed=42)
    done = False

    reward_tot = 0
    loss_avg = 0
    for episode in range(episodes):
        exploration_count = 0
        exploitation_count = 0
        action_count = 0
        loss_tot = 0
        while not done:
            action, type = model.get_action(obs)
            if type == "exploitation":
                exploitation_count += 1
            elif type == "exploration":
                exploration_count += 1
            new_state, reward, done, trunc, info = env.step(action)
            model.replay_buffer.push(obs, action, reward, new_state, done)
            loss_tot += model.update_network()
            reward_tot += reward
            action_count += 1
            if done or trunc:
                obs, info = env.reset(seed=42)
                print('___________________________________________')
                print(f'Episode {episode + 1} Reward: {reward_tot}')
                print(f'Loss: {loss_tot / action_count}')
                reward_tot = 0                                
            else:
                obs = new_state
        done = False
        print(f'Exploration: {exploration_count} \nExploitation: {exploitation_count}')
        print(f'Total Moves: ', action_count)
        print('___________________________________________')
    
        torch.save(model.state_dict(), "model.pth")
    env.close()

if __name__ == "__main__":
    main()
