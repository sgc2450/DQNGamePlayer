import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.notebook import tqdm
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import random
from collections import deque, Counter
import copy
import gymLogger

logger = gymLogger.logger


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

    def __init__(self, env: gym.Env, device, lr: float = 0.0001, eps: float = 0.50, 
                 gamma: float = 0.995, buffer_size: int = 10000, 
                 batch_size: int = 32, optimizer = None, loss = None):
        
        super(Agent,self).__init__()
        self.env = env
        self.device = device
        self.lr = lr   # Learning rate
        self.eps = eps # Exploration/Exploitation policy
        self.eps_decay = 0.9995
        self.eps_min = 0.01
        self.gamma = gamma # Reward Decay
        self.n_actions = env.action_space.n
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size



        # Q function approximator -> CNN to FC Linear NN
        self.q_net = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=10, stride=5, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(4096, 1228),
                nn.ReLU(),
                nn.Linear(1228, 512),
                nn.ReLU(),
                nn.Linear(512, self.n_actions)
            ).to(device)
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = Adam(self.q_net.parameters(), lr=self.lr)

        if loss is not None:
            self.loss = loss
        else:
            self.loss = nn.SmoothL1Loss()

        self.target_q_net = copy.deepcopy(self.q_net)

    def forward(self, x):
        x = self.q_net(x)
        return x

    """
        Implementing greedy epsilon eploration
    """
    def get_action(self, state, training = True):

        action_type = None
        #relative_action = None
        action = None

        if training:
            if (random.random() < self.eps):
                #relative_action = random.choice(list(RELATIVE_ACTIONS.keys()))
                action = self.env.action_space.sample()
                action_type = "exploration"
        if action is None:
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device) # Configure tensor dimensions
            q_actions = self.forward(state) 
            #print(f'\r{q_actions}\n', end='', flush=True)
            action = torch.argmax(q_actions).item()  
            #print(f'\n\r{action}', end='', flush=True)  
            action_type = "exploitation"

        if training:
            if (self.eps > self.eps_min):
                self.eps *= self.eps_decay # Decay eps upon exploration
        # print(f'Action {action}, {action_type}')
        return action, action_type
    
    def update_network(self):

        if len(self.replay_buffer) < self.batch_size:
            return 0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = torch.clamp(rewards / 10.0, min=-1.0, max=1.0)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.forward(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Compute value update for q value
            next_q_values = self.target_q_net(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            # We don't use bellman as explicitly as it is defined
            # We compute the target q instead, compute loss, and back propagate
            # Target q = R + ymax(a)Q(s',a')
            target_q_values = rewards + (self.gamma * max_next_q_values * (1-dones))
            
        self.optimizer.zero_grad()
        loss = self.loss(state_action_values, target_q_values)    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        #print(loss)
        return loss.item()

def main():
    env_name = "ALE/Pacman-v5"
    episodes = 1500
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    env = gym.make(env_name, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, scale_obs=True,screen_size=84, grayscale_obs=True)
    env = FrameStack(env, num_stack=4)
    model = Agent(env, device)
    obs, info = env.reset(seed=42)
    done = False
    update_frequency = 10

    for episode in range(episodes):
        exploration_count = 0
        exploitation_count = 0
        action_count = 0
        loss_tot = 0
        reward_tot = 0
        lives = 4
        score = 0
        episode_actions = []
        while not done:
            action, type = model.get_action(obs, training=True)
            episode_actions.append(action)
            if type == "exploitation":
                exploitation_count += 1
            elif type == "exploration":
                exploration_count += 1

            new_state, reward, done, trunc, info = env.step(action)

            score += reward
            if info['lives'] < lives:
                reward = -20
                lives = info['lives']
            if action == 0:
                reward = -10
            if reward == 0:
                reward = -1
            if action != 0:
                reward += 1
            if reward == 1:
                reward += 1


            model.replay_buffer.push(obs, action, reward, new_state, done)
            reward_tot += reward
            action_count += 1
            loss_tot += model.update_network()
            
            if done or trunc:
                for action, count in Counter(episode_actions).items():  
                    if count > 0.5 * len(episode_actions):
                        msg = (f'\n*********\nEpisode {episode + 1} ONE MOVE ERROR\n'
                               f'MOVE: {action}\n*********\n')          
                        logger.warning(msg)      
                msg = (f'\n___________________________________________\n'
                       f'Episode {episode + 1} Reward: {reward_tot/2}\n'
                       f'Loss: {loss_tot / action_count}\n'
                       f'Total Score: {score}\n' 
                       f'___________________________________________\n') 
                obs, info = env.reset()                             
            else:
                obs = new_state
        msg += (f'Exploration: {exploration_count}\n'
               f'Exploitation: {exploitation_count}\n'
               f'Total Moves: {action_count}\n'
               f'___________________________________________\n')
        logger.info(msg)
        done = False
        episode_actions = []

        
        # Update target q net with active q net
        if episode % update_frequency == 0:
            model.target_q_net.load_state_dict(model.q_net.state_dict())
            torch.save(model.state_dict(), "td_model.pth")

    env.close()

if __name__ == "__main__":
    main()
