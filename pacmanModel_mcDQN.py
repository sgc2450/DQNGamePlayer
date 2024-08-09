import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm.notebook import tqdm
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import random
from collections import deque, Counter
import gymLogger



logger = gymLogger.logger

ORIENTATIONS = ["noop", "up", "right", "left", "down"]

RELATIVE_ACTIONS = {
    "noop" : 0,
    "forward": 1,
    "right": 2,
    "left": 3,
    "backward": 4
}

def get_absolute_action(orientation, relative_action):
    orientation_index = ORIENTATIONS.index(orientation)
    absolute_action = (orientation_index + RELATIVE_ACTIONS[relative_action]) % 5
    return absolute_action

def update_orientation(orientation, relative_action):
    orientation_index = ORIENTATIONS.index(orientation)
    new_orientation_index = (orientation_index + RELATIVE_ACTIONS[relative_action]) % 5
    return ORIENTATIONS[new_orientation_index]

# Add experience buffer
# Make sure you get new state by env, not NN

#Adds short memory of expeirences to sample from 
class Agent(nn.Module):

    def __init__(self, env: gym.Env, device, lr: float = 0.1, eps: float = 0.8, 
                 gamma: float = 0.905, batch_size: int = 32, optimizer = None, loss = None):
        
        super(Agent,self).__init__()
        self.env = env
        self.device = device
        self.lr = lr   # Learning rate
        self.eps = eps # Exploration/Exploitation policy
        self.gamma = gamma # Reward Decay
        self.n_actions = env.action_space.n
        self.batch_size = batch_size
        #self.orientation = "up"

        # Q function approximator -> CNN to FC Linear NN
        self.q_net = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=10, stride=5, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(6272, 3136),
                nn.ReLU(),
                nn.Linear(3136, 512),
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
            self.loss = nn.MSELoss()

        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []

    def forward(self, x):
        x = self.q_net(x)
        return x
    
    def store_transition(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    """
        Implementing greedy epsilon eploration
    """
    def get_action(self, state, training=True):

        action_type = None
        #relative_action = None
        action = None

        if training is True:
            if (random.random() < self.eps):
                #relative_action = random.choice(list(RELATIVE_ACTIONS.keys()))
                action = self.env.action_space.sample()
                action_type = "exploration"
        if action is None:
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32).permute(0, 1, 2).unsqueeze(0).to(self.device) # Configure tensor dimensions
            q_actions = self.forward(state) 
            #print(f'\r{q_actions}', end='', flush=True)
            action = torch.argmax(q_actions).item()  
            #print(f'\n\r{action}', end='', flush=True)  
            #relative_action_index = torch.argmax(q_actions).item()
            #relative_action = list(RELATIVE_ACTIONS.keys())[relative_action_index % 5]
            action_type = "exploitation"

        if (self.eps > .01):
            self.eps *= 0.99995 # Decay eps upon exploration

        #absolute_action = get_absolute_action(self.orientation, relative_action)
        #self.orientation = update_orientation(self.orientation, relative_action)

        return action, action_type
    
    def update_network(self):

        G = 0
        returns = []
        loss_tot = 0

        # Discount the rewards from the episode
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0,G)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize the rewards
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() - 1e-5)

        
        for state, action, G in zip(self.episode_states, self.episode_actions, returns):
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32).permute(0, 1, 2).unsqueeze(0).to(self.device)
            q_values = self.forward(state)
            state_action_value = q_values[0, action]
            loss = self.loss(state_action_value, G)
            loss_tot += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        return loss_tot

        
def main():
    env_name = "ALE/Pacman-v5"
    episodes = 1000
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Training on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training on CPU.")

    env = gym.make(env_name, frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, scale_obs=True ,screen_size=84)
    env = FrameStack(env, num_stack=4)
    model = Agent(env, device)
    obs, info = env.reset()
    done = False

    reward_tot = 0
    loss_avg = 0
    for episode in range(episodes):
        exploration_count = 0
        exploitation_count = 0
        action_count = 0
        loss_tot = 0
        #model.orientation = "up"
        lives = 3
        while not done:
            action, type = model.get_action(obs)
            if type == "exploitation":
                exploitation_count += 1
            elif type == "exploration":
                exploration_count += 1

            new_state, reward, done, trunc, info = env.step(action)

            if info['lives'] < lives:
                reward = -1.0
                lives = info['lives']
            #if reward == 0:
                #reward = -0.10
            
            model.store_transition(obs, action, reward) 
            reward_tot += reward
            action_count += 1
            if done or trunc:
                for action, count in Counter(model.episode_actions).items():  
                    if count > 0.5 * len(model.episode_actions):
                        msg = (f'\n*********\nEpisode {episode + 1} ONE MOVE ERROR\n'
                               f'MOVE: {action}\n*********\n')          
                        logger.warning(msg)
                loss_tot = model.update_network()
                obs, info = env.reset()
                msg = (f'\n___________________________________________\n'
                       f'Episode {episode + 1} Reward: {reward_tot}\n'
                       f'Loss: {loss_tot / action_count}\n'
                       f'___________________________________________\n') 
                reward_tot = 0                                
            else:
                obs = new_state
        done = False
        msg += (f'Exploration: {exploration_count}\n'
               f'Exploitation: {exploitation_count}\n'
               f'Total Moves: {action_count}\n'
               f'___________________________________________\n')
        logger.info(msg)
        torch.save(model.state_dict(), "model.pth")
    env.close()

if __name__ == "__main__":
    main()
