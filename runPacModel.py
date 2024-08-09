import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import numpy as np
import torch.nn as nn
from pacmanModel_mcDQN import Agent


def load_model(model_path, env, device):
    model = Agent(env, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

def main():
    env_name = "ALE/Pacman-v5"
    model_path = 'model.pth'
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Running on GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Running on CPU.")

    env = gym.make(env_name, render_mode='human', frameskip=1)
    env = AtariPreprocessing(env, frame_skip=8, scale_obs=True ,screen_size=84)
    env = FrameStack(env, num_stack=4)

    model = load_model(model_path, env, device)

    obs, info = env.reset(seed=42)
    done = False

    while True:
        action, info = model.get_action(obs, training=False)
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        if done or trunc:
            obs, info = env.reset(seed=42)

if __name__ == "__main__":
    main()