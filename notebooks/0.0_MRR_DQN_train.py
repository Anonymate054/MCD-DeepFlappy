import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import flappy_bird_gymnasium
import matplotlib.pyplot as plt

from models.dqn import DQN
from models.replay_buffer import ReplayBuffer

# Hiperparámetros
EPISODES = 5000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 5000
LR = 1e-4
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 100

# Inicializar entorno
env = gym.make("FlappyBird-v0", render_mode="human")
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Redes
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
steps_done = 0

# Cargar checkpoint si existe
load_checkpoint = True
checkpoint_path = "models/flappy_dqn_checkpoint_ep1000.pth"

if load_checkpoint and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode_rewards = checkpoint['rewards']
    start_episode = checkpoint['episode'] + 1
    print(f"✅ Checkpoint cargado desde episodio {checkpoint['episode']}")
else:
    episode_rewards = []
    start_episode = 0

def epsilon_by_frame(frame_idx):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * frame_idx / EPSILON_DECAY)

# Entrenamiento
for episode in range(start_episode, EPISODES):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0

    while not done:
        steps_done += 1
        epsilon = epsilon_by_frame(steps_done)

        # Selección de acción
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor([state])
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        # Paso en el entorno
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state, dtype=np.float32)

        # ✅ Ajuste de recompensa
        reward += 0.1  # incentivo por sobrevivir
        if done:
            reward -= 1.0  # castigo por morir

        total_reward += reward

        # Guardar experiencia
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # Entrenamiento
        if len(replay_buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = policy_net(states).gather(1, actions).squeeze()
            next_q_values = target_net(next_states).max(1)[0]
            target = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Actualizar red objetivo
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)

    # Guardar checkpoint cada 500 episodios
    if episode % 500 == 0 and episode != 0:
        checkpoint = {
            'episode': episode,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rewards': episode_rewards
        }
        checkpoint_path = f"models/flappy_dqn_checkpoint_ep{episode}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"📌 Checkpoint guardado: {checkpoint_path}")

    print(f"Ep {episode}, Recompensa: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

env.close()

# Guardar el modelo final
model_path = "models/flappy_dqn.pth"
torch.save(policy_net.state_dict(), model_path)
print(f"✅ Modelo final guardado en: {model_path}")

# Visualización de recompensas
def plot_rewards(rewards, window=100):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Recompensa por episodio')
    
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Media móvil ({window})', linewidth=2)

    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Desempeño del agente DQN')
    plt.legend()
    plt.grid()
    plt.savefig("flappy_rewards_plot.png")
    plt.show()

plot_rewards(episode_rewards)