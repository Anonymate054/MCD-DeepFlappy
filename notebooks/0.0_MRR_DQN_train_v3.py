# deepflappy_v3_train.py - Ajustado con Etapa 1
# Reducción de epsilon y castigo por morir antes del primer tubo

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
EPISODES = 10000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.001  # Más bajo para menos exploración
EPSILON_DECAY = 5000
LR = 5e-5
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000
TARGET_UPDATE_FREQ = 200

# Inicializar entorno
env = gym.make("FlappyBird-v0", render_mode="rgb_array")
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
checkpoint_path = "models/flappy_dqn_checkpoint_ep5000.pth"

if load_checkpoint and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episode_rewards = checkpoint['rewards']
    episode_scores = checkpoint.get('scores', [])
    start_episode = checkpoint['episode'] + 1
    print(f"✅ Checkpoint cargado desde episodio {checkpoint['episode']}")
else:
    episode_rewards = []
    episode_scores = []
    start_episode = 0

def epsilon_by_frame(frame_idx):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * frame_idx / EPSILON_DECAY)

# Entrenamiento
for episode in range(start_episode, EPISODES):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0
    score = 0

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
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state, dtype=np.float32)

        # Obtener score real si existe
        if 'score' in info:
            score = info['score']

        reward += 0.1
        if done:
            if score < 1:
                reward -= 3.0  # castigo fuerte por morir antes del primer tubo
            else:
                reward -= 1.0  # castigo normal

        total_reward += reward

        # Guardar experiencia
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        # Entrenamiento de la red
        if len(replay_buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            q_values = policy_net(states).gather(1, actions).squeeze()

            # Double DQN
            next_actions = policy_net(next_states).argmax(1)
            next_q_values = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()

            target = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.SmoothL1Loss()(q_values, target.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    episode_scores.append(score)

    if episode % 500 == 0 and episode != 0:
        checkpoint = {
            'episode': episode,
            'model_state_dict': policy_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rewards': episode_rewards,
            'scores': episode_scores
        }
        torch.save(checkpoint, f"models/flappy_dqn_checkpoint_ep{episode}.pth")
        print(f"📌 Checkpoint guardado: episodio {episode}")

    print(f"Ep {episode}, Recompensa: {total_reward:.2f}, Tubos: {score}, Epsilon: {epsilon:.4f}")

env.close()

# Estadísticas
max_reward = max(episode_rewards)
min_reward = min(episode_rewards)
avg_reward = sum(episode_rewards) / len(episode_rewards)

max_score = max(episode_scores)
min_score = min(episode_scores)
avg_score = sum(episode_scores) / len(episode_scores)

print("\n📊 Estadísticas finales del entrenamiento:")
print(f"   🏆 Máxima recompensa: {max_reward:.2f}")
print(f"   😬 Mínima recompensa: {min_reward:.2f}")
print(f"   📈 Promedio recompensa: {avg_reward:.2f}")
print(f"   🎯 Máximo tubos pasados: {max_score}")
print(f"   🪵 Mínimo tubos pasados: {min_score}")
print(f"   🧮 Promedio tubos pasados: {avg_score:.2f}")

# Guardar el modelo final
torch.save(policy_net.state_dict(), "models/flappy_dqn.pth")
print("✅ Modelo final guardado")

# Gráfica final
def plot_rewards(rewards, window=100):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Recompensa por episodio')
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Media móvil ({window})', linewidth=2)
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa')
    plt.title('Aprendizaje del agente DQN')
    plt.legend()
    plt.grid()
    plt.savefig("flappy_rewards_plot.png")
    plt.show()

plot_rewards(episode_rewards)
