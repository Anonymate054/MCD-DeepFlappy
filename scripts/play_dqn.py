# play_dqn.py actualizado
# Visualiza al agente jugando y muestra tubos pasados y recompensa real

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from models.dqn import DQN

# Configuración
MODEL_PATH = "models/flappy_dqn.pth"
RENDER_MODE = "human"  # Mostrar juego en pantalla

# Inicializar entorno
env = gym.make("FlappyBird-v0", render_mode=RENDER_MODE)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Cargar modelo entrenado
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("\n🚀 Modelo cargado. Ejecutando partidas...\n")

# Jugar episodios
NUM_EPISODES = 3
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    done = False
    total_reward = 0
    score = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor([state])
            q_values = model(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = np.array(next_state, dtype=np.float32)

        if 'score' in info:
            score = info['score']

    print(f"🏁 Episodio {episode + 1} terminado:")
    print(f"   🎯 Tubos pasados: {score}")
    print(f"   💰 Recompensa total: {total_reward:.2f}\n")

env.close()