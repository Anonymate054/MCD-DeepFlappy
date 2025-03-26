# play_dqn_save_results.py
# Ejecuta varios episodios y guarda tubos + recompensa en un CSV con nombre único

import csv
import time
import torch
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
import sys
import os

# Asegurarse de que el directorio raíz del proyecto esté en sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ahora sí puedes importar desde models
from models.dqn import DQN

# Configuración
MODEL_PATH = "models/flappy_dqn.pth"
RENDER_MODE = "rgb_array"  # Sin ventana
TOTAL_EPISODES = 50
TIMESTAMP = int(time.time())
CSV_PATH = f"flappy_test_results_{TIMESTAMP}.csv"

# Inicializar entorno
env = gym.make("FlappyBird-v0", render_mode=RENDER_MODE)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

# Cargar modelo entrenado
model = DQN(input_dim, output_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print(f"🚀 Modelo cargado. Jugando {TOTAL_EPISODES} episodios...\n")

results = []

for episode in range(TOTAL_EPISODES):
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

    print(f"🎯 Episodio {episode + 1}: Tubos = {score}, Recompensa = {total_reward:.2f}")
    results.append((episode + 1, score, total_reward))

env.close()

# Guardar CSV con nombre único
with open(CSV_PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Episodio", "Tubos Pasados", "Recompensa Total"])
    writer.writerows(results)

print(f"\n✅ Resultados guardados en: {CSV_PATH}")
