import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import numpy as np

# ✅ Usa un checkpoint que tenga recompensas
checkpoint_path = "models/flappy_dqn_checkpoint_ep4500.pth"

# Cargar checkpoint
checkpoint = torch.load(checkpoint_path)
print(f"🔍 Claves del checkpoint: {checkpoint.keys()}")

if 'rewards' in checkpoint:
    episode_rewards = checkpoint['rewards']
    max_score = max(episode_rewards)
    max_index = episode_rewards.index(max_score)
    avg_score = sum(episode_rewards) / len(episode_rewards)

    print(f"\n📊 Resultados del entrenamiento:")
    print(f"   🏆 Máxima recompensa: {max_score:.2f} (episodio {max_index})")
    print(f"   📈 Promedio general: {avg_score:.2f}")
    print(f"   📎 Total episodios: {len(episode_rewards)}\n")

    def plot_rewards(rewards, window=100):
        plt.figure(figsize=(12, 6))
        plt.plot(rewards, label='Recompensa por episodio', alpha=0.5)
        
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Media móvil ({window})', linewidth=2)

        # Línea horizontal con el máximo
        plt.axhline(max_score, color='red', linestyle='--', label=f'Máximo: {max_score:.2f}')
        plt.scatter(max_index, max_score, color='red')
        
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.title('Desempeño del agente DQN')
        plt.legend()
        plt.grid()
        plt.savefig("flappy_rewards_loaded_plot.png")
        plt.show()

    plot_rewards(episode_rewards)

else:
    print("Este archivo no contiene 'rewards'. Usa un checkpoint generado durante el entrenamiento.")