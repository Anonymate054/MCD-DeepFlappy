import sys
import os

# Agrega la ruta raíz del proyecto para poder importar bien
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dqn import DQN
import torch

model = DQN(input_dim=8, output_dim=2)
sample_state = torch.rand((1, 8))
q_values = model(sample_state)

print("Valores Q generados:", q_values)