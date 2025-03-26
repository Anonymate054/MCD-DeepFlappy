import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from models.dqn_1 import DQN
from datetime import datetime, timedelta
import argparse
import itertools
import flappy_bird_gymnasium

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class Agent():

    def __init__(self, hyperparameter_set):
        base_path = os.path.dirname(__file__)
        with open(os.path.join(base_path, 'hyperparameters.yml'), 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.stop_on_reward     = hyperparameters['stop_on_reward']
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params', {})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def run(self, is_training=True, render=False, episodes=None):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)
        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]

        rewards_per_episode = []
        epsilon_history = []
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            step_count = 0
            best_reward = -9999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        episode_range = range(episodes) if episodes is not None else itertools.count()
        for episode in episode_range:
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                if is_training and random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                new_state, reward, terminated, truncated, info = env.step(action.item())
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)
            if is_training and len(memory) > self.mini_batch_size:
                epsilon_history.append(epsilon)

            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

        if is_training and rewards_per_episode:
            rewards_array = np.array(rewards_per_episode)

            max_reward = np.max(rewards_array)
            min_reward = np.min(rewards_array)
            avg_reward = np.mean(rewards_array)

            tubes_passed = rewards_array[rewards_array > 0]
            max_tubes = np.max(tubes_passed) if len(tubes_passed) > 0 else 0
            min_tubes = np.min(tubes_passed) if len(tubes_passed) > 0 else 0
            avg_tubes = np.mean(tubes_passed) if len(tubes_passed) > 0 else 0

            resumen = (
                "\nEstadísticas del entrenamiento:\n"
                f"Recompensa máxima: {max_reward:.2f}\n"
                f"Recompensa mínima: {min_reward:.2f}\n"
                f"Recompensa promedio: {avg_reward:.2f}\n"
                f"Máx. tubos pasados: {max_tubes:.2f}\n"
                f"Mín. tubos pasados (con éxito): {min_tubes:.2f}\n"
                f"Prom. tubos pasados (con éxito): {avg_tubes:.2f}"
            )

            print(resumen)
            with open(self.LOG_FILE, 'a') as file:
                file.write(resumen + '\n')

            # Guardar CSV
            csv_path = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_rewards.csv')
            with open(csv_path, 'w') as f:
                f.write('episodio,recompensa,epsilon,tubos_pasados\n')
                for idx, reward in enumerate(rewards_per_episode):
                    eps = epsilon_history[idx] if idx < len(epsilon_history) else ''
                    tubos = reward if reward > 0 else 0
                    f.write(f'{idx},{reward:.2f},{eps:.4f},{int(tubos)}\n')
            print(f"Datos guardados en: {csv_path}")
            print("Entrenamiento finalizado")

    def save_graph(self, rewards_per_episode, epsilon_history):
        fig = plt.figure(1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).max(1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Nombre del conjunto de hiperparámetros')
    parser.add_argument('--train', help='Modo entrenamiento', action='store_true')
    parser.add_argument('--episodes', type=int, default=10000, help='Número de episodios de entrenamiento')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True, episodes=args.episodes)
    else:
        dql.run(is_training=False, render=True)
