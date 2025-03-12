import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import neat
import pickle
import fs
MODELS_DIR = fs.open_fs("../models/")
CONFIG_FILES_DIR = fs.open_fs("config_files/")

def eval_genomes(genomes, config):
    """Función de evaluación de genomas durante la evolución."""
    # env = gym.make("FlappyBird-v0", use_lidar=True)
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)


    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # Crea la red neuronal
        observation, _ = env.reset()
        fitness = 0  # Puntuación del agente

        done = False
        while not done:
            obs = observation[:180]  
            action_prob = net.activate(obs)  # Predicción de la red
            action = 1 if action_prob[0] > 0.5 else 0  # Salta si la salida > 0.5

            observation, reward, done, _, _ = env.step(action)
            fitness += reward  # Acumulamos la recompensa

        genome.fitness = fitness  # Asigna el fitness final al genoma

    env.close()

def run_neat(config_path):
    """Ejecuta la evolución NEAT y guarda el mejor genoma."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    population = neat.Population(config)

    # Agregar reportes para visualizar progreso
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Ejecutar la evolución
    best_genome = population.run(eval_genomes, 50)  # Evalúa 50 generaciones

    print("\nMejor genoma encontrado:")
    print(best_genome)

    # Guardar el mejor genoma con su clave como nombre de archivo
    filename = f"best_genome_{best_genome.key}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(best_genome, f)

    print(f"El mejor genoma ha sido guardado en '{filename}'.")

def load_and_run_best_genome(config_path, filename):
    """Carga el mejor genoma guardado y lo ejecuta en Flappy Bird."""
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Cargar el mejor genoma desde el archivo
    with open(filename, "rb") as f:
        best_genome = pickle.load(f)

    print(f"\nEjecutando el mejor genoma desde '{filename}' (key={best_genome.key})")

    # Crear la red neuronal a partir del mejor genoma
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)

    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    observation, _ = env.reset()
    
    done = False
    while not done:
        obs = observation[:180]  # Tomamos solo altura del pájaro y distancia al tubo
        action_prob = net.activate(obs)  # Predicción de la red
        action = 1 if action_prob[0] > 0.5 else 0  # Salta si la salida > 0.5

        observation, _, done, _, _ = env.step(action)

    env.close()
    print("Ejecución finalizada.")

if __name__ == "__main__":
    # config_path = "config-feedforward.txt"
    config_path = CONFIG_FILES_DIR.getsyspath("config-feedforward_01.txt")
    # Opción 1: Entrenar y guardar el mejor genoma
    # run_neat(config_path)

    # Opción 2: Cargar el mejor genoma guardado y ejecutarlo
    best_genome_filename = MODELS_DIR.getsyspath("best_genome_2392.pkl") 
    load_and_run_best_genome(config_path, best_genome_filename)
