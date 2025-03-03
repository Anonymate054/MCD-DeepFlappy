import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import neat
import pickle

def eval_genomes(genomes, config):
    # env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)
    env = gym.make("FlappyBird-v0", use_lidar=True)

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # Crea la red neuronal
        observation, _ = env.reset()
        fitness = 0  # Puntuación del agente

        done = False
        while not done:
            obs = observation[:180]  # Tomamos solo altura del pájaro y distancia al tubo
            action_prob = net.activate(obs)  # Predicción de la red
            action = 1 if action_prob[0] > 0.5 else 0  # Salta si la salida > 0.5

            observation, reward, done, _, _ = env.step(action)
            fitness += reward  # Acumulamos la recompensa

        genome.fitness = fitness  # Asigna el fitness final al genoma

    env.close()

def run_neat(config_path):
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

    # Obtener el identificador del mejor genoma
    genome_key = str(best_genome.key)

    # Guardar en un archivo con el nombre basado en la clave
    filename = f"best_genome_{genome_key}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(best_genome, f)

    print(f"El mejor genoma ha sido guardado en '{filename}'.")

if __name__ == "__main__":
    run_neat("config-feedforward.txt")
