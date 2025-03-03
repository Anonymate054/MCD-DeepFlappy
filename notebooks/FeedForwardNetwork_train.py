import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import neat
import pickle
import os
import fs
MODELS_DIR = fs.open_fs("../models")

def eval_genomes(genomes, config):
    env = gym.make("FlappyBird-v0", use_lidar=True)

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)  
        observation, _ = env.reset()
        fitness = 0  

        done = False
        while not done:
            obs = observation[:180]  
            action_prob = net.activate(obs)  
            action = 1 if action_prob[0] > 0.5 else 0  

            observation, reward, done, _, _ = env.step(action)
            fitness += reward  

        genome.fitness = fitness  

    env.close()

def run_neat(config_path, genome_filename=None):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # Inicializar una población completa
    population = neat.Population(config)

    if genome_filename and os.path.exists(genome_filename):
        print(f"Cargando el mejor genoma desde '{genome_filename}'...")
        with open(genome_filename, "rb") as f:
            best_genome = pickle.load(f)

        best_genome.fitness = None  # Resetear fitness

        # Asignar un ID único al genoma cargado para evitar conflictos
        best_genome.key = max(population.population.keys()) + 1  # Generar un ID único
        population.population[best_genome.key] = best_genome  # Insertar el mejor genoma en la población

        # Especiar la población correctamente
        population.species.speciate(config, population.population, population.generation)

    else:
        print("Iniciando nuevo entrenamiento desde cero...")

    # Agregar reportes
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Continuar el entrenamiento
    best_genome = population.run(eval_genomes, 500)

    print("\nMejor genoma encontrado:")
    print(best_genome)

    # Guardar el nuevo mejor genoma
    genome_key = str(best_genome.key)
    filename = f"best_genome_{genome_key}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(best_genome, f)

    print(f"El mejor genoma ha sido guardado en '{filename}'.")

if __name__ == "__main__":
    config_path = "config-feedforward.txt"
    best_genome_filename = MODELS_DIR.getsyspath("best_genome_2285.pkl")
    run_neat(config_path, best_genome_filename)