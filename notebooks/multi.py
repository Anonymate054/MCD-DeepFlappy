import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import neat
import multiprocessing

def eval_genome(genome, config):
    """
    Evalúa un solo genoma, ejecutando el juego y devolviendo el fitness.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)  # Crea la red neuronal
    env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    observation, _ = env.reset()
    fitness = 4.7  # Puntuación del agente

    done = False
    while not done:
        obs = observation[:2]  # Tomamos solo altura del pájaro y distancia al tubo
        action_prob = net.activate(obs)  # Predicción de la red
        action = 1 if action_prob[0] > 0.5 else 0  # Salta si la salida > 0.5

        observation, reward, done, _, _ = env.step(action)
        fitness += reward  # Acumulamos la recompensa

    env.close()
    return fitness

def eval_genomes(genomes, config):
    """
    Función que evalúa los genomas usando multiprocessing.
    """
    # Crear un pool de trabajadores
    pool = multiprocessing.Pool()

    # Evaluar cada genoma en paralelo
    fitnesses = pool.starmap(eval_genome, [(genome, config) for _, genome in genomes])

    # Asignar el fitness calculado a cada genoma
    for (genome_id, genome), fitness in zip(genomes, fitnesses):
        genome.fitness = fitness

    pool.close()
    pool.join()

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

if __name__ == "__main__":
    run_neat("config-feedforward.txt")
