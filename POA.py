import numpy as np
import time


# Pelican Optimization Algorithm (POA)
def POA(population, objective_function, ub, lb, max_iterations):
    population_size, dimension = population.shape
    alpha = 0.1
    beta = 0.1
    best_solution = float('inf')
    cg_curve = np.zeros(max_iterations)
    fitness = np.zeros(population_size)

    start_time = time.time()

    for i in range(population_size):
        fitness[i] = objective_function(population[i])

    # Main loop
    for iter in range(max_iterations):
        # Update best solution
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        best_solution = population[best_index].copy()

        # Update pelican positions
        for i in range(population_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            rand_pelican = population[np.random.randint(population_size)]

            velocity = (alpha * population[i] +
                        beta * (best_solution - population[i]) +
                        beta * (rand_pelican - population[i]))
            population[i] += velocity

        # Clip positions to stay within bounds
        population = np.clip(population, lb, ub)

        # Evaluate fitness of the new population
        for i in range(population_size):
            fitness[i] = objective_function(population[i])

        # Track convergence
        cg_curve[iter] = np.min(fitness)

    elapsed_time = time.time() - start_time
    best_fitness = cg_curve[-1]

    return best_solution, cg_curve, best_fitness, elapsed_time
