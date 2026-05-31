import time

import numpy as np


# Golf Optimization Algorithm (GOA)
def GOA(SearchAgents, fobj, lowerbound, upperbound, Max_iterations):
    N, dimension = SearchAgents.shape[0], SearchAgents.shape[1]
    X = SearchAgents

    fitness = fobj(SearchAgents[:])

    fbest = np.inf
    Xbest = np.zeros(dimension)

    GOA_curve = np.zeros(Max_iterations)

    # Main loop
    ct = time.time()
    for t in range(Max_iterations):
        # Update the best member and worst member
        best = np.min(fitness)
        blocation = np.argmin(fitness)

        if t == 0:
            Xbest = X[blocation, :]
            fbest = best
        elif best < fbest:
            fbest = best
            Xbest = X[blocation, :]

        # Phase 1: Exploration (global search)
        for i in range(SearchAgents.shape[0]):
            if np.random.rand() < 0.5:
                I = np.round(1 + np.random.rand(1))
                RAND = np.random.rand(1)
            else:
                I = np.round(1 + np.random.rand(1, dimension))
                RAND = np.random.rand(1, dimension)

            # Update position based on Eq (4)
            X_P1 = X[i, :] + RAND * (Xbest - I * X[i, :])
            X_P1 = np.clip(X_P1, lowerbound, upperbound)

            # Update fitness
            F_P1 = fobj(X_P1)
            if F_P1[i] < fitness[i]:
                X[i, :] = X_P1[0]
                fitness[i] = F_P1[i]

        # Phase 2: Exploitation (local search)
        for i in range(SearchAgents.shape[0]):
            X_P2 = X[i, :] + (1 - 2 * np.random.rand(1)) * (
                    lowerbound / (t + 1) + np.random.rand(1) * (upperbound / (t + 1) - lowerbound / (t + 1)))
            X_P2 = np.clip(X_P2, lowerbound / (t + 1), upperbound / (t + 1))
            X_P2 = np.clip(X_P2, lowerbound, upperbound)

            # Update fitness
            F_P2 = fobj(X_P2)
            if F_P2[i] < fitness[i]:
                X[i, :] = X_P2[0]
                fitness[i] = F_P2[i]

        GOA_curve[t] = fbest
    ct = time.time() - ct

    return fbest, GOA_curve, Xbest, ct
