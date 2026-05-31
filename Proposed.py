import time

import numpy as np


# Enhanced Starfish Optimization Algorithm (ESOA) Starting Line No. 55
def Proposed(Xpos, fobj, lb, ub, Max_it):
    Npop, nD = Xpos.shape[0], Xpos.shape[1]
    GP = 0.5  # parameter
    if len(ub) == 1:
        lb = np.full(nD, lb)
        ub = np.full(nD, ub)

    fvalbest = np.inf
    Curve = np.zeros(Max_it)

    # Fitness evaluation
    Fitness = np.array([fobj(Xpos[i, :]) for i in range(Npop)])
    fvalbest = np.min(Fitness)  # global best fitness
    order = np.argmin(Fitness)
    xposbest = Xpos[order, :]  # global best position
    newX = np.zeros_like(Xpos)

    # Evolution
    T = 1
    ct = time.time()
    while T <= Max_it:
        theta = np.pi / 2 * T / Max_it
        tEO = (Max_it - T) / Max_it * np.cos(theta)

        if np.random.rand() < GP:  # Exploration of starfish
            for i in range(Npop):
                if nD > 5:
                    jp1 = np.random.choice(nD, 5, replace=False)
                    for j in jp1:
                        pm = (2 * np.random.rand() - 1) * np.pi
                        if np.random.rand() < GP:
                            newX[i, j] = Xpos[i, j] + pm * (xposbest[j] - Xpos[i, j]) * np.cos(theta)
                        else:
                            newX[i, j] = Xpos[i, j] - pm * (xposbest[j] - Xpos[i, j]) * np.sin(theta)
                        # Boundary check
                        newX[i, j] = np.clip(newX[i, j], lb[j], ub[j])
                else:
                    jp2 = np.random.randint(0, nD)
                    im = np.random.permutation(Npop)
                    rand1, rand2 = 2 * np.random.rand() - 1, 2 * np.random.rand() - 1
                    newX[i, jp2] = tEO * Xpos[i, jp2] + rand1 * (Xpos[im[0], jp2] - Xpos[i, jp2]) + rand2 * (
                                Xpos[im[1], jp2] - Xpos[i, jp2])
                    # Boundary check
                    newX[i, jp2] = np.clip(newX[i, jp2], lb[jp2], ub[jp2])
        else:  # Exploitation of starfish
            df = np.random.choice(Npop, 5, replace=False)
            dm = np.array([xposbest - Xpos[df[j], :] for j in range(5)])  # Five arms of starfish
            for i in range(Npop):
                sorted_indices = np.argsort(Fitness)
                F1 = Fitness[sorted_indices[0]]  # Best
                F2 = Fitness[sorted_indices[-1]]  # Worst
                r1, r2 = F2 / (2 * F1), F2 / (2 * F1)
                kp = np.random.choice(5, 2, replace=False)
                newX[i, :] = Xpos[i, :] + r1 * dm[kp[0], :] + r2 * dm[kp[1], :]
                if i == Npop - 1:
                    newX[i, :] = np.exp(-T * Npop / Max_it) * Xpos[i, :]  # Regeneration of starfish
                # Boundary check
                newX[i, :] = np.clip(newX[i, :], lb, ub)

        # Fitness evaluation
        for i in range(Npop):
            newFit = fobj(newX[i, :])
            if newFit < Fitness[i]:
                Fitness[i] = newFit
                Xpos[i, :] = newX[i, :]
                if newFit < fvalbest:
                    fvalbest = Fitness[i]
                    xposbest = Xpos[i, :]

        Curve[T - 1] = fvalbest
        T += 1
    ct = time.time() - ct
    return fvalbest, Curve, xposbest, ct
