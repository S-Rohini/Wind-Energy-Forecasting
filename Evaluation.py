import math
import numpy as np


def evaluate_error(sp, act) -> object:
    r = np.squeeze(act)
    x = np.squeeze(sp)
    points = np.zeros(len(x))
    abs_r = np.zeros(len(x))
    abs_x = np.zeros(len(x))
    abs_r_x = np.zeros(len(x))
    abs_x_r = np.zeros(len(x))
    abs_r_x__r = np.zeros(len(x))
    for j in range(1, len(x)):
        points[j] = abs(x[j] - x[j - 1])
    for i in range(len(r)):
        abs_r[i] = abs(r[i])
    for i in range(len(r)):
        abs_x[i] = abs(x[i])
    for i in range(len(r)):
        abs_r_x[i] = abs(r[i] - x[i])
    for i in range(len(r)):
        abs_x_r[i] = abs(x[i] - r[i])
    for i in range(len(r)):
        abs_r_x__r[i] = abs((r[i] - x[i]) / r[i])
    md = (100 / len(x)) * sum(abs_r_x__r)
    smape = (1 / len(x)) * sum(abs_r_x / ((abs_r + abs_x) / 2))
    mase = sum(abs_r_x) / ((1 / (len(x) - 1)) * sum(points))
    mae = sum(abs_r_x) / len(r)
    rmse = (sum(abs_x_r ** 2) / len(r)) ** 0.5
    mse = rmse ** 2
    accuracy = (1 - (np.abs(sum(r - x)) / sum(r))) * 100

    EVAL_ERR = [md, smape, mase, mae, rmse, mse, accuracy]
    return EVAL_ERR