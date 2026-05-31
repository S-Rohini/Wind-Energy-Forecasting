import numpy as np
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import pylab
from matplotlib.patches import Polygon

warnings.filterwarnings("ignore")

No_of_Dataset = 3


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'GOA-ExSGADTN', 'NBOA-ExSGADTN', 'POA-ExSGADTN', 'SOA-ExSGADTN', 'RFSOA-ExSGADTN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(No_of_Dataset):
        Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
        for j in range(len(Algorithm) - 1):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        fig = plt.figure()
        fig.canvas.manager.set_window_title('Dataset-' + str(i + 1) + ' Convergence Curve')
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label=Algorithm[5])
        plt.xlabel('No. of Iteration', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.ylabel('Cost Function', fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.yticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.xticks(fontname="Arial", fontsize=14, fontweight='bold', color='k')
        plt.legend(loc=1, prop={'weight': 'bold', 'size': 12})
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_Alg_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Algorithm = ['GOA-ExSGADTN', 'NBOA-ExSGADTN', 'POA-ExSGADTN', 'SOA-ExSGADTN', 'RFSOA-ExSGADTN']
    Graph_Term = [2, 3, 4, 5, 6]
    arrow_width = 0.1
    tip_height_ratio = 0.0001
    wave_amplitude = 0.01
    wave_freq = 10
    spacing = 0.15
    colors = ['#20B2AA', '#E69F00', '#80B918', '#A52A2A', 'k']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Term[j]]

            Graph = Graph[:, :5]
            bars_per_category = Graph.shape[1]

            fig, ax = plt.subplots(figsize=(8, 6))
            X = np.arange(Graph.shape[0])
            for x_idx in range(len(X)):
                for alg_idx in range(bars_per_category):
                    value = Graph[x_idx, alg_idx]
                    offset = (alg_idx - (bars_per_category - 1) / 2) * spacing
                    base_x = X[x_idx] + offset

                    y = np.linspace(0, value, 200)
                    tip_height = value * tip_height_ratio

                    left_x = base_x - arrow_width / 2 + wave_amplitude * np.sin(2 * np.pi * wave_freq * y / value)
                    right_x = base_x + arrow_width / 2 + wave_amplitude * np.sin(2 * np.pi * wave_freq * y / value)
                    tip_top = value + tip_height

                    points = list(zip(left_x, y)) + [(base_x, tip_top)] + list(zip(right_x[::-1], y[::-1]))

                    arrow_poly = Polygon(points, facecolor=colors[alg_idx], edgecolor='black', linewidth=1)
                    ax.add_patch(arrow_poly)

            ax.set_ylim(0, np.max(Graph) * 1.3)
            ax.set_xticks(X)
            ax.set_xticklabels(['Cross \n Entropy Loss', 'Huber Loss', 'Hinge Loss', 'Wasserstein Loss'],
                               fontname="Arial",
                               fontsize=14, fontweight='bold', color='#14213d')
            ax.set_xlabel('Loss Function', fontname="Arial", fontsize=14,
                          fontweight='bold', color='k')
            ax.set_ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14,
                          fontweight='bold', color='k')
            ax.tick_params(axis='y', labelsize=14, colors='#14213d')
            padding = spacing * (bars_per_category / 2 + 0.5)
            ax.set_xlim(-padding, len(X) - 1 + padding)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='#14213d')
            legend_markers = [plt.Line2D([0], [0], marker='^', color='w',
                                         markerfacecolor=col, markersize=18)
                              for col in colors[:bars_per_category]]
            ax.legend(legend_markers, Algorithm[:bars_per_category], loc='upper center',
                      bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,
                      prop={'weight': 'bold', 'size': 12})

            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Alg_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            fig.canvas.manager.set_window_title('Dataset - ' + str(i+1) + 'Algorithm Comparison of Loss Function vs ' + Terms[Graph_Term[j]])
            plt.show()


def Plot_Mod_Results():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Classifier = ['DNN', 'ConvLSTM', 'CNN', 'ExSGADTN ', 'RFSOA-ExSGADTN']
    arrow_width = 0.1
    tip_height_ratio = 0.0001
    wave_amplitude = 0.01
    wave_freq = 10
    spacing = 0.15
    Graph_Term = [2, 3, 4, 5, 6]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Term[j]]

            Graph = Graph[:, 5:]
            bars_per_category = Graph.shape[1]
            color = ['darkmagenta', 'darkorange', 'darkslateblue', '#80B918', 'k']
            fig, ax = plt.subplots(figsize=(8, 6))
            X = np.arange(Graph.shape[0])
            for x_idx in range(len(X)):
                for alg_idx in range(bars_per_category):
                    value = Graph[x_idx, alg_idx]
                    offset = (alg_idx - (bars_per_category - 1) / 2) * spacing
                    base_x = X[x_idx] + offset

                    y = np.linspace(0, value, 200)
                    tip_height = value * tip_height_ratio

                    left_x = base_x - arrow_width / 2 + wave_amplitude * np.sin(2 * np.pi * wave_freq * y / value)
                    right_x = base_x + arrow_width / 2 + wave_amplitude * np.sin(2 * np.pi * wave_freq * y / value)
                    tip_top = value + tip_height

                    points = list(zip(left_x, y)) + [(base_x, tip_top)] + list(zip(right_x[::-1], y[::-1]))

                    arrow_poly = Polygon(points, facecolor=color[alg_idx], edgecolor='black', linewidth=1)
                    ax.add_patch(arrow_poly)

            ax.set_ylim(0, np.max(Graph) * 1.3)
            ax.set_xticks(X)
            ax.set_xticklabels(['Cross \n Entropy Loss', 'Huber Loss', 'Hinge Loss', 'Wasserstein Loss'],
                               fontname="Arial",
                               fontsize=14, fontweight='bold', color='#14213d')
            ax.set_xlabel('Loss Function', fontname="Arial", fontsize=14,
                          fontweight='bold', color='k')
            ax.set_ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=14,
                          fontweight='bold', color='k')
            ax.tick_params(axis='y', labelsize=14, colors='#14213d')
            padding = spacing * (bars_per_category / 2 + 0.5)
            ax.set_xlim(-padding, len(X) - 1 + padding)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.yticks(fontname="Arial", fontsize=15, fontweight='bold', color='#14213d')
            legend_markers = [plt.Line2D([0], [0], marker='^', color='w',
                                         markerfacecolor=col, markersize=18)
                              for col in color[:bars_per_category]]
            ax.legend(legend_markers, Classifier[:bars_per_category], loc='upper center',
                      bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,
                      prop={'weight': 'bold', 'size': 12})
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Mod_bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Dataset - ' + str(i+1) + 'Method Comparison of Loss Function vs ' + Terms[Graph_Term[j]])
            plt.show()


def Table():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Algorithm = ['Step Per Epochs', 'GOA-ExSGADTN', 'NBOA-ExSGADTN', 'POA-ExSGADTN', 'SOA-ExSGADTN', 'RFSOA-ExSGADTN']
    Classifier = ['Step Per Epochs', 'DNN', 'ConvLSTM', 'CNN', 'ExSGADTN ', 'RFSOA-ExSGADTN']
    Graph_Term = np.array([2, 3, 4, 5, 6]).astype(int)
    Table_Terms = [2, 3, 4, 5, 6]
    table_terms = [Terms[i] for i in Table_Terms]
    StepPerEpochs = ['100', '200', '300', '400', '500']
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, :]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], StepPerEpochs)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Term[k]])
            print('----------------------------Dataset -  ', i + 1, table_terms[k],
                  ' - Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], StepPerEpochs)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Term[k]])
            print('---------------------------Dataset -  ', i + 1, table_terms[k],
                  ' - Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plot_Proposed_Results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'MSE', 'Accuracy']
    Algorithm = ['GOA-ExSGADTN', 'NBOA-ExSGADTN', 'POA-ExSGADTN', 'SOA-ExSGADTN', 'RFSOA-ExSGADTN']
    Classifier = ['DNN', 'ConvLSTM', 'CNN', 'ExSGADTN ', 'RFSOA-ExSGADTN']
    Graph_Term = [4, 6]
    Optimizer = ['Adam', 'SGD', 'RMSProp']
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Term[j]]

            fig = plt.figure(figsize=(10, 8))
            fig.canvas.manager.set_window_title('Dataset - ' + str(i + 1) + 'Algorithm Comparison of Optimizer vs ' + Terms[Graph_Term[j]])
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            X = np.arange(len(Optimizer))
            ax.barh(X + 0.00, Graph[:, 0], height=0.15, color='darkmagenta', label="LSTM")
            ax.barh(X + 0.15, Graph[:, 1], height=0.15, color='darkorange', label="Faster R-CNN")
            ax.barh(X + 0.30, Graph[:, 2], height=0.15, color='darkslateblue', label="GRU")
            ax.barh(X + 0.45, Graph[:, 3], height=0.15, color='#80B918', label="RD-SA")
            ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label="GSSA-ARD-SA")
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)

            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['darkmagenta', 'darkorange', 'darkslateblue', '#80B918', 'k']]
            plt.legend(dot_markers, Algorithm, loc='upper center', bbox_to_anchor=(0.5, 1.20), fontsize=9,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.yticks(X + 0.30, ['Adam', 'SGD', 'RMSProp'], fontsize=15,
                       fontname="Arial",
                       fontweight='bold', color='k')
            plt.ylabel('Optimizer', fontname="Arial", fontsize=15, fontweight='bold', color='#14213d')
            plt.xlabel(Terms[Graph_Term[j]], fontsize=15, fontname="Arial", fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Prop_Alg_Bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            # ------------------------------------- Methods ------------------------------------------------
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
            fig.canvas.manager.set_window_title('Dataset - ' + str(i + 1) + ' Method Comaprison of Loss Function vs ' + Terms[Graph_Term[j]])
            X = np.arange(len(Optimizer))
            ax.barh(X + 0.00, Graph[:, 5], height=0.15, color='royalblue', label="LSTM")
            ax.barh(X + 0.15, Graph[:, 6], height=0.15, color='violet', label="Faster R-CNN")
            ax.barh(X + 0.30, Graph[:, 7], height=0.15, color='palegreen', label="GRU")
            ax.barh(X + 0.45, Graph[:, 8], height=0.15, color='crimson', label="RD-SA")
            ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label="GSSA-ARD-SA")
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)

            dot_markers = [plt.Line2D([2], [2], marker='s', color='w', markerfacecolor=color, markersize=10) for color
                           in ['royalblue', 'violet', 'palegreen', 'crimson', 'k']]
            plt.legend(dot_markers, Classifier, loc='upper center', bbox_to_anchor=(0.5, 1.20), fontsize=9,
                       frameon=False, ncol=3, prop={'weight': 'bold', 'size': 12})
            plt.yticks(X + 0.30, ['Adam', 'SGD', 'RMSProp'], fontsize=15,
                       fontname="Arial",
                       fontweight='bold', color='k')
            plt.ylabel('Optimizer', fontname="Arial", fontsize=15, fontweight='bold', color='#14213d')
            plt.xlabel(Terms[Graph_Term[j]], fontsize=15, fontname="Arial", fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=15, fontweight='bold', color='#35530a')
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.tight_layout()
            path = "./Results/Dataset_%s_%s_Prop_Mod_Bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    Plot_Alg_Results()
    Plot_Mod_Results()
    Table()
    Plot_Proposed_Results()
