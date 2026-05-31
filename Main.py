import os
from numpy import matlib
from sklearn.preprocessing import StandardScaler
import pandas as pd
from GOA import GOA
from Global_Vars import Global_Vars
from Model_CNN import Model_CNN
from Model_ConvLSTM import Model_ConvLSTM
from Model_DNN import Model_DNN
from Model_ExSGADTN import Model_ExSGADTN
from NBOA import NBOA
from POA import POA
from Plot_Results import *
from Proposed import Proposed
from SOA import SOA
from objfun_feat import objfun

No_of_Dataset = 3

# Read Dataset 1
an = 0
if an == 1:
    Datas = []
    Target = []
    Path = './Dataset 1'
    InDir = os.listdir(Path)
    for i in range(len(InDir) - 1):
        Files = Path + '/' + InDir[i]
        DataFrame = pd.read_csv(Files)
        Data = DataFrame.values[:20000, :-1]
        Tar = DataFrame.values[:20000, -1]
        Datas.append(Data)
        Target.append(Tar)

    FullData = np.asarray(Datas)
    data = np.reshape(FullData, (FullData.shape[0] * FullData.shape[1], FullData.shape[2]))
    for i in range(data.shape[1]):
        if data[:, i][0] == str(data[:, i][0]):
            uniq = np.unique(data[:, i])
            value = data[:, i]
            final_data = np.zeros((value.shape[0]))  # create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            data[:, i] = final_data
    FullTar = np.asarray(Target)
    Targets = np.reshape(FullTar, (FullTar.shape[0] * FullTar.shape[1], 1))
    Targets = np.nan_to_num(Targets, nan=0.0)

    np.save('Data_1.npy', data)
    np.save('Target_1.npy', Targets)

# Read Dataset 2
an = 0
if an == 1:
    DataFrame = pd.read_csv('./Dataset 2/Turbine_Data.csv')
    DropData = DataFrame.drop(['ActivePower'], axis=1)
    Data = DropData.values[:, :-1]
    for i in range(Data.shape[1]):
        if Data[:, i][0] == str(Data[:, i][0]):
            Data[:, i] = [str(x) if x == 0.0 else x for x in Data[:, i]]
            uniq = np.unique(Data[:, i])
            value = Data[:, i]
            final_data = np.zeros((value.shape[0]))  # Create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            Data[:, i] = final_data

    Target = DataFrame.get(['ActivePower']).values
    Target = np.nan_to_num(Target, nan=0.0)

    np.save('Data_2.npy', Data)
    np.save('Target_2.npy', np.reshape(Target, (-1, 1)))

# Read Dataset 3
an = 0
if an == 1:
    DataFrame = pd.read_csv('./Dataset 3/T1.csv')
    DropData = DataFrame.drop(['LV ActivePower (kW)'], axis=1)
    Data = DataFrame.values[:]
    for i in range(Data.shape[1]):
        if Data[:, i][0] == str(Data[:, i][0]):
            Data[:, i] = [str(x) if x == 0.0 else x for x in Data[:, i]]
            uniq = np.unique(Data[:, i])
            value = Data[:, i]
            final_data = np.zeros((value.shape[0]))  # create within rage zero values
            for uni in range(len(uniq)):
                index = np.where(value == uniq[uni])
                final_data[index[0]] = uni + 1
            Data[:, i] = final_data
    Target = DataFrame.get(['LV ActivePower (kW)']).values
    np.save('Data_3.npy', Data)
    np.save('Target_3.npy', Target)

# Data Preprocessing
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True).astype(np.float64)
        data_clean = np.where(Data < 0, np.nan, Data).astype(float)
        nan_mask = np.isnan(data_clean)
        column_means = np.nanmean(data_clean, axis=0)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(nan_mask)
        np.save('Pre_Data_' + str(n + 1) + '.npy', normalized_data)

# Optimization for Prediction
an = 0
if an == 1:
    for n in range(No_of_Dataset):
        Feat = np.load('Pre_Data_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Global_Vars.Data = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden neuron count, No. of Epochs, Activation Function
        xmin = matlib.repmat([5, 5, 1], Npop, 1)
        xmax = matlib.repmat([255, 50, 5], Npop, 1)
        fname = objfun
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print('GOA....')
        [bestfit1, fitness1, bestsol1, Time1] = GOA(initsol, fname, xmin, xmax, Max_iter)

        print('NBOA....')
        [bestfit2, fitness2, bestsol2, Time2] = NBOA(initsol, fname, xmin, xmax, Max_iter)

        print('POA....')
        [bestfit3, fitness3, bestsol3, Time3] = POA(initsol, fname, xmin, xmax, Max_iter)

        print('SOA....')
        [bestfit4, fitness4, bestsol4, Time4] = SOA(initsol, fname, xmin, xmax, Max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Prediction
an = 0
if an == 1:
    Eval_all = []
    for n in range(No_of_Dataset):
        Feat = np.load('Pre_Data_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        BestSol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        EVAL = []
        StepPerEpochs = [100, 200, 300, 400, 500]
        for act in range(len(StepPerEpochs)):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((10, 7))
            for j in range(BestSol.shape[0]):
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_ExSGADTN(Train_Data, Train_Target, Test_Data, Test_Target, StepPerEpochs[act],
                                                  sol)  # With optimization
            Eval[5, :], pred1 = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                          StepPerEpochs[act])
            Eval[6, :], pred2 = Model_ConvLSTM(Train_Data, Train_Target, Test_Data,
                                               Test_Target, StepPerEpochs[act])
            Eval[7, :], pred3 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target,
                                          StepPerEpochs[act])
            Eval[8, :], pred4 = Model_ExSGADTN(Train_Data, Train_Target, Test_Data, Test_Target,
                                               StepPerEpochs[act])  # Without optimization
            Eval[9, :] = Eval[4, :]
            EVAL.append(Eval)
        Eval_all.append(EVAL)
    np.save('Evaluate.npy', np.asarray(Eval_all))  # Save Eval all

plotConvResults()
Plot_Alg_Results()
Plot_Mod_Results()
Table()
Plot_Proposed_Results()
