import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from Evaluation import evaluate_error


def Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target, StepPerEpochs):
    # Build the DNN model
    model = Sequential()
    model.add(Dense(64, input_dim=Train_Data.shape[1], activation='relu'))  # Input layer + hidden layer
    model.add(Dense(32, activation='relu'))  # Hidden layer
    model.add(Dense(Train_Target.shape[1], activation='softmax'))  # Output layer
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Print model summary
    model.summary()
    # Train the model
    TrainX = np.asarray(Train_Data).astype(np.float32)
    TestX = np.asarray(Test_Data).astype(np.float32)
    model.fit(TrainX, Train_Target, epochs=100, steps_per_epoch=StepPerEpochs)
    pred = model.predict(TestX)
    Eval = evaluate_error(pred, Test_Target)
    return Eval, pred