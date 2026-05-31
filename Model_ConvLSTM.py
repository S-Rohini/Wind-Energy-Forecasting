import numpy as np
from keras.models import Model
from keras.layers import (
    Input, Dense, ConvLSTM2D, BatchNormalization,
    Flatten, Reshape, LayerNormalization,
    MultiHeadAttention, Dropout, GlobalAveragePooling1D
)
from keras.optimizers import Adam

from Evaluation import evaluate_error


def transformer_block(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
    attn_output = Dropout(dropout)(attn_output)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)

    x = LayerNormalization(epsilon=1e-6)(x + ffn)
    return x


def LSTM_train(trainX, trainY, testX, testY, steps_per_epoch):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1], 1, 1))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1], 1, 1))

    train = np.asarray(trainX).astype(np.float32)
    test = np.asarray(testX).astype(np.float32)
    inputs = Input(shape=(train.shape[1], train.shape[2], train.shape[3], train.shape[4]))

    # ConvLSTM
    x = ConvLSTM2D(
        filters=32,
        kernel_size=(1, 1),
        activation='relu',
        return_sequences=False
    )(inputs)

    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Transformer
    x = Reshape((1, x.shape[-1]))(x)
    x = transformer_block(x)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(trainY.shape[1], activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.01),
        metrics=['accuracy']
    )
    model.fit(train, trainY, epochs=100, steps_per_epoch=steps_per_epoch)
    pred = model.predict(test)
    return pred, model


def Model_ConvLSTM(train_data, train_target, test_data, test_target, steps_per_epoch):
    out, model = LSTM_train(
        train_data, train_target,
        test_data, test_target, steps_per_epoch
    )
    pred = np.asarray(out)
    Eval = evaluate_error(pred, test_target)

    return Eval, pred
