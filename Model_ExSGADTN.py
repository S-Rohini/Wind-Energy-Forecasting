import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Conv1D, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam

from Evaluation import evaluate_error


# Dilated Transformer Block
def dilated_transformer_block(x, head_size=64, num_heads=4, ff_dim=128, rate=2, dropout=0.1):
    conv_out = Conv1D(filters=ff_dim, kernel_size=3,
                      dilation_rate=rate, padding='same',
                      activation='relu')(x)

    attn_out = MultiHeadAttention(num_heads=num_heads,
                                  key_dim=head_size)(conv_out, conv_out)

    attn_out = Dropout(dropout)(attn_out)

    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)

    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x


# Graph Attention
def graph_attention_layer(inputs, units=64):
    h = Dense(units)(inputs)

    N = tf.shape(h)[1]

    h1 = tf.repeat(h, repeats=N, axis=1)
    h2 = tf.tile(h, [1, N, 1])

    concat = tf.concat([h1, h2], axis=-1)

    e = Dense(1, activation="relu")(concat)
    attention = tf.nn.softmax(e, axis=1)

    output = tf.matmul(attention, h)

    return output


def ExSGADTN_train(trainX, trainY, testX, testY, StepPerEpoch, Sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1], 1, 1))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1], 1, 1))
    trainX = trainX.astype(np.float32)
    testX = testX.astype(np.float32)
    Act = ['linear', 'relu', 'Tanh', 'softmax', 'sigmoid']

    inputs = Input(shape=(trainX.shape[1], trainX.shape[2]))

    # Graph Attention
    x = graph_attention_layer(inputs)

    # Dilated Transformer
    x = dilated_transformer_block(x, rate=1)
    x = dilated_transformer_block(x, rate=2)
    x = dilated_transformer_block(x, rate=4)

    # Classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(Sol[0], activation=Sol[Act[2]])(x)
    x = Dropout(0.3)(x)

    outputs = Dense(trainY.shape[1], activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        trainX, trainY,
        validation_data=(testX, testY),
        epochs=Sol[1],
        batch_size=64, steps_per_epoch=StepPerEpoch
    )

    pred = model.predict(testX)

    return pred, model


def Model_ExSGADTN(train_data, train_target, test_data, test_target, StepPerEpoch=100, Sol=None):
    if Sol is None:
        Sol = [5, 5, 1]
    pred, model = ExSGADTN_train(train_data, train_target, test_data, test_target, StepPerEpoch, Sol)

    pred = np.asarray(pred)
    print("\nRunning SHAP...")
    background = train_data[np.random.choice(train_data.shape[0],
                                             min(100, train_data.shape[0]),
                                             replace=False)]

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(test_data[:50])

    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, test_data[:50], show=False)
    plt.savefig("Results/Explainable Graph/SHAP_SummaryPlot.png", bbox_inches='tight')
    plt.close()

    # Dependence Plot
    plt.figure()
    shap.dependence_plot(0, shap_values[0], test_data[:50], show=False)
    plt.savefig("Results/SHAP_DependencePlot.png", bbox_inches='tight')
    plt.close()

    # Waterfall Plot
    shap.initjs()
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[0],
        shap_values[0][0],
        feature_names=[f"F{i}" for i in range(test_data.shape[-1])]
    )
    plt.savefig("Results/Explainable Graph/SHAP_WaterFallPlot.png", bbox_inches='tight')
    plt.close()
    Eval = evaluate_error(pred, test_target)
    return Eval, pred

