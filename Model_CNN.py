import tensorflow as tf

from Evaluation import evaluate_error


def Model_CNN(Train_Data, Train_Target, Test_Data, TestTarget, StepPerEpochs):
    X_train = Train_Data.reshape((Train_Data.shape[0], Train_Data.shape[1], 1, 1))
    X_test = Test_Data.reshape((Test_Data.shape[0], Test_Data.shape[1], 1, 1))

    num_classes = Train_Target.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], 1, 1)),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    model.fit(X_train,Train_Target, epochs=100, batch_size=4, steps_per_epoch=StepPerEpochs)
    pred = model.predict(X_test)
    Eval = evaluate_error(pred, TestTarget)

    return Eval, pred