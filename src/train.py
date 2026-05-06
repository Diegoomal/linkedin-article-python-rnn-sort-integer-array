import numpy as np                              # type: ignore
from tensorflow.keras import layers, models     # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore

N_CLASSES, ARRAY_SIZE, N_SAMPLES = 10, 10, 1_000_000
EPOCHS, BATCH_SIZE = 50, 256

X = np.random.randint(0, N_CLASSES, size=(N_SAMPLES, ARRAY_SIZE))
y = np.sort(X, axis=1)

X = to_categorical(X, num_classes=N_CLASSES)

model = models.Sequential(
    [
        layers.Input(shape=(ARRAY_SIZE, N_CLASSES)),
        layers.Bidirectional(layers.SimpleRNN(256, return_sequences=True)),
        layers.Bidirectional(layers.SimpleRNN(256, return_sequences=True)),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

hist = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

model.save('model.keras')

print(f"Histórico de treinamento: {hist.history}")