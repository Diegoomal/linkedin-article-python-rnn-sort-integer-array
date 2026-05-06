import sys
import numpy as np  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore


ARRAY_SIZE, N_CLASSES = 10, 10

test = np.random.randint(0, 10, size=(1, 10), dtype=np.int32)

print(f"Teste:\n{test}\n")

model = load_model('model.keras', compile=False)

model_input = to_categorical(test, num_classes=N_CLASSES)

print(f"Entrada one-hot:\n{model_input}\n")

probabilities = model.predict(model_input, verbose=0)[0]

print(f"Probabilidades:\n{probabilities}\n")

pred = np.argmax(probabilities, axis=1).astype(np.int32)

print(f"valor da classe (Probabilidades & argmax):\n{pred}\n")

print(f"Entrada:\n{test[0]}")
print(f"Ordenado pela RNN:\n{pred}")
print(f"Ordenado real:\n{np.sort(test[0])}")
print(f"Predição correta: {np.array_equal(pred, np.sort(test[0]))}")