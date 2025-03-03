import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import os

def load_data(json_folder):
    x, y = [], []  # Списки для входных данных (координат) и выходных меток (классов)

    label_map = {"Ничего": 0}  # Неизвестный элемент теперь 0
    label_map.update({letter: i + 1 for i, letter in enumerate("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")})
    print(label_map)
    for i in range(34):
        path = json_folder + '\\' + str(i)
        for filename in os.listdir(path):
            try:
                with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for letter, points in data.items():
                        label = label_map.get(letter, 33)  # Если буква неизвестна → 33
                        features = []
                        for i in range(21):  # 21 точки
                            joint = points[str(i)]
                            features.extend([joint["x"], joint["y"], joint["z"]])
                        x.append(features)
                        y.append(label)
            except Exception as e:
                pass
    X = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    y = tf.keras.utils.to_categorical(y, num_classes=34)  # One-hot encoding
    return X, y

# Параметры модели
time_steps = 1  # Каждый жест — это один набор координат (не видео)
input_shape = (time_steps, 63)  # 63 входных признака (21 точка × 3 координаты)

model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPooling1D(pool_size=2),

    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),

    layers.Dense(128, activation='relu'),
    layers.Dense(34, activation='softmax')  # 34 класса (33 буквы + 1 "неизвестная буква")
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Вывод структуры модели
model.summary()

x, y = load_data(os.path.abspath(os.path.join(os.getcwd(), "..", "references")))

for i in range(5):
    print(x[i])
    print(y[i])
