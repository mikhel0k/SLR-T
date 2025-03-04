import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


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
