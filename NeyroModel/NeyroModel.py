import tensorflow as tf  # Импортируем TensorFlow
from tensorflow import keras  # Импортируем Keras из TensorFlow
from tensorflow.keras import layers  # Импортируем слои Keras
import numpy as np  # Импортируем NumPy для работы с массивами
import json  # Импортируем json для работы с файлами JSON
from sklearn.model_selection import train_test_split  # Импортируем функцию для разбиения данных на обучающую и тестовую выборки


def load_dataset(file_path="dataset.json"):
    """Загружает данные из JSON и преобразует их в NumPy массивы"""
    with open(file_path, "r", encoding="utf-8") as file:  # Открываем JSON-файл
        data = json.load(file)  # Загружаем данные из файла

    X = np.array(data["X"], dtype=np.float32)  # Преобразуем X в массив NumPy (координаты суставов)
    Y = np.array(data["Y"], dtype=np.int32)  # Преобразуем Y в массив NumPy (классы)
    Y = tf.keras.utils.to_categorical(Y, num_classes=34)  # 34 класса (33 буквы + "ничего")

    return X, Y  # Возвращаем обработанные данные


# Загружаем данные из dataset.json
X, Y = load_dataset()

# Разделение данных на обучающую и тестовую выборки (80% на обучение, 20% на тест)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Изменяем форму данных для Conv1D + LSTM (добавляем ось временного шага)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Параметры модели
time_steps = 1  # Каждый жест — это один набор координат (не видео)
input_shape = (time_steps, 63)  # 63 входных признака (21 точка × 3 координаты)

# Создаем модель нейросети
model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),  # Сверточный слой 1
    layers.MaxPooling1D(pool_size=2),  # Пулинговый слой 1

    layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),  # Сверточный слой 2
    layers.MaxPooling1D(pool_size=2),  # Пулинговый слой 2

    layers.LSTM(128, return_sequences=True),  # Первый LSTM слой
    layers.LSTM(64),  # Второй LSTM слой

    layers.Dense(128, activation='relu'),  # Полносвязный слой
    layers.Dense(34, activation='softmax')  # Выходной слой (34 класса)
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Добавляем callbacks для ранней остановки и уменьшения LR при плато
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

# Обучение модели
history = model.fit(X_train, Y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_test, Y_test),
                    callbacks=callbacks)

# Сохранение модели в файл
model.save("gesture_model.h5")

# Оценка точности модели на тестовом наборе
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Точность модели: {accuracy:.4f}")

import matplotlib.pyplot as plt  # Импортируем библиотеку для построения графиков

# График точности
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("График точности модели")
plt.show()

# График потерь
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("График потерь модели")
plt.show()
