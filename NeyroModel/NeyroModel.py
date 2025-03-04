import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_dataset(file_path="dataset.json"):
    """Загружает данные из JSON и преобразует их в NumPy массивы"""
    with open(file_path, "r", encoding="utf-8") as file:  # Открываем JSON-файл
        data = json.load(file)  # Загружаем данные из файла

    # Преобразуем координаты суставов (X) в массив NumPy (float32 для ускорения вычислений)
    X = np.array(data["X"], dtype=np.float32)

    # Преобразуем метки классов (Y) в массив NumPy
    Y = np.array(data["Y"], dtype=np.int32)

    # Преобразуем метки классов в формат one-hot encoding (34 класса: 33 буквы + "ничего")
    Y = tf.keras.utils.to_categorical(Y, num_classes=34)

    return X, Y  # Возвращаем обработанные данные


# Загружаем данные из dataset.json
X, Y = load_dataset()

# Разделение данных на обучающую и тестовую выборки (80% на обучение, 20% на тест)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Изменяем форму данных для Conv1D (добавляем ось канала признаков)
X_train = X_train.reshape((X_train.shape[0], 63, 1))  # (количество примеров, 63 признака, 1 канал)
X_test = X_test.reshape((X_test.shape[0], 63, 1))  # То же самое для тестового набора


# Параметры модели
time_steps = 1  # Один временной шаг (не используется, но оставлено для совместимости)
input_shape = (63, 1)  # 63 входных признака (21 сустав × 3 координаты, 1 канал)

# Улучшенная модель CNN
model = keras.Sequential([
    # Первый сверточный блок
    layers.Conv1D(64, kernel_size=3, padding='same', input_shape=input_shape),  # Первый сверточный слой
    layers.BatchNormalization(),  # Нормализация активаций для ускорения обучения
    layers.LeakyReLU(alpha=0.1),  # Улучшенный ReLU (убирает "мёртвые нейроны")
    layers.MaxPooling1D(pool_size=2),  # Пулинг для уменьшения размерности признаков

    # Второй сверточный блок
    layers.Conv1D(128, kernel_size=3, padding='same'),  # Второй сверточный слой
    layers.BatchNormalization(),  # Нормализация для стабилизации градиентов
    layers.LeakyReLU(alpha=0.1),  # Нелинейность
    layers.MaxPooling1D(pool_size=2),  # Пулинг

    # Третий сверточный блок (новый)
    layers.Conv1D(256, kernel_size=3, padding='same'),  # Третий сверточный слой (углубляем анализ признаков)
    layers.BatchNormalization(),  # Нормализация
    layers.LeakyReLU(alpha=0.1),  # Нелинейность
    layers.MaxPooling1D(pool_size=2),  # Пулинг

    layers.Flatten(),  # Разворачиваем тензор в одномерный вектор для подачи в полносвязные слои

    # Полносвязные слои
    layers.Dense(128),  # Полносвязный слой с 128 нейронами
    layers.BatchNormalization(),  # Нормализация
    layers.LeakyReLU(alpha=0.1),  # Улучшенный ReLU
    layers.Dropout(0.3),  # Dropout (обнуляет 30% нейронов для предотвращения переобучения)

    layers.Dense(64),  # Полносвязный слой с 64 нейронами
    layers.BatchNormalization(),  # Нормализация
    layers.LeakyReLU(alpha=0.1),  # Улучшенный ReLU
    layers.Dropout(0.3),  # Dropout (дополнительная защита от переобучения)

    layers.Dense(34, activation='softmax')  # Выходной слой (34 класса)
])


# Компиляция модели
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),  # Оптимизатор Adam с LR 0.001
              loss='categorical_crossentropy',  # Функция потерь для многоклассовой классификации
              metrics=['accuracy'])  # Метрика — точность классификации

# Вывод структуры модели
model.summary()

# Добавляем callbacks для ранней остановки и уменьшения LR при плато
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),  # Ранняя остановка (если `val_loss` не уменьшается 5 эпох)
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)  # Уменьшение learning rate при плато
]

# Обучение модели
history = model.fit(X_train, Y_train,
                    epochs=100,  # Количество эпох
                    batch_size=32,  # Размер пакета
                    validation_data=(X_test, Y_test),  # Проверка модели на тестовом наборе во время обучения
                    callbacks=callbacks)  # Callbacks для управления обучением

# Сохранение модели в файл
model.save("gesture_model.keras")  # Сохранение модели для последующего использования

# Оценка точности модели на тестовом наборе
loss, accuracy = model.evaluate(X_test, Y_test)  # Оцениваем модель на тестовых данных
print(f"Точность модели: {accuracy:.4f}")  # Выводим точность


# График точности
plt.plot(history.history['accuracy'], label='Train Accuracy')  # Точность на обучающей выборке
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Точность на тестовой выборке
plt.legend()
plt.title("График точности модели")
plt.show()

# График потерь
plt.plot(history.history['loss'], label='Train Loss')  # Функция потерь на обучающей выборке
plt.plot(history.history['val_loss'], label='Validation Loss')  # Функция потерь на тестовой выборке
plt.legend()
plt.title("График потерь модели")
plt.show()
