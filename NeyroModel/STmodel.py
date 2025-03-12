import json
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
from tensorflow.keras.utils import register_keras_serializable


def load_json(file_path="dataset.json"):
    """Загружает большой JSON-файл и преобразует его в массивы X и y."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    X_data = []
    y_data = np.array(data["Y"], dtype=np.int32)  # Метки классов

    for entry in data["X"]:
        if isinstance(entry, dict) and "X" in entry:
            coords_list = entry["X"]  # Берём массив координат
        else:
            coords_list = entry  # Если это просто список, берём его

        coords = np.array(coords_list).reshape(-1, 21, 3)  # (T, 21, 2)
        coords = np.transpose(coords, (2, 0, 1))  # Преобразуем в (C=2, T, V=21)
        X_data.append(coords)

    return np.array(X_data, dtype=np.float32), y_data


#  Загрузка данных
X, Y = load_json()
# Проверка формата
# print("Форма X:", X.shape)  # Ожидается (N, 2, T, 21)
# print("Форма y:", Y.shape)  # Ожидается (N,)
# print("Пример X:", X[0][:, :5, :5])  # Выведем кусочек данных
# print("Пример y:", Y[:5])  # Выведем первые 5 меток

# Находим минимальные и максимальные значения
def normalize_coordinates(X):
    """Нормализует координаты X в диапазон [0, 1]"""
    min_vals = np.min(X, axis=(0, 2, 3), keepdims=True)
    max_vals = np.max(X, axis=(0, 2, 3), keepdims=True)
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-6)  # Добавляем eps для избежания деления на 0
    return X_norm

X = normalize_coordinates(X)
# print("Мин X:", np.min(X))
# print("Макс X:", np.max(X))

# 80% train, 10% val, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

def create_tf_dataset(X, y, batch_size=32):
    """Создаёт tf.data.Dataset из numpy-массивов"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(len(y)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Создание датасетов
batch_size = 32
train_ds = create_tf_dataset(X_train, y_train, batch_size)
val_ds = create_tf_dataset(X_val, y_val, batch_size)
test_ds = create_tf_dataset(X_test, y_test, batch_size)


# соединяем пальцы и запястье
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),  # Указательный
    (0, 9), (9, 10), (10, 11), (11, 12),  # Средний
    (0, 13), (13, 14), (14, 15), (15, 16),  # Безымянный
    (0, 17), (17, 18), (18, 19), (19, 20)  # Мизинец
]

V = 21  # Число вершин (точек руки)
A = np.zeros((V, V))  # Матрица смежности

for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1  # Граф неориентированный

# Нормализация
D = np.diag(np.sum(A, axis=1))  # Степенная матрица
D_inv = np.linalg.inv(D + np.eye(V))  # Инверсия
A_hat = D_inv @ A  # Нормализованная матрица

# print("Размерности A_hat:", A_hat.shape)  # Должно быть (21, 21)

@register_keras_serializable()
class STGCNLayer(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, dropout_rate=0.3):
        super(STGCNLayer, self).__init__()
        self.out_channels = out_channels  # Здесь задаем атрибут
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Инициализация слоев
        self.gcn = layers.Conv2D(self.out_channels, kernel_size=(1, 1), padding="same")
        self.tcn = layers.Conv2D(self.out_channels, kernel_size=(self.kernel_size, 1), padding="same")
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.dropout = layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        # В данном случае метод build не нужно переопределять, так как слои уже инициализированы в __init__
        pass

    def call(self, x, A):
        # Применяем матрицу смежности A
        num_nodes = A.shape[0]  # 21
        x_channels = x.shape[-1]  # 64

        # Если число каналов не совпадает с числом узлов, приводим его к нужному формату
        if x_channels != num_nodes:
            x = layers.Dense(num_nodes)(x)

        # Перемещаем оси (N, C, T, V) -> (N, C, V, T)
        x = tf.transpose(x, perm=[0, 1, 3, 2])

        # Применяем матрицу смежности A
        x = tf.einsum("ncvt,vw->ncwt", x, A)

        # Возвращаем в изначальный формат (N, C, T, V)
        x = tf.transpose(x, perm=[0, 1, 3, 2])

        return x


# print("Размерности x перед STGCNLayer:", X.shape)
# print("Размерности A_hat:", A_hat.shape)

def build_stgcn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)  # (C=2, T, V=21)

    x = STGCNLayer(64)(inputs, A_hat)  # Первый ST-GCN слой
    x = STGCNLayer(128)(x, A_hat)  # Второй ST-GCN слой
    x = STGCNLayer(256)(x, A_hat)  # Третий ST-GCN слой

    x = layers.GlobalAveragePooling2D()(x)  # Усреднение
    x = layers.Dense(128, activation="relu")(x)  # Полносвязный слой
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # Классификация

    return keras.Model(inputs, outputs)

# Построение модели
num_classes = len(set(Y))  # Количество жестов
input_shape = (3, X.shape[2], 21)  # (C, T, V)
model = build_stgcn(input_shape, num_classes)

# Компиляция
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Вывод информации
model.summary()

# Добавляем callbacks для ранней остановки и уменьшения LR при плато
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),  # Ранняя остановка (если `val_loss` не уменьшается 5 эпох)
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)  # Уменьшение learning rate при плато
]
# Обучение модели
history = model.fit(
    train_ds,
    epochs=100,  # Количество эпох
    batch_size=32, # Размер пакета
    validation_data=val_ds,# Проверка модели на тестовом наборе во время обучения
    callbacks=callbacks # Callbacks для управления обучением
)

# Сохранение модели
model.save("stgcn_model.keras")

# Оценка точности модели на тестовом наборе
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# График точности
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.show()