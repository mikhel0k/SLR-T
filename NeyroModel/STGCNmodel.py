import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

def load_and_process_data(file_path="dataset.json", sequence_length=30, test_size=0.2):
    # 1. Загрузка данных из JSON
    with open(file_path, 'r') as f:
        data = json.load(f)

    X = data['X']
    Y = data['Y']

    # 2. Проверка и очистка данных
    X_clean, Y_clean = [], []
    for x, y in zip(X, Y):
        if len(x) == 63:  # Проверка корректной размерности
            X_clean.append(x)
            Y_clean.append(y)

    # 3. Нормализация данных
    def normalize_frame(frame):
        frame = np.array(frame).reshape(21, 3)
        wrist = frame[0]
        frame_centered = frame - wrist
        max_val = np.max(np.abs(frame_centered)) + 1e-8  # Защита от деления на 0
        return (frame_centered / max_val).flatten()

    X_normalized = [normalize_frame(x) for x in X_clean]

    # 4. Формирование временных последовательностей
    # Преобразуем в numpy array
    X_np = np.array(X_normalized).reshape(len(X_normalized), -1, 63)  # (samples, frames, 63)

    # Дополнение/обрезание до фиксированной длины последовательности
    processed_sequences = []
    for seq in X_np:
        num_frames = seq.shape[0]

        if num_frames < sequence_length:
            # Добавляем нулевые кадры в конец
            pad = np.zeros((sequence_length - num_frames, 63))
            processed = np.vstack([seq, pad])
        else:
            # Обрезаем до нужной длины
            processed = seq[:sequence_length]

        processed_sequences.append(processed)

    X_sequences = np.array(processed_sequences)

    # 5. Построение графа (матрицы смежности)
    # Определяем связи между суставами для MediaPipe Hands
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # большой палец
        (0, 5), (5, 6), (6, 7), (7, 8),  # указательный
        (0, 9), (9, 10), (10, 11), (11, 12),  # средний
        (0, 13), (13, 14), (14, 15), (15, 16),  # безымянный
        (0, 17), (17, 18), (18, 19), (19, 20)  # мизинец
    ]

    num_nodes = 21
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # Неориентированный граф

    # 6. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, Y_clean,
        test_size=test_size,
        stratify=Y_clean,
        random_state=42
    )

    # 7. Преобразование формата для ST-GCN
    def reshape_for_stgcn(data):
        # Исходная форма: (samples, frames, 63)
        # Новая форма: (samples, channels, frames, nodes)
        return data.reshape(*data.shape[:2], 3, 21).transpose(0, 2, 1, 3)

    X_train = reshape_for_stgcn(X_train)
    X_test = reshape_for_stgcn(X_test)

    # 8. Преобразование меток
    num_classes = len(np.unique(Y_clean))
    y_train =tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return X_train, X_test, y_train, y_test, A


# Использование
X_train, X_test, y_train, y_test, adjacency_matrix = load_and_process_data()

# Вывод информации о данных
# print(f"Train shape: {X_train.shape}")
# print(f"Test shape: {X_test.shape}")
# print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
# print(f"Train labels shape: {y_train.shape}")

class GraphConv(layers.Layer):
    def __init__(self, output_dim, adjacency_matrix, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.adjacency_matrix = adjacency_matrix  # Храним как numpy array

    def build(self, input_shape):
        # Преобразуем в тензор при создании весов
        self.A = tf.convert_to_tensor(self.adjacency_matrix, dtype=tf.float32)

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # Динамическое преобразование в тензор в текущем контексте
        A = tf.convert_to_tensor(self.adjacency_matrix, dtype=tf.float32)

        x = tf.einsum('ij,bkjf->bkif', A, inputs)
        x = tf.einsum('bkif,fg->bkig', x, self.kernel)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "adjacency_matrix": self.adjacency_matrix.tolist()
        })
        return config


class STGCNBlock(layers.Layer):
    def __init__(self, filters, adjacency_matrix, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.adjacency_matrix = adjacency_matrix
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Инициализируем все слои внутри build()
        self.graph_conv = GraphConv(self.filters, self.adjacency_matrix)
        self.temporal_conv = layers.Conv2D(
            self.filters,
            kernel_size=(self.kernel_size, 1),
            padding='same'
        )
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        x = self.graph_conv(inputs)
        x = self.temporal_conv(x)
        x = self.bn(x)
        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "kernel_size": self.kernel_size
        })
        return config


def build_stgcn_model(input_shape, adjacency_matrix, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Permute((2, 3, 1))(inputs)

    # Инициализация матрицы смежности для каждого блока
    x = STGCNBlock(64, adjacency_matrix.numpy() if tf.is_tensor(adjacency_matrix) else adjacency_matrix)(x)
    x = layers.MaxPool2D((2, 1))(x)

    x = STGCNBlock(128, adjacency_matrix.numpy() if tf.is_tensor(adjacency_matrix) else adjacency_matrix)(x)
    x = layers.MaxPool2D((2, 1))(x)

    x = STGCNBlock(256, adjacency_matrix.numpy() if tf.is_tensor(adjacency_matrix) else adjacency_matrix)(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# Использование
adjacency_matrix = np.array(adjacency_matrix)  # Убедитесь, что это numpy array
model = build_stgcn_model(
    input_shape=X_train.shape[1:],
    adjacency_matrix=adjacency_matrix,
    num_classes=y_train.shape[1]
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
)

model.save("stgcn_model2.keras", save_format="keras")

# Оценка точности модели на тестовом наборе
loss, accuracy = model.evaluate(X_test, y_test)  # Оцениваем модель на тестовых данных
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