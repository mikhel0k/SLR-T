import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Определение пользовательских слоев с учётом передачи параметра training

class GraphConv(tf.keras.layers.Layer):
    def __init__(self, output_dim, adjacency_matrix, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.adjacency_matrix = adjacency_matrix  # храним матрицу как numpy array

    def build(self, input_shape):
        # Преобразуем матрицу смежности в тензор
        self.A = tf.convert_to_tensor(self.adjacency_matrix, dtype=tf.float32)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, training=None):
        # Передаем параметр training, если потребуется
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


class STGCNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, adjacency_matrix, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.adjacency_matrix = adjacency_matrix
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # Инициализация внутренних слоев
        self.graph_conv = GraphConv(self.filters, self.adjacency_matrix)
        self.temporal_conv = tf.keras.layers.Conv2D(
            self.filters,
            kernel_size=(self.kernel_size, 1),
            padding='same'
        )
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        x = self.graph_conv(inputs, training=training)
        x = self.temporal_conv(x)
        # Передаем параметр training в BatchNormalization
        x = self.bn(x, training=training)
        return tf.nn.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "kernel_size": self.kernel_size
        })
        return config


# Загрузка обученной модели с использованием кастомных слоев
model = tf.keras.models.load_model(
    "gesture_model.keras",
    custom_objects={
        'GraphConv': GraphConv,
        'STGCNBlock': STGCNBlock
    }
)

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Определение классов жестов
gesture_classes = ['Ничего', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
                   'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
                   'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

# Открываем камеру
cap = cv2.VideoCapture(0)
prev_gesture = ''

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Сбор координат 21 точки (x, y, z) -> 63 элемента
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Подготовка данных для модели:
            # Преобразуем 63 элемента в форму (63, 1), что соответствует ожиданиям модели
            X_input = np.array(landmarks, dtype=np.float32).reshape(63, 1)
            # Добавляем размер батча, итоговая форма (1, 63, 1)
            X_input = np.expand_dims(X_input, axis=0)

            # Получение предсказания в режиме инференса
            prediction = model.predict(X_input)
            class_index = np.argmax(prediction)
            gesture = gesture_classes[class_index]

            if prev_gesture != gesture:
                print(f"Распознанный жест: {gesture}")
                prev_gesture = gesture

            # Вывод предсказанного жеста на изображении
            cv2.putText(img, f"Жест: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            # Рисуем скелет руки
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", img)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
