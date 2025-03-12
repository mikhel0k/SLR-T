import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Загрузка обученной модели
model = tf.keras.models.load_model("gesture_model.keras")

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Классы (жесты)
gesture_classes = ['Ничего', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
                   'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц',
                   'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

# Открываем камеру
cap = cv2.VideoCapture(0)
a = ''

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Список для хранения координат (21 точки × 3 координаты)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Преобразуем в NumPy-массив и подготавливаем для нейросети
            X_input = np.array(landmarks, dtype=np.float32).reshape(1, 63, 1)

            # Получаем предсказание
            prediction = model.predict(X_input)
            class_index = np.argmax(prediction)  # Индекс предсказанного класса
            gesture = gesture_classes[class_index]  # Название жеста

            if a != gesture:
                # Выводим предсказание в консоль
                print(f"Распознанный жест: {gesture}")
                a = gesture

            # Отображаем предсказанный жест на изображении
            cv2.putText(img, f"Жест: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Рисуем точки на руке
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Отображаем изображение
    cv2.imshow("Gesture Recognition", img)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
