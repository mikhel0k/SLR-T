import cv2
import mediapipe as mp
import os
import json


# Класс для трекинга рук
class Tracker:
    # Словарь, где каждой букве соответствует числовой идентификатор
    letters = {
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18,
        'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26, 'Nothing': 0
    }

    def __init__(self, camera):
        """ Инициализация трекера рук (с ограничением на 1 руку) """
        self.mp_hands = mp.solutions.hands  # MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1  # Ограничиваем обработку до одной руки
        )
        self.mp_draw = mp.solutions.drawing_utils  # Утилиты для рисования
        self.cap = cv2.VideoCapture(camera)  # Захват видео с камеры

        # Создаем основную папку для английских референсов
        self.create_directories()

    def create_directories(self):
        """Создает все необходимые директории для сохранения данных"""
        # Создаем основную папку english_references на уровень выше
        parent_dir = os.path.join(os.path.dirname(__file__), '..', 'english_references')
        parent_dir = os.path.normpath(parent_dir)  # Нормализуем путь
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Created parent directory: {parent_dir}")

        # Создаем подпапки для каждой буквы
        for letter, number in self.letters.items():
            folder_path = os.path.join(parent_dir, str(number))
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created directory: {folder_path}")

    def find_hands(self, img):
        """ Обрабатывает изображение и возвращает результаты детекции рук """
        return self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def get_hand_coords(self, results, bg):
        """ Возвращает список найденных рук, если они есть """
        coords = []
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks  # Возвращаем объект с координатами рук
        return coords  # Если рук нет, возвращаем пустой список

    def save_references(self, letter: str, start: int, finish: int):
        """
        Захватывает кадры с камеры, отслеживает руки и сохраняет данные о координатах
        в JSON и изображения с разметкой в папку english_references/
        """
        start = (start - 1) * 2  # Коррекция номера кадра

        # Всегда получаем числовой идентификатор буквы
        letter_number = self.letters[letter]

        # Создаем пути относительно родительской директории
        parent_dir = os.path.join(os.path.dirname(__file__), '..', 'english_references')
        parent_dir = os.path.normpath(parent_dir)

        json_path = os.path.join(parent_dir, str(letter_number), '')
        img_path = os.path.join(parent_dir, str(letter_number), '')

        print(f"Starting capture for letter: {letter} (ID: {letter_number})")
        print(f"Saving to: {json_path}")

        frame_count = 0
        saved_count = 0

        while saved_count < finish:  # Сохраняем нужное количество файлов
            # Проверяем, существует ли фоновое изображение
            bg = None
            if os.path.exists('background.jpg'):
                bg = cv2.imread('background.jpg')
            elif os.path.exists('HandTracking/background.jpg'):
                bg = cv2.imread('HandTracking/background.jpg')
            else:
                # Создаем черный фон если файл не найден
                import numpy as np
                bg = np.zeros((480, 640, 3), dtype=np.uint8)
                print("Background image not found, using black background")

            success, img = self.cap.read()  # Считываем кадр с камеры
            if not success:
                print("Failed to read frame from camera")
                break

            results = self.find_hands(img)  # Обрабатываем изображение и ищем руки
            coords = self.get_hand_coords(results, bg.shape)  # Получаем координаты найденных рук

            if coords:  # Если руки найдены
                hand_data = {}  # Создаём словарь для хранения координат
                frame_count += 1

                # Проходим по всем найденным рукам
                for handLms in coords:
                    for id, lm in enumerate(handLms.landmark):  # Перебираем точки руки (21 точка)
                        hand_data[id] = {
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z
                        }
                    # Рисуем точки и соединения на изображении
                    self.mp_draw.draw_landmarks(bg, handLms, self.mp_hands.HAND_CONNECTIONS)

                # Сохраняем каждый 2-й кадр с рукой
                if frame_count % 2 == 0:
                    file_number = saved_count + 1
                    img_filename = os.path.join(img_path, f'{file_number}.jpg')
                    json_filename = os.path.join(json_path, f'{file_number}.json')

                    # Сохраняем изображение
                    cv2.imwrite(img_filename, bg)

                    # Записываем данные в JSON-файл
                    with open(json_filename, "w", encoding="utf-8") as file:
                        data_to_save = {
                            letter: hand_data
                        }
                        json.dump(data_to_save, file, indent=4, ensure_ascii=False)

                    print(f"Saved: {img_filename} and {json_filename}")
                    saved_count += 1

            # Показываем обработанный кадр
            cv2.imshow("Hand Tracking - Press ESC to exit", bg)

            # Проверяем нажатие ESC для выхода
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

        print(f"Completed! Saved {saved_count} files for letter '{letter}'")
        self.cap.release()
        cv2.destroyAllWindows()


# Основной блок кода
if __name__ == '__main__':
    tracker = Tracker(0)  # Создаём объект трекера рук с камерой 0 (по умолчанию)

    # Тестируем с буквой 'A' вместо 'Nothing'
    tracker.save_references('A', 1, 5)  # Сохраняем 5 файлов для буквы A