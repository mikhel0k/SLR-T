import cv2
import mediapipe as mp
import os
import json


# Класс для трекинга рук
class Tracker:
    # Словарь, где каждой букве соответствует числовой идентификатор
    letters = {
        'А': 1, 'Б': 2, 'В': 3, 'Г': 4, 'Д': 5,'Е': 6, 'Ё': 7, 'Ж': 8, 'З': 9,
        'И': 10, 'Й': 11, 'К': 12, 'Л': 13, 'М': 14, 'Н': 15, 'О': 16, 'П': 17, 'Р': 18,
        'С': 19, 'Т': 20, 'У': 21, 'Ф': 22, 'Х': 23, 'Ц': 24, 'Ч': 25, 'Ш': 26, 'Щ': 27,
        'Ъ': 28, 'Ы': 29, 'Ь': 30, 'Э': 31, 'Ю': 32, 'Я': 33, 'Ничего': 0,
    }

    def __init__(self, camera):
        """ Инициализация трекера рук (с ограничением на 1 руку) """
        self.mp_hands = mp.solutions.hands  # MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1  # Ограничиваем обработку до одной руки
        )
        self.mp_draw = mp.solutions.drawing_utils  # Утилиты для рисования
        self.cap = cv2.VideoCapture(camera)  # Захват видео с камеры

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
        в JSON и изображения с разметкой в папку references/
        """
        start = (start - 1) * 2  # Коррекция номера кадра
        if letter != '0':
            letter_number = self.letters[letter]  # Получаем числовой идентификатор буквы
            json_path = f'references/{letter_number}/'  # Формируем путь для сохранения JSON-файлов

        while start < finish * 2:  # Цикл записи данных
            # Проверяем, существует ли фоновое изображение
            if os.path.exists('background.jpg'):
                bg = cv2.imread('background.jpg')
            else:
                bg = cv2.imread('HandTracking/background.jpg')

            success, img = self.cap.read()  # Считываем кадр с камеры
            results = self.find_hands(img)  # Обрабатываем изображение и ищем руки
            coords = self.get_hand_coords(results, bg.shape)  # Получаем координаты найденных рук

            if coords:  # Если руки найдены
                hand_data = {}  # Создаём словарь для хранения координат
                start += 1  # Увеличиваем счётчик кадров

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

                # Сохраняем изображение и JSON каждые 2 кадра
                if start % 2 == 0 and letter != '0':
                    if not os.path.exists(f'references/{letter_number}/{start // 2}.jpg'):
                        # Сохраняем изображение с нанесёнными точками руки
                        cv2.imwrite(f'references/{letter_number}/{start // 2}.jpg', bg)

                    # Записываем данные в JSON-файл
                    with open(json_path + f'{start // 2}.json', "w", encoding="utf-8") as file:
                        hand_data = {
                            letter: hand_data
                        }
                        print(start // 2, hand_data)  # Выводим отладочную информацию
                        json.dump(hand_data, file, indent=4, ensure_ascii=False)  # Сохраняем координаты в JSON

                elif letter == '0':  # Если передан символ '0', сбрасываем счётчик
                    start = 0

            cv2.imshow("Image", bg)  # Показываем обработанный кадр
            cv2.waitKey(1)  # Ожидание нажатия клавиши (для обновления окна)
            if cv2.waitKey(1) & 0xFF == 27:  # Нажатие ESC закроет программу
                break

        self.cap.release()  # Закрываем камеру
        cv2.destroyAllWindows()


# Основной блок кода
if __name__ == '__main__':
    tracker = Tracker(0)  # Создаём объект трекера рук с камерой 0 (по умолчанию)
    tracker.save_references('0', 0, 10)  # Запускаем процесс сохранения данных
