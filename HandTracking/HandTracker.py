import cv2
import mediapipe as mp
import keyboard
import os


# Класс для трэкинга
class Tracker:
    letters = {
        'А' : 1,
        'Б' : 2,
        'В' : 3,
        'Г' : 4,
        'Д' : 5,
        'Ж' : 6,
        'З' : 7,
        'Е' : 8,
        'Ё' : 9,
        'И' : 10,
        'Й' : 11,
        'К' : 12,
        'Л' : 13,
        'М' : 14,
        'Н' : 15,
        'О' : 16,
        'П' : 17,
        'Р' : 18,
        'С' : 19,
        'Т' : 20,
        'У' : 21,
        'Ф' : 22,
        'Х' : 23,
        'Ц' : 24,
        'Ч' : 25,
        'Ш' : 26,
        'Щ' : 27,
        'Ъ' : 28,
        'Ы' : 29,
        'Ь' : 30,
        'Э' : 31,
        'Ю' : 32,
        'Я' : 3,
    }
    mp_hands = mp.solutions.hands  # Загружается модуль, который отвечает за отслеживание рук на изображениях
    hands = mp_hands.Hands()  # Создается объект hands для детекции и трекинга рук
    mp_draw = mp.solutions.drawing_utils  # Загружается модуль, который содержит инструменты для рисования

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def save_references(self, letter:str, start:int, finish:int):
        start -= 1
        if letter != '0':
            letter_number = self.letters[letter]
        while start < finish*5:
            if os.path.exists('background.jpg'):
                bg = cv2.imread('background.jpg')
            else:
                bg = cv2.imread('HandTracking/background.jpg')

            success, img = self.cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(imgRGB)

            if results.multi_hand_landmarks:
                start += 1
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        print(id, lm)
                        h, w, c = bg.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id == 0:
                            cv2.circle(bg, (cx, cy), 30, (255, 0, 0), cv2.FILLED)
                    self.mp_draw.draw_landmarks(bg, handLms, self.mp_hands.HAND_CONNECTIONS)

                if start % 5 == 0 and letter != '0':
                    if not os.path.exists(f'references/{letter_number}/{start // 5}.jpg'):
                        cv2.imwrite(f'references/{letter_number}/{start // 5}.jpg', bg)

                else:
                    if start % 5 == 0:
                        start = 0

            cv2.imshow("Image", bg)
            cv2.waitKey(1)


if __name__ == '__main__':
    tracker = Tracker()
    tracker.save_references('0', 0, 10)