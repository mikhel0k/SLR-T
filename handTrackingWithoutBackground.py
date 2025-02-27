import cv2
import mediapipe as mp
import time
import keyboard

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    bg = cv2.imread('background.jpg')
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = bg.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    cv2.circle(bg, (cx, cy), 30, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(bg, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(bg, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", bg)
    cv2.waitKey(1)

    if keyboard.is_pressed("space"):
        break