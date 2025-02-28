import cv2
import mediapipe as mp
import keyboard

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
count = 0

while True:
    bg = cv2.imread('background.jpg')
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        count += 1
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = bg.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 0:
                    cv2.circle(bg, (cx, cy), 30, (255, 0, 0), cv2.FILLED)
            mpDraw.draw_landmarks(bg, handLms, mpHands.HAND_CONNECTIONS)

        if count % 5 == 0 and count > 0:
            cv2.imwrite(f'images/2-{count//5}.jpg', bg)

    cv2.imshow("Image", bg)
    cv2.waitKey(1)

    if keyboard.is_pressed("space"):
        break
