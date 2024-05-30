import numpy as np
import cv2
import autopy
import mediapipe as mp
import urllib.request
import time

url = "http://192.168.4.1/cam-hi.jpg"

wCam, hCam = 800, 600

plocX, plocY = 0, 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

wScr, hScr = autopy.screen.size()

landmarks_data = ""

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_data = ",".join([f"{int(landmark.x * img.shape[1])},{int(landmark.y * img.shape[0])}" for landmark in hand_landmarks.landmark])
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if landmarks_data:  # Check if the pressed key is a number and landmarks_data is not empty
        with open('hand_landmarks_DATA.csv', 'a', newline='') as file:  # Open the CSV file in append mode
            file.write(f"{0},{landmarks_data}\n")  # Write the data with the key pressed at the beginning of the line
            print(f"Data saved for key '{0}'")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27: # ASCII code for Esc key
        break

cv2.destroyAllWindows()
