import numpy as np
import cv2
import autopy
import mediapipe as mp
import urllib.request
import time
import tensorflow as tf

url = "http://192.168.4.1/cam-hi.jpg"

wCam, hCam = 800, 600
frameR = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

wScr, hScr = autopy.screen.size()

# Load mô hình đã lưu
model = tf.keras.models.load_model("MODEL/hand_model")

while True:
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    landmarks_data = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                image_height, image_width, _ = img.shape
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                landmarks_data += f"{x},{y},"
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_data = landmarks_data[:-1]
    
    # Đảm bảo rằng landmarks_data không rỗng trước khi dự đoán
    if landmarks_data:
        # Chuyển landmarks_data thành mảng numpy 2D
        landmarks = np.array([list(map(int, point.split(','))) for point in landmarks_data.split(',')])

        # Dự đoán bằng mô hình đã tải
        predicted_label = np.argmax(model.predict(landmarks.reshape(1, -1)))

        # Hiển thị dự đoán
        print("Predicted fingers:", predicted_label)
        
        cv2.putText(img, str(predicted_label), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)
    if key == 27: # ASCII code for Esc key
        break

cv2.destroyAllWindows()
