import numpy as np
import cv2
import autopy
import mediapipe as mp
import urllib.request
import tensorflow as tf
import requests

# Địa chỉ IP của ESP32-CAM
esp32cam_ip = "192.168.4.1"
url = f"http://{esp32cam_ip}/cam-hi.jpg"

# Cấu hình camera và điều khiển mượt mà
wCam, hCam = 800, 600
frameR = 100
smoothening = 7
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Thiết lập Mediapipe và Autopy
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils
wScr, hScr = autopy.screen.size()

# Tải mô hình TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="MODEL/hand_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

while True:
    # Lấy hình ảnh từ ESP32-CAM
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)

    # Xử lý hình ảnh bằng Mediapipe
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    landmarks_data = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                image_height, image_width, _ = img.shape
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                landmarks_data += f"{x},{y},"
            #mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks_data = landmarks_data[:-1]
    
    # Dự đoán và gửi dữ liệu nếu landmarks_data không rỗng
    if landmarks_data:
        landmarks = np.array([list(map(int, point.split(','))) for point in landmarks_data.split(',')]).reshape(1, -1)
        input_data = np.array(landmarks, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = int(np.argmax(output_data))

        post_url = f"http://{esp32cam_ip}/post-data"
        post_data = {"Predict": predicted_label}
        requests.post(post_url, json=post_data)

    # Hiển thị ảnh nền
    imageNEN = cv2.imread('Logo_PTIT_University.jpg')
    cv2.imshow("Image", imageNEN)

    # Thoát nếu nhấn phím Esc
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
