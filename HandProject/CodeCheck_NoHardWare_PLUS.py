import tensorflow as tf
import numpy as np
import requests
from datetime import datetime
import cv2

# Load mô hình đã lưu (TensorFlow Lite)
interpreter = tf.lite.Interpreter(model_path="MODEL/hand_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example input data
# Example Number 6
x = np.array([[600,264,610,209,587,148,566,105,561,70,512,153,463,126,429,114,398,103,493,186,469,175,511,196,544,210,481,220,454,211,495,226,528,235,473,252,426,236,400,225,375,211]], dtype=np.float32)

# Chuẩn bị dữ liệu đầu vào cho mô hình TensorFlow Lite
input_data = np.array(x.reshape(1, -1), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Thực thi mô hình TensorFlow Lite
interpreter.invoke()

# Lấy kết quả từ mô hình TensorFlow Lite
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)
predicted_label2 = output_data

# Hiển thị dự đoán
print("Predicted fingers:", predicted_label)
print("Predicted fingers2:", predicted_label2)

# Chuyển đổi predicted_label sang kiểu dữ liệu int
predicted_label = int(predicted_label)

# Địa chỉ IP của ESP32-CAM
esp32cam_ip = "192.168.4.1"

# Gửi yêu cầu GET để nhận ảnh từ ESP32-CAM
image_url = f"http://{esp32cam_ip}/cam-mid.jpg"
image_response = requests.get(image_url)

# Gửi dữ liệu dự đoán lên ESP32-CAM thông qua yêu cầu POST
post_url = f"http://{esp32cam_ip}/post-data"
post_data = {
    "Predict": predicted_label
}
post_response = requests.post(post_url, json=post_data)

# In ra phản hồi từ ESP32-CAM
print(post_response.text)
