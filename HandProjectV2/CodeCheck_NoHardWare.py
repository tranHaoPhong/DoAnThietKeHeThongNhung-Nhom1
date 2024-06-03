import tensorflow as tf
import numpy as np
import pandas as pd

# Đường dẫn tới file CSV
csv_path = "image_data.csv"

# Đọc dữ liệu từ file CSV
data = pd.read_csv(csv_path)

# Chọn một hàng bất kỳ
row_index = 5  # bạn có thể thay đổi chỉ số này để chọn hàng khác
selected_row = data.iloc[row_index]

# Cột đầu tiên là label, các cột còn lại là dữ liệu điểm ảnh
label = selected_row[0]
x = selected_row[1:].values

# Chuẩn bị dữ liệu đầu vào
x = x.reshape(1, 28, 28, 1) / 255.0  # Chuẩn hóa dữ liệu như khi huấn luyện
x = (x * 255 - 128).astype(np.int8)  # Chuyển đổi dữ liệu sang int8

# Load mô hình đã lưu (TensorFlow Lite)
interpreter = tf.lite.Interpreter(model_path="HandModel.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Kiểm tra hình dạng đầu vào mong đợi của mô hình
input_shape = input_details[0]['shape']

# Chuẩn bị dữ liệu đầu vào cho mô hình TensorFlow Lite với hình dạng phù hợp
interpreter.set_tensor(input_details[0]['index'], x)

# Thực thi mô hình TensorFlow Lite
interpreter.invoke()

# Lấy kết quả từ mô hình TensorFlow Lite
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)
predicted_label2 = output_data

# Hiển thị dự đoán
print("Actual label:", label)
print("Predicted label:", predicted_label)
print("Model output probabilities:", predicted_label2)
