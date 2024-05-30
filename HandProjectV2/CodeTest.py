import requests
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time

# URL của server ESP32
url = 'http://192.168.4.1/capture'

# Tên file CSV để lưu dữ liệu ảnh
csv_filename = 'image_data.csv'
label = 0  # Thay thế 'label_value' bằng giá trị label thực tế của bạn

# Kiểm tra xem file CSV đã tồn tại chưa
file_exists = os.path.isfile(csv_filename)

# Mở file CSV ở chế độ append
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Ghi header vào file CSV nếu file chưa tồn tại
    if not file_exists:
        header = ['label'] + [f'pixel_{i}' for i in range(32 * 24)]
        csv_writer.writerow(header)
    
    # Lặp lại 10 lần
    for i in range(20):
        # Gửi yêu cầu và nhận JSON phản hồi
        response = requests.get(url)
        data = response.json()

        if i%2 == 1:# Chống dội ảnh
            # Lấy dữ liệu ảnh từ JSON
            image_data = data['image']

            # Chuyển đổi dữ liệu ảnh thành numpy array
            image_array = np.array(image_data, dtype=np.uint8)
            image_array = image_array.reshape((24, 32))

        
            # Hiển thị ảnh
            plt.imshow(image_array, cmap='gray')
            plt.show()

            # Ghi dữ liệu ảnh thành một hàng vào file CSV
            row = [label] + image_array.flatten().tolist()
            csv_writer.writerow(row)

        # Đợi một thời gian ngắn trước khi chụp lần tiếp theo (nếu cần)
        #time.sleep(1)  # Đợi 1 giây (có thể điều chỉnh thời gian nếu cần)

print(f"Dữ liệu ảnh đã được ghi vào file {csv_filename}")
