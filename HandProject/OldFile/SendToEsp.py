import requests
from datetime import datetime

# Địa chỉ IP của ESP32-CAM
esp32cam_ip = "192.168.4.1"

# Gửi yêu cầu GET để nhận ảnh từ ESP32-CAM
image_url = f"http://{esp32cam_ip}/cam-mid.jpg"
image_response = requests.get(image_url)

# Lấy ngày tháng hiện tại
current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Gửi dữ liệu ngày tháng lên ESP32-CAM thông qua yêu cầu POST
post_url = f"http://{esp32cam_ip}/post-data"
post_data = {
    "date": current_date
}
post_response = requests.post(post_url, json=post_data)

# In ra phản hồi từ ESP32-CAM
print(post_response.text)
