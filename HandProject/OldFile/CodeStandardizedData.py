import pandas as pd

# Đọc file CSV
file_path = 'hand_landmarks_DATA.csv'
df = pd.read_csv(file_path, header=None)

# Hàm để chuẩn hóa tọa độ
def normalize_coordinates(row):
    new_row = [row[0]]  # Giữ nguyên cột đầu tiên
    for i in range(1, len(row)):
        if i % 2 == 1:  # tọa độ x
            new_row.append(row[i] / 800)
        else:           # tọa độ y
            new_row.append(row[i] / 600)
    return new_row

# Áp dụng hàm normalize_coordinates lên từng dòng của dataframe
df = df.apply(normalize_coordinates, axis=1, result_type='expand')

# Ghi lại file CSV đã chuẩn hóa
output_file_path = 'hand_landmarks_DATA(fix).csv'
df.to_csv(output_file_path, header=False, index=False)

print(f'Dữ liệu đã được chuẩn hóa và lưu tại: {output_file_path}')
