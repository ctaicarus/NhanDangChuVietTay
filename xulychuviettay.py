import numpy as np
import cv2
from matplotlib import pyplot as ptl

img = cv2.imread('resource/dulieu.png', 0);  # ảnh nhị phân, ảnh xám, kích thước 2000*1000, 50 hàng, 50 cột
cells = [np.hsplit(row, 50) for row in np.vsplit(img, 50)]  # cắt các ảnh

# print(cell) # cell là ma trận số 2 chiều
# print(cell[0][0]) ma trận của điểm 0 0

cv2.imwrite('anhso00.png', cells[0][0])  # lưu ảnh điểm 0 0, kích thước 20*20 pixels

x = np.array(cells)  # chuyển cell mảng 2 chiều thành x mảng 1 chiều

x00 = x[0, 0].reshape(-1, 400)  # -1 sẽ biến đổi theo tham số phía sau, 400 = 20*20, ảnh sau sẽ là 1 hàng 400 cột
# print(x00) # ma trận 1 chiều
cv2.imwrite('anhso00new.png', x00)  # lưu ảnh về dạng 1*400

# dữ liệu train
train = x[:, :25].reshape(-1, 400).astype(np.float32)  # lấy nửa bên trái: cột 0-25, có

# dữ liệu test
test = x[:, 25:50].reshape(-1, 400).astype(np.float32)  # lấy nửa bên phải: cột 25-50

# dữ liệu chuẩn
k = np.arange(10)
# print(numbers)

# gán nhãn dữ liệu train
train_labels = np.repeat(k, 125)[:, np.newaxis]  # chuyển 2 chiều thành 1 chiều, và gán nhãn với dữ liệu chuẩn, 125 = 5*25 mỗi kí tự
# print(train_labels)

# nhận dạng
knn = cv2.ml.KNearest_create()
knn.train(train, 0, train_labels)  # học dữ liệu truyền vào dữ liệu train và dữ liệu chuẩn
kq1, kq2, kq3, kq4 = knn.findNearest(test, 5) # 5 là tìm 5 láng giềng gần nó nhất
print(kq2[570]) # kết quả từ test
print(train_labels[570]) # kết quả chuẩn

