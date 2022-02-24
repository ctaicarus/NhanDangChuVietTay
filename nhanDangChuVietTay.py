import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('resource/dulieu.png', 0)  # dữ liệu, 50*50 pixels

imgInput = cv2.imread('resource/test/test_so0.png', 0)  # ảnh đầu vào, 1*1 pixels

input = np.array(imgInput)  # chuyển ảnh về ma trận
#print(input)

cells = [np.hsplit(row, 50) for row in np.vsplit(img, 50)]  # cắt các ảnh có độ phân giải 1*1 pixels

x = np.array(cells)  # chuyển cells 2 chiều thành về mảng 1 chiều

# tạo data train và test
train = x[:, :50].reshape(-1, 400).astype(np.float32)
test = input.reshape(-1, 400).astype(np.float32)

# dữ liệu chuẩn, gán nhãn cho dữ liệu train
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]  # gán nhãn với dữ liệu chuẩn, 250 = 5*50 mỗi kí tự

# nhận dạng
knn = cv2.ml.KNearest_create()
knn.train(train, 0, train_labels)
kq1, kq2, kq3, kq4 = knn.findNearest(test, 5) # thuật toán knn, 5 là số láng giềng

print(kq1)
print(kq2) # kết quả cần tìm
print(kq3) # hàng xóm của nó
print(kq4) # khoảng cách đến hàng xóm của nó
