# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image 

# List1 是準備接train轉換成array資料
# List2 是準備接train Label轉換成array資料
list1 = []
list2 = []

# 讀取目標資料夾
path = "./data-release/traindata/" 
# 取得資料夾下所有label資料夾名稱
files= os.listdir(path)

for i in files:
  path = "./data-release/traindata/{}".format(i)
  # print(i)
  # 取得每個label資料夾下所有檔案名稱
  files1= os.listdir(path)
  for a in files1:
    # 打開該圖檔資料
    img = Image.open(path+"/"+a)
    # print(a)
    # 將圖檔轉換成array
    img = np.array(img)
    # 將資料加進全域的list1 製作可放進model的長array
    list1.append(img)
    # 將該資料對應label 加到全域的list2裡
    list2.append([i])

# 將承載所有資料的list1和list2轉成array 
x_train = np.array(list1)
y_train = np.array(list2)
print(x_train.shape)
print(y_train.shape)
# 存成npy檔案以供model調用
np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)

