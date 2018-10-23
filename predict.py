# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator     #導入kearas圖像預處理模組
from PIL import Image 
import os 
import csv

# list1 = []
# 讀取訓練好的model
model = tf.keras.models.load_model('DAT264-x_CNN.h5')

router = './data-release/testdata' #資料目錄

# print(files)
#打開測試資料集的csv檔案
with open('./data-release/submission_format.csv') as csvfile:
    # 讀取csv 檔案的內容
    rows = csv.reader(csvfile)
    # writer = csv.writer(csvfile)
    for row in rows:
        try:
            # 迴圈讀取每一個csv檔案中相對應的測試資料
            img = Image.open('{}/{}.png'.format(router,row[0]))
            # print('{}/{}.png'.format(router,row[0]))
            # 轉換資料為 array 
            img = np.array(img)
            # 跟訓練資料集一樣轉換資料型態以符合訓練資料的型態
            img = img.astype('float32')
            img = img / 255
            img = img.reshape(1, 256, 118, 3)
            # print(img.shape)
            results = np.argmax(model.predict(img)) #預測
            confidence = np.max(model.predict(img))*100  #信心指數
            confidence = '%.2f' % confidence
            # 打印出結果
            print(results)
            # print(confidence) 打印該結果信心指數
            # list1.append(results)
        except:
            continue

# print(list1)