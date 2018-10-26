# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator     #導入kearas圖像預處理模組
from sklearn.model_selection import train_test_split #隨機切分資料用
from PIL import Image 
import os 
import csv


"""### 建立模型"""
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, activation=tf.nn.relu, kernel_size=(3, 3), input_shape=(256,118,3)))  #新增卷積層
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))   #最大池化
model.add(keras.layers.Dropout(0.25))                 #防止過擬合

model.add(keras.layers.Conv2D(64, activation=tf.nn.relu,kernel_size=(3, 3)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(128, activation=tf.nn.relu,kernel_size=(3, 3)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(256, activation=tf.nn.relu,kernel_size=(3, 3)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(512, activation=tf.nn.relu,kernel_size=(3, 3)))
model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(550, activation=tf.nn.relu))  #普通的全連接層
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(330, activation=tf.nn.relu))  #普通的全連接層
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(110, activation=tf.nn.relu))  #普通的全連接層
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(44, activation=tf.nn.relu))  #普通的全連接層
model.add(keras.layers.Dense(11, activation=tf.nn.softmax))
model.summary()

# load weights 載入模型權重
model.load_weights('weights.best.hdf5')

# compile模型
model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('載入模型和權重')


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