# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split #隨機切分資料用
from tensorflow import keras
from keras import regularizers
from keras.callbacks import ModelCheckpoint


# 讀取array檔
x = np.load('x_train.npy')
y = np.load('y_train.npy')

#隨機切分資料分成訓練和驗證
# x_train = 訓練資料
# y_train = 訓練資料的label
# x_test = 測試資料的
# x_test = 測試資料的label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#轉換資料型態成浮點數
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#資料正規化(將RGB色碼數字最大為255 => 除以255 除以最大數字就是1,所有數字都會介於0~1之間)
x_train = x_train / 255
x_test = x_test / 255


#oneHot 編碼:去掉不重要的特徵值,或將特徵值二元化 ex.00010  01000
NUM_CLASSES = 11  #總共11個分類
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
# print(x_train.shape[1:])

"""### 建立模型"""
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, activation=tf.nn.relu, kernel_size=(3, 3), input_shape=x_train.shape[1:]))  #新增卷積層
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
model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))
model.summary()


# compile模型
model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',             # loss函數選擇adam
              metrics=['accuracy'])

#設立檢查點
filepath='weights.best.hdf5'

# 有一次提升,則將該次權重儲存
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

# 執行訓練
history = model.fit(x_train, y_train, callbacks=callbacks_list, validation_data = (x_test, y_test),epochs=1000)


### 保存模型
# model.save('DAT264-x_CNN.h5')   #該檔案保存模型結構、模型權重