# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import csv

################ 處理train資料 #########################

#取得data資料目錄
router = './data-release/train/' #資料目錄
classifire = './data-release/traindata/' #目標目錄


#開啟 label的 CSV 檔案
with open('./data-release/train_labels.csv') as csvfile:

    #  如果没有某個格式的資料夾,則新增這個資料夾
    if not os.path.exists(classifire):
        os.mkdir(classifire)

    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)

    for row in rows:
        # print(row)
        try:
            files = '{}/{}'.format(router,row[0]) #檔案名稱路徑
            files2  = classifire + '{}'.format(row[1]) #分類資料夾路徑
            
            #去掉csv檔的標頭
            if row[0] == 'id':
                continue
            #確認分類資料夾路徑是否存在
            if not os.path.exists(files2):
                os.mkdir(files2)
            #讀取圖片資料
            img1 = cv2.imread(files+'_c.png')
            img2 = cv2.imread(files+'_v.png')
            # img2 = cv2.imread(files+'_v.png', cv2.IMREAD_GRAYSCALE) #灰階

            #各種不同的拼接方式
            image = np.concatenate((img1, img2))  #縱向拼接
            # image = np.concatenate([img1, img2], axis=1)                     #橫向拼接
            # image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)  #融合圖片

            #將拼接好的檔案丟到分類好的路徑資料夾下
            cv2.imwrite('{}/{}.png'.format(files2,row[0]), image)

        except:
            continue


################ 處理test資料 #########################

#取得data資料目錄
router = './data-release/test/' #資料目錄
classifire = './data-release/testdata/' #目標目錄


#開啟 label的 CSV 檔案
with open('./data-release/submission_format.csv') as csvfile:

    #  如果没有某個格式的資料夾,則新增這個資料夾
    if not os.path.exists(classifire):
        os.mkdir(classifire)

    # 讀取 CSV 檔案內容
    rows = csv.reader(csvfile)

    for row in rows:
        try:
            files = '{}/{}'.format(router,row[0]) #檔案名稱路徑

            #讀取圖片資料
            img1 = cv2.imread(files+'_c.png')
            img2 = cv2.imread(files+'_v.png')
            # img2 = cv2.imread(files+'_v.png', cv2.IMREAD_GRAYSCALE) #灰階

            #各種不同的拼接方式
            image = np.concatenate((img1, img2))  #縱向拼接
            # image = np.concatenate([img1, img2], axis=1)                     #橫向拼接
            # image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)  #融合圖片

            #將拼接好的檔案丟到分類好的路徑資料夾下
            cv2.imwrite('{}/{}.png'.format(classifire,row[0]), image)

        except:
            continue
