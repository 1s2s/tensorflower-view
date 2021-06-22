# coding=utf-8

from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import glob
import math

#读图片的数量
readImgCount = 2000
#从哪张开始读图片
appendStart = readImgCount*3

#保存txt的名称
txtname = '-cut-4-4kli'
#图像格式
picType = '/*.bmp'
#图像大小
w=100
h=100
c=3

doc=open("min_cols-ave "+str(appendStart)+"-"+str(appendStart+readImgCount)+txtname+'.txt','w')
bpdpath= '.\\Cut-4pic\\1bpd\\'
cjwpath = '.\\Cut-4pic\\4kli\\'
####非设置选项
def read_img(path):
    imgs=[]
    readImg = 0
    readStart = 0
    for im in glob.glob(path+picType):
        if readStart>=appendStart:
            if readImg >= readImgCount:
                return np.asarray(imgs,np.float32)
            readImg=readImg+1
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
        readStart = readStart+1
        print(readStart)
    return np.asarray(imgs,np.float32)
with tf.Session() as sess:

    bpddata = read_img(bpdpath)
    cjwdate = read_img(cjwpath)
    saver = tf.train.import_meta_graph('.\\t-2d-cnn\\4Pic\\tfdroid.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('.\\t-2d-cnn\\4Pic\\'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    logits = graph.get_tensor_by_name("logits_eval:0")
    bpd_feed_dict = {x:bpddata}
    bpd_classification_result = sess.run(logits,bpd_feed_dict)
    sum = 0
    bpd_output=[]
    bpd_output = tf.argmax(bpd_classification_result,1).eval()
    cjw_output=[]
    cjw_feed_dict = {x:cjwdate}
    cjw_classification_result = sess.run(logits,cjw_feed_dict)
    cjw_output = tf.argmax(cjw_classification_result,1).eval()


    O = []
    cjwisbpdCount = 0
    cjwisNotbpdCount = 0
    colsCount = 0
    colsNotCount = 0
    sum = 0
    count = 0
    Cols_O = []
    #求所有欧式距离
    for i in range(len(cjw_output)):
        Cols = []
        for j in range(len(bpd_output)):
            if bpd_output[j] == 0:
                if cjw_output[i]==0:
                    loss = math.sqrt(math.pow(abs(bpd_classification_result[j][0]-cjw_classification_result[i][0]),2)+\
                                     math.pow(abs(bpd_classification_result[j][1]-cjw_classification_result[i][1]),2))
                    Cols.append(loss)
                    cjwisbpdCount =cjwisbpdCount+1
                    colsCount = colsCount+1
            elif bpd_output[j] == 1:
                if cjw_output[i]==1:
                    loss = math.sqrt(math.pow(abs(cjw_classification_result[i][0]-bpd_classification_result[j][0]),2)+\
                                     math.pow(abs(cjw_classification_result[i][1]-bpd_classification_result[j][1]),2))
                    Cols.append(loss)
                    cjwisNotbpdCount = cjwisNotbpdCount+1
                    colsNotCount = colsNotCount+1
        #列最小值,即一张2cjw对所有的1pbd的欧式距离的最小值
        min = Cols[0]
        while 1:
            if count >= colsCount:
                count = 0
                break
            if(Cols[count]<=min):
                min = Cols[count]
            count = count+1
        Cols_O.append(min)
        colsNotCount = 0
        colsCount = 0
        O.append(Cols)
    print("cjwisbpdCount      "+str(cjwisbpdCount))
    print("cjwisnotbpdCount   "+str(cjwisNotbpdCount))
    print("count" +str(count))
    print("len(Cols_O)" +str(len(Cols_O)))
    #print(Cols_O)
    #求列最小值的平均值
    while 1:
        if count >= cjwisbpdCount/len(Cols_O):
            count = 0
            break
        sum = sum +Cols_O[count]
        count = count+1
    ave = sum/(cjwisbpdCount/len(Cols_O))
    print("ave  "+str(ave))
    print(ave,file=doc)

    #求全部最大值
    #max = O[0]
    #while 1:
    #    if count >= cjwisbpdCount:
    #        count = 0
    #        break
    #    if(O[count]>=max):
    #        max = O[count]
    #    count = count+1
    #print("max  "+str(max))
    #print(max,file=doc)
    #求全部最小值
    #min = O[0]
    #while 1:
    #    if count >= cjwisbpdCount:
    #        count = 0
    #        break
    #    if(O[count]<=min):
    #        min = O[count]
    #    count = count+1
    #print("min  "+str(min))
    #print(min,file=doc)
doc.close()
