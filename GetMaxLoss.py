# coding=utf-8

from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import glob
import math

doc=open('InputText1.txt','w')
path='.\\1'

flower_dict = {0:'两创',1:'白宫'}
#sum_dict= {0:'first',1:'second'}
w=100
h=100
c=3

#标准矩阵
liangchuang = [13.465392, -13.404057]
baigong = [-14.382368,15.3616295]

def read_img(path):
    imgs=[]
    for im in glob.glob(path+'/*.jpg'):
        print('reading the images:%s'%(im))
        img=io.imread(im)
        img=transform.resize(img,(w,h))
        imgs.append(img)
    return np.asarray(imgs,np.float32)


with tf.Session() as sess:

    data = read_img(path)

    saver = tf.train.import_meta_graph('.\\tfdroid.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('.\\'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")
    Officecount = 0
    Complexcount = 0
    classification_result = sess.run(logits,feed_dict)
    #sum = 0
    output=[]
    output = tf.argmax(classification_result,1).eval()
    #print(output)
    for i in range(len(output)):
        if output[i] == 0:
            #两创矩阵
            print(classification_result[i])
            loss = math.sqrt(math.pow(abs(classification_result[i][0]-liangchuang[0]),2)+math.pow(abs(classification_result[i][1]-liangchuang[1]),2))
            print(loss,file=doc)
        elif output[i]==1:
            #白宫矩阵
            print(classification_result[i])
            loss = math.sqrt(math.pow(abs(classification_result[i][0]-baigong[0]),2)+math.pow(abs(classification_result[i][1]-baigong[1]),2))
            print(loss,file=doc)
doc.close()
