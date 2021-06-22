# coding=utf-8
from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import glob


class TestPhoto:
    def __init__(self,testpath,savePoint,restoreCheckP,readImgCount):
        self.path = testpath
        self.savePoint = savePoint
        self.readImgCount = readImgCount
        self.restoreCheckP = restoreCheckP

        #self.testPhoto()


    def read_img(self):
        w=100
        h=100
        c=3
        imgs=[]
        readImg = 0
        for im in glob.glob(self.path+'/*.jpg'):
            if readImg >= self.readImgCount:
                return np.asarray(imgs,np.float32)
            readImg=readImg+1
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
        return np.asarray(imgs,np.float32)

    def testPhoto(self):
        self.data = self.read_img()
        doc=open('all_2bpg_pre.txt','w')
        flower_dict = {0:'1bpd',1:'1WRP'}
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(self.savePoint)
            saver.restore(sess,tf.train.latest_checkpoint(self.restoreCheckP))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            logits = graph.get_tensor_by_name("logits_eval:0")
            feed_dict = {x:self.data}
            classification_result = sess.run(logits,feed_dict)
            output = []
            output = tf.argmax(classification_result,1).eval()
            count = 0
            acc = 0
            for i in range(len(output)):
                if flower_dict[output[i]] == '1bpd':
                    count+=1
                print('the',i,'flower:'+flower_dict[output[i]],file=doc)
            acc=count/len(output)
            print(count,acc,file=doc)
        doc.close()
        return

    def setreadImgCount(readImgCount):
        self.readImgCount = readImgCount
        return

    def settestpath(testpath):
        self.path = testpath
        return

if __name__ == "__main__":
    testedImg = TestPhoto("D:\\tensorflow\\cutPic\\original_pictrues\\1bpdresultpicture\\",\
                        'D:\\tensorflow\\cutPic\\original_result\\tfdroid.ckpt.meta',\
                            'D:\\tensorflow\\cutPic\\original_result\\',\
                            2000)
