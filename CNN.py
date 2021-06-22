# -*- coding: utf-8 -*-
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time


class CNN:
    def __init__(self,trainpath,savePoint,width,height,channels):
        self.path = trainpath +"\\"
        self.savePoint = savePoint
        self.Width = width
        self.Height= height
        self.Channels = channels
        print(self.Width)
        print(self.Height)
        self.isDrop     = 0
        self.classnum   = 2
        self.n_epoch    = 20
        self.batch_size = 64
        self.ratio      = 0.8

    def Train(self):
        self.crateData()
        #打乱顺序
        num_example=self.data.shape[0]
        arr=np.arange(num_example)
        np.random.shuffle(arr)
        self.data=self.data[arr]
        self.label=self.label[arr]
        #将所有数据分为训练集和验证集
        self.ratio = 0.8
        s=np.int(num_example*self.ratio)
        x_train=self.data[:s]
        y_train=self.label[:s]
        x_val=self.data[s:]
        y_val=self.label[s:]

        #网络部分
        x=tf.placeholder(tf.float32,shape=[None,self.Width,self.Height,self.Channels],name='x')
        y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
        logits = self.Build_CNN(x)
        b = tf.constant(value=1,dtype=tf.float32)
        logits_eval = tf.nn.softmax(tf.multiply(logits,b,name='logits_eval'),name='output')
        loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
        train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
        acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        loss_summary = tf.summary.scalar(loss.op.name,loss)
        accuracy_summary = tf.summary.scalar(acc.op.name, acc)
        saver = tf.train.Saver()

        sess=tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        merged = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter('C:\\tensorboard\\log\\train',sess.graph)
        test_summary_writer = tf.summary.FileWriter('C:\\tensorboard\\log\\test')
        for epoch in range(self.n_epoch):
            start_time = time.time()
            #training
            train_loss, train_acc, n_batch = 0, 0, 0
            for x_train_a, y_train_a in self.minibatches(x_train, y_train, self.batch_size, 1):
                _,summary_str,err,ac=sess.run([train_op,merged,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
                train_summary_writer.add_summary(summary_str,epoch)
                train_loss += err; train_acc += ac; n_batch += 1

            print("   train loss: %f" % (train_loss/n_batch),n_batch)
            print("   train acc: %f" % (train_acc/n_batch),epoch)
            #validation
            val_loss, val_acc, n_batch = 0, 0, 0
            for x_val_a, y_val_a in self.minibatches(x_val, y_val, self.batch_size, 0):
                _,summary_str,err,ac=sess.run([train_op,merged,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
                test_summary_writer.add_summary(summary_str,epoch)
                val_loss += err; val_acc += ac; n_batch += 1
            print("   validation loss: %f" % (val_loss/ n_batch),n_batch)
            print("   validation acc: %f" % (val_acc/ n_batch),epoch)
        tf.train.write_graph(sess.graph_def, '.', self.savePoint+'tfdroid.pbtxt')
        saver.save(sess, self.savePoint+'tfdroid.ckpt')
        train_summary_writer.close()
        test_summary_writer.close()
        sess.close()

    def readImg(self):
        cate=[self.path+x for x in os.listdir(self.path) if os.path.isdir(self.path+x)]
        imgs=[]
        labels=[]
        for idx,folder in enumerate(cate):
            for im in glob.glob(folder+'/*.jpg'):
                print('reading the images:%s'%(im))
                img=io.imread(im)
                img=transform.resize(img,(int(self.Width),int(self.Height)))
                imgs.append(img)
                labels.append(idx)
        return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

    def crateData(self):
        self.data,self.label = self.readImg()

    #卷积层
    def Conv_pool(self,input_X,Fliters,K_size,Padding,Sttdev,conv_name,P_size,Stride,pool_name):
        conv=tf.layers.conv2d(
              inputs=input_X,
              filters=Fliters,
              kernel_size=K_size,
              padding=Padding,
              activation=tf.nn.relu,
              kernel_initializer=tf.truncated_normal_initializer(stddev=Sttdev))
        conv_name= tf.multiply(conv,1,name=conv_name)
        pool=tf.layers.max_pooling2d(inputs=conv, pool_size=P_size, strides=Stride)
        pool_conv = tf.multiply(pool,1,name=pool_name)
        return pool
    #全连接层
    def Dense_Drop(self,Input_X,Units,Sttdev,dense_name,isDrop):
        dense = tf.layers.dense(inputs=Input_X,
                              units=Units,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=Sttdev),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        dense_layer = tf.multiply(dense,1,name=dense_name)
        if isDrop == 1:
            dropout = tf.layers.dropout(inputs=dense,rate=0.5)
            return dropout
        return dense
    #构建CNM网络
    def Build_CNN(self,x):
        pool1_conv1 = self.Conv_pool(x,32,[5,5],"same",0.01,'conv_11',[2, 2],2,'pool1_conv1')
        pool2_conv2 = self.Conv_pool(pool1_conv1,64,[5,5],"same",0.01,'conv_22',[2,2],2,'pool2_conv2')
        pool3_conv3 = self.Conv_pool(pool2_conv2,128,[3,3],"same",0.01,'conv_33',[2,2],2,'pool3_conv3')
        pool4_conv4 = self.Conv_pool(pool3_conv3,128,[3,3],"same",0.01,'conv_44',[2,2],2,'pool4_conv4')
        self.re1 = tf.reshape(pool4_conv4, [-1, 6 * 6 * 128])
        dense1_layer1_drop = self.Dense_Drop(self.re1,1024,0.01,'dense1_layer1',self.isDrop)
        dense2_layer2_drop = self.Dense_Drop(dense1_layer1_drop,512,0.01,'dense2_layer2',self.isDrop)
        dense3_layer3_drop = self.Dense_Drop(dense2_layer2_drop,self.classnum,0.01,'logits',0)
        return dense3_layer3_drop

    def minibatches(self,inputs,targets,batch_size,shuffle):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]


if __name__ == "__main__":
    cnn = CNN('C:\\Users\\bao\\Desktop\\毕设+实习\\python可视化\\chemistry_pic\\',\
                '.\\cnn_result\\',100,100,3)
    cnn.Train()
