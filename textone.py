# -*- coding: utf-8 -*-

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
#from tensorflow.python.framework import graph_util
#from tensorflow.python.platform import gfile
path='C:\\Users\\bao\\Desktop\\毕设+实习\\python可视化\\chemistry_pic\\'

#将所有的图片resize成100*100
w=100
h=100
c=3


#读取图片
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)


#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#将所有数据分为训练集和验证集
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

#-----------------构建网络----------------------
#占位符
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='input')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')
b = tf.constant(value=1,dtype=tf.float32)
#第一个卷积层（100——>50)
conv1=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
conv_11= tf.multiply(conv1,b,name='conv_11')
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
pool1_conv1 = tf.multiply(pool1,b,name='pool1_conv1')
#第二个卷积层(50->25)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
conv_22= tf.multiply(conv2,b,name='conv_22')
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_conv2 = tf.multiply(pool2,b,name='pool2_conv2')
#第三个卷积层(25->12)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
conv_33= tf.multiply(conv3,b,name='conv_33')
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
pool3_conv3 = tf.multiply(pool3,b,name='pool3_conv3')
#第四个卷积层(12->6)
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
conv_44= tf.multiply(conv4,b,name='conv_44')
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
pool4_conv4 = tf.multiply(pool4,b,name='pool4_conv4')
re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

#全连接层
dense1 = tf.layers.dense(inputs=re1,
                      units=1024,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense1_layer1 = tf.multiply(dense1,b,name='dense1_layer1')
#dropout = tf.layers.dropout(inputs=dense1,rate=0.5)
dense2= tf.layers.dense(inputs=dense1,
                      units=512,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
dense2_layer2 = tf.multiply(dense2,b,name='dense2_layer2')
#dropout1 = tf.layers.dropout(inputs=dense2,
#                      rate=0.5)
logits= tf.layers.dense(inputs=dense2,
                        units=2,
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
#---------------------------网络结束---------------------------
b = tf.constant(value=1,dtype=tf.float32)
#logits_eval = tf.nn.softmax(tf.multiply(logits,b,name='logits_eval'),name='output')
logits_eval = tf.nn.softmax(logits,name='output')
loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
loss_summary = tf.summary.scalar(loss.op.name,loss)
accuracy_summary = tf.summary.scalar(acc.op.name, acc)
saver = tf.train.Saver()
#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
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

#训练和测试数据，可将n_epoch设置更大一些

n_epoch=20
batch_size=64
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#设置0p和保存路径
summary_op = tf.summary.merge([loss_summary, accuracy_summary])
merged = tf.summary.merge_all()
train_summary_writer = tf.summary.FileWriter('C:\\tensorboard\\log\\train',sess.graph)
test_summary_writer = tf.summary.FileWriter('C:\\tensorboard\\log\\test')
for epoch in range(n_epoch):
    start_time = time.time()

    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,summary_str,err,ac=sess.run([train_op,merged,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_summary_writer.add_summary(summary_str,epoch)
        train_loss += err; train_acc += ac; n_batch += 1

    print("   train loss: %f" % (train_loss/n_batch),n_batch)
    print("   train acc: %f" % (train_acc/n_batch),epoch)

    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        _,summary_str,err,ac=sess.run([train_op,merged,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        test_summary_writer.add_summary(summary_str,epoch)
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (val_loss/ n_batch),n_batch)
    print("   validation acc: %f" % (val_acc/ n_batch),epoch)
#output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=['output'])
#with tf.gfile.FastGFile('C:\\Users\\harderman\\data\\graph.pb', mode='wb') as f:
    #f.write(output_graph_def.SerializeToString())
tf.train.write_graph(sess.graph_def, '.', 'tfdroid.pbtxt')
saver.save(sess, './tfdroid.ckpt')
train_summary_writer.close()
test_summary_writer.close()
sess.close()
