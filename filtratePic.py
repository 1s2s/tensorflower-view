# coding=utf-8

from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import glob
from PIL import Image

path = "D:\\tensorflow\\cutPic\\Cut-8pic\\2bpg\\"
save_path ="D:\\tensorflow\\cutPic\\8_drop_white\\test_image\\2bpg\\ "
#path2 = "C:\\Users\\harderman\\Desktop\\2.jpg"
flower_dict = {0:'1bpd',1:'1WRP'}
#sum_dict= {0:'first',1:'second',2:'third',3:'fourth',4:'fifth'}
w=100
h=100
c=3



def read_img(path):
    imgs=[]
    i = 0
    for im in glob.glob(path+'/*.bmp'):
        print('reading the images:%s'%(im))
        img = Image.open(im)
        print (img.size)#打印图片大小
        if drop_white_pic(img):
            img = img.convert("RGB")#把图片强制转成RGB
            img.save(save_path+str(i)+".jpg")#保存修改像素点后的图片
            i=i+1
        #img=transform.resize(img,(w,h))
        #imgs.append(img)
    return




def drop_white_pic(img):
    width = img.size[0]#长度
    height = img.size[1]#宽度
    for i in range(0,width):#遍历所有长度的点
        for j in range(0,height):#遍历所有宽度的点
            data = (img.getpixel((i,j)))#打印该图片的所有点
            print (data)#打印每个像素点的颜色RGBA的值(r,g,b,alpha)
            print (data[0])#打印RGBA的r值
            if (data[0]!=255 or data[0]!=255 or data[1] != 255):
                return 1
    return 0

def main():
    read_img(path)

if __name__ == '__main__':
    main()
        #err2=sess.run([loss], feed_dict)

        #print(err2)
        #if err2<=0.0000001:
            #for i in range(len(output)):
                #print('the','flower:'+flower_dict[output[i]])
        #elif i==1:
        #    print("to learn!")
