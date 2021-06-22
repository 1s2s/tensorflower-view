# -*- coding: utf-8 -*-
from PIL import Image
from skimage import io,transform
import glob
import os
import math
import numpy as np
import main_window as mw

#读取图片
class cutPhotos:
    def __init__(self,path,savePath,rowSum,columnSum):
        self.path = path
        self.savePath = savePath +'\\'
        self.rowSum = rowSum
        self.columnSum = columnSum
        #self.read_cut_img()

    #读图片和切图片
    def read_cut_img(self):
        imgs=[]
        pic_sum = 0
        #readOnePicAndCut
        for im in glob.glob(self.path+'/*.jpg'):
            print('reading the images:%s'%(im))
            pix=io.imread(im)
            [row,column,dep]=pix.shape
            row_1=0
            row_1_flag = 0
            row_2=0
            row_2_flag = 0
            column_1=0
            column_1_flag =0
            column_2=0
            column_2_flag = 0
            row=row-1
            column=column-1
            little_pic_sum = 10
            #TOP
            for x in range(1,row-1,1):
                for y in range(1,column-1,1):
                    r, g, b = pix[x, y]
                    if(r!=255 and g!=255 and b!=255):
                        row_1 = x
                        row_1_flag = 1
                        break
                if(row_1_flag):
                    break
            #RIGHT
            for y in range(1,column-1,1):
                for x in range(1,row-1,1):
                    r, g, b = pix[x, y]
                    if(r!=255 and g!=255 and b!=255):
                        column_1 = y
                        column_1_flag = 1
                        break
                if(column_1_flag):
                    break
            #BUTTOM
            for x in range(row-1,1,-1):
                for y in range(column-1,column_1,-1):
                    r, g, b = pix[x, y]
                    if(r!=255 and g!=255 and b!=255):
                        row_2 = x
                        row_2_flag = 1
                        break
                if(row_2_flag):
                    break
            #LEFT
            for y in range(column-1,column_1,-1):
                for x in range(row-1,1,-1):
                    r, g, b = pix[x, y]
                    if(r!=255 and g!=255 and b!=255):
                        column_2 = y
                        column_2_flag = 1
                        break;
                if(column_2_flag):
                    break

            img=Image.fromarray(pix)
            #裁剪并保存图片
            box = (column_1, row_1, column_2, row_2)
            #print(box)
            cropped = img.crop(box)  # (left, upper, right, lower)
            #cropped.save(self.savePath+"_"+str(pic_sum)+"_"+'.jpg')
            L = cropped.size
            max_row = self.rowSum#floor(L(1)/height);
            max_col = self.columnSum#floor(L(2)/width);
            #ceil()向上取整,floor()向下取整
            #print("cropped.size"+str(cropped.size))
            #print("cropped.size0 "+str(cropped.size[0]))
            #print("cropped.size1 "+str(cropped.size[1]))
            height = math.floor(cropped.size[1]/max_row)
            width  =  math.floor(cropped.size[0]/max_col)
            #print(height)
            #print(width)
            #分块
            for row in range(max_row):
                for col in range(max_col):
                    box1 = (width*row, height*col, width*(row+1), height*(col+1))
                    #print(box1)
                    cropped1 = cropped.crop(box1)
                    cropped1.save(self.savePath+"_"+str(pic_sum)+"_"+str(row)+"_"+str(col)+'.jpg')
            pic_sum = pic_sum+1
        return

    #过滤图片
    def drop_white_pic(img):
        width = img.size[0]#长度
        height = img.size[1]#宽度
        for i in range(0,width):#遍历所有长度的点
            for j in range(0,height):#遍历所有宽度的点
                data = (img.getpixel((i,j)))#打印该图片的所有点
                if (data[0]!=255 or data[0]!=255 or data[1] != 255):
                    return 1
        return 0

#if __name__ == "__main__":
#    cutImg = cutPhotos('D:\\tensorflow\\cutPic\\original_pictrues\\4kliresultpicture\\',\
#                        'D:\\tensorflow\\python可视化\\cut\\',3,3)
