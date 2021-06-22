# -*- coding: utf-8 -*-
# data :2019-6-4
import wx
import cutPhotos as cp
import _thread as th
import threading
import time
import inspect
import ctypes
import CNN as Cnn
import testPhotos as tP
import os

os.system("mode con cols=80 lines=20")

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class MainWindow(wx.Frame):
    """We simply derive a new class of Frame."""
    def __init__(self, parent, title):
        wx.Frame.__init__(self, parent, title = title, size = (800, 600),style=wx.MINIMIZE_BOX|
        wx.SYSTEM_MENU|wx.CAPTION|wx.CLOSE_BOX)
        self.panel = None
        self.trainPanel = None

        #CutPhtoto模块
        self.tc1 = None             #Path Choose:
        self.tc2 = None             #Path Save
        self.Cut4 =None              #Cut1
        self.Cut1 = None            #Cut4
        self.CuttingButton = None   #切割
        self.SaveButton    = None   #保存

        #训练图片模快
        self.trainPath1 = None
        self.tc11 = None
        self.tc21 = None
        self.tc31 = None
        self.Train = None
        self.EndTrain = None
        self.ReadImg = None

        #测试图片
        self.trainTest1 = None
        self.Test =None
        self.EndTest =None

        #路径
        self.train_path =None

        #线程
        self.cut_save_thread = None
        self.train_save_thread = None
        self.test_img_thread = None

        #初始化
        self.cut1 = "False"
        self.cut4 = "False"

        self.SetBackgroundColour('gray')
        self.CreateStatusBar()
        self.InitMenuUI()
        self.cutPanel()
        self.Centre()
        self.Show(True)
        #self.DisableAllCom()

    def InitMenuUI(self):
      menubar = wx.MenuBar()
      fileMenu = wx.Menu()
      quit = wx.MenuItem(fileMenu, wx.ID_EXIT, '&Quit\tCtrl+Q')
      fileMenu.AppendItem(quit)

      menubar.Append(fileMenu, '&Menu')
      self.SetMenuBar(menubar)

    def cutPanel(self):
        self.panel = wx.Panel(self,wx.ID_ANY,pos=(0,0))
        vbox = wx.BoxSizer(wx.HORIZONTAL)
        view =  wx.GridBagSizer(0,0)


        #裁剪图片
        nm = wx.StaticBox(self.panel, wx.ID_ANY, 'Cut Pics:',\
                        size = (350, 200),style=0)
        nmSizer = wx.StaticBoxSizer(nm, wx.VERTICAL)
        fgs = wx.FlexGridSizer(4, 1, 10,10)
        title = wx.StaticText(self.panel, label = "Path Choose:")
        author = wx.StaticText(self.panel, label = "Path Save:")
        self.tc1 = wx.DirPickerCtrl(self.panel, wx.ALL|wx.LEFT, path='',name ="Path Choose:")
        self.tc2 = wx.DirPickerCtrl(self.panel,  wx.ALL|wx.LEFT, path='',name =":")
        fgs.AddMany([(title), (self.tc1), (author), (self.tc2)])
        nmSizer.Add(fgs, 0,wx.ALIGN_CENTER_HORIZONTAL, 10)
        fgs_cut = wx.BoxSizer(wx.HORIZONTAL)
        self.Cut4 = wx.Button(self.panel,  -1, 'Cut-4')
        fgs_cut.Add(self.Cut4, 0,wx.ALL|wx.RIGHT, 10)
        self.Cut1 = wx.Button(self.panel, -1, 'Cut-1')

        fgs_cut.Add(self.Cut1, 0,wx.ALL|wx.LEFT, 10)
        nmSizer.Add(fgs_cut, 0, wx.ALIGN_CENTER_HORIZONTAL, 10)
        fgs_cut1 = wx.BoxSizer(wx.HORIZONTAL)
        self.CuttingButton = wx.Button(self.panel, -1, 'Cutting')
        fgs_cut1.Add(self.CuttingButton, 0,wx.ALL|wx.RIGHT, 10)
        self.SaveButton = wx.Button(self.panel, -1, 'Save')
        fgs_cut1.Add(self.SaveButton, 0,wx.ALL|wx.LEFT, 10)
        nmSizer.Add(fgs_cut1, 0, wx.ALIGN_CENTER_HORIZONTAL, 10)


        #训练图片
        nm = wx.StaticBox(self.panel, wx.ID_ANY, 'Train Pics:',\
                        size = (350, 200),style=0)
        nmSizer1 = wx.StaticBoxSizer(nm, wx.VERTICAL)
        fgs_train = wx.BoxSizer(wx.VERTICAL)
        trainPath = wx.StaticText(self.panel,  wx.CENTER|wx.LEFT,  label = "Path Train:")
        self.trainPath1 = wx.DirPickerCtrl(self.panel, wx.CENTER, path='',name ="Path Train:")
        fgs_train.Add(trainPath, 0, wx.CENTER, 10)
        fgs_train.Add(self.trainPath1, 0,wx.CENTER, 10)
        nmSizer1.Add(fgs_train, 0, wx.ALIGN_CENTER_VERTICAL|wx.CENTER, 10)
        fgs1 = wx.FlexGridSizer(3, 2, 10,10)
        title1 = wx.StaticText(self.panel, label = "Photo Width")
        author1 = wx.StaticText(self.panel, label = "Photo Height")
        review1 = wx.StaticText(self.panel, label = "Photo Chan")
        self.tc11 = wx.TextCtrl(self.panel)
        self.tc21 = wx.TextCtrl(self.panel)
        self.tc31 = wx.TextCtrl(self.panel)
        fgs1.AddMany([(title1), (self.tc11, 1, wx.CENTER), (author1),
            (self.tc21, 1, wx.CENTER), (review1, 1, wx.CENTER), (self.tc31, 1, wx.CENTER)])
        nmSizer1.Add(fgs1, 0, wx.ALL|wx.CENTER, 10)
        fgs_train1 = wx.BoxSizer(wx.HORIZONTAL)
        self.ReadImg = wx.Button(self.panel,  -1, 'ReadImg')
        fgs_train1.Add(self.ReadImg, 0,wx.ALL|wx.RIGHT, 10)
        self.Train = wx.Button(self.panel,  -1, 'Train')
        fgs_train1.Add(self.Train, 0,wx.ALL|wx.RIGHT, 10)
        self.EndTrain  = wx.Button(self.panel,  -1, 'End')
        fgs_train1.Add(self.EndTrain, 0,wx.ALL|wx.RIGHT, 10)
        nmSizer1.Add(fgs_train1, 0, wx.ALIGN_CENTER_VERTICAL|wx.CENTER, 10)


        #测试图片
        nm = wx.StaticBox(self.panel, wx.ID_ANY, 'Test Pics:',\
                        size = (350, 200),style=0)
        nmSizer11 = wx.StaticBoxSizer(nm, wx.VERTICAL)
        fgs_test = wx.BoxSizer(wx.VERTICAL)
        trainTest = wx.StaticText(self.panel,  wx.CENTER|wx.LEFT,  label = "Path Test:")
        self.trainTest1 = wx.DirPickerCtrl(self.panel, wx.CENTER, path='',name ="Path Test:")
        fgs_test.Add(trainTest, 0, wx.CENTER, 10)
        fgs_test.Add(self.trainTest1, 0,wx.CENTER, 10)
        nmSizer11.Add(fgs_test, 0, wx.ALIGN_CENTER_VERTICAL|wx.CENTER, 10)
        fgs_test = wx.BoxSizer(wx.HORIZONTAL)
        self.Test = wx.Button(self.panel,  -1, 'Test')
        fgs_test.Add(self.Test, 0,wx.ALL|wx.RIGHT, 10)
        self.EndTest  = wx.Button(self.panel,  -1, 'End Test')
        fgs_test.Add(self.EndTest, 0,wx.ALL|wx.RIGHT, 10)
        nmSizer11.Add(fgs_test, 0, wx.ALIGN_CENTER_VERTICAL|wx.CENTER, 10)



        #控制台
        sbox2 = wx.StaticBox(self.panel, wx.ID_ANY, 'Console Show:',\
                            style = 0)
        sboxSizer2 = wx.StaticBoxSizer(sbox2, wx.HORIZONTAL)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.textReadOnly = wx.TextCtrl(self.panel,value = "只读文本\n",size = (380,205),style =wx.TE_MULTILINE|wx.TE_WORDWRAP|wx.TE_READONLY|wx.TE_RICH2)
        hbox3.Add(self.textReadOnly,1,wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5)
        sboxSizer2.Add(hbox3, 0, wx.ALL|wx.CENTER, 5)

        view.Add(nmSizer, pos   = (0,  0), flag = wx.ALL, border = 0)
        view.Add(nmSizer1, pos  = (1, 0), flag = wx.ALL, border = 0)
        view.Add(nmSizer11, pos = (1, 1), flag = wx.RIGHT, border = 0)
        view.Add(sboxSizer2, pos = (0, 1), flag = wx.RIGHT, border = 0)
        vbox.Add(view, proportion = 2, flag = wx.ALL|wx.EXPAND, border = 10)
        self.panel.SetSizerAndFit(vbox)

        #事件绑定
        self.Cut1.Bind(wx.EVT_BUTTON,self.OnCut1ButtonClick)
        self.Cut4.Bind(wx.EVT_BUTTON,self.OnCut4ButtonClick)
        self.Train.Bind(wx.EVT_BUTTON,self.OnTrainButtonClick)
        self.Test.Bind(wx.EVT_BUTTON,self.OnTestButtonClick)
        self.CuttingButton.Bind(wx.EVT_BUTTON,self.OnCuttingButtonClick)
        self.SaveButton.Bind(wx.EVT_BUTTON,self.OnSaveButtonClick)
        self.ReadImg.Bind(wx.EVT_BUTTON,self.OnReadImgButtonOnClick)
        self.EndTrain.Bind(wx.EVT_BUTTON,self.OnEndTrainButtonClick)
        self.EndTest.Bind(wx.EVT_BUTTON,self.OnEndTestButtonClick)
        return

    def Console(self, text):
        wx_text = text + '\n'
        self.textReadOnly.AppendText(wx_text)
        return

    def OnCut1ButtonClick(self,event):
        self.path = self.tc1.GetPath()
        self.save_path = self.tc2.GetPath()
        if self.path  or self.save_path :
            self.Console("不切图片，图片路径为")
            self.Console(self.path)
            self.Console("保存图片路径为")
            self.Console(self.save_path)
            self.cut1 = "True"
            self.cut4 = "False"
        return

    def OnCut4ButtonClick(self,event):
        self.path = self.tc1.GetPath()
        self.save_path = self.tc2.GetPath()
        if self.path  or self.save_path :
            self.Console("选择一张切四张，图片路径为")
            self.Console(self.path)
            self.Console("保存图片路径为")
            self.Console(self.save_path)
            self.cut4 = "True"
            self.cut1 = "False"
        return

    def OnCuttingButtonClick(self,event):
        self.Console("切割中.....")
        if self.cut4 == "True":
            cutImg = cp.cutPhotos(self.path,self.save_path ,2,2)
            try:
                self.cut_save_thread = threading.Thread(target=cutImg.read_cut_img, args=())
                self.cut_save_thread.start()
                self.Console("开启线程")
                self.Console("cut4切割中.....")
            except:
                self.Console("不能开启线程")
        elif self.cut1 == "True":
            cutImg = cp.cutPhotos(self.path,self.save_path,1,1)
            try:
                self.cut_save_thread = threading.Thread(target=cutImg.read_cut_img, args=())
                self.cut_save_thread.start()
                self.Console("开启线程")
                self.Console("cut1切割中.....")
            except:
                self.Console("不能开启线程")
        self.DisableAllCom()
        self.EnableOneCom("SaveButton")
        return

    def OnSaveButtonClick(self,event):
        self.Console("保存中.....")
        if self.cut_save_thread.is_alive():
            time.sleep(0.5)
            stop_thread(self.cut_save_thread)
        #stopThread.stop()
        self.EnableAllCom()
        return

    def OnEndTrainButtonClick(self,event):
        self.Console("结束训练.....")
        if self.train_save_thread.is_alive():
            time.sleep(0.5)
            stop_thread(self.train_save_thread)
        #stopThread.stop()
        self.EnableAllCom()

    def OnEndTestButtonClick(self,event):
        self.Console("结束测试.....")
        if self.test_img_thread.is_alive():
            time.sleep(0.5)
            stop_thread(self.test_img_thread)
        #stopThread.stop()
        self.EnableAllCom()

    def OnReadImgButtonOnClick(self,event):
        self.Console("读取图片开始")
        return

    def OnTrainButtonClick(self,event):
        self.train_path = self.trainPath1.GetPath()
        w = self.tc11.GetValue()
        h = self.tc21.GetValue()
        c = self.tc31.GetValue()
        if self.train_path and int(w)>=0 and int(h) >=0 and int(c)==3:
            self.Console("训练图片路径为")
            self.Console(self.train_path)
            self.Console("训练图片格式为w："+w+" "+"h:"+h+" "+"c:"+c)
            self.cnn = Cnn.CNN(self.train_path,'.\\cnn_result\\',w,h,c)
            self.train_save_thread = threading.Thread(target=self.cnn.Train, args=())
            self.train_save_thread.start()
            self.Console("训练开始")
            self.DisableAllCom()
            self.EnableOneCom("End")
        elif int(c)!=3:
            self.Console("图片色彩通道不对")
        return

    def OnTestButtonClick(self,event):
        test_path = self.trainTest1.GetPath()
        if test_path:
            self.Console("测试图片路径为")
            self.Console(test_path)
            self.test = tP.TestPhoto(test_path,\
                                '.\\cnn_result\\tfdroid.ckpt.meta',\
                                    '.\\cnn_result\\',\
                                    2000)
            self.test_img_thread = threading.Thread(target=self.test.testPhoto, args=())
            self.test_img_thread.start()
            self.Console("测试开始")
            self.DisableAllCom()
            self.EnableOneCom("EndTest")
        return

    def DisableAllCom(self):
        self.tc1.Enable(False)
        self.tc2.Enable(False)
        self.Cut4.Enable(False)
        self.Cut1.Enable(False)
        self.CuttingButton.Enable(False)
        self.SaveButton.Enable(False)
        #训练图片模快
        self.trainPath1.Enable(False)
        self.tc11.Enable(False)
        self.tc21.Enable(False)
        self.tc31.Enable(False)
        self.Train.Enable(False)
        self.EndTrain.Enable(False)
        self.ReadImg.Enable(False)
        #测试图片
        self.trainTest1.Enable(False)
        self.Test.Enable(False)
        self.EndTest.Enable(False)

    def EnableAllCom(self):
        self.tc1.Enable(True)
        self.tc2.Enable(True)
        self.Cut4.Enable(True)
        self.Cut1.Enable(True)
        self.CuttingButton.Enable(True)
        self.SaveButton.Enable(True)

        #训练图片模快
        self.trainPath1.Enable(True)
        self.tc11.Enable(True)
        self.tc21.Enable(True)
        self.tc31.Enable(True)
        self.Train.Enable(True)
        self.EndTrain.Enable(True)
        self.ReadImg.Enable(True)
        #测试图片
        self.trainTest1.Enable(True)
        self.Test.Enable(True)
        self.EndTest.Enable(True)

    def EnableOneCom(self,name):
        if name == "tcl1":
            self.tc1.Enable(True)
        elif name == "tc2":
            self.tc2.Enable(True)
        elif name == "Cut4":
            self.Cut4.Enable(True)
        elif name == "Cut1":
            self.Cut1.Enable(True)
        elif name == "SaveButton":
            self.SaveButton.Enable(True)
        elif name == "trainPath1":
            self.trainPath1.Enable(True)
        elif name == "tc11":
            self. tc11.Enable(True)
        elif name == "tc21":
            self.tc21.Enable(True)
        elif name == "tc31":
            self.tc31.Enable(True)
        elif name == "Trian":
            self.Train.Enable(True)
        elif name == "trainTest1":
            self.trainTest1.Enable(True)
        elif name == "Test":
            self.Test.Enable(True)
        elif name == "EndTest":
            self.EndTest.Enable(True)
        elif name == "End":
            self.EndTrain.Enable(True)
        return


if __name__ == "__main__":
    app = wx.App(False)
    frame = MainWindow(None, 'CNN System')
    app.MainLoop()
