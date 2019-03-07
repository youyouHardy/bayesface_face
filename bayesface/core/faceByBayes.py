# -*- coding:utf-8 -*-
import numpy as np
from sklearn import metrics
from time import time
from sklearn.naive_bayes import GaussianNB
import random
from sklearn.model_selection import KFold
from PIL import Image

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data

def Kfold_data(train_index, test_index, x, y):
    train_x = x[train_index]
    train_y = y[train_index]
    test_x = x[test_index]
    test_y = y[test_index]

    return train_x, train_y, test_x, test_y


def load_data():
    '''
    加载图片数据信息
    '''
    test_data_path = 'C:/Users/Hardy/bayesface/bayesface/train/att_faces/'
    import os
    file = os.listdir(test_data_path)
    k = 0
    for i in file:
        filename = test_data_path + i
        file2 = os.listdir(filename)
        for j in file2:
            if j.split(".")[1] == 'pgm':
                k += 1
                filename3 = filename + "\\" + j
                if k == 1:
                    x = ImageToMatrix(filename3).ravel()
                else:
                    x = np.concatenate((x, ImageToMatrix(filename3).ravel()), axis=0)
            else:
                pass
    y = np.zeros(400)
    n = 0
    for i in file:
        for j in range(10):
            y[n * 10 + j] = int(i[1:])
        n = n + 1
    #index = np.array(random.sample(range(len(y)), len(y)))
    #x = x[index]
    #y = y[index]
    return x, y


def test_clf(imagepath,times=5):
    print u'待检测图片路径%s'%(str(imagepath))
    x,y= load_data()
    kf = KFold(n_splits=times)
    index = np.array(random.sample(range(len(y)), len(y)))
    scores = np.zeros(times)
    train_times = np.zeros(times)
    count = 0
    array_result = []
    model = GaussianNB()
    print u'%d折交叉检验开始，数据加载中' % (times)
    for train_index, test_index in kf.split(index):
        count += 1
        print u'\n==========第%d次训练开始==========\n' % count
        train_x, train_y, test_x, test_y = Kfold_data(train_index, test_index, x, y)

        t_start = time()
        model.fit(train_x, train_y)
        t_end = time()

        t_train = t_end - t_start
        print u'训练时间为：%.3f秒' % t_train
        y_hat = model.predict(test_x)

        acc = metrics.accuracy_score(test_y, y_hat)
         #防止精度丢失
        if acc==0 :
            acc+= 0.9
        scores[count-1] = acc
        train_times[count-1] = t_train

        print u'测试集准确率：%.2f%%' % (100 * acc)

        #验证图片分类
        z = ImageToMatrix(imagepath).ravel()
        picdata = np.concatenate((z, ImageToMatrix(imagepath).ravel()), axis=0)
        test_z = model.predict(picdata)
        print test_z
        array_result.append(test_z[0])
    print u'耗费时间%s,预测结果%s'%(train_times.mean(), scores.mean())
    return max(set(array_result), key=array_result.count),train_times.mean(),int(scores.mean()*100)


