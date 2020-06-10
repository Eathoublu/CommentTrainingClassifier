# coding:utf8
import pandas as pd
import numpy as np
from gensim.models import word2vec
import re
import pickle as pkl
from tqdm import tqdm
from sklearn.externals import joblib
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.layers import Embedding
import time
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, LSTM
import os
import sys

reload(sys)
sys.setdefaultencoding('utf8')


rule = re.compile(ur'[^a-zA-Z\u4e00-\u9fa5]')
rule2 = re.compile('~+')
# r1 = rule.sub('~', unicode('「最懂苹果的分析师」Ming-Chi Kuo（郭明錤）9 月 4 日早上向投资者发布的一份新消息称，苹果公司 2019 年款 iPhone 应该不支持采用屏下指纹，而是继续使用 Face ID。 郭明錤认为 Android 制造商距离 iPhone 的先进 Face ID 技术还有几年的时间。这可能是 Android 厂商专注于屏下指纹技术的一个原因。明年 iPhone 相机或大升级：三个后置摄像头据《经济日报》表示，明年（2019）发布的 iPhone 有望升级为三后置摄像头的配置，除了成像表现更强之外，明年的新款 iPhone 还或将搭载 TOF 3D 识别技术。该技术与目前主流的 3D 结构光技术比较而言，具备工作距离更远（可以达到 5 米甚至更远）的优点，但缺点是近距离的精度相对较差。而且明年的 LCD 版 iPhone 也很可能从单摄像头升级到双摄像头，这些设计会为新 iPhone 带来更创新的增强现实功能。苹果结束一项针对 Apple Watch 用户心脏健康的研究根据外媒 MacRumors 的报道，一项针对 Apple Watch 用户心脏健康的大规模研究即将结束，苹果公司已经通知一些注册用户，他们的贡献已经完成。这项名为「苹果心脏研究」(Apple Heart Study)的研究是苹果与斯坦福大学合作进行的，它于 2017 年 11 月推出。心律不整的佩戴者可以在他们的 Apple Watch 和 iPhone 上收到通知，并且可以免费咨询研究医生，还可以通过心电图贴片进行额外监测。这项研究于上月初对新参与者关闭。关注 AppSo，回复「8888」获取最新苹果壁纸Apple Watch 将引入 always-on 息屏显示模式一项新的专利申请显示，苹果正在为 Apple Watch 开发一种 always-on 息屏显示模式。这项功能可以让用户不必抬起手腕就能看到时间，可能还有其他信息。有传闻称苹果将会在 9 月 13 日的发布会上推出 Apple Watch Series 4，新机型将配备更大尺寸的显示屏幕，至于有没有 always-on 息屏显示功能，可能要等到发布会到来之后才能确定。滴滴出行 9 月 8 日至 '))
# print(rule2.sub('', r1))


# quit()
def wash(text):
    r1 = rule.sub(u'~', unicode(text))
    r2 = rule2.sub(' ', unicode(r1))
    return r2

def txt2list(text, gram=2):
    returnlist = []
    for idx in range(len(text)-1):
        returnlist.append(text[idx:idx+gram])
    return returnlist

def loaddata(filename):
    df = pd.read_csv(filename)
    df.columns = ['no1','no2', 'content', 'score']
    return df


def makeslist(df):
    slist = []
    for idx in tqdm(range(len(df['content'])-1)):
        slist.append(txt2list(wash(df['content'][idx])))
    return slist

def getscore(df):
    scorelist = []
    for idx in tqdm(range(len(df['score']) - 1)):
        scorelist.append(toonehot(float(df['score'][idx])))
    return scorelist

def trainw2v(corpus):
    model = word2vec.Word2Vec(sentences=corpus, size=100)
    model.save('words.w2v')
    return model

def loadw2v(filename='word.w2v'):
    return word2vec.Word2Vec.load(filename)

def sen2mat(sen, v2w, max=30):
    mat = []
    for s in sen:
        tmp = []
        k = 0
        for word in s:
            k += 1
            if word in v2w:
                tmp.append(v2w[word])
            else:
                tmp.append(np.zeros(100))
            if k >= max:
                break
        if k < max:
            for _ in range(max-k):
                tmp.append(np.zeros(100))
        # tmp = np.array(tmp)
        # print(tmp.shape)
        mat.append(tmp)
    return mat

def makemodel(shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(shape)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    # model.add(LSTM(256))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128))
    # model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

def toonehot(tar):
    # print('TAR', tar)
    if tar == 1.:
        return [0, 0, 1]
    if tar == 0.:
        return [0, 1, 0]
    if tar == -1.:
        return [1, 0, 0]
    return [0, 1, 0]

MAX = 582
SPLIT = 0.9

def makevalidationlist():
    df = loaddata(filename='1.csv')[:MAX]
    slist = makeslist(df)
    glist = np.array(getscore(df))
    return slist, glist


rule = re.compile(ur'[^a-zA-Z\u4e00-\u9fa5]')
rule2 = re.compile('~+')
def wash(text):
    r1 = rule.sub(u'~', unicode(text))
    r2 = rule2.sub(' ', unicode(r1))
    return r2


def txt2list(text, gram=2):
    returnlist = []
    for idx in range(len(text) - 1):
        returnlist.append(text[idx:idx + gram])
    return returnlist


def loaddata(filename, content_idx, score_idx, count=False):
    df = pd.read_csv(filename)
    c = [str(i) for i in range(df.shape[1])]
    """	标题	来源	时间	内容	态度"""
    # df.columns = ['no', 'title', 'origin', 'time', 'content', 'score']
    c[content_idx] = 'content'
    c[score_idx] = 'score'
    df.columns = c
    if count:
        j = 0
        for i in tqdm(range(len(c[score_idx]))):
            if c[score_idx][i]:
                j = i+1
            else:
                break
        return df, j
    return df


def makeslist(df):
    slist = []
    for idx in tqdm(range(len(df['content']) - 1)):
        slist.append(txt2list(wash(df['content'][idx])))
    return slist


def getscore(df):
    scorelist = []
    for idx in tqdm(range(len(df['score']) - 1)):
        scorelist.append(toonehot(df['score'][idx]))
    return scorelist


def trainw2v(corpus):
    model = word2vec.Word2Vec(sentences=corpus, size=D2)
    model.save('words3.w2v')
    return model


def loadw2v(filename='words3.w2v'):
    return word2vec.Word2Vec.load(filename)


def sen2mat(sen, v2w, max=30):
    mat = []
    for s in tqdm(sen):
        tmp = []
        k = 0
        for word in s:
            k += 1
            if word in v2w:
                tmp.append(v2w[word])
            else:
                tmp.append(np.zeros(D2))
            if k >= max:
                break
        if k < max:
            for _ in range(max - k):
                tmp.append(np.zeros(D2))
        # tmp = np.array(tmp)
        # print(tmp.shape)
        mat.append(tmp)
    return mat


def makemodel(shape):
    sgd = SGD(lr=0.0001, decay=1e-9, nesterov=True, momentum=0.9)

    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(shape)))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    # model.add(LSTM(256))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(128))
    # model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  # optimizer=sgd,
                  metrics=['accuracy'])

    return model

def tocategory(res):
    _res = []
    for i in res:
        _i = i.tolist()
        if _i.index(max(i)) == 0:
            _res.append(NEGATIVE)
        elif _i.index(max(i)) == 1:
            _res.append(CENTER)
        else:
            _res.append(POSITIVE)
    return _res


def toonehot(tar):
    # print('TAR', tar)
    if tar == POSITIVE:
        return [0, 0, 1]
    if tar == CENTER:
        return [0, 1, 0]
    if tar == NEGATIVE:
        return [1, 0, 0]
    return [0, 1, 0]
TITLE = '文本分类器'
VERSION = 'V1.0'

MAX = 10000
SPLIT = 0.95
D1 = 200
D2 = 100
WMR = 'words3.w2v'
KMR = 'clf.ks'
EPOCH = 2
BATCH_SIZE = 128
POSITIVE = '积极'
NEGATIVE = '消极'
CENTER = '中立'

if __name__ == '__main__':

    print('欢迎使用{}{}!'.format(TITLE, VERSION))

    while True:
        print("""
        1）训练模型
        2）修改关键参数
        3) 查看使用说明
        4) 使用模型预测数据
        """)
        status = raw_input('>>>')
        if '2' in status:
            print('当前参数如下：\nMAX:{}\nSPLIT:{}\nD1:{}\nD2:{}\nWMR:{}\nKMR:{}\nEPOCH:{}\nBATCH_SIZE:{}\n正面标签:{}\n中性标签:{}\n反面标签:{}\n').format(MAX, SPLIT, D1, D2, WMR, KMR, EPOCH, BATCH_SIZE, POSITIVE, CENTER, NEGATIVE)
            D1 = input('>>>请输入D1的值:')
            D2 = input('>>>请输入D2的值:')
            MAX = input('>>>请输入最大可用数据条数：')
            SPLIT = input('>>>请输入测试/训练划分比例：')
            WMR = raw_input('>>>请输入词向量模型默认路径：')
            KMR = raw_input('>>>请输入神经网络模型默认路径：')
            EPOCH = input('>>>请输入迭代次数：')
            BATCH_SIZE = input('>>>请输入批大小：')
            POSITIVE = raw_input('>>>正面标签：')
            CENTER = raw_input('>>>中性标签：')
            NEGATIVE = raw_input('>>>负面标签：')
            if POSITIVE.isdigit():
                POSITIVE = int(POSITIVE)
            if CENTER.isdigit():
                CENTER = int(CENTER)
            if NEGATIVE.isdigit():
                NEGATIVE = int(NEGATIVE)
            print('更新完毕')
            continue
        elif '3' in status:
            print('-'*150+"""
            -*- 使用说明 -*-
    欢迎使用{}{}。这是一个用于快速训练文本分类模型的脚本。
    使用方法：在主界面输入1后，按照提示填写信息。训练分为word2vec模型以及神经网络模型两部分，其中，word2vec模型可以使用现有的，也可以重新训练。
    程序默认识别的打分标签是积极、中立、消极，可以在修改关键参数中修改成数据集中提供的标签。
""".format(TITLE, VERSION)+'-'*150)
            continue
        elif '1' in status:
            filename = raw_input(">>>Please enter CSV file's name")
            while not os.path.exists(filename):
                print('刚刚输入的文件路径有误，请重新输入！')
                filename = raw_input(">>>Please enter CSV file's name")
            content_idx = input('>>>在您的数据中，文本正文是在第几列？(从0开始计算)')
            score_idx = input('>>>在您的数据中，分数是在第几列？(从0开始计算)')
            print('正在加载数据...')
            df = loaddata(filename=filename, content_idx=content_idx, score_idx=score_idx)
            slist = makeslist(df)
            glist = np.array(getscore(df))
            print('数据加载完完毕')
            MAX = raw_input('>>>请输入用于训练的数据条数（全部请直接回车）:')
            if not MAX:
                MAX = len(slist)
            has_w_m = raw_input('请问是否有现成的word2vec模型？(Y/n)')
            if 'y' in has_w_m.lower():
                mname = raw_input('>>>请输入模型路径(回车表示默认路径):')
                if not mname:
                    mname = WMR
                elif not os.path.exists(mname):
                    print('刚刚输入的文件路径有误，请重新输入！')
                    mname = raw_input('>>>请输入模型路径:')
                print('正在加载模型，请稍后...')
                wmodel = loadw2v(mname)
                print('模型加载完成')
            else:
                mname = raw_input('>>>请输入将要生成的模型的名称(回车表示默认路径)：')
                if not mname:
                    mname = WMR
                print('正在训练模型，请稍后...')
                wmodel = trainw2v(slist)
                print('模型训练完成')
            mn = raw_input('>>>请输入神经网络模型的保存名（回车可使用默认）：')
            if not mn:
                mn = KMR
            # slist = makeslist(df)
            # glist = np.array(getscore(df))
            print('正在准备数据...')
            smat = np.array(sen2mat(slist, wmodel, max=D1))
            print('数据准备完成')
            print('正在初始化参数...')
            splitlen = int(MAX * SPLIT)
            print('初始化参数完成')
            print('正在创建神经网络模型...')
            nmodel = makemodel((D1, D2))
            print('神经网络模型创建完成')
            print('即将开始训练...')
            nmodel.fit(smat[:splitlen], glist[:splitlen], epochs=EPOCH, batch_size=BATCH_SIZE,
                       validation_data=(smat[splitlen: MAX], glist[splitlen: MAX]))  # 16 0.75
            print('训练完毕，正在保存模型...')
            nmodel.save(KMR)
            print('保存完毕，进入测试阶段...')
            #
            _, score = nmodel.evaluate(smat[splitlen: MAX], glist[splitlen: MAX], verbose=1)
            #
            # nmodel.save('model2/e20b128_300_{}_model.ks'.format(score))

            print('测试正确率为：{}'.format(score))
            print('任务完成，感谢您的使用。欢迎再次使用！')
        elif '4' in status:
            print('使用已训练的模型预测数据...')
            path_ks = raw_input('>>>请输入神经网络模型的路径：(默认请回车)')
            if path_ks != '':
                while not os.path.exists(path_ks):
                    print('刚刚输入的文件路径有误，请重新输入！')
                    path_ks = raw_input(">>>请输入神经网络模型的路径：(默认请回车)")
            else:
                path_ks = KMR
            path_w2v = raw_input('>>>请输入词向量模型的路径：（默认请回车）')
            if path_w2v != '':
                while not os.path.exists(path_w2v):
                    print('刚刚输入的文件路径有误，请重新输入！')
                    path_w2v = raw_input(">>>请输入词向量模型的路径：（默认请回车）")
            else:
                path_w2v = WMR
            path_csv = raw_input('>>>请输入数据集的路径：')
            while not os.path.exists(path_w2v):
                print('刚刚输入的文件路径有误，请重新输入！')
                path_csv = raw_input(">>>请输入数据集的路径：")
            content_idx = input('>>>请问评论是在第几列？')
            score_idx = input('>>>请问分数是在第几列？')

            output = raw_input('>>>请输入输出文件保存名：')

            w2v_model = loadw2v(path_w2v)
            print('Word2Vec模型加载完成！')
            ks_model = load_model(path_ks)
            print('神经网络模型加载完成！')
            data, last_score = loaddata(path_csv, content_idx, score_idx, count=True)
            print('数据加载完成！')
            # sen2mat(data['content'].tolist()[last_score:-1], w2v_model, 1000000)
            slist = makeslist(data)
            # last_score = 500
            print('正在处理数据...')
            smat = np.array(sen2mat(slist, w2v_model, max=D1)[last_score:-1])
            # print('smat shape', smat.shape)
            # print(smat[1])
            print('开始预测数据...')
            res = ks_model.predict(smat)
            print('预测完毕！')
            _res = tocategory(res)
            _res = data['score'].tolist()[0:last_score] + _res
            print('正在输出文件...')
            pd.DataFrame(_res).to_csv(output)
            print('输出完毕！')

# 2.csv 4 5
