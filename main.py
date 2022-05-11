import pandas
import pandas as pd
from snownlp import SnowNLP
import jieba
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
import os
from multiprocessing import Pool, TimeoutError
import re
import pyLDAvis.gensim_models
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

stop_list = './stoplist.txt'


def tp(x):
    print(type(x))


def l(x):
    print(len(x))


def to_percent(y, position):
    return str(10 * y) + "%"


def ts(timee=1000):
    time.sleep(timee)


def mycut(s): return ' '.join(jieba.cut(s))  # 分词函数


def snow_nlp(data_path):
    startt = time.time()
    print(f'start analyse{data_path}')
    pic_path = "./核子频率直方图/" + data_path[0:-5] + ".png"
    # print(data_path[0:-5], pic_path)
    # return
    # time.sleep(100)
    df = pd.read_excel(data_path)
    df = df.dropna()
    print(df.values)
    data1 = pandas.DataFrame(
        df.iloc[:, 0].unique())[0:30]  # 去重,处理所有行第一列，取前300
    # write_comments=p
    comments = data1.iloc[:, 0].apply(lambda x: SnowNLP(x).sentiments)
    tp(comments)  # series
    tp(comments[0])
    tp(str(comments))
    print(comments[1])
    print('--------------------')
    print(str(comments))
    ts()
    # print(len(data1[comments>=0.9]))
    # print(len(data1))
    # print(comments)  # 数值
    # plt.hist(comments, density=True)
    # plt.legend()

    comments.plot(kind='kde', label=data_path[2:-5])
    plt.xlabel(f"{data_path[0:-5]}牙膏情感指数")
    plt.ylabel("比率")
    plt.xticks([0, 1])

    fomatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)

    plt.savefig(pic_path)
    plt.show()
    plt.clf()
    print(f'{data_path}用时', time.time() - startt)


def testt(data_path='test1.xlsx'):
    file_name=data_path[0:-5]
    print(file_name)
    startt = time.time()
    df = pd.read_excel(data_path)
    df = df.dropna()
    data1 = pandas.DataFrame(
        df.dropna().iloc[:, 0].unique())  # 去重,处理所有行第一列，取前300
    l(data1)
    comments = data1.iloc[:, 0].apply(
        lambda x: SnowNLP(x).sentiments)  # series,做筛选条件

    stops = pd.read_csv(stop_list, encoding='gbk', header=None,
                        sep='mz666', engine='python')  # 读取失败？ 导致停用词未去除s
    stops = [' ', ''] + list(stops[0])

    positive_df = data1[comments >= 0.9]
    negative_df = data1[comments < 0.3]  # 后期改0.1

    positive_df.to_excel("./df/pos_df_{}".format(data_path))
    negative_df.to_excel("./df/neg_fg_{}".format(data_path))

    positive_ser = positive_df.iloc[:, 0].apply(mycut)  # series
    negative_ser = negative_df.iloc[:, 0].apply(mycut)
    ser=data1.iloc[:, 0].apply(mycut)

    positive_df = pd.DataFrame(positive_ser)  # df 存分词
    negative_df = pd.DataFrame(negative_ser)
    df=pd.DataFrame(ser)


    df[1] = df[0].apply(
        lambda s: s.split(' '))
    df[2] = df[1].apply(
        lambda x: [i for i in x if i not in stops])
    dict = Dictionary(df[2])  # 建立词典
    corpus = [dict.doc2bow(i) for i in df[2]]  # 建立语料库
    lda = LdaModel(corpus, num_topics=3, id2word=dict)  # LDA 模型训练
    plot = pyLDAvis.gensim_models.prepare(lda, corpus, dict)
    pyLDAvis.save_html(plot, './result/{}_all.html'.format(file_name))
    return

    positive_df[1] = positive_df[0].apply(
        lambda s: s.split(' '))  # 定义一个分割函数，然后用apply广播
    positive_df[2] = positive_df[1].apply(
        lambda x: [i for i in x if i not in stops]) # 将stops转为gbk了已经，不需要再转为utf-8

    # for i in positive_df[2]:
    #     print(i)
    # return
    # print('去停用词后positive_df')
    # print(positive_df[0]) #原
    # print(positive_df[1]) #分词
    # print(positive_df[2]) #去停用词
    # print(positive_df)
    # return
    negative_df[1] = negative_df[0].apply(
        lambda s: s.split(' '))  # 定义一个分割函数，然后用apply广播
    negative_df[2] = negative_df[1].apply(
        lambda x: [i for i in x if i not in stops])




    pos_dict = Dictionary(positive_df[2])
    # print(pos_dict)
    # print(pos_dict[0])
    # tp(pos_dict[0])
    # for i in pos_dict:
    #     if   '牙刷' in pos_dict[i]:
    #         # print('有的')
    #         print(pos_dict[i])
    # return
    pos_corpus = [pos_dict.doc2bow(i) for i in positive_df[2]] #词袋实现BOW模型
    pos_lda = LdaModel(pos_corpus, num_topics=3, id2word=pos_dict)
    pos_theme= pos_lda.show_topics()
    print(pos_theme)
    pos_plot=pyLDAvis.gensim_models.prepare(pos_lda,pos_corpus,pos_dict)
    pyLDAvis.save_html(pos_plot,'./result/{}_pos.html'.format(file_name))

    # return

    neg_dict = Dictionary(negative_df[2])  # 建立词典
    neg_corpus = [neg_dict.doc2bow(i) for i in negative_df[2]]  # 建立语料库
    neg_lda = LdaModel(neg_corpus, num_topics=3, id2word=neg_dict)  # LDA 模型训练
    neg_plot = pyLDAvis.gensim_models.prepare(neg_lda, neg_corpus, neg_dict)
    pyLDAvis.save_html(neg_plot, './result/{}_neg.html'.format(file_name))



    print('#正面主题分析{}'.format(data_path[0:-5]))
    for i in range(2):
        print('topic', i)
        print(pos_lda.print_topic(i))  # 输出每个主题
    print('#负面主题分析{}'.format(data_path[0:-5]))
    for i in range(3):
        print('topic', i)
        print(neg_lda.print_topic(i))  # 输出每个主题


    # ts()
    # print(len(data1[comments>=0.9]))
    # print(len(data1))
    # print(comments)  # 数值
    plt.hist(comments, density=True)
    # plt.legend()

    comments.plot(kind='kde', label=data_path[0:-5])
    plt.xlabel(f"{data_path[2:-5]}牌牙膏情感指数")
    plt.ylabel("比率")
    plt.xticks([0, 1])

    fomatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    # print("./test1/" + data_path[2:-5] + ".png")
    # print(data_path)
    # time.sleep(1111)
    # plt.savefig("./test1/" + data_path[0:-5] + ".png") #去后缀 .xlsx
    plt.show()
    print(f'{data_path}用时', round(time.time() - startt,2),'s')


def start_pool():
    data_paths = []
    with Pool() as pool:
        for data_path in os.listdir('./'):
            if 'xlsx' in data_path and 'test1' not in data_path:
                data_paths.append(data_path)
        print(data_paths)
        pool.map(testt, data_paths)
        pool.close()
        pool.join()


def fenci():
    pass


if __name__ == '__main__':

    start_pool()
    # snow_nlp('./test1.xlsx')
    # testt('中华.xlsx')
    for data_path in os.listdir('./'):
        pass
        # tp(comments)  # Dataframe
        # pos_comments = data1[comments >= 0.9]
        # neg_comments = data1[comments <= 0.3]
        #
        # my_cut = lambda s: ' '.join((jieba.cut(s)))
        # pos_cut = pos_comments.iloc[:, 0].apply(my_cut)
        # pos_cut = pos_comments.iloc[:, 0].apply(my_cut)
        # neg_cut = neg_comments.iloc[:, 0].apply(my_cut)
        # # print(pos_cut)
        # # print(neg_cut)
        # # 怎么存，直接把情感写入excel?
    # print("展示结果：")
    # plt.show()
    # stops = pd.read_csv(stop_list, encoding='gbk', header=None,
    #                     sep='mz', engine='python')
    # tp(stops) # dateframe
    # print(stops)
    # stops=[' ','']+list(stops[0]) #此时stops转为list了
    # # print(stops[0])
    # l(stops)
    # print(stops)
    # testt('./中华.xlsx')
    # testt()
