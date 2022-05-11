from snownlp import SnowNLP


def tp(x):
    print(type(x))


def l(x):
    print(len(x))

"""
把原来的xlsx分割，
新建积极的和消极的xlsx
然后对新xlsx(保存情感指数)进行LDA主题分析
"""
if __name__ == '__main__':
    text1 = "虽然换包装了，但是希望品质不变！用了好久纳美牙膏和牙刷了，全家都在用，非常好用，刷牙口感好，不杀沙口，味道清新，价格美丽！"
    text2="已经是第二次购买了，这款牙膏非常适合我，刷完了以后，牙齿非常干净，口气也很清新。非常不错，推荐给大家，同时还买了他们家的牙刷，用起来也是不错的。挺好的，这个品质能够经得起使用。"
    text=[text1,text2]
    s = SnowNLP(text1)
    print(s)
    print(s.sentiments)
    print(type(s))
    # with open('snownlpp.txt','w') as f:
    #     f.write(s)
