import MySearch

@MySearch.pr_runtime
def test():
    '''
     尽可能多的展示了MySearch中的功能
    '''
    Corpus = ['这是第一个文档：新年快乐。',
              '那是第一个：春节过年好',
              '这不是第二个文档吧：又一年过去，又长了一岁',
              '这不是最后的文档：我在北京过年',
              'Happy new year!',
              '我爱北京天安门',
              'I love Peking University.']

    t = MySearch.MySearch()
    default_corpus = t.GetDefaultCorpusName()
    if default_corpus:
        try:
            print("在默认语料库：%s 中查询'过年好，北京'：" % default_corpus)
            res = t.Query('过年好，北京')
            show(res)
        except:
            print('默认语料库受损')

    t.Train(Corpus) #建立检索模型，本次输入类型是[str, str, ...str]，也可以是代表语料路径名的str
    t.SaveModel('新语料库') #保存模型,参数1为新库名称，省略时将以当前时间命名，参数2为将文档保存的文件名，缺省则按照index命名
    t.AdjustDefaultCorpus('新语料库') #设置默认搜索库为默认语料库，此种情况下由于在SaveModel中设置了当前t类的语料库名，可以缺省参数，仍然可以成功设置
    print('保存了“新语料库”，并将其设置为默认检索语料库，下面是搜索“过年好，北京”的结果：')
    res = t.Query('过年好，北京')
    show(res)

    t1 = MySearch.MySearch('pkuseg') #使用pkuseg切词
    print('在默认搜索库中查询“我在北京读文档”：')
    show(t1.Query('我在北京读文档')) #查询保存的默认搜索库

    t1.Train(('今年冬天北京天气很好，可惜北京没有下雪',
              '可惜北京没有下雪',
              '南方下雪了吗',
              '杭州的雪景特别漂亮',
              '等到北京下雪的时候，我要堆个雪人'))
    # 本次Train()的参数是元组tuple。(str,str,...,str)
    print('在元组的元素中检索“北京下雪”的结果：')
    show(t1.Query('北京下雪'))
    print('向“新语料库”中添加文档，文档来自刚才的元组元素，这里进行了命名：')
    t1.AddCorpus('新语料库', filename=['v.txt', 'w.txt', 'x.txt', 'y.txt', 'z.txt']) #将当前训练的检索模型和语料库(参数2不缺省则将使用参数2指定的检索模型和语料库)加入到'新语料库'中去，加入时将这几个文档命名为参数3filename指定的名称
    t1.SaveModel(filename=['a.txt', 'b.txt', 'c.txt', 'd.txt', 'e.txt']) #保存当前训练的检索模型和语料库，参数1缺省，将以时间命名，参数2filename指定了文档名称
    print('在“新语料库”中检索“北京下雪”的结果：')
    show(t1.Query('北京 下雪', '新语料库')) #指定在 '新语料库' 中检索
    print('在“新语料库”中删除了z.txt后，检索“北京下雪”：')
    t1.DelDocument(['z.txt'], '新语料库') #在新语料库中删除z.txt
    show(t1.Query('北京 下雪', '新语料库')) #指定在 '新语料库' 中检索

    t2 = MySearch.MySearch('pkuseg')
    print('在“新语料库”中检索“北京”：')
    show(t2.Query('北京'))
    t2.add_userword('北京天安门') #添加“北京天安门”以后，再搜“北京”应该搜不到“我爱北京天安门”
    print('添加新词“北京天安门”后，在“新语料库”中检索“北京”')
    t2.Train('新语料库_corpus') #本次输入是代表路径名的str
    show(t2.Query('北京'))

    return t1.corpus_name

def test1(t1_corpus_name):
    '''
    删除保存的名为t1_corpus_name的搜索库
    '''
    t = MySearch.MySearch()
    t.RemoveCorpus(t1_corpus_name)

def show(res):
    for document in res:
        print('**********')
        if document.get('filename'):
            print(document['filename'])
        print(document.get('content'))
    print('共' + str(len(res)) + '个检索结果')
    print('**********************************')

if __name__ == '__main__':
    t1_corpus_name = test()
    print(t1_corpus_name)
    test1(t1_corpus_name)
