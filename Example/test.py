import MySearch

@MySearch.pr_runtime
def test():
    '''
     MySearch类中seg为缺省状态，默认为'jiaba'，使用jieba进行中文分词，启用seg='pkuseg'时将使用pkuseg进行中文分词
     下边的例子中，pkuseg的结果要好于jieba。
    '''
    Corpus = ['这是第一个文档：新年快乐。',
              '那是第一个：春节过年好',
              '这不是第二个文档吧：又一年过去，又长了一岁',
              '这是最后的文档：我爱北京天安门']

    t = MySearch.MySearch('pkuseg')#参数缺省或指定'jieba'时使用jieba
    t.Train(Corpus)
    res = t.Query('新年好，今年我要在北京读更多文档')

    for document in res:
        print('**********')
        print(document.get('score'))
        print(document.get('content'))
    print('共' + str(len(res)) + '个检索结果')

if __name__ == '__main__':
    test()
