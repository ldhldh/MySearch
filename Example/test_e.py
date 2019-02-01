import MySearch

@MySearch.pr_runtime
def test():
    '''
     Train()中启用e='e'时，将不使用中文分词库（将按空格分词），提高效率，不启用'e'也可以处理英文
    '''
    Corpus = ['this is the first document.',
              'this is the second text',
              'the third document?',
              'the last one!']

    t = MySearch.MySearch()
    # 也可以直接在初始化MySearch时使用参数'e'，如果此时使用'e'，不仅训练时会按照空格切词，在查询时也将使用空格切词

    t.Train(Corpus, 'e')#仅在Train时使用参数'e'时，查询时，仍将使用jieba或pkuseg（取决于MySearch的参数）对查询字串进行切词
    res = t.Query('the document')

    for document in res:
        print('**********')
        print(document.get('content'))
    print('共' + str(len(res)) + '个检索结果')

if __name__ == '__main__':
    test()
