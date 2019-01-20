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
    t.Train(Corpus, 'e')
    res = t.Query('the document')

    print(len(res))
    for document in res:
        print('**********')
        print(document.get('content'))

if __name__ == '__main__':
    test()