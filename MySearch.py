from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csc_matrix
import jieba.posseg as pseg
import jieba
import codecs
from collections import Iterable
import os
import time
import copy
import shutil

def pr_runtime(func):
    '''
    装饰器：增加运行时间打印
    '''
    def runtime(*argv, **kw):
        t_begin = time.time()
        func(*argv, **kw)
        print('*******************')
        print(str(time.time() - t_begin) + 's')
    return runtime

class MySearch(object):
    '''
    该class可以为中/英文语料库（文件夹/Iterable对象）建立基于tf-idf的检索模型，若涉及文件操作（除stop_words.txt，userdict.txt）
    外，只支持编码utf-8，其中中文分词依赖jieba，tf-idf依赖sklearn
    实现以下功能：
    1.检索模型、语料库的建立（训练） -- 保存 -- 使用 -- 添加语料（从列表/目录） -- 删除 -- 部分删除
    2.根据搜索词汇，对Iterable元素（元素为字符串）进行相关性排序，返回序列号等
    3.根据搜索词汇，对指定目录下的文件进行相关性排序，返回文件名序列号等

    :param 参数同 sklearn.feature_extraction.text.TfidfTransformer(), 主要是为了支持不同的tf-idf模式

    Example：见test(),test2(),go()

    函数简介：

    关于停用词和用户词汇：
        默认停用词存放在stop_words.txt，可自行修改，也可以在使用 add_stopwords() 函数添加
        默认用户词存放在userdict.txt，可自行修改，也可以在使用 add_userword() 函数添加
        add_stopwords()或add_userword() 使用后，将在下一次self.Train()使用时生效

        add_stopwords(self, my_stopword_list) 添加停用词
            my_stopword_list: list[str,str,...,str] 待添加的停用词
            :return: 成功返回True，失败返回False

        add_userword(self, my_word_list) 添加词汇，用来优化中文分词，使jieba语料库中没有的词能正确切分。
            当self.Train()使用e='e'时，不使用jieba，用户词汇无效。
            my_word_list: list[str,str,...,str] 待添加的分词
            :return:成功返回True，失败返回False

    Train(self, argc, e=None) 训练函数，用于建立检索模型
        argc支持str类型和其它Iterable（list, tuple等，其element应为str类型，表示文本）型变量：
            argc为str类型时，argc代表语料文件夹目录，Train()将使用该目录下的文档建立检索模型。
            argc为Iterable类型时，argc即待训练语料库，argc中的element应为str类型，表示语料文本，
        Train()将使用argc中的element作为语料文本建立检索模型。
        e默认为None，启用jieba库，当待使用文本为纯英文或其它 *由空格隔开* 无需分词的语料时，可
        以令e='e',将不使用jieba库，可一定程度提高效率，但用户词汇无效。

    Query(self, query_str, corpus_name=None) 查询函数
        query_str为查询字符串，corpus_name为查询语料库名称，corpus_name为None时优先查询self.Train(),
        其次查询默认语料库，若corpus_name不为空，则查询corpus_name指向的语料库，若没有该语料库将出错。
        :return: list[dict,dict,...,dict]
            其中dict{'index', 'score', 'content'} or dict{'index', 'score', 'filename', 'content'}
        注：以'_corpus'结尾的文件夹将被作为保存好的语料库，文件夹名即为语料库名，查询保存好的
    语料库时，corpus_name结尾可以带'_corpus'也可以不带。另，'_model'结尾的文件被作为保存好的模型。

    SaveModel(self, corpus_name=None, filename=None) 保存模型
        self.Train()之后才能self.SaveModel(),否则将出错
        corpus_name:str 保存成名为corpus_name的语料库，缺省时将根据时间生成一个语料库名称
        filename: list[str,str,...,str] 用于给语料文档document命名，缺省时使用原名称，没有原名称时用index
        :return:成功将返回True，失败会报错

    AddCorpus(self, corpus_name, corpus_name2=None, filename=None) 向指定语料库添加语料文档，并更新检索模型
        向corpus_name指定的语料库中添加语料文档，语料文档来自corpus_name2指定的语料库，当corpus_name2为None时，
        默认为self.Train()生成的检索模型和其语料库。filename参数可用于给新添加的文档命名，格式错误时视为没有该参数,
        默认为保存原名称或者按其在原语料库中的Index序号命名。另，名称与新语料库中的名称相同时，将增加‘-副本’后缀
        corpus_name1: name of Corpus1    str
        corpus_name2: name of Corpus2    str
        filename: filename of Corpus2's file     list [str,str...，str]
        :return: 成功返回True，失败将报错

    GetDefaultCorpusName(self)
        获取当前默认的搜索库名称

    AdjustDefaultCorpus(self, default_corpus)
        调整默认搜索库，参数为新库名称,有默认搜索库时，将调整为新搜索库，没有默认搜索库时将设置
        default_corpus: str,
                    name of the new default corpus
        :return: 成功返回True，失败返回False

    RemoveCorpus(self, corpus_name=None) 删除保存好的语料库
        删除保存好的语料库，参数为None时删除self.corpus_name指定的语料库，语料库不存在将报错
        corpus_name:想要删除的语料库名称
        :return:成功返回True,失败将报错

    DelDocument(self, documents, corpus_name=None) 删除保存好的语料库的部分语料文档document
        删除保存好的语料库的部分语料文档document
        documents: list[str,str,...,str] 待删除的语料文档document的文件名
        corpus_name: str  语料库名
        :return: 成功将返回True,失败会报错
    '''
    def __init__(self, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
        self.stop_word_path = "stop_words.txt"
        self.stopwords = self.__get_stopwords(self.stop_word_path)
        self.corpus = []
        self.corpus_name = ''
        self.files = []
        self.tfidf = None
        self.word_dict = {}
        self.my_word_list = []
        jieba.load_userdict("userdict.txt")
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def __get_stopwords(self, stop_words):
        stopwords = codecs.open(stop_words, 'r').readlines()
        stopwords = [w.strip() for w in stopwords]
        return ['', ' '] + stopwords

    def add_stopwords(self, my_stopword_list):
        '''
        添加停用词，add_stopwords()使用后，将在下一次self.Train()使用时生效
        :param my_stopword_list: list[str,str,...,str] 待添加的停用词
        :return: 成功返回True，失败返回False
        '''
        for word in my_stopword_list:
            if type(word) != str:
                return False
        self.stopwords += my_stopword_list
        return True

    def add_userword(self, my_word_list):
        '''
        添加词汇，用来优化中文分词，使jieba语料库中没有的词能正确切分，
        add_userword() 使用后，将在下一次self.Train()使用时生效，当self.Train()使用e='e'时，不使用jieba，用户词汇无效。
        :param my_word_list: list[str,str,...,str] 待添加的分词
        :return:成功返回True，失败返回False
        '''
        for word in self.my_word_list:
            if type(word) != str:
                return False
            jieba.add_word(word)
        self.my_word_list = my_word_list
        return True

    def __Path2Corpus(self):
        self.files = os.listdir(self.corpus_name)
        for file in self.files:
            f = open(self.corpus_name + '/' + file, 'r', -1, encoding='utf-8')
            self.corpus.append(f.read())
            f.close()

    def __cut_str(self, text):
        res = ''
        words = pseg.cut(text)
        for word, flag in words:
            if flag not in self.stop_flag and word not in self.stopwords:
                word_for_search = jieba.cut_for_search(word)
                for n in word_for_search:
                    res += n + ' '
        return res

    def __cut_corpus(self):
        corpus_cut = []
        for s in self.corpus:
            corpus_cut.append(self.__cut_str(s))
        return corpus_cut

    def __cut_for_e(self):
        corpus_cut = []
        for text in self.corpus:
            content = ''
            for word in text.split(' '):
                if word not in self.stopwords:
                    content += word + ' '
            corpus_cut.append(content)
        return corpus_cut

    def __sort_seed(self, document):
        return document['score']

    def __show(self, document_scores):
        if not len(document_scores):
            print('Sorry,the words you queried are not in the corpus')
            return []
        res = []
        for document in document_scores:
            res.append({'index': document, 'score': document_scores[document]})

        res = sorted(res, key=self.__sort_seed, reverse=True)
        if self.corpus_name:
            try:
                self.files = os.listdir(self.corpus_name)
            except:
                self.files = []
        if self.corpus:
            for document in res:
                document['content'] = self.corpus[document['index']]
        elif self.files:
            try:
                for document in res:
                    f = open(self.corpus_name + '/' + self.files[document['index']], 'r', encoding='utf-8')
                    document['content'] = f.read()
                    f.close()
            except:
                pass

        if self.files:
            try:
                for document in res:
                    document['filename'] = self.files[document['index']]
                    print(document['filename'])
            except:
                pass
        return res

    def __save_select(self, corpus_path):
        print("%s has existed!"
              "Choosing to continue will delete the old one!\n"
              "Maybe you can use 'AddCorpus()' or change another corpus_name.\n"
              "Do you want to continue?(y/n) :" % corpus_path, end='')
        choose = input()
        if choose == 'y' or choose == 'Y':
            shutil.rmtree(corpus_path)
        else:
            raise ValueError('User termination.')

    def __creat_corpus(self, corpus_name=None, filename=None):
        if not self.corpus_name:
            if corpus_name:
                self.corpus_name = corpus_name
                if self.corpus_name[-7:] == '_corpus':
                    self.corpus_name = self.corpus_name[:-7]
            else:
                self.corpus_name = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
            corpus_path = self.corpus_name + '_corpus'
            if os.path.exists(corpus_path):
                self.__save_select(corpus_path)
            os.mkdir(corpus_path)
            self.files = filename
            if not self.files or len(self.files) != len(self.corpus):
                print('No filename or filename is not complete')
                self.files = []
                for i in range(len(self.corpus)):
                    self.files.append(str(i)+'.txt')
            for index in range(len(self.corpus)):
                try:
                    f = open(corpus_path + '/' + self.files[index], 'w', encoding='utf-8')
                    f.write(self.corpus[index])
                    f.close()
                except:
                    continue
        else:
            old_path = self.corpus_name
            self.corpus_name = self.corpus_name.replace('\\', '/')
            if '/' in self.corpus_name:
                self.corpus_name = self.corpus_name.split('/')[-1]
            if self.corpus_name[-7:] == '_corpus':
                self.corpus_name = self.corpus_name[:-7]
            corpus_path = self.corpus_name + '_corpus'
            if os.path.exists(corpus_path):
                self.__save_select(corpus_path)
            shutil.move(old_path, corpus_path)
        self.corpus_name = corpus_path
        return self.corpus_name[:-7]

    def SaveModel(self, corpus_name=None, filename=None):
        '''
        self.Train()之后才能self.SaveModel(),否则将出错
        :param corpus_name:str 保存成名为corpus_name的语料库，缺省时将根据时间生成一个语料库名称
        :param filename: list[str,str,...,str] 用于给语料文档document命名，缺省时使用原名称，没有原名称时用index
        :return:成功将返回True，失败会报错
        '''
        if self.tfidf == None:
            ValueError("SaveModel() must be used after Train()")
        path = self.__creat_corpus(corpus_name, filename)
        sparse_matrix_2_save = []
        sparse_matrix_2_save.append((list(self.tfidf.data), list(self.tfidf.indices), list(self.tfidf.indptr)))
        sparse_matrix_2_save.append(self.tfidf.shape)
        sparse_matrix_2_save.append(self.word_dict)
        try:
            f = open(path + '_model', 'w', encoding='utf-8')
            f.write(str(sparse_matrix_2_save))
            return True
        except:
            self.__del_corpus(path)
            raise ValueError('save errors!')

    def AddCorpus(self, corpus_name, corpus_name2=None, filename=None):
        '''
        向corpus_name指定的语料库中添加语料文档，语料文档来自corpus_name2指定的语料库，当corpus_name2为None时，
        默认为self.Train()生成的检索模型和其语料库。filename参数可用于给新添加的文档命名，格式错误时视为没有该参数,
        默认为保存原名称或者按其在原语料库中的Index序号命名。另，名称与新语料库中的名称相同时，将增加‘-副本’后缀
        :param corpus_name1: name of Corpus1    str
        :param corpus_name2: name of Corpus2    str
        :param filename: filename of Corpus2's file     list [str,str...]
        :return: 成功返回True，失败将报错
        '''
        exist_list = self.__is_corpus_exit(corpus_name)
        if not exist_list[0]:
            raise ValueError('no corpus found, %s' % (corpus_name + '_corpus'))
        if not exist_list[1]:
            raise ValueError('no model found, %s' % (corpus_name + '_model'))
        corpus_name = exist_list[0]
        if corpus_name2:
            exist_list = self.__is_corpus_exit(corpus_name2)
            if not exist_list[0]:
                raise ValueError('no corpus found, %s' % (corpus_name2 + '_corpus'))
            if not exist_list[1]:
                raise ValueError('no model found, %s' % (corpus_name2 + '_model'))
            temp_name = exist_list[0]
        else:
            TempSearch = copy.deepcopy(self)
            temp_name = str(time.time()) + '_corpus'
            TempSearch.SaveModel(temp_name, filename)
        try:
            files = os.listdir(temp_name)
            files2 = os.listdir(corpus_name)
            if filename:
                for i in filename:
                    if type(i) != str:
                        ValueError('Element of filename should be str!')
            if filename and len(filename) != len(files):
                print('No filename or filename is not complete')
                filename = files
            for i in range(len(files)):
                while filename[i] in files2:
                    filename[i] = filename[i].split('.')[0] + '-副本' + filename[i].split('.')[1] if '.' in filename[i] else filename[i] + '-副本'
                os.rename(temp_name + '/' + files[i], corpus_name + '/' + filename[i])
            NewSearch = MySearch()
            NewSearch.Train(corpus_name)
            self.__del_corpus(temp_name)
        except:
            ValueError("AddCorpus() error!")
        else:
            print('Added!')
            return True

    def GetDefaultCorpusName(self):
        '''
        :return: str
                当前默认的搜索库名称
        '''
        try:
            f = open('venv/default_corpus', 'r', encoding='utf-8')
            default_corpus = f.read()
            f.close()
        except:
            raise ValueError('Get default_corpus error! '
                             'Make sure venv/default_corpus exist')
        return default_corpus

    def AdjustDefaultCorpus(self, default_corpus):
        '''
        调整默认搜索库，参数为新库名称,有默认搜索库时，将调整为新搜索库，没有默认搜索库时将设置
        :param default_corpus: str,
                    name of the new default corpus
        :return: 成功返回True，失败返回False
        '''
        exist_list = self.__is_corpus_exit(default_corpus)
        if not exist_list[0]:
            print('Adjust default corpus failed!'
                             'Make sure "%s_corpus" exist.\n'
                  'Maybe you can use "SaveModel()" first.' % default_corpus)
            return False
        elif not exist_list[1]:
            print('Adjust default corpus failed!'
                             'Make sure "%s_model" exist.\n'
                  'Maybe you can use "SaveModel()" first.' % default_corpus)
            return False
        else:
            f = open('venv/default_corpus', 'w', encoding='utf-8')
            f.write(default_corpus)
            f.close()
            return True

    def __is_corpus_exit(self, corpus_name=None):
        if corpus_name == None:
            corpus_name = self.GetDefaultCorpusName()
        corpus_path = corpus_name + '_corpus'
        model_path = corpus_name + '_model'
        fs = os.listdir('.')
        exist_list = [corpus_path in fs, model_path in fs]
        if exist_list[0]:
            exist_list[0] = corpus_path
        if exist_list[1]:
            exist_list[1] = model_path
        return exist_list

    def __generate_csc_matrix(self, model_path):
        f = open(model_path, 'r', encoding='utf-8')
        sparse_matrix_saved = eval(f.read())
        self.tfidf = csc_matrix(sparse_matrix_saved[0], sparse_matrix_saved[1])
        self.word_dict = sparse_matrix_saved[2]

    def __use_model(self, corpus_name=None):
        if corpus_name == None:
            corpus_name = self.GetDefaultCorpusName()
        exist_list = self.__is_corpus_exit(corpus_name)
        if not exist_list[0]:
            raise ValueError('no corpus found, %s' % (corpus_name + '_corpus'))
        if not exist_list[1]:
            raise ValueError('no model found, %s' % (corpus_name + '_model'))
        self.corpus_name = exist_list[0]
        files = os.listdir(self.corpus_name)
        self.__generate_csc_matrix(exist_list[1])
        if self.tfidf.shape[1] != len(files) or self.tfidf.shape[1] != len(self.word_dict):
            ValueError("Error! Corpus was destroyed!\n"
                       "Suggest you .Train() the corpus.")

    def __del_corpus(self, corpus_name=None):
        if corpus_name == None:
            corpus_name = self.corpus_name
        if corpus_name[-7:] == '_corpus':
            corpus_name = corpus_name[:-7]
        if corpus_name[-6:] == '_model':
            corpus_name = corpus_name[:-6]
        if corpus_name[-10:] == '_model':
            corpus_name = corpus_name[:-10]
        files = os.listdir('.')
        try:
            if corpus_name + '_corpus' in files:
                shutil.rmtree(corpus_name + '_corpus')
            if corpus_name + '_model' in files:
                os.unlink(corpus_name + '_model')
        except:
            pass
        else:
            return corpus_name + '_corpus and ' + corpus_name + '_model'

    def RemoveCorpus(self, corpus_name=None):
        '''
        删除保存好的语料库，参数为None时删除self.corpus_name指定的语料库，语料库不存在将报错
        :param corpus_name:想要删除的语料库名称
        :return:成功返回True,失败将报错
        '''
        res = self.__del_corpus(corpus_name)
        if res:
            print("%s have Removed!" % res)
            return True
        else:
            ValueError('Remove Error!')

    def DelDocument(self, documents, corpus_name=None):
        '''
        删除保存好的语料库的部分语料文档document
        :param documents: list[str,str,...,str] 待删除的语料文档document的文件名
        :param corpus_name: str  语料库名
        :return: 成功将返回True,失败会报错
        '''
        if not corpus_name:
            corpus_name = self.corpus_name
        files = os.listdir('.')
        if corpus_name in files:
            files2 = os.listdir(corpus_name)
            try:
                for file in documents:
                    if file in files2:
                        os.unlink(corpus_name + '/' + file)
                    else:
                        print(file + 'not in ' + corpus_name + '/')
            except:
                ValueError('unlink error.')
            print('Deleted!')
            return True
        else:
            ValueError('DelDocument() error, for wrong corpus_name.\n'
                  'Please indicate the correct corpus_name.')

    def Train(self, argc, e=None):
        if type(argc) == str:
            self.corpus_name = argc
            self.__Path2Corpus()
        elif isinstance(argc, Iterable):
            self.corpus = argc
        if e == 'e':
            corpus_cut = self.__cut_for_e()
        else:
            corpus_cut = self.__cut_corpus()

        vectorizer = CountVectorizer()
        words = vectorizer.fit_transform(corpus_cut)
        self.word_dict = vectorizer.vocabulary_
        self.tfidf = TfidfTransformer(self.norm, self.use_idf, self.smooth_idf,
                 self.sublinear_tf).fit_transform(words)
        self.tfidf = self.tfidf.tocsc()

    def __get_scores(self, query_list):
        document_scores = {}
        for word in query_list:
            index = self.word_dict.get(word)
            if index == None:
                continue
            documents = self.tfidf.indices[self.tfidf.indptr[index]:self.tfidf.indptr[index + 1]]
            data = self.tfidf.data[self.tfidf.indptr[index]:self.tfidf.indptr[index + 1]]
            for i in range(len(documents)):
                document_scores[documents[i]] = document_scores.get(documents[i], 0) + data[i]
        return document_scores

    def Query(self, query_str, corpus_name=None):
        '''
        查询函数
        :param query_str: str query_str为查询字符串
        :param corpus_name: str corpus_name为查询语料库名称，corpus_name为None时优先查询self.Train(),
        其次查询默认语料库，若corpus_name不为空，则查询corpus_name指向的语料库，若没有该语料库将出错。
        :return: list[dict,dict,...,dict]
            其中dict{'index', 'score', 'content'} or dict{'index', 'score', 'filename', 'content'}
        注：以'_corpus'结尾的文件夹将被作为保存好的语料库，文件夹名即为语料库名，查询保存好的
    语料库时，corpus_name结尾可以带'_corpus'也可以不带。另，'_model'结尾的文件被作为保存好的模型。
        '''
        if self.corpus_name == corpus_name:
            corpus_name = None
        if self.tfidf != None and corpus_name == None:
            temp = self.__cut_str(query_str).split(' ')
            query_list = [x for x in temp if x not in self.stopwords]
            document_scores = self.__get_scores(query_list)
            res = self.__show(document_scores)
        elif self.tfidf != None:
            newSerch = MySearch()
            res = newSerch.Query(query_str, corpus_name)
        else:
            self.__use_model(corpus_name)
            res = self.Query(query_str)
        return res

@pr_runtime
def test():
    '''
    运行时在本目录下应存在名为Poetries1的文件夹，文件夹中存放几个文档
    '''
    corpus = [
        '这是第一个文档。',
        '这是第二个文档呗！',
        '老三.',
        '难道这就是文档1？',
    ]
    t0 = MySearch()
    t0.Train('Poetries1')
    t0.SaveModel()

    t = MySearch()
    t.AdjustDefaultCorpus('Poetries1')
    t.Train(corpus)
    t.AddCorpus('Poetries1', filename=['w', 'x', 'y', 'z'])
    res = t.Query('文档')
    for document in res:
        print(document.get('content'))
    t.SaveModel('test', ['a.txt', 'b.txt', 'c.txt', 'd.txt'])
    res = t.Query('文档')
    for document in res:
        print(document.get('content'))
    res = t.Query('黄叶', 'Poetries1')
    for document in res:
        print(document.get('content'))
    res = t.Query('文档', 'Poetries1')
    for document in res:
        print(document.get('content'))
    res = t.Query('恶意代码', 'Poetries1')
    for document in res:
        print(document.get('content'))
    t.AdjustDefaultCorpus('test')
    t2 = MySearch()
    res = t2.Query('文档')
    for document in res:
        print(document.get('content'))
    t2.RemoveCorpus('test')
    res = t2.Query('文档')
    for document in res:
        print(document.get('content'))
    t2.DelDocument(['x'], 'Poetries1_corpus')
    t2.AdjustDefaultCorpus('Poetries')

@pr_runtime
def test2():
    Corpus = ['this is the first document.',
              'this is the second text',
              'the third document?',
              'the last one!']
    t = MySearch()
    t.Train(Corpus)
    res = t.Query('the document')
    print(len(res))
    for document in res:
        print('**********')
        print(document.get('content'))

@pr_runtime
def go():
    '''
    运行时应存在保存好的默认搜索语料库
    '''
    t = MySearch()
    res = t.Query('年华')

    for document in res:
        print('**********')
        # print(document)
        content = document.get('content')
        if content:
            print(content.replace('<br/>', '\n').replace('\n\n', '\n').replace('\n\n', '\n'))
    print(len(res))

if __name__ == '__main__':
    test2()