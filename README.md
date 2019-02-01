# MySearch描述
本地语料很多？爬到的文档很多？运行出很多无序结果？我们经常面对一些搜索引擎无法检索的文本/或其它程序运行结果，想要对这些内容进行检索、按相关性排序等。MySearch是我用python3写的，目的在于方便中英文检索的小脚本，中文分词支持jieba或者pkuseg，相关性排序基于sklearn的tf-idf计算。另，想要使用其他切词包的，可以先自行切词，调整输入格式后，再使用.Train(Iterable, 'e')，仍可以运用搜索。

## 实现以下功能：	

    1.检索模型、语料库的建立（训练） -- 保存 -- 使用 -- 添加语料（从列表/目录） -- 删除 -- 部分删除
	
    2.根据搜索词汇，对Iterable元素（元素为字符串）进行相关性排序，返回序列号等，输入为该Iterable对象。
	
    3.根据搜索词汇，对指定目录下的文件进行相关性排序，返回文件名序列号等，输入为目录名str。
	
上述功能基本已经实现，but目前 添加语料（从列表/目录）到已有库中，实现起来比较垃圾（需要重新训练Train()一次），另外，待检索内容如果在文件中，那么只支持encoding='utf-8'，或'gbk'编码（垃圾），我可能会抽空把这个改好的，，呵呵吧。


## 关于停用词和用户词汇：

        默认停用词存放在stop_words.txt，可自行修改，也可以在使用 add_stopwords() 函数添加
		
        默认用户词存放在userdict.txt，可自行修改，也可以在使用 add_userword() 函数添加
		
        add_stopwords()或add_userword() 使用后，将在下一次self.Train()使用时生效
		

### 注：欢迎使用、改进和提意见，尤其是标注"垃圾"的地方很有提升空间，欢迎小伙伴们帮忙改进，转载请注明出处，不过代码目前比较垃圾，你们可能也不会转。

# 用法
不支持安装成python包。需要手动（垃圾）将"MySearch.py","stop_words.txt","userdict.txt"置于python3项目文件夹中，即可在项目中使用import MySearch加载MySearch.py，之后使用MySearch.func()即可使用MySearch中的函数func()。



# 具体操作
## 示例详见Example文件夹的test.py,test_e.py,test_all.py


## 1.第一步通常是生成Mysearch class

  	t = MySearch() #不带参数表示默认使用jieba进行中文分词
	
	or
	
	t = MySearch('pkuseg') #使用pkuseg进行中文分词
  
  
## 2.介绍一下MySearch.Query(self, query_str, corpus_name=None)吧

        2.1 如果当前这个t类使用了t.Train(),那么优先从刚建立好的检索模型中检索query_str

        2.2 即使当前这个t类使用了t.Train(),但是检索的时候给了corpus_name参数，那么会从corpus_name指定的语料库中检索query_str，如果corpus_name指向的语料库不存在或者其检索模型受损，将会查询失败。

        2.3 如果当前这个t类没使用t.Train(),而且也没有给出corpus_name，那么会从默认的语料库中检索，如果默认库不存在或者受损，将会查询失败。

        2.4 如果当前这个t类没使用t.Train(),而给出了corpus_name参数，那么当然会从corpus_name指定的语料库中检索query_str，同样的，如果corpus_name指向的语料库不存在或者其检索模型受损，将会查询失败。

        2.5 返回值：list[dict, dict, ..., dict]，其中：dict{'index', 'score', 'content'} or dict{'index', 'score', 'filename', 'content'}。list按照dict的score从大到小排序，score为该dict对应的文档/待搜索内容的tf-idf得分，index为其在原序列中的序列号，content即是内容，存在文件时，filename为文件名。

  注一下：保存好的语料库是以'_corpus'结尾的文件夹，检索模型是'_model'结尾的文件，你可以使用.SaveModel()来保存语料库和检索模型，但是t.SaveModel()之前得t.Train()一下，否则没有可保存的模型，也会发生错误。.Train()的用法？往下看...
  
 
## 3.介绍一下MySearch.Train(self, argc, e=None)
 
  Train()将使用argc中的element作为语料文本建立检索模型。
  
      :param argc: argc支持str类型和其它Iterable（list, tuple等，其element应为str类型，表示文本）型变量：
	  
          argc为str类型时，argc代表语料文件夹目录，Train()将使用该目录下的文档建立检索模型。
		  
          argc为Iterable类型时，argc即待训练语料库，argc中的element应为str类型，表示语料文本，
		  
      :param e:  e默认为None，启用jieba或pkuseg，当待使用文本为纯英文或其它 *由空格隔开* 无需分词的语料时，可以令e='e',将不使用jieba、pkuseg，可一定程度提高效率，但用户词汇无效。
	  
      :return:成功返回True，失败将报错。
	  

## 4.其它函数在这里简单罗列一下，具体参数和使用方法可以在MySearch.py的注释中查看

### 1.add_stopwords(self, my_stopword_list) 
  
        添加停用词
		
### 2.add_userword(self, my_word_list) 
  
        添加词汇，用来优化中文分词，使jieba、pkuseg语料库中没有的词能正确切分。
		
### 3.AddCorpus(self, corpus_name, corpus_name2=None, filename=None) 
  
        向指定语料库添加语料文档，并更新检索模型 （垃圾）
		
        向corpus_name指定的语料库中添加语料文档，语料文档来自corpus_name2指定的语料库，当corpus_name2为None时，
		
        默认为self.Train()生成的检索模型和其语料库。filename参数可用于给新添加的文档命名，格式错误时视为没有该参数,
		
        默认为保存原名称或者按其在原语料库中的Index序号命名。另，名称与新语料库中的名称相同时，将增加‘-副本’后缀
		
### 4.GetDefaultCorpusName(self)
  
        获取当前默认的搜索库名称
		
### 5.AdjustDefaultCorpus(self, default_corpus)
  
        调整默认搜索库，参数为新库名称,有默认搜索库时，将调整为新搜索库，没有默认搜索库时将设置
		
### 6.RemoveCorpus(self, corpus_name=None) 
  
        删除保存好的语料库，参数为None时删除self.corpus_name指定的语料库，语料库不存在将报错
		
### 7.DelDocument(self, documents, corpus_name=None) 
  
        删除保存好的语料库的部分语料文档document，documents是文档名称列表，corpus_name指定的语料库不存在将报错
	
	
## 修改日志
2019.1

	创建富含bug的代码，并修补了一些bug
	
	完善对功能边界的定义、拓展了对pkuseg分词的支持等

2019.2.2

	修复SaveModel()中的一处bug

	对于文件夹中的训练语料，扩展支持gbk编码

## 作者: ldhldh
