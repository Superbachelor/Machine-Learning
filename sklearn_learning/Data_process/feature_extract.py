from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn_learning.Data_process import jieba_tool
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
"""


def sklearn_dataset():
    iris = datasets.load_iris()

    print("特征值名字:", iris.feature_names)
    print("特征值,和shape", iris.data, iris.data.shape)

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    Bunch = datasets.base.Bunch  # 字典类型
    Bunch.iris = iris['data']
    print("\n", Bunch.iris)
    return None


def sklearn_feature():
    data = [{'city': '北京', 'temperature': 100}, {'city': '北京2', 'temperature': 50},
            {'city': '北京3', 'temperature': 20}]
    # transfer = DictVectorizer(sparse=False)  # 大矩阵
    transfer = DictVectorizer(sparse=True)  # 减少内存, 自动过滤掉为0的值
    data_new = transfer.fit_transform(data)
    print("new data:", data_new)
    return None


def count_vector():  # 英文
    data = ["life is short, I like the python, I like the life",
            "life is too long, I dislike the python"]
    transfer = CountVectorizer()  # 统计样本出现特征词的次数
    data_new = transfer.fit_transform(data)

    print("特征词:\n", transfer.get_feature_names())
    print("new data:\n", data_new.toarray())
    print("new data(sparse):\n", data_new)
    return None


def count_vector_Chinese():  # 中文
    data = ["我 爱的 是 你, 你 却 不爱 我",
            "我 爱的 就是 你"]
    transfer = CountVectorizer(stop_words=[])  # 统计样本出现特征词的次数
    data_new = transfer.fit_transform(data)

    print("特征词:\n", transfer.get_feature_names())
    print("new data:\n", data_new.toarray())
    print("new data(sparse):\n", data_new)
    return None


def count_vector_jieba_Chinese():  # 中文自动分词
    data = ["统计学作为数据分析的入门知识，非常的重要，作为入门，必须要掌握描述性统计以及里面各类图表的应用场景和理解。"
            "而再深入到，如线性回归，贝叶斯，假设检验等，则是为以后成为高级数据分析师做铺垫，在未来做到建模和预测时，"
            "会用到很多这类知识，同时在未来进阶过程中，学习机器学习的一些经典算法时，也需要这些知识来帮助理解和学习。",

            "Excel作为数据分析的基础，是众多数据分析工具的入门工具，而且它的功能非常的强大，具有非常多的实用性，在快速处理一些数据，"
            "快速出图的时候，非常的灵活，也非常的便捷，其中也有很多的函数，包括max,min,average,find,match,vlookup等，"
            "可以非常灵活的查询数值或者进行统计分析，同时Excel的数据透视表功能也非常的强大，可以快速的选取所需元素进行分析。"
            "非常适合用来做快速的数据清洗，入门门槛低，而且实用性非常强"]
    new_data = []
    for element in data:
        sentence = jieba_tool.separate_word(element)
        new_data.append(sentence)

    transfer = CountVectorizer(stop_words=[])  # 统计样本出现特征词的次数
    data_new = transfer.fit_transform(new_data)

    print("特征词:\n", transfer.get_feature_names())
    print("new data:\n", data_new.toarray())
    print("new data(sparse):\n", data_new)
    return None


def TF_IDF():
    # TF: 某一个词在文本中的频率
    # IDF: 总文件数目除以包含改词语的文件数目, 再将得到的结果取10为底数的对数
    # TF-IDF = TF * IDF (前面两个相乘)
    data = ["统计学作为数据分析的入门知识，非常的重要，作为入门，必须要掌握描述性统计以及里面各类图表的应用场景和理解。"
            "而再深入到，如线性回归，贝叶斯，假设检验等，则是为以后成为高级数据分析师做铺垫，在未来做到建模和预测时，"
            "会用到很多这类知识，同时在未来进阶过程中，学习机器学习的一些经典算法时，也需要这些知识来帮助理解和学习。",

            "Excel作为数据分析的基础，是众多数据分析工具的入门工具，而且它的功能非常的强大，具有非常多的实用性，在快速处理一些数据，"
            "快速出图的时候，非常的灵活，也非常的便捷，其中也有很多的函数，包括max,min,average,find,match,vlookup等，"
            "可以非常灵活的查询数值或者进行统计分析，同时Excel的数据透视表功能也非常的强大，可以快速的选取所需元素进行分析。"
            "非常适合用来做快速的数据清洗，入门门槛低，而且实用性非常强"]
    new_data = []
    for element in data:
        sentence = jieba_tool.separate_word(element)
        new_data.append(sentence)

    transfer = TfidfVectorizer(stop_words=["一些", "众多"])
    data_new = transfer.fit_transform(new_data)
    print("特征词:\n", transfer.get_feature_names())
    print("new data:\n", data_new.toarray())
    print("new data(sparse):\n", data_new)
    return None


if __name__ == '__main__':
    # sklearn_dataset()
    # sklearn_feature()
    # count_vector()
    # count_vector_Chinese()
    # count_vector_jieba_Chinese()
    TF_IDF()

