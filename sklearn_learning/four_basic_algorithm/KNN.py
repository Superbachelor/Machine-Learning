from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
# KNN也叫K近邻算法(简单容易的算法)
# K值如果过小, 容易受到异常值的影响,
# K值过大容易受到样本不均衡的情况的影响
原理: 
    如果一个样本在特征空间中的K个最相似（即特征空间中最邻近）
    的样本中的大多数属于某一个类别，则该样本也属于这个类别
"""


def load_data():
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = \
        train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)  # 测试集不要用fit, 因为要保持和训练集处理方式一致
    return x_train, x_test, y_train, y_test


def KNN_test():
    x_train, x_test, y_train, y_test = load_data()

    # KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 传入测试值通过前面的预估器获得预测值
    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    score = estimator.score(x_test, y_test)
    print("准确率为: ", score)
    return None


def KNN_optimal():  # 模型选择和调优
    # 网格搜索和交叉验证
    x_train, x_test, y_train, y_test = load_data()
    estimator = KNeighborsClassifier()  # 默认都是欧式距离, 采用的是minkowski推广算法,p=1是曼哈顿, p=2是欧式, 而默认值为2
    # 开始调优
    # 第一个参数是estimator
    # 第二个是估计器参数，参数名称（字符串）作为key，要测试的参数列表作为value的字典，或这样的字典构成的列表
    # 第三个是指定cv=K,  K折交叉验证
    # https://www.cnblogs.com/dblsha/p/10161798.html
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)
    # 结束调优
    estimator.fit(x_train, y_train)

    # 传入测试值通过前面的预估器获得预测值
    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    score = estimator.score(x_test, y_test)
    print("准确率为: ", score)
    # ------------------
    print("最佳参数:\n", estimator.best_params_)
    print("最佳结果:\n", estimator.best_score_)
    print("最佳估计器:\n", estimator.best_estimator_)
    print("交叉验证结果:\n", estimator.cv_results_)
    # -----------------以上是自动筛选出的最佳参数, 调优结果

    return None


if __name__ == '__main__':
    # KNN_test()
    KNN_optimal()
