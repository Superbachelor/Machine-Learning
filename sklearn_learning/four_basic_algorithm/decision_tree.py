from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
学习要点:信息熵 信息增益
"""


def show_tree(estimator, feature_name):
    export_graphviz(estimator, out_file="../tree.dot", feature_names=feature_name)  # 生成树文件, 可以用图像识别软件来画树
    return None


def decision_tree_test():
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)
    show_tree(estimator, iris.feature_names)

    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    score = estimator.score(x_test, y_test)
    print("准确率为: ", score)
    return None


if __name__ == '__main__':
    decision_tree_test()
