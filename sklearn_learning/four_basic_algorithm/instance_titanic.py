import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
import numpy as np
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
"""


def load_data():
    data = pd.read_csv("../../resources/titanic/titanic.csv")
    titanic = data.copy()

    # 方法一: 过滤掉空的值的数据组, 准确率高点
    data_used = titanic[["pclass", "age", "sex", "survived"]]
    real_data = pd.DataFrame(columns=["pclass", "age", "sex", "survived"])
    for row in data_used.values:
        if not np.isnan(row[1]):
            real_data = real_data.append([{'pclass': row[0], 'age': row[1],
                                           'sex': row[2], 'survived': row[3]}],
                                         ignore_index=True)
    x = real_data[["pclass", "age", "sex"]].to_dict(orient="records")
    y = real_data["survived"]

    # 方法二: 对空数据设置个非0值
    # x = titanic[["pclass", "age", "sex"]]  # 只提取这一些特征
    # y = titanic["survived"]  # 目标值
    # x["age"].fillna(x["age"].mean(), inplace=True)
    # x = x.to_dict(orient="records")

    x_train, x_test, y_train, y_test = train_test_split(x, y.astype('int'), random_state=22)
    return x_train, x_test, y_train, y_test


def show_tree(estimator, feature_name):
    export_graphviz(estimator, out_file="../titanic_tree.dot", feature_names=feature_name)
    return None


def titanic_test():
    x_train, x_test, y_train, y_test = load_data()

    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=12)
    estimator.fit(x_train, y_train)
    show_tree(estimator, transfer.get_feature_names())

    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    score = estimator.score(x_test, y_test)
    print("准确率为: ", score)
    return None


if __name__ == '__main__':
    titanic_test()
