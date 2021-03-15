import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
# 随机森林就是多个树, 最后通过投票选择多数的那个决策
# 随机有两种方式
# 1: 每一个树训练集不同
# 2: 需要训练的特征进行随机分配 从特定的特征集里面抽取一些特征来分配
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


def titanic_ramdo_test():
    x_train, x_test, y_train, y_test = load_data()

    transfer = DictVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = RandomForestClassifier()
    # 默认bootstrap 表示为true,也就是说默认情况下放回抽样

    param_dict = {"n_estimators": [120, 200, 300, 500, 800, 1200],
                  "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(x_train, y_train)  # 训练集里面的数据和目标值

    # 传入测试值通过前面的预估器获得预测值
    y_predict = estimator.predict(x_test)
    print("预测值为:", y_predict, "\n真实值为:", y_test, "\n比较结果为:", y_test == y_predict)
    score = estimator.score(x_train, y_train)
    print("准确率为: ", score)
    # ------------------
    print("最佳参数:\n", estimator.best_params_)
    print("最佳结果:\n", estimator.best_score_)
    print("最佳估计器:\n", estimator.best_estimator_)
    print("交叉验证结果:\n", estimator.cv_results_)

    return None


if __name__ == '__main__':
    titanic_ramdo_test()
