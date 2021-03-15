# 逻辑回归一般是二分类问题
"""
这一部分用逻辑回归来分类breast是否良性
这里需要注意的有一下:
    LogisticRegression方法相当于SGDClassifier(loss="log",penalty=" ")
    SGDClassifier实现了一个普通的随机梯度下降学习,
    也支持平均梯度下降ASGD, 可以设置average=True来开启
"""
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn_learning.model_load_store.Util_model import *
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
"""


def load_data():
    """
    先获取数据
    处理数据
        有缺失值
    数据集划分 测试 训练
    特征工程
        无量纲化-标准化(不要用归一化 之前有笔记)
    逻辑回归预估器
    模型评估
    :return x_train, x_test, y_train, y_test:
    """
    column_name = ['Sample code number', 'Clump Thickness',
                   'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
                   'Single Epithelial Cell Size',
                   'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    # # 网上直接下载
    # path = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    # original_data = pd.read_csv(path, names=column_name)

    # 文件读取
    original_data = pd.read_csv("../../resources/cancer/breast-cancer-wisconsin.data", names=column_name)

    # 缺失值处理
    # 第一步先替换 ? 为 nan
    data = original_data.replace(to_replace="?", value=np.nan)
    # 第二步可以选择前面笔记里面自己写的的过滤nan也可以用简单的方法如下:
    data.dropna(inplace=True)
    print("检测是否还有缺失值(全为false表示没有缺失值)\n", data.isnull().any())  # 检测是否还有缺失值

    # 第三步 筛选特征值和目标值
    x = data.iloc[:, 1:-1]  # 表示每一行数据都要, 从第一列到倒数第二列的column字段也要
    y = data["Class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    return x_train, x_test, y_train, y_test


def logic_Regression():
    """
    逻辑回归的真实值是分类, 也就是是否属于某一个类别,和线性回归不一样
    线性回归损失函数: (y_predict-y_true)平方和/总数
    逻辑回归损失函数: 对数似然损失(https://blog.csdn.net/u014182497/article/details/82252456)
    逻辑回归用sigmoid函数为例子: 需要把结果映射到sigmoid函数上
    分两种情况:(见截图) y轴表示损失值, 横轴x表示映射结果.(分段函数)
    当真实值为1 见 对数似然损失-1.png
    当真实值为0 见 对数似然损失-2.png
    不难理解,需要对着图看
    逻辑回归损失值得到后需要用梯度下降来优化
    后续就差不多
    :return:
    """

    x_train, x_test, y_train, y_test = load_data()
    # 第四步: 开始特征工程
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 第五步, 预估器流程
    estimator = LogisticRegression()  # 默认参数
    estimator.fit(x_train, y_train)
    print("逻辑回归_权重系数为: ", estimator.coef_)
    print("逻辑回归_偏置为:", estimator.intercept_)

    # store_model(estimator, "logic_regression_model01.pkl")  # 保存模型
    # estimator = load_model("logic_regression_model01.pkl")  # 加载模型

    # 第六步, 模型评估
    y_predict = estimator.predict(x_test)
    print("逻辑回归_预测结果", y_predict)
    print("逻辑回归_预测结果对比:", y_test == y_predict)
    score = estimator.score(x_test, y_test)
    print("准确率为:", score)
    # 2是良性的 4是恶性的
    """
    但是实际上这个预测结果不是我们想要的, 以上只能说明预测的正确与否,
    而事实上, 我们需要一种评估方式来显示我们对恶性breast的预测成功率, 也就是召回率
    同时可以查看F1-score的稳健性
    (召回率和精确率看笔记和截图)
    所以下面换一种评估方法
    """

    Score = classification_report(y_test, y_predict, labels=[2, 4],
                                  target_names=["良性", "恶性"])
    print("查看精确率,召回率,F1-score\n", Score)
    # support表示样本量

    """
    ROC曲线和AUC指标(样本分类不均衡的情况下,可以使用这种方法)
    AUC = 0.5 是瞎猜模型
    AUC = 1 是最好的模型
    AUC < 0.5 属于反向毒奶
    更多的看截图
    """
    # 需要转换为0,1表示
    y_true = np.where(y_test > 3, 1, 0)  # 表示大于3为1,反之为0(class值为2和4)
    return_value = roc_auc_score(y_true, y_predict)
    print("ROC曲线和AUC返回值为(三角形面积)", return_value)

    fpr, tpr, thresholds = roc_curve(y_true, y_predict)
    plt.plot(fpr, tpr)
    plt.show()
    return None


if __name__ == '__main__':
    logic_Regression()
