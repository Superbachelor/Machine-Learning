from sklearn.externals import joblib
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
这是个工具文件, 用于保存和加载模型
前面部分没有使用这个工具, 只在逻辑回归里面使用了
"""


def store_model(estimator, name):
    joblib.dump(estimator, "../../models/"+name)
    return "SUCCESS"


def load_model(name):
    model = joblib.load("../../models/"+name)
    return model



