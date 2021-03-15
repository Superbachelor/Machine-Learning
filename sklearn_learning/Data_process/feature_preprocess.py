from sklearn.preprocessing import MinMaxScaler  # 归一化
from sklearn.preprocessing import StandardScaler  # 标准化
import pandas as pd
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
# 归一化和标准化 (是为了让所有特征都能得到学习,
# 有的特征值非常大而有的特征值非常小就会被忽略(如欧氏距离算法))
"""


def read_text():
    data = pd.read_csv("../../resources/datingTestSet2.csv")
    data.iloc[:, :3]
    print(type(data))
    print("data:\n", data)
    return data


def preprocess_MinMaxScaler():
    data_numpy_array = read_text()
    transfer = MinMaxScaler(feature_range=[0, 1])  # 默认 MIN=0, MAX=1
    new_array = transfer.fit_transform(data_numpy_array)  # 需要传numpy array格式, 返回array
    print("data new:\n", new_array)
    return None


#  一般都用标准化 不用归一化
def preprocess_StandardScaler():
    data_numpy_array = read_text()
    transfer = StandardScaler()  # 值都在0附近,所以有负数是正常的
    new_array = transfer.fit_transform(data_numpy_array)
    print("data new:\n", new_array)
    return None


if __name__ == '__main__':
    # preprocess_MinMaxScaler()
    preprocess_StandardScaler()



