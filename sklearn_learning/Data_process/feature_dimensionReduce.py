from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr  # 皮尔逊相关系数
import matplotlib.pyplot as plt
import pandas as pd
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
#  降维是为了降低特征的个数, 随机变量, 得到一组不相关的主变量的过程
#  特征选择分嵌入式(embeded)和过滤式(filter)
#  过滤式分 1: 方差选择法  2: 相关系数
"""


def variance_filter():  # 低方差过滤
    data = pd.read_csv("../../resources/datingTestSet2.csv")
    data = data.iloc[:, 0:-1]  # 获取左边第0个右边第一个中间的所有
    transfer = VarianceThreshold(threshold=0)  # threshold过滤掉不太重要的特征
    data_new = transfer.fit_transform(data)
    print("new data: ", data_new)
    pearson_relation(data)  # 使用原数据data
    return None


def pearson_relation(data):
    # 皮尔逊相关系数范围[-1,1], 如果大于0就是正相关(越接近1就越相关), 反之亦然
    r = pearsonr(data["milage"], data["Liters"])
    print("milage和Liters的相关系数为:\n", r)
    # show_relation(data["milage"], data["milage"])
    r = pearsonr(data["milage"], data["Consumtime"])
    print("milage和Liters的相关系数为:\n", r)
    # 如果相关性高可用以下方法:
    # 1 选取其中一个特征
    # 2 两个特征加权求和
    # 3 主成分分析(高维数据变低维,舍弃原由数据,创造新数据,如: 压缩数据维数,降低原数据复杂度,损失少了信息)
    return None


def show_relation(data1, data2):
    plt.figure(figsize=(20, 8), dpi=100)
    plt.scatter(data1, data2)
    plt.show()


# 主成分分析: PCA(高维数据变低维,舍弃原由数据,创造新数据,
# 如: 压缩数据维数,降低原数据复杂度,损失少了信息)
def decomposition_PCA():  # PCA 降维
    data = [[2, 8, 4, 5], [3, 8, 5, 5], [10, 5, 1, 0]]  # 3*4矩阵 包含四个特征
    N = 3  # N为整数就是转为多少个特征  保留的至少都比原特征值少一个
    # N = 0.95  # N为小数就是保留百分之多少的信息
    transfer = PCA(n_components=N)
    data_new = transfer.fit_transform(data)
    print("(主成分分析)PCA降维:", data_new)
    return None


def decomposition_test():
    order_product = pd.read_csv("../../resources/instacart/order_products__prior.csv")
    aisles = pd.read_csv("../../resources/instacart/aisles.csv")
    orders = pd.read_csv("../../resources/instacart/orders.csv")
    products = pd.read_csv("../../resources/instacart/products.csv")
    # 先合并表
    table_1 = pd.merge(aisles, products, on=["aisle_id", "aisle_id"])  # 让aisles 和product_id一起
    table_2 = pd.merge(table_1, order_product, on=["product_id", "product_id"])
    table_3 = pd.merge(table_2, orders, on=["order_id", "order_id"])
    # 交叉表
    result_table = pd.crosstab(table_3["user_id"], table_3["aisle"])
    # print(result_table)  # 处理后的最终数据

    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(result_table)
    print("PCA降维结果:\n", data_new)
    return None


if __name__ == '__main__':
    # variance_filter()
    # decomposition_PCA()
    decomposition_test()
