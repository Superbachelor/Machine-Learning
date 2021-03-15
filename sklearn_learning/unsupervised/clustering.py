"""
无监督学习:
    PCA降维
    聚类算法 K-means
        目标: 一开始数据是整个的, 然后需要分成 K 个组
        步骤: 见截图
        判定结果: 下一次的中心点和上次的基本上一样, 就可以结束cluster
        中心点求法:(平均值算法)
"""
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  # 所有模型评估都在metrics里面
# 对用户的商品进行个分类来推荐商品, 资源文件里面的instacart
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
"""


# 先降维
def decomposition():
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
    result_table = result_table[:10000]  # 只取10000组数据, 防止后面预测的时候内存爆炸
    transfer = PCA(n_components=0.95)
    data_new = transfer.fit_transform(result_table)
    print("数据样本", data_new.shape)
    print("PCA降维结果:\n", data_new)
    return data_new


def KMeans_test():
    data = decomposition()

    # 预估器流程:
    estimator = KMeans(n_clusters=3, init='k-means++')
    estimator.fit(data)

    predict = estimator.predict(data)  # 分成的组, 0 1 2
    print("展示前300个用户的类别", predict[:300])

    # 模型评估:
    """
    引入轮廓系数分析和对应公式, 见截图
    """
    score = silhouette_score(data, predict)  # 小心你的内存爆炸.....
    print("模型轮廓系数为(1 最好, -1 最差):", score)

    return None


if __name__ == '__main__':
    KMeans_test()

