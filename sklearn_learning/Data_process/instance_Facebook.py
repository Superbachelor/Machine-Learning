import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
# Facebook 案例, 分析数据, 然后预测签到位置,
# 筛选数据集中出现的范围,缩小数据范围
# 处理时间为有意义的数据
# 过滤签到次数少的地点
"""


def load_data():
    data = pd.read_csv("../../resources/FBlocation/train.csv")
    # data = data.query("x<2.5 & x>1 & y<1.5 & y>1.0")  # query 方法处理数据
    data = data.copy()
    # 处理时间特征
    time_value = pd.to_datetime(data["time"], unit="s")  # 通用datetime时间类型数据
    date = pd.DatetimeIndex(time_value)  # 转换为可筛选的时间格式
    data["day"] = date.day
    data["weekday"] = date.weekday
    data["hour"] = date.hour
    # 过滤签到次数少的地点
    print("计数count统计\n", data.groupby("place_id").count())  # 展示为可观测列表数据, 这里计数, 并且后面的所有的字段数据全是代表出现的总次数
    place_count = data.groupby("place_id").count()[
        "row_id"]  # 签到place的次数统计, 方便直观展示而只过滤place和次数, row_id是随便加的, 现在所有字段都代表count值, 所以可以取其他的也行
    print("签到place的次数统计\n", place_count)

    place_count[place_count > 3]  # 过滤所有数据筛选出签到(这里是row_id>3)次数大于3的
    print("过滤所有数据,筛选出签到次数大于10的\n", place_count[place_count > 10])

    data["place_id"].isin(place_count[place_count > 3].index.values)  # 布尔值索引
    print("布尔值索引\n", data["place_id"].isin(place_count[place_count > 10].index.values))

    final_data = data[data["place_id"].isin(place_count[place_count > 10].index.values)]  # 通过布尔索引筛选
    print("处理后的data:\n", final_data)

    return final_data


def implement():
    used_data_x = load_data()[["x", "y", "accuracy", "day", "weekday", "hour"]]
    used_data_y = load_data()["place_id"]

    x_train, x_test, y_train, y_test = \
        train_test_split(used_data_x, used_data_y)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = KNeighborsClassifier()
    param_dict = {"n_neighbors": [5, 10, 15, 20]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=4)
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
    # -----------------以上是自动筛选出的最佳参数, 调优结果

    return None


if __name__ == '__main__':
    implement()
