# 线性模型包括线性关系和非线性关系两种
# 线性模型包括参数一次幂和自变量一次幂
# 线性关系一定是线性模型, 反之不一定
# 优化方法有两种: 一种是正规方程, 第二种是梯度下降

# 这部分用来训练预测房价
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error  # 均方误差
"""
Author: Siliang Liu
15/08/2020
Reference itheima.com
"""


def load_data():
    boston_data = load_boston()
    print("特征数量为:(样本数,特征数)", boston_data.data.shape)
    x_train, x_test, y_train, y_test = train_test_split(boston_data.data,
                                                        boston_data.target, random_state=22)
    return x_train, x_test, y_train, y_test


# 正规方程
def linear_Regression():
    """
    正规方程的优化方法
    不能解决拟合问题
    一次性求解
    针对小数据
    :return:
    """
    x_train, x_test, y_train, y_test = load_data()
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    print("正规方程_权重系数为: ", estimator.coef_)
    print("正规方程_偏置为:", estimator.intercept_)

    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("正规方程_房价预测:", y_predict)
    print("正规方程_均分误差:", error)
    return None


# 梯度下降
def linear_SGDRegressor():
    """
    梯度下降的优化方法
    迭代求解
    针对大数据
    :return:
    """
    x_train, x_test, y_train, y_test = load_data()
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 建议看下这个函数的api, 这些值都是默认值
    # estimator = SGDRegressor(loss="squared_loss", fit_intercept=True, eta0=0.01,
    #                          power_t=0.25)

    estimator = SGDRegressor(learning_rate="constant", eta0=0.01, max_iter=10000)
    # estimator = SGDRegressor(penalty='l2', loss="squared_loss")  # 这样设置就相当于岭回归, 但是建议用Ridge方法
    estimator.fit(x_train, y_train)

    print("梯度下降_权重系数为: ", estimator.coef_)
    print("梯度下降_偏置为:", estimator.intercept_)

    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降_房价预测:", y_predict)
    print("梯度下降_均分误差:", error)

    return None


def linear_Ridge():
    """
    Ridge: 岭回归方法
    :return:
    """
    x_train, x_test, y_train, y_test = load_data()
    transfer = StandardScaler()  # 建议使用标准化处理数据
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = Ridge(max_iter=10000, alpha=0.5)  # 岭回归
    # estimator = RidgeCV(alphas=[0.1, 0.2, 0.3, 0.5])  # 加了交叉验证的岭回归
    estimator.fit(x_train, y_train)

    print("岭回归_权重系数为: ", estimator.coef_)
    print("岭回归_偏置为:", estimator.intercept_)

    y_predict = estimator.predict(x_test)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归_房价预测:", y_predict)
    print("岭回归_均分误差:", error)

    return None


if __name__ == '__main__':
    linear_Regression()
    linear_SGDRegressor()
    linear_Ridge()
