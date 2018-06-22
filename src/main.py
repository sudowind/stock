import tushare as ts
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import reduce
from scipy.optimize import minimize


def gen_weight(l):
    weight = [random.random() for _ in range(l)]
    weight = np.array([_ / sum(weight) for _ in weight])
    return weight


def series_completion(l):
    """
    complete series by date
    :param l: list of series
    :return: np.array of completed series
    """
    indexes = set()
    for s in l:
        for i in s.index:
            indexes.add(i)
    # print(indexes)
    sorted_index = sorted(indexes, reverse=True)
    # print(sorted_index)
    data = []
    for s in l:
        tmp = []
        for i in sorted_index:
            if i in s:
                tmp.append(s[i])
            else:
                tmp.append(0)
        data.append(np.array(tmp))
    return np.array(data)


def calc_max_draw_down(l):
    """
    计算最大回撤
    :param l: 给定一个series
    :return: 返回最大回撤值
    """
    res = 0
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] - l[j] > res:
                res = l[i] - l[j]
    return res


def show_stock(c):
    """
    折线图展示一系列stock
    :param c:
    :return:
    """
    stocks = []
    price = []
    for i in c:
        hist = ts.get_hist_data(i, start='2018-01-01', end='2018-06-01')
        stocks.append(hist)
        price.append(hist['close'])
    price = series_completion(price)
    plt.plot(price.T)
    plt.show()


def back_test(c, w):
    """
    计算一个投资组合的实际投资回报
    可以考虑的指标包括：sharpe_ratio，max_draw_down rate_of_return，volatility，liquidity
    :param c:
    :param w:
    :return:
    """
    stocks = []
    data = []
    for i in c:
        hist = ts.get_hist_data(i, start='2018-01-01', end='2018-06-01')
        stocks.append(hist)
        data.append(hist['p_change'])
    data = series_completion(data)
    rate = w.dot(data)
    # print(rate)
    volatility = np.var(rate)
    max_draw_down = calc_max_draw_down(rate)
    sharpe_ratio = sum(rate) / volatility
    rate_of_return = sum(rate)
    print('volatility', volatility)
    print('max draw down', -max_draw_down)
    print('sharpe ratio', sharpe_ratio)
    print('rate of return', rate_of_return)


def get_stocks():
    """
    获取某一部分股票
    :return:
    """
    sz50s = ts.get_sz50s()
    data = []
    for i in sz50s['code']:
        # print(i)
        hist = ts.get_hist_data(i, start='2017-01-01', end='2018-01-01')
        data.append(hist['close'])
        # for k in hist.keys():
        #     print(k)
        # print(hist['open']['2017-12-29'])
    data = series_completion(data)
    print(data.shape)


def main():
    """
    主函数
    :return:
    """
    # res = ts.get_hist_data('600848')
    # print(res['high'][0])
    # 确定股票之后计算有效边界，计算最优组合
    sz50s = ts.get_sz50s()
    stocks = []
    codes = []
    for i in sz50s['code'][20:35]:
        # print(i)
        hist = ts.get_hist_data(i, start='2017-01-01', end='2018-01-01')
        length = len(hist['close'])
        if length == 244:
            codes.append(i)
            stocks.append(hist)
        # print(var)

    data = []
    for i in stocks:
        data.append(np.array(i['p_change']))
    data = np.array(data)
    mean = np.array([np.average(l) for l in data])
    print(mean)
    cov = np.cov(data)
    # print(data.shape)
    # print(cov)
    stock_count = cov.shape[0]
    x_data = []
    y_data = []

    def sd(w):
        return np.sqrt(reduce(np.dot, [w, cov, w.T]))

    # for i in range(100000):
    #     weight = [random.random() for _ in range(stock_count)]
    #     weight = np.array([_ / sum(weight) for _ in weight])
    #     r = sum(mean * weight)
    #     s = sd(weight)
    #     x_data.append(s)
    #     y_data.append(r)
    #
    # plt.plot(x_data, y_data, 'ro')
    #
    # max_r = max(y_data)
    # min_r = min(y_data)
    # given_r = np.arange(min_r, max_r, 0.005)

    given_r = [0.15]
    risk = []

    for i in given_r:
        x0 = np.array([1.0 / stock_count for _ in range(stock_count)])
        bounds = tuple((0, 1) for x in range(stock_count))
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                       {'type': 'eq', 'fun': lambda x: sum(x * mean) - i}]

        outcome = minimize(sd, x0=x0, constraints=constraints, bounds=bounds)
        # print(sum(outcome.x * mean))
        # print(outcome)
        # print(sd(outcome.x))
        risk.append(outcome.fun)
        back_test(codes, outcome.x)
    weight = gen_weight(stock_count)
    print(weight)
    back_test(codes, weight)
    # plt.plot(risk, given_r, 'x')
    # plt.show()
    # show_stock(codes)


if __name__ == '__main__':
    main()
    # get_stocks()

