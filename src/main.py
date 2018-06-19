import tushare as ts

if __name__ == '__main__':
    res = ts.get_hist_data('600848')
    print(res['high'][0])
