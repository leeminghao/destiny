import numpy as np
import pandas as pd
import scipy.stats as scs
import tushare
import matplotlib.pyplot as plt

from numpy import random

print(tushare.__version__)

# Step1: Download datas
try:
    import tushare as ts
    token='d5d542a0b4534c9de59f80040c743845fbe611cae04d5b278860374a'
    pro = ts.pro_api(token)
    df = pro.query('daily',ts_code='300323.SZ', start_date='20210609', end_date='20211102')
    # Save data
    df.to_csv('stockdata.csv') 
    print('Save data successfully!')
except:
    print('Failed to get data')

# Step2:
# 定义检验显著性的函数
def normality_test(array): 
    print('Norm test p-value %14.3f' % scs.normaltest(array)[1])

# 要求时间向后递增，原数据是向上递增的
data = pd.read_csv(r'stockdata.csv')[::-1]
# 获得收益率字段数据
df0 = data['pct_chg']
# 注意，这里使用的是对数收益率
log_data = np.array(np.log(df0 + 2).dropna())

 # 检验正态分布
normality_test(log_data)    


# Step3:     
data = pd.read_csv(r'stockdata.csv')
# 波动率
sigma = (data['pct_chg']/100).std()
# 历史价格时间长度
n = 96
# 单位时间
dt = 1/n
# 漂移项
sigs = sigma*np.sqrt(dt)
# 期望收益率
mu = (data['pct_chg']/100).mean()
# 扰动项
drt = mu*dt
# 最后一天股价
pe = data['close'].iloc[0]

# Step4
pt = [] # 全部模拟数据列表
# 蒙特卡洛模拟
for i in range(1000): # 控制次数
    pn = pe # 初始化股价
    p = [] # 单次模拟情况
    p.append(pe) # 计入初始股价
    for days in range(1,365): # 控制天数
        pn = pn + pn*(random.normal(drt, sigs)) # 产生新股价
        if pn < 0.1: # 确保股价大于等于一毛钱
           pn = p[-1]
        p.append(pn)
    pt.append(p)
pt = pd.DataFrame(pt).T  # 把全部历史数据转置一下便于作图时数据与matplotlib库要求的形式一致
simulations = pt.iloc[-1:].T  # simulations为模拟的n次的n个最终价格


# Step5
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.subplot(1,2,1)
plt.plot(pt)
plt.title('300323-MonteCarlo-Simulator')
plt.xlabel('time/day')
plt.ylabel('price/yuan')

plt.subplot(1,2,2)
q = np.percentile(simulations, 1) # 1%分位数位置
# figtext头两个参数为文本位置，自行调整到合适的位置即可
plt.figtext(0.85, 0.8, "Start price: %.2f" % data['close'].iloc[0])
plt.figtext(0.85, 0.7, "Avg price: %.2f" % simulations.mean())
plt.figtext(0.85, 0.6, "VaR(0.99): %.2f" % (data['close'].iloc[0] - q))
plt.figtext(0.85, 0.5, "CI(0.99): %.2f" % q)
plt.axvline(x=q, linewidth=4, color='r') # 置信区间位置

plt.title('300323-MonteCarlo-Histogram')
plt.xlabel('Price range')
plt.ylabel('Frequency')
plt.hist(simulations,bins=200)
plt.show()
