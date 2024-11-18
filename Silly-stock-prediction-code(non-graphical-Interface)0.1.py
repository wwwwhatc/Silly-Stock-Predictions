# stock_analysis_and_prediction.py
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates
from datetime import timedelta

# 设置Tushare API Token
ts.set_token('你的API token')  # https://tushare.pro/document/2?doc_id=27进入该网注册，获取API Token，替换为你的Tushare API Token
pro = ts.pro_api()

# 获取股票数据
ts_code = ' 000682.SZ'  # 请替换为你感兴趣的股票代码
start_date = '20200101'
end_date = '20241118'
df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

# 将日期列转换为日期格式并设置为索引
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)

# 保存为CSV文件
df.to_csv(f'{ts_code}_daily_data.csv', encoding='utf-8-sig')  # 保存为CSV文件，编码为UTF-8

# 设置支持中文的字体
def set_chinese_font():
    font_properties = FontProperties()
    font_properties.set_family('sans-serif')
    font_properties.set_name(['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'])  # 优先级顺序
    plt.rcParams['font.sans-serif'] = font_properties.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

set_chinese_font()

# 绘制收盘价
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df.index, df['close'], label='收盘价', color='blue')

# 设置日期格式
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每三个月一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日期格式为YYYY-MM-DD

# 显示起始点和终点的具体日期
start_date_str = df.index[0].strftime('%Y-%m-%d')
end_date_str = df.index[-1].strftime('%Y-%m-%d')
plt.title(f'{ts_code} 收盘价走势图 ({end_date_str} 至 {start_date_str})')
plt.xlabel('日期')
plt.ylabel('价格 (元)')
plt.legend()
plt.xticks(rotation=45)  # 旋转日期标签以便更好地显示
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格线
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()

# 绘制成交量
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df.index, df['vol'], label='成交量', color='orange')

# 设置日期格式
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每三个月一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日期格式为YYYY-MM-DD

# 显示起始点和终点的具体日期
plt.title(f'{ts_code} 成交量走势图 ({end_date_str} 至 {start_date_str})')
plt.xlabel('日期')
plt.ylabel('成交量')
plt.legend()
plt.xticks(rotation=45)  # 旋转日期标签以便更好地显示
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格线
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()

# 创建特征和目标变量
X = df[['open', 'high', 'low', 'vol']].values  # 特征：开盘价、最高价、最低价、成交量
y = df['close'].values  # 目标变量：收盘价

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'均方误差 (MSE): {mse:.2f}')

# 可视化预测结果
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(y_test, label='实际收盘价', color='green')
ax.plot(y_pred, label='预测收盘价', linestyle='--', color='red')

plt.title('实际 vs 预测收盘价')
plt.xlabel('样本索引')
plt.ylabel('价格 (元)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格线
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()

# 预测未来第二天的价格
latest_data = df.iloc[0]  # 获取最新一天的数据（使用df的第一行）
next_day_features = np.array(
    [latest_data['open'], latest_data['high'], latest_data['low'], latest_data['vol']]).reshape(1, -1)
predicted_next_day_close = model.predict(next_day_features)[0]

# 计算价格波动
price_change = predicted_next_day_close - latest_data['close']
price_change_percent = (price_change / latest_data['close']) * 100

# 绘制预测未来第二天的价格波动
fig, ax = plt.subplots(figsize=(10, 6))
dates = [latest_data.name, latest_data.name + timedelta(days=1)]
prices = [latest_data['close'], predicted_next_day_close]

# 绘制实际收盘价和预测收盘价
bar_width = 0.35
index = np.arange(len(dates))
rects = ax.bar(index, prices, bar_width, color=['blue', 'red'])

# 添加标签
ax.set_xlabel('日期')
ax.set_ylabel('价格 (元)')
ax.set_title(f'实际收盘价 vs 预测收盘价 ({latest_data.name.strftime("%Y-%m-%d")} vs {dates[1].strftime("%Y-%m-%d")})')
ax.set_xticks(index)
ax.set_xticklabels([date.strftime("%Y-%m-%d") for date in dates])

# 在每个柱状图上方添加数值
for rect in rects:
    height = rect.get_height()
    ax.annotate(f'{height:.2f} 元',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# 显示价格波动
ax.annotate(f'价格变化: {price_change:.2f} 元\n变化百分比: {price_change_percent:.2f}%',
            xy=(1, predicted_next_day_close),
            xytext=(0, -30),  # 调整垂直偏移量，避免与柱状图重叠
            textcoords="offset points",
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# 显示前一天的收盘价数值
ax.annotate(f'前一天收盘价: {latest_data["close"]:.2f} 元',
            xy=(0, latest_data['close']),
            xytext=(0, -20),  # 调整垂直偏移量，避免与柱状图重叠
            textcoords="offset points",
            ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# 添加网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 自动调整子图参数
plt.tight_layout()

# 显示图表
plt.show()
