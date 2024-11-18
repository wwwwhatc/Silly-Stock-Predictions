import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties


class StockApp:
    def __init__(self, master):
        self.master = master
        self.master.title("傻瓜股票分析与预测")

        # GUI元素
        self.create_widgets()

    def create_widgets(self):
        # Tushare API Token 输入框
        self.api_label = ttk.Label(self.master, text="Tushare API Token:")
        self.api_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.api_entry = ttk.Entry(self.master, width=40)
        self.api_entry.grid(row=0, column=1, padx=5, pady=5, columnspan=2)

        # 股票代码 输入框
        self.stock_label = ttk.Label(self.master, text="股票代码:")
        self.stock_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.stock_entry = ttk.Entry(self.master, width=40)
        self.stock_entry.grid(row=1, column=1, padx=5, pady=5, columnspan=2)

        # 日期选择
        self.date_label = ttk.Label(self.master, text="选择日期:")
        self.date_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.start_date_label = ttk.Label(self.master, text="开始日期:")
        self.start_date_label.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        self.start_date = ttk.Entry(self.master, width=15)
        self.start_date.grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)

        self.end_date_label = ttk.Label(self.master, text="结束日期:")
        self.end_date_label.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        self.end_date = ttk.Entry(self.master, width=15)
        self.end_date.grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)

        # 预测日期选择
        self.predict_date_label = ttk.Label(self.master, text="预测日期:")
        self.predict_date_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.predict_date = ttk.Entry(self.master, width=15)
        self.predict_date.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

        # 执行按钮
        self.run_button = ttk.Button(self.master, text="执行分析", command=self.run_analysis)
        self.run_button.grid(row=5, column=0, columnspan=3, pady=10)

        # 结果显示区域
        self.result_frame = ttk.Frame(self.master)
        self.result_frame.grid(row=6, column=0, columnspan=3, pady=10)

    def run_analysis(self):
        # 获取用户输入
        api_token = self.api_entry.get().strip()
        stock_code = self.stock_entry.get().strip()
        start_date = self.start_date.get().strip()
        end_date = self.end_date.get().strip()
        predict_date_str = self.predict_date.get().strip()

        if not (api_token and stock_code and start_date and end_date and predict_date_str):
            messagebox.showwarning("输入错误", "请确保所有字段都已填写。")
            return

        # 设置Tushare API Token
        ts.set_token(api_token)
        pro = ts.pro_api()

        try:
            # 获取股票数据
            df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)

            # 检查数据集大小
            if len(df) < 5:
                messagebox.showwarning("数据不足", "数据集太小，无法进行训练和测试。请选择更长的时间段。")
                return

            # 保存为CSV文件
            df.to_csv(f'{stock_code}_daily_data.csv', encoding='utf-8-sig')

            # 设置支持中文的字体
            def set_chinese_font():
                font_properties = FontProperties()
                font_properties.set_family('sans-serif')
                font_properties.set_name(['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'])  # 优先级顺序
                plt.rcParams['font.sans-serif'] = font_properties.get_name()
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            set_chinese_font()

            # 绘制收盘价
            fig, axs = plt.subplots(3, 1, figsize=(14, 18))

            axs[0].plot(df.index, df['close'], label='收盘价', color='blue')
            axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每三个月一个刻度
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日期格式为YYYY-MM-DD
            axs[0].set_title(f'{stock_code} 收盘价走势图 ({start_date} 至 {end_date})')
            axs[0].set_xlabel('日期')
            axs[0].set_ylabel('价格 (元)')
            axs[0].legend()
            axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格线
            axs[0].tick_params(axis='x', rotation=45)  # 旋转日期标签以便更好地显示

            # 绘制成交量
            axs[1].plot(df.index, df['vol'], label='成交量', color='orange')
            axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每三个月一个刻度
            axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日期格式为YYYY-MM-DD
            axs[1].set_title(f'{stock_code} 成交量走势图 ({start_date} 至 {end_date})')
            axs[1].set_xlabel('日期')
            axs[1].set_ylabel('成交量')
            axs[1].legend()
            axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格线
            axs[1].tick_params(axis='x', rotation=45)  # 旋转日期标签以便更好地显示

            # 创建特征和目标变量
            X = df[['open', 'high', 'low', 'vol']].values  # 特征：开盘价、最高价、最低价、成交量
            y = df['close'].values  # 目标变量：收盘价

            # 检查数据集大小并调整测试集比例
            test_size = 0.2
            if len(df) < 10:
                test_size = 0.1  # 如果数据集小于10个样本，减少测试集比例

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # 检查训练集是否为空
            if len(X_train) == 0:
                messagebox.showwarning("数据不足", "训练集为空，请选择更长的时间段。")
                return

            # 训练线性回归模型
            model = LinearRegression()
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)

            # 评估模型
            mse = mean_squared_error(y_test, y_pred)
            print(f'均方误差 (MSE): {mse:.2f}')

            # 绘制历史实际 vs 预测收盘价
            axs[2].plot(y_test, label='实际收盘价', color='green')
            axs[2].plot(y_pred, label='预测收盘价', linestyle='--', color='red')
            axs[2].set_title('实际 vs 预测收盘价')
            axs[2].set_xlabel('样本索引')
            axs[2].set_ylabel('价格 (元)')
            axs[2].legend()
            axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)  # 添加网格线

            # 预测未来指定日期的价格
            predict_date = datetime.strptime(predict_date_str, '%Y%m%d')
            latest_data = df.iloc[0]  # 获取最新一天的数据
            next_day_features = np.array(
                [latest_data['open'], latest_data['high'], latest_data['low'], latest_data['vol']]).reshape(1, -1)
            predicted_next_day_close = model.predict(next_day_features)[0]

            # 计算价格波动
            price_change = predicted_next_day_close - latest_data['close']
            price_change_percent = (price_change / latest_data['close']) * 100

            # 绘制预测未来指定日期的价格波动
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            dates = [latest_data.name, predict_date]
            prices = [latest_data['close'], predicted_next_day_close]

            # 绘制实际收盘价和预测收盘价
            bar_width = 0.35
            index = np.arange(len(dates))
            rects = ax2.bar(index, prices, bar_width, color=['blue', 'red'])

            # 添加标签
            ax2.set_xlabel('日期')
            ax2.set_ylabel('价格 (元)')
            ax2.set_title(
                f'实际收盘价 vs 预测收盘价 ({latest_data.name.strftime("%Y-%m-%d")} vs {predict_date.strftime("%Y-%m-%d")})')
            ax2.set_xticks(index)
            ax2.set_xticklabels([date.strftime("%Y-%m-%d") for date in dates])

            # 在每个柱状图上方添加数值
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'{height:.2f} 元',
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3),  # 3 points vertical offset
                             textcoords="offset points",
                             ha='center', va='bottom')

            # 显示价格波动
            ax2.annotate(f'价格变化: {price_change:.2f} 元\n变化百分比: {price_change_percent:.2f}%',
                         xy=(1, predicted_next_day_close),
                         xytext=(0, -30),  # 调整垂直偏移量，避免与柱状图重叠
                         textcoords="offset points",
                         ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

            # 显示前一天的收盘价数值
            ax2.annotate(f'前一天收盘价: {latest_data["close"]:.2f} 元',
                         xy=(0, latest_data['close']),
                         xytext=(0, -20),  # 调整垂直偏移量，避免与柱状图重叠
                         textcoords="offset points",
                         ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

            # 添加网格线
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

            # 自动调整子图参数
            plt.tight_layout()

            # 显示图表
            self.display_results(fig, fig2)
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {str(e)}")

    def display_results(self, fig, fig2):
        # 清除之前的图表
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # 创建一个新的Frame来容纳图表
        chart_frame = ttk.Frame(self.result_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True)

        # 创建Canvas和Scrollbar
        canvas = tk.Canvas(chart_frame, width=1450, height=500)  # 增大Canvas的大小
        scrollbar = ttk.Scrollbar(chart_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 显示第一个图表
        canvas1 = FigureCanvasTkAgg(fig, master=scrollable_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 显示第二个图表
        canvas2 = FigureCanvasTkAgg(fig2, master=scrollable_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # 调整图表大小
        fig.set_size_inches(14, 12)
        fig2.set_size_inches(10, 6)

        # 布局Canvas和Scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


# 创建主窗口并运行应用
root = tk.Tk()
app = StockApp(root)
root.mainloop()

