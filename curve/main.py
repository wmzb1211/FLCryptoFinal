# acc_ = [90.23, 95.41, 96.77, 97.10, 97.21, 97.61, 97.70, 97.82, 97.91, 97.90]
# acc = [90.99, 96.88, 96.94, 97.19, 97.42, 97.65, 97.97, 98.05, 98.24, 98.29]
# xvalue = [5,10,15,20,25,30,35,40,45,50]
# acc__ = [90.13, 95.51, 96.67, 96.70, 97.029, 97.11, 97.56, 97.62, 97.40, 97.52]
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure()
# # 加坐标点标识，三角形和空心圆
# # 一个是橘色，一个是蓝色，漂亮的颜色而非标准色
# # 三角形，橘色
# plt.title('三种算法准确率对比图')
# plt.plot(xvalue, acc_, 'd-', color='#1f77b4', label='acc_EVSFL')
# # 空心圆，蓝色
# plt.plot(xvalue, acc, 'o-', color='#d62728', label='acc_fedavg')
# plt.plot(xvalue, acc__, 's-', color='#2ca02c', label='acc_HEFL')
#
# plt.xlabel('轮数')
# plt.ylabel('准确率')
# plt.legend()
# plt.savefig('acc.png')
# plt.show()
import numpy as np
epoch = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
num_servers = [6, 16, 20, 24, 26, 28, 31, 32, 34, 36]
cost = [11.5, 16.8, 19.0, 23.5, 27, 28, 31.6, 32.8, 34.1, 35.2 ]
plt.figure()
plt.title('最优开销曲线图')
fig, ax1 = plt.subplots()

# 绘制cost的柱状图
bar_width = 20
bar_positions = np.array(epoch)
ax1.bar(bar_positions, cost, width=bar_width, label='客户端开销', color='lightskyblue')

# 设置第一个Y轴的标签和图例
ax1.set_xlabel('客户端数量')
ax1.set_ylabel('客户端开销/s', color='black')
ax1.legend(loc='upper left')

# 创建一个新的Y轴并绘制num_servers的曲线
ax2 = ax1.twinx()
ax2.plot(epoch, num_servers, linestyle='--', marker='o', label='聚合服务器数量', color='r')

# 设置第二个Y轴的标签和图例
ax2.set_ylabel('聚合服务器数量', color='black')
ax2.legend(loc='upper right')
plt.savefig('cost.svg', format = 'svg', dpi = 1200)
# 显示图表
plt.show()
