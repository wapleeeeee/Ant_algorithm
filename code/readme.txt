姓名：王枫
学号：2017202110031
专业：计算机软件与理论

#数学建模课程作业：应用蚁群算法解决tsp问题

###参数设置
ants = 40  #蚁群蚂蚁数量
iterator = 200  #迭代次数
alpha = 1  #信息素启发因子
beta = 2  #城市距离衰退因子
decrease_ratio = 0.8  #信息素每轮衰退因子
Q = 1  #信息素总量

###蚁群算法函数功能
init_disdict(): 初始化城市距离
init_etatable(): 初始化信息素因子
first_step(): 蚁群随机游走第一步
next_step(node, res_list): 蚁群下一步选择
refresh_matrix(path_table): 更新信息素
refresh_statistic(iter, dis_table, path_table): 更新统计值
one_iteration(iter): 每一步迭代
start_cal(): 开始计算
draw_plot(filepath): 画出对应图像

###实验结果
由于算法时间限制，实现了部分题目的计算
../ans目录中
filename_average_best是平均距离和最优距离根据迭代次数变化而变化的图像
filename_best_path是最优路径的可视化图像