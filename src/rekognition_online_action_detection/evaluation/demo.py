import pulp

# 假设数据
d = {
    (1, 1): 240, (1, 2): 720, (1, 3): 960,(1, 4): 552, (1, 5): 252,
    (2, 1): 240, (2, 2): 720, (2, 3): 960,(2, 4): 552, (2, 5): 252,
    (3, 1): 456, (3, 2): 360, (3, 3): 936,(3, 4): 360, (3, 5): 120,
    (4, 1): 456, (4, 2): 360, (4, 3): 936,(4, 4): 360, (4, 5): 120,
    (5, 1): 768, (5, 2): 240, (5, 3): 936,(5, 4): 240, (5, 5): 396,
    (6, 1): 768, (6, 2): 240, (6, 3): 936, (6, 4): 240, (6, 5): 396,
    (7, 1): 960, (7, 2): 1200, (7, 3): 360,(7, 4): 600, (7, 5): 264,
    (8, 1): 960, (8, 2): 1200, (8, 3): 360,(8, 4): 600, (8, 5): 264,
    (9, 1): 1080, (9, 2): 960, (9, 3): 360,(9, 4): 408, (9, 5): 120,
    (10, 1): 1080, (10, 2): 960, (10, 3): 360,(10, 4): 408, (10, 5): 120,
}

v = 1.2  # 步行速度
P = [500,500,500,500,500,500,500,500,500,500]  # 宿舍楼人数
C = [460, 1410, 4896,1176,576]  # 集结点容量
K = 5  # 批次数

# 创建问题实例
prob = pulp.LpProblem("EvacuationPlan", pulp.LpMinimize)

# 决策变量
x = pulp.LpVariable.dicts("x", ((i, j, k) for i in range(len(P)) for j in range(len(C)) for k in range(K)), cat='Binary')
t = pulp.LpVariable.dicts("t", ((j, k) for j in range(len(C)) for k in range(K)), lowBound=0)

# 目标函数
prob += pulp.lpSum([t[(j, K-1)] for j in range(len(C))])

# 约束条件
for i in range(len(P)):
    prob += pulp.lpSum([x[(i, j, k)] for j in range(len(C)) for k in range(K)]) == P[i], "疏散约束_{}".format(i)

for j in range(len(C)):
    prob += pulp.lpSum([x[(i, j, k)] for i in range(len(P)) for k in range(K)]) <= C[j], "容量约束_{}".format(j)

for i in range(len(P)):
    for k in range(1, K):
        prob += pulp.lpSum([t[(j, k)] for j in range(len(C)) if d.get((i, j), 0) != 0]) >= pulp.lpSum([t[(j, k-1)] + (d.get((i, j), 0) / v) * x[(i, j, k)] for j in range(len(C))]), "时间计算约束_{}_{}".format(i, k)

# 求解问题
prob.solve()

# 输出结果
for j in range(len(C)):
    for k in range(K):
        for index in range(len(P)):
            # print(index)
            # if index == 0:
            #     print(pulp.value(x[(0, j, k)]))
            if pulp.value(x[(index, j, k)]):
                print(f"宿舍楼{index+1}在批次{k+1}疏散到集结点{j+1}")
            # if pulp.value(x[(1, j, k)]):
            #     print(f"宿舍楼2在批次{k+1}疏散到集结点{j+1}")