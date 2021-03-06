'''
数据归一化
输入：台风list和划分点索引(测试集第一个样本索引)
输出：最大值最小值以及归一化后的台风list
'''
import numpy as np
def norm(typhoon, divide_id):

    divide_id = divide_id
    ty_train = typhoon[:divide_id]
    ty_test = typhoon[divide_id:]

    # 数据归一化
    alldata = list()
    for i in range(len(ty_train)):
        for j in range(len(ty_train[i])):
            for k in range(1, len(ty_train[i][j]), 1):
                alldata.append(ty_train[i][j][k])
    alldata = np.array(alldata)
    alldata = alldata.reshape([-1, len(ty_train[0][0])-1])
    
    # 求最大最小值
    maxdata = np.max(alldata, axis = 0)
    maxdata = maxdata.reshape([1, -1])
    mindata = np.min(alldata, axis = 0)
    mindata = mindata.reshape([1, -1])
    
    # 训练集归一化
    for i in range(len(ty_train)):
        for j in range(len(ty_train[i])):
            for k in range(1, len(ty_train[i][j]), 1):
                ty_train[i][j][k] = (ty_train[i][j][k] - mindata[0,k-1]) / (maxdata[0,k-1] - mindata[0,k-1])
    # 测试集归一化
    for i in range(len(ty_test)):
        for j in range(len(ty_test[i])):
            for k in range(1, len(ty_test[i][j]), 1):
                ty_test[i][j][k] = (ty_test[i][j][k] - mindata[0,k-1]) / (maxdata[0,k-1] - mindata[0,k-1])

    return maxdata, mindata, ty_train, ty_test