## 主程序
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import load_data, organize_data, get_idx, plot_path, organize_data_48
from normalize import norm
from variable_dict import variable_list, variable_dict_max, variable_dict_min
from net_2D_1979_2016 import myDataset, myModel

import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch
import torch.nn as nn
import torch.utils.data as Data


long = 16 #长度小于long的台风数据弃用
filename = 'D:/typhoon/CMABSTdata/1979_2016_1.txt'
data = load_data(filename, long)
typhoon = copy.deepcopy(data)

#plot_path(data, 0, 50)
#pyplot.savefig('E:/台风实践/2006-2010.pdf')

# 归一化
maxdata, mindata, ty_train, ty_test = norm(typhoon, divide_id=746)

# 组织样本
sequence_length = 5
train_n = organize_data(ty_train, sequence_length)
test_n = organize_data(ty_test, sequence_length)

# 网络输入输出
X = train_n[:, :sequence_length, :]
Y = train_n[:, sequence_length+3, 6]
Y = Y.reshape((Y.shape[0], 1))
X_test = test_n[:, :sequence_length, :]
Y_test = test_n[:, sequence_length+3, 6]
Y_test = Y_test.reshape((Y_test.shape[0], 1))

# 训练集真值
Y_true = np.zeros_like(Y)
for i in range(Y.shape[0]):
    Y_true[i][0] = Y[i][0] * (maxdata[0, 5] - mindata[0, 5]) + mindata[0, 5]
# 测试集真值
Y_true_test = np.zeros_like(Y_test)
for i in range(Y_test.shape[0]):
    Y_true_test[i][0] = Y_test[i][0] * (maxdata[0, 5] - mindata[0, 5]) + mindata[0, 5]


# 2D卷积输入（训练集）
'深度'
sequence_samples = list()
for time_idx in range(sequence_length):
    '样本数'
    samples = list()
    for i in range(X.shape[0]):
        lat_value = X[i, time_idx, 3] * (maxdata[0, 2] - mindata[0, 2]) + mindata[0, 2]
        lon_value = X[i, time_idx, 4] * (maxdata[0, 3] - mindata[0, 3]) + mindata[0, 3]
        lat_idx, lon_idx = get_idx(lat_value, lon_value)
        '通道数'
        cuboid = list()
        for variable in variable_list:
            temp = sio.loadmat('E:/CFSR/' + variable + '/' + str(int(X[i, time_idx, 0])) + '.mat')['mat']
            square = temp[(lat_idx-30):(lat_idx+31), (lon_idx-30):(lon_idx+31)]
#            square = square.astype(np.float16)
            cuboid.append(square)
        cuboid = np.array(cuboid)
        samples.append(cuboid)
        print('>>>训练集时间点{}--数据{}--done'.format(time_idx, i))
    samples = np.array(samples)
    sequence_samples.append(samples)
sequence_samples = np.array(sequence_samples)

# 2D卷积输入（测试集）
'深度'
sequence_samples_test = list()
for time_idx in range(sequence_length):
    '样本数'
    samples = list()
    for i in range(X_test.shape[0]):
        lat_value = X_test[i, time_idx, 3] * (maxdata[0, 2] - mindata[0, 2]) + mindata[0, 2]
        lon_value = X_test[i, time_idx, 4] * (maxdata[0, 3] - mindata[0, 3]) + mindata[0, 3]
        lat_idx, lon_idx = get_idx(lat_value, lon_value)
        '通道数'
        cuboid = list()
        for variable in variable_list:
            temp = sio.loadmat('E:/CFSR/' + variable + '/' + str(int(X_test[i, time_idx, 0])) + '.mat')['mat']
            square = temp[(lat_idx-30):(lat_idx+31), (lon_idx-30):(lon_idx+31)]
#            square = square.astype(np.float16)
            cuboid.append(square)
        cuboid = np.array(cuboid)
        samples.append(cuboid)
        print('>>>测试集时间点{}--数据{}--done'.format(time_idx, i))
    samples = np.array(samples)
    sequence_samples_test.append(samples)
sequence_samples_test = np.array(sequence_samples_test)

# 2D输入归一化
for time_idx in range(sequence_samples.shape[0]):
    for i in range(sequence_samples.shape[1]):
        for variable_idx in range(sequence_samples.shape[2]):
            variable = variable_list[variable_idx]
            if variable_dict_min[variable] == 'nan':
                max_abs = variable_dict_max[variable]
                sequence_samples[time_idx, i, variable_idx] = ((sequence_samples[time_idx, i, variable_idx] / max_abs) + 1) / 2
            else:
                max_value = variable_dict_max[variable]
                min_value = variable_dict_min[variable]
                sequence_samples[time_idx, i, variable_idx] = (sequence_samples[time_idx, i, variable_idx] - min_value) / (max_value - min_value)
        print('>>>训练集时间点{}--数据{}--done'.format(time_idx, i))
        
for time_idx in range(sequence_samples_test.shape[0]):
    for i in range(sequence_samples_test.shape[1]):
        for variable_idx in range(sequence_samples_test.shape[2]):
            variable = variable_list[variable_idx]
            if variable_dict_min[variable] == 'nan':
                max_abs = variable_dict_max[variable]
                sequence_samples_test[time_idx, i, variable_idx] = ((sequence_samples_test[time_idx, i, variable_idx] / max_abs) + 1) / 2
            else:
                max_value = variable_dict_max[variable]
                min_value = variable_dict_min[variable]
                sequence_samples_test[time_idx, i, variable_idx] = (sequence_samples_test[time_idx, i, variable_idx] - min_value) / (max_value - min_value)
        print('>>>测试集时间点{}--数据{}--done'.format(time_idx, i))

# 改变一下数据类型
X = X.astype(np.float32)
Y = Y.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.float32)
#sequence_samples = sequence_samples.astype(np.float32)
#sequence_samples_test = sequence_samples_test.astype(np.float32)

################################   LSTM   #####################################
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu') 

torch.set_default_tensor_type('torch.FloatTensor')

# 数据集
dataset_train = myDataset(input_lstm=X[:, :, 1:], input_2d = sequence_samples, output_data=Y)
dataset_test = myDataset(input_lstm=X_test[:, :, 1:], input_2d = sequence_samples_test, output_data=Y_test)

# 加载数据集
dataloader_train = Data.DataLoader(dataset=dataset_train, batch_size=50, shuffle=True, drop_last=False)
dataloader_val = Data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=False)
dataloader_test = Data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

# 模型
model = myModel(feature_num=70, hidden_units=256, variable_num=len(variable_list)).to(device)
print(model)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 损失函数
loss_function = nn.L1Loss()

# 训练测试
distance_train = list()
distance_test = list()
epoch_num = 30
for epoch in range(30,40,1):
    loss_total = 0
    for iteration, sample in enumerate(dataloader_train):
        x = sample['x'].requires_grad_().to(device)
        y = sample['y'].requires_grad_().to(device)
        x_2d = sample['x_2d'].requires_grad_().to(device)
        optimizer.zero_grad()
        out = model(x, x_2d)
        loss = loss_function(out, y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        
        if (iteration+1) % 10 == 0 or (iteration+1) == len(dataloader_train):
            print('==>>> Epoch: [{}/{}]  |  Iteration: {}  |  Train_loss_batch: {:.6f}'.format\
                  (epoch+1, epoch_num, iteration+1, loss.item()))
    
    print('')
    print('>>>>> Epoch{} has finished, train_loss_average is {:.6f}.'.format(epoch+1, loss_total/(len(dataloader_train))))
    
    #一个epoch验证一次训练集
    error_train = list()
    with torch.no_grad():
        for iteration, sample in enumerate(dataloader_val):
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            x_2d = sample['x_2d'].to(device)
            pre = model(x, x_2d) #对于单个样本，输出pre为torch.Size([1, 1])
            pre = pre.item()
            true = y.item()
            pre = pre * (maxdata[0,5]-mindata[0,5]) + mindata[0,5]
            true = true * (maxdata[0,5]-mindata[0,5]) + mindata[0,5]
            error = abs(pre - true)
            error_train.append(error)
    print('>>>>> Epoch{} has finished, error_24h_train is {:.6f}.'.format(epoch+1, sum(error_train)/len(error_train)))
    
    #一个epoch测试一次
    error_test = list()
    with torch.no_grad():
        for iteration, sample in enumerate(dataloader_test):
            x = sample['x'].to(device)
            y = sample['y'].to(device)
            x_2d = sample['x_2d'].to(device)
            pre = model(x, x_2d) #对于单个样本，输出pre为torch.Size([1, 1])
            pre = pre.item()
            true = y.item()
            pre = pre * (maxdata[0,5]-mindata[0,5]) + mindata[0,5]
            true = true * (maxdata[0,5]-mindata[0,5]) + mindata[0,5]
            error = abs(pre - true)
            error_test.append(error)
    print('>>>>> Epoch{} has finished, error_24h_test is {:.6f}.'.format(epoch+1, sum(error_test)/len(error_test)))
    print('')
    
    distance_train.append(sum(error_train)/len(error_train))
    distance_test.append(sum(error_test)/len(error_test))

# 误差曲线
x_axis = list()
for i in range(epoch_num+10):
    x_axis.append(i+1)
plt.plot(x_axis, distance_train)
plt.plot(x_axis, distance_test)
plt.title('LSTM_2D_Model Intensity Error-48h')
plt.xlabel('Epoch')
plt.ylabel('Error (m/s)')
plt.legend(['Train', 'Test'], loc='upper right')
plt.xticks(range(0, epoch_num+1+10, 5))
plt.grid(axis='both')
plt.show()

# 保存
#torch.save(model.state_dict(), 'E:\台风实践\pyFile-强度\model_lstm_2D_48_param.pkl')
