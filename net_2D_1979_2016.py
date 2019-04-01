'''数据集和模型'''
import torch.utils.data as Data
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 自定义数据集
class myDataset(Data.Dataset):
    
    def __init__(self, input_lstm, input_2d, output_data):
        self.input_2d = input_2d
        self.input_lstm = input_lstm
        self.output_data = output_data
        
    def __len__(self):
        return self.input_lstm.shape[0]
    
    def __getitem__(self, idx):
        x = self.input_lstm[idx, :, :]
        x_2d = self.input_2d[:, idx, :, :, :]
        y = self.output_data[idx, :]
        sample = {'x': x, 'x_2d': x_2d, 'y': y}
        return sample
    
# 自定义模型
class myModel(nn.Module):
    def __init__(self, feature_num, hidden_units, variable_num):
        super(myModel, self).__init__()
        self.feature_num = feature_num
        self.hidden_units = hidden_units
        self.variable_num = variable_num
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.variable_num, 32, (3,3)),#输入num*61*61，输出32*59*59
                                   nn.ReLU())#输出32*59*59
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, (3,3)),#输入32*59*59，输出32*57*57
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2), stride=(2,2)))#输出32*28*28
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, (3,3)),#输入32*28*28，输出64*26*26
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2), stride=(2,2)))#输出64*13*13
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, (3,3)),#输入64*13*13，输出64*11*11
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2), stride=(2,2)))#输出64*5*5
        self.conv5 = nn.Sequential(nn.Conv2d(64, 128, (3,3)),#输入64*5*5，输出128*3*3
                                   nn.ReLU())#输出128*3*3
        self.fc1 = nn.Sequential(nn.Linear(in_features=1152, out_features=256),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(in_features=256, out_features=64),
                                 nn.Sigmoid())
        
        self.lstm = nn.LSTM(input_size=self.feature_num, hidden_size=self.hidden_units, num_layers=2, batch_first=True)
        
        self.fc01 = nn.Sequential(nn.Linear(in_features=self.hidden_units, out_features=16),
                                  nn.ReLU())
        self.fc02 = nn.Linear(in_features=16, out_features=1)
        
        
    def forward(self, x, x_2d):#x_2d:(batch, time_idx, c, h, w); x:(batch, l, f)
        #2D卷积
        outputs_conv = torch.zeros((5, x.size(0), 64), requires_grad=True).to(device)
        for time_idx in range(5):
            input_conv = x_2d[:, time_idx, :, :, :]
            output_conv = self.conv1(input_conv)
            output_conv = self.conv2(output_conv)
            output_conv = self.conv3(output_conv)
            output_conv = self.conv4(output_conv)
            output_conv = self.conv5(output_conv)
            output_conv = output_conv.view(output_conv.size(0), -1)
            output_conv = self.fc1(output_conv)
            output_conv = self.fc2(output_conv)#输出:(batch, 64)
            outputs_conv[time_idx, :, :] = output_conv#输出:(5, batch, 64)
        outputs_conv = outputs_conv.permute(1, 0, 2)#输出:(batch, 5, 64)
        #lstm
        x_cat = torch.cat((x, outputs_conv), 2)
        h0 = torch.zeros((2, x.size(0), self.hidden_units), requires_grad=True).to(device)
        c0 = torch.zeros((2, x.size(0), self.hidden_units), requires_grad=True).to(device)
        x_cat, (hn, cn) = self.lstm(x_cat, (h0, c0))
        #lstm输入x_cat:(batch, seq_len, feature),输出x_cat:(batch, seq_len, hidden_size),输出x_cat是lstm最后一层所有时间步的输出
        #h与c:(num_layers, batch, hidden_size)
        out_final = x_cat[:, -1, :]#y输出:(batch*256)
        out_final = self.fc01(out_final)
        out_final = self.fc02(out_final)
        
        return out_final