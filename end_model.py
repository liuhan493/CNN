import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch.optim as optim
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]='1'


# 检查是否有CUDA-compatible的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device: {device}')

#CNN模型
class BinaryClassCNN(nn.Module):
    def __init__(self):
        super(BinaryClassCNN, self).__init__()
        # 假设输入通道数为1，输出通道数可以自己定义
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=24, stride=1,padding=0,bias=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=256, kernel_size=5, stride=1, padding=0,bias=True)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0,bias=True)
        self.dropout = nn.Dropout(0.2)
        # 这里要计算第三个卷积层后的输出维度, 以决定全连接层的输入大小
        self.fc_input_size = self._get_conv_output_size(200)
        self.fc = nn.Linear(self.fc_input_size, 1024)
        self.fc1 = nn.Linear(1024, 64)  # 用于二分类的输出层数为2
        self.fc2 = nn.Linear(1024, 256)  # 用于z值估算
        self.fc3 = nn.Linear(1024, 256)  # 用于EW2796计算
        self.fc4 = nn.Linear(1024, 256)  # 用于EW2803计算
        #三个任务的输出层
        self.out_task1 = nn.Linear(64,1)
        self.out_task2 = nn.Linear(256,1)
        self.out_task3 = nn.Linear(256,1)
        self.out_task4 = nn.Linear(256,1)
    def _get_conv_output_size(self, input_size):
        # 假设出于简单考虑不使用padding和stride，则只需模拟网络结构即可
        size = input_size
        size = size - 24  # 第一个卷积层kernel_size=3
        size = size // 2  # 第一个池化层kernel_size=2
        size = size - 5  # 第二个卷积层kernel_size=3
        size = size // 2  # 第二个池化层kernel_size=2
        size = size - 3  # 第三个卷积层kernel_size=3
        size = size // 2  # 第三个池化层kernel_size=2
        # print(size)
        return 20 * 128  # 维度乘以最后一层卷积的输出通道数

    def forward(self, x):
        x = x.unsqueeze(1)
        # x shape is (batch_size, channels, length)
        x = F.relu(self.conv1(x))
        # print(x.shape)
#        x = self.dropout(x)
        x = self.pool1(x)
        # print(x.shape)
        x = F.relu(self.conv2(x))
#        x = self.dropout(x)
        # print(x.shape)
        x = self.pool2(x)
        # print(x.shape)
        x = F.relu(self.conv3(x))
#        x = self.dropout(x)
        # print(x.shape)
        x = self.pool3(x)
        # print(x.shape)
        # 展平操作，准备接入全连接层
        # print(self.fc_input_size)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        x = F.relu(self.fc(x))

        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x))
        x3 = torch.relu(self.fc3(x))
        x4 = torch.relu(self.fc4(x))
        out1 = self.out_task1(x1)
        out2 = self.out_task2(x2)
        out3 = self.out_task3(x3)
        out4 = self.out_task4(x4)
        return F.sigmoid(out1),out2,out3,out4

# 实例化模型并转到GPU
model = BinaryClassCNN().to(device)


# 数据集
# 创建一个数据集类
class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y


x_flux_train = []
y_label_train = []
y_z_train = []
y_ew_2796_train = []
y_ew_2803_train = []
t_path = "/home/liuhan493/True1.npy"
t_f = np.load(t_path,allow_pickle=True).item()
t_f = dict(t_f)

f_path = "/home/liuhan493/Flase1.npy"
f_f = np.load(f_path,allow_pickle=True).item()
f_f = dict(f_f)
sum_true = 0
for t_id in range(0,55000):   #len(t_f["list_id"])
    if t_f["QSO_SN"][t_id] > 5:
        if t_f["EW_2796"][t_id] >= 0.3 and t_f["EW_2803"][t_id] >= 0.3:
            x_flux_train.append(t_f["FLUX"][t_id])
            y_label_train.append(t_f["label"][t_id])
            y_z_train.append(t_f["Z_ABS"][t_id])

            # ew = t_f["EW_2796"][t_id] + t_f["EW_2803"][t_id]
            y_ew_2796_train.append(t_f["EW_2796"][t_id])
            y_ew_2803_train.append(t_f["EW_2803"][t_id])

            x_flux_train.append(f_f["FLUX"][t_id])
            y_label_train.append(f_f["label"][t_id])
            y_z_train.append(f_f["Z_ABS"][t_id])

            # ew = f_f["EW_2796"][t_id] + f_f["EW_2803"][t_id]
            y_ew_2796_train.append(f_f["EW_2796"][t_id])
            y_ew_2803_train.append(f_f["EW_2803"][t_id])
            sum_true+=1

x_flux_test = []
y_label_test = []
y_z_test = []
y_ew_2796_test = []
y_ew_2803_test = []
for t_id in range(50000,60000):   #len(t_f["list_id"])
    if t_f["QSO_SN"][t_id] > 5:
        if t_f["EW_2796"][t_id] >= 0.3 and t_f["EW_2803"][t_id] >= 0.3:
            x_flux_test.append(t_f["FLUX"][t_id])
            y_label_test.append(t_f["label"][t_id])
            y_z_test.append(t_f["Z_ABS"][t_id])

            ew = t_f["EW_2796"][t_id] + t_f["EW_2803"][t_id]
            y_ew_2796_test.append(t_f["EW_2796"][t_id])
            y_ew_2803_test.append(t_f["EW_2803"][t_id])

            x_flux_test.append(f_f["FLUX"][t_id])
            y_label_test.append(f_f["label"][t_id])
            y_z_test.append(f_f["Z_ABS"][t_id])

            ew = f_f["EW_2796"][t_id] + f_f["EW_2803"][t_id]
            y_ew_2796_test.append(f_f["EW_2796"][t_id])
            y_ew_2803_test.append(f_f["EW_2803"][t_id])
            sum_true+=1

print(f'train数量:{len(y_ew_2803_train)}')
print(f'test数量:{len(y_ew_2796_test)}')

# f_path = r"C:\Users\Administrator\Desktop\i\SDSS_spec_downloader-main\Flase.npy"
# f_f = np.load(f_path,allow_pickle=True).item()
# f_f = dict(f_f)
# sum_false = 0
# for f_id in range(0,len(f_f["list_id"])): #len(f_f["list_id"])
#     if f_f["QSO_SN"][f_id] >5:
#         x_flux.append(f_f["FLUX"][f_id])
#         y_label.append(f_f["label"][f_id])
#         y_z.append(f_f["Z_ABS"][f_id])
#
#         ew = f_f["EW_2796"][f_id] + f_f["EW_2803"][f_id]
#         y_ew.append(ew)
#         y_s = [y_label, y_z, y_ew]
#         y.append(y_s)
#         # y_z.append(t_f["Z_ABS"][t_id])
#         # ew_t = t_f["EW_2796"][t_id] + t_f["EW_2803"][t_id]
#         # y_ew.append(ew_t)
#         sum_false+=1
#     if sum_false> sum_true:
#         break





print("数据集制作完成")


class MultiTaskDataset(Dataset):
    def __init__(self, x_flux, y_label, y_z, y_ew_2796,y_ew_2803):
        self.flux = x_flux
        self.y_label = y_label
        self.y_z = y_z
        self.y_ew_2796 = y_ew_2796
        self.y_ew_2803 = y_ew_2803


    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        # 获取索引对应的数据
        flux = self.flux[idx]
        y_label = self.y_label[idx]
        y_z = self.y_z[idx]
        y_ew_2796 = self.y_ew_2796[idx]
        y_ew_2803 = self.y_ew_2803[idx]
        # 通量和三个任务的标签
        return flux, y_label, y_z, y_ew_2796,y_ew_2803
# 创建数据集
train_dataset = MultiTaskDataset(x_flux_train, y_label_train, y_z_train, y_ew_2796_train,y_ew_2803_train)
test_dataset = MultiTaskDataset(x_flux_test, y_label_test, y_z_test, y_ew_2796_test,y_ew_2803_test)

# # 分割为训练集和测试集
# print("1")
# X_train, X_test, y_train, y_test = train_test_split(x_flux, y, test_size=0.1, random_state=0,shuffle=True)
#
# print(y_test)
# print(y_train)


# 将训练数据和测试数据转换为tensor数据集

# train_dataset = SimpleDataset(X_train, y_train)
# test_dataset = SimpleDataset(X_test, y_test)

# 创建DataLoader
batch_size = 256
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# tt_loader =
# 模型训练
optimizer = Adam(model.parameters(), lr=0.001)
criterion_task1 = nn.BCELoss()  #二分类
criterion_task2 = nn.MSELoss()  #回归问题
criterion_task3 = nn.MSELoss()  #回归问题O
criterion_task4 = nn.MSELoss()
#自适应学习率
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)



total_l = []
# 训练模型3
def train(model, criterion_task1, criterion_task2, criterion_task3, criterion_task4, optimizer, train_loader, epochs, device):
    model.train()
    for epoch in range(epochs):
        # running_loss = 0.0
        total_loss = 0
        for i, (inputs, labels,z,ew_2796,ew_2803) in enumerate(train_loader):#,z,ew_total
            # labels=0
            # z=0
            # ew_total=0
            # 把数据和标签转移到GPU上
            inputs = inputs.to(device).float()
            # labels = labels.to(device).float()
            labels = labels.float().to(device)
            z = z.float().to(device)
            # ew_total = ew_total.float().to(device)
            ew_2796 = ew_2796.float().to(device)
            ew_2803 = ew_2803.float().to(device)
            optimizer.zero_grad()
            outputs_task1,outputs_task2,outputs_task3,outputs_task4 = model(inputs) #
            outputs_task1 = outputs_task1.squeeze()
            outputs_task2 = outputs_task2.squeeze()
            outputs_task3 = outputs_task3.squeeze()
            outputs_task4 = outputs_task4.squeeze()
            loss_1 = criterion_task1(outputs_task1, labels)
            loss_2 = criterion_task2(outputs_task2, z)
            loss_3 = criterion_task3(outputs_task3, ew_2796)
            loss_4 = criterion_task3(outputs_task3, ew_2803)
            loss = loss_1+10*loss_2+10*loss_3+10*loss_4   #+loss_2+loss_3
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item()
            #if (i + 1) % 10 == 0:
            #    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        scheduler.step()  #学习率调整
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        # print(total_loss)
        total_l.append(total_loss)
        # 打印统计信息（例如每个epoch的平均损失）
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss:.5f}")
        
        evaluate1(model,test_loader)

# 训练模型
#epochs = 10
#train(model, criterion_task1,criterion_task2, criterion_task3, optimizer, train_loader, epochs,device)

acc = []
z_err = []
ew_2796_err = []
ew_2803_err = []


def evaluate1(model, test_loader):
    model.eval()
    me_ew = []
    me_z = []
    with torch.no_grad():
        correct = 0
        p_z = 0
        p_ew_2796 = 0
        p_ew_2803 = 0
        pre_z_err = 0
        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0
        total = 0
        for i, (inputs, labels,z,ew_2796,ew_2803) in enumerate(test_loader):
            labels = np.array(labels)
            z = z.to(device)
            ew_2796 = ew_2796.to(device)
            ew_2803 = ew_2803.to(device)
            inputs = inputs.to(device)
            outputs_task1,outputs_task2,outputs_task3,outputs_task4= model(inputs.float())
            # print(outputs)
            predicted_classification = (outputs_task1.data > 0.5).float().squeeze().cpu().numpy()
            predicted_z = criterion_task2(outputs_task2.squeeze(),z)
            predicted_ew_2796 = criterion_task3(outputs_task3.squeeze(),ew_2796)
            predicted_ew_2803 = criterion_task3(outputs_task3.squeeze(),ew_2803)
            total += labels.shape[0]

            # correct += (predicted == labels).sum().item()

            correct += np.count_nonzero(predicted_classification == labels)



#            predicted_classification = (outputs_task1.data > 0.5).float().squeeze()
            # 计算True Positives (TP)
#            TP = torch.sum((predicted_classification == 1) & (labels == 1))

            # 计算True Negatives (TN)
#            TN = torch.sum((predicted_classification == 0) & (labels == 0))

            # 计算False Positives (FP)
#            FP = torch.sum((predicted_classification == 1) & (labels == 0))
                        # 计算False Negatives (FN)
#            FN = torch.sum((predicted_classification == 0) & (labels == 1))
#            # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
            p_z+=predicted_z.cpu()
            p_ew_2796+=predicted_ew_2796.cpu()
            p_ew_2803+=predicted_ew_2803.cpu()


#            outputs_task2 = outputs_task2.tolist()
#            task2 = []
#            for ii in range(0,len(outputs_task2)):
#                task2.append(outputs_task2[ii][0])
#            z = z.tolist()

#            for ii in range(0,len(z)):
#                err_z = (z[ii] - task2[ii])
             #   print(err_z)
#                me_z.append(err_z)

#            outputs_task3 = outputs_task3.tolist()
#            task3 = []
#            for ii in range(0,len(outputs_task3)):
#                task3.append(outputs_task3[ii][0])
#            ew_total = ew_total.tolist()

#            for ii in range(0,len(z)):
#                err_ew = (ew_total[ii] - task3[ii])
            #    print(err_ew)
#                me_ew.append(err_ew)

        acc.append(correct/total)
        z_err.append(p_z/total)
        ew_2796_err.append(p_ew_2796/total)
        ew_2803_err.append(p_ew_2803/total)
        print(f'Accuracy of the model on the test spec: {100 * correct / total} %')
        print(f'error of the model on the test z_err: {p_z / total} ' )
        print(f'error of the model on the test 2796_err: {p_ew_2796 / total} ')
        print(f'error of the model on the test 2803_err: {p_ew_2803 / total} ')
# 模型的验证
def evaluate(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        p_z = 0
        p_ew_2796 = 0
        p_ew_2803 = 0
        pre_z_err = 0
        TPsum=0
        TNsum=0
        FPsum=0
        FNsum=0
        total = 0
        for i, (inputs, labels,z,ew_2796,ew_2803) in enumerate(test_loader):
            labels = np.array(labels)
            z = z.to(device)
            ew_2796 = ew_2796.to(device)
            ew_2803 = ew_2803.to(device)
            inputs = inputs.to(device)
            outputs_task1,outputs_task2,outputs_task3,outputs_task4= model(inputs.float())
            # print(outputs)
            predicted_classification = (outputs_task1.data > 0.5).float().squeeze().cpu().numpy()
            predicted_z = criterion_task2(outputs_task2.squeeze(),z)
            predicted_ew_2796 = criterion_task3(outputs_task3.squeeze(),ew_2796)
            predicted_ew_2803 = criterion_task3(outputs_task3.squeeze(),ew_2803)

            total += labels.shape[0]

            # correct += (predicted == labels).sum().item()
        
            correct += np.count_nonzero(predicted_classification == labels)
            
            predicted_classification = torch.from_numpy(predicted_classification)
            labels = torch.from_numpy(labels)

#            predicted_classification = (outputs_task1.data > 0.5).float().squeeze()
            # 计算True Positives (TP)
            TP = torch.sum((predicted_classification == 1) & (labels == 1))

            # 计算True Negatives (TN)
            TN = torch.sum((predicted_classification == 0) & (labels == 0))

            # 计算False Positives (FP)
            FP = torch.sum((predicted_classification == 1) & (labels == 0))

            # 计算False Negatives (FN)
            FN = torch.sum((predicted_classification == 0) & (labels == 1))
            # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
            p_z+=predicted_z.cpu()
            p_ew_2796+=predicted_ew_2796.cpu()
            p_ew_2803+=predicted_ew_2803.cpu()


            outputs_task2 = outputs_task2.tolist()
            task2 = []
            for ii in range(0,len(outputs_task2)):
                task2.append(outputs_task2[ii][0])
            z = z.tolist()

            for ii in range(0,len(z)):
                err_z = (z[ii] - task2[ii])
             #   print(err_z)
                me_z.append(err_z)

            outputs_task3 = outputs_task3.tolist()
            task3 = []
            for ii in range(0,len(outputs_task3)):
                task3.append(outputs_task3[ii][0])
            ew_2796 = ew_2796.tolist()

            for ii in range(0,len(z)):
                err_ew_2796 = (ew_2796[ii] - task3[ii])
            #    print(err_ew)
                me_ew_2796.append(err_ew_2796)
            outputs_task4 = outputs_task4.tolist()
            task4 = []
            for ii in range(0,len(outputs_task4)):
                task4.append(outputs_task4[ii][0])
            ew_2803 = ew_2803.tolist()

            for ii in range(0,len(z)):
                err_ew_2803 = (ew_2803[ii] - task4[ii])
            #    print(err_ew)
                me_ew_2803.append(err_ew_2803)
            TPsum+=TP
            TNsum+=TN
            FPsum+=FP
            FNsum+=FN
        acc.append(correct/total)
        z_err.append(p_z/total)
        ew_2796_err.append(p_ew_2796/total)
        ew_2803_err.append(p_ew_2803/total)
        me_z.append(err_z)
        # me_ew_2796.append(err_ew_2796)
        # me_ew_2803.append(err_ew_2803)
        tp.append(TPsum)
        tn.append(TNsum)
        fp.append(FPsum)
        fn.append(FNsum)
        print(f'Accuracy of the model on the test spec: {100 * correct / total} %')
        print(f'error of the model on the test z_err: {p_z / total} ' )
        print(f'error of the model on the test 2796_err: {p_ew_2796 / total} ')
        print(f'error of the model on the test 2803_err: {p_ew_2803 / total} ')

epochs = 2000
train(model,criterion_task1,criterion_task2,criterion_task3,criterion_task4,optimizer,train_loader,epochs,device)
# 验证模型

tp = []
tn = []
fp = []
fn = []
me_ew_2796 = []
me_ew_2803 = []
me_z = []

evaluate(model, test_loader)
#print(total_l)
print(len(me_z))

# 保存整个模型
torch.save(model, '/home/liuhan493/mode/model_end.pt')

# 加载整个模型
#model = torch.load('/home/liuhan493/mode/complete_model.pt')
# 如前所述，如果你希望继续训练该模型，需要调用model.train()
#model.train()
m = {'acc':acc,'loss':total_l,'z_err':me_z,'ew_err_2796':me_ew_2796,'ew_err_2803':me_ew_2803}
np.save("/home/liuhan493/mode/data_end.npy", m)


#torch.save(model,'/home/liuhan493/model')
plt.figure(figsize=(10, 5))
plt.plot(total_l,label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Total_Loss')
plt.legend()
plt.grid(True)

plt.savefig('/home/liuhan493/mode/loss_end.png')  




plt.figure(figsize=(10, 5))

plt.plot(100*(acc-0.02),label='Accuary Rate')
#plt.title('Acc vs. Number of Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuary Rate')
plt.legend()
plt.grid(True)
plt.savefig('/home/liuhan493/mode/acc_end.png')


plt.figure(figsize=(10, 5))
plt.plot(z_err,label='z_err')
plt.title('P_z_err vs. Number of Epochs')
plt.xlabel('Epoch')
plt.ylabel('P_z_err')
plt.legend()
plt.grid(True)
plt.savefig('/home/liuhan493/mode/p_z_err_end.png')



# plt.figure(figsize=(10, 5))
# plt.plot(ew_err,label='ew_err')
# plt.title('P_ew_err vs. Number of Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('P_ew_err')
# plt.legend()
# plt.grid(True)
# plt.savefig('/home/liuhan493/mode/p_ew_err2.png')
#
# plt.figure(figsize=(10, 5))
# plt.plot(ew_err,label='ew_err')
# plt.title('P_ew_err vs. Number of Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('P_ew_err')
# plt.legend()
# plt.grid(True)
# plt.savefig('/home/liuhan493/mode/p_ew_err2.png')



#print("me_z:",me_z)
#m = {'acc':acc,'loss':total_l,'z_err':me_z,'ew_err_2796':me_ew_2796,'ew_err_2803':me_ew_2803}
#np.save("/home/liuhan493/mode/data_end.npy", m)


print("com/purity")
print(tp[0]/(tp[0]+fn[0]))
print(tp[0]/(tp[0]+fp[0]))
