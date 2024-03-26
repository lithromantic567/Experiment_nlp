import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import EnvDataset
from Param import *
import numpy as np
from torchvision import models
import random
'''
# 定义卷积神经网络（agent A）
class ConvA(nn.Module):
    def __init__(self):
        super(ConvA, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, Param.room_emb_size)
        )
    def forward(self, x, tgt_rooms):
        env_info=x
        x= x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        x=x.transpose(1,3).transpose(2,3)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x=x.reshape(env_info.shape[0],env_info.shape[1],Param.room_emb_size)
        results=x[np.arange(tgt_rooms.shape[0]), tgt_rooms, :]
        return results
        
        self.vgg = models.vgg16(pretrained=True)
        self.vgg.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 32)
        )
    def forward(self, x, tgt_rooms):
        env_info = x
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x = x.transpose(1, 3).transpose(2, 3)
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        x = x.reshape(env_info.shape[0], env_info.shape[1], 32)
        results = x[np.arange(tgt_rooms.shape[0]), tgt_rooms, :]
        return results
'''
# 定义卷积神经网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,16,(3,3)),
            nn.ReLU(),
            nn.MaxPool2d((3,3)),
            nn.Conv2d(16,32,(2,2)),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, Param.room_emb_size)
        )
    def forward(self, env_info):
        x=env_info
        x= x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        x=x.transpose(1,3).transpose(2,3)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x=x.reshape(env_info.shape[0],env_info.shape[1],Param.room_emb_size)
        #results=x[np.arange(tgt_rooms.shape[0]), tgt_rooms, :]
        return x
        '''
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, Param.room_emb_size)
        )
        '''

'''
# 定义全连接神经网络（agent B）
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 9)  # 输出层为九个图像的类别数
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x
'''

def classify_room(query, features):
    # 计算余弦相似度
    similarities=nn.functional.cosine_similarity(query.unsqueeze(1), features,dim=2)
    #max_index=torch.max(similarities,dim=1)
    #prob=nn.functional.softmax(similarities,dim=1)
    return similarities

def guess_room_train():
    # 加载训练数据集
    train_dataset = EnvDataset(Param.env_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=Param.batch_size)

    # 初始化agent A和agent B的模型，并定义损失函数和优化器
    agent_a = ConvNet()
    agent_b = ConvNet()
    #conv_net=ConvB()
    
    criterion = nn.CrossEntropyLoss()
    optimizer_a = optim.Adam(agent_a.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    optimizer_b = optim.Adam(agent_b.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    #optimizer = optim.Adam(conv_net.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)

    #optimizer_a = optim.SGD(agent_a.parameters(), lr=0.001, momentum=0.9)
    #optimizer_b = optim.SGD(agent_b.parameters(), lr=0.001, momentum=0.9)

    # 训练agent A和agent B

    accum_tgt = []; accum_pred = []
    #用早停方法，防止过拟合
    best_val_acc = 0.0  # 记录最佳验证集准确率
    patience = 100  # 设置耐心值，即连续多少个训练周期验证集准确率没有提高时停止训练
    counter = 0  # 用于计数连续没有提高的训练周期
    for i in range(Param.epoch):
        running_loss_a = 0.0
        running_loss_b = 0.0
        #running_loss = 0.0
        total = 0
        correct = 0
        tgt = []
        pred = []
        total_loss_A = 0; total_loss_B = 0
        for step,images in enumerate(train_dataloader): 
            
            tgt_rooms = []
            tgt_rooms = np.random.randint(np.zeros(images.shape[0]), Param.room_num)
            tgt.append(tgt_rooms)
            
            images=images.to(dtype=torch.float32)
            
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            #optimizer.zero_grad()

            # agent A的前向传播
            features_a= agent_a(images)[np.arange(tgt_rooms.shape[0]), tgt_rooms, :]
            # agent B的前向传播
            features_b = agent_b(images)
            #features_b = conv_net(images)
            #features_a = features_b[np.arange(tgt_rooms.shape[0]), tgt_rooms, :]
            outputs=classify_room(features_a,features_b)

            # 计算损失和准确率
            tgt_rooms=torch.tensor(tgt_rooms).long()
            #loss_a = criterion(features_a, tgt_rooms)
            loss_a = criterion(outputs, tgt_rooms)
            loss_b = criterion(outputs, tgt_rooms)
            #loss = criterion(outputs, tgt_rooms)
            running_loss_a += loss_a.item()
            running_loss_b += loss_b.item()
            #running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            pred.append(predicted)
            total += tgt_rooms.size(0)
            correct += (predicted == tgt_rooms).sum().item()

            # 反向传播和优化
            loss_a.backward(retain_graph=True)
            loss_b.backward()
            optimizer_a.step()
            optimizer_b.step()
            total_loss_A += loss_a; total_loss_B += loss_b
            #loss.backward()
            #optimizer.step()
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        accum_tgt.append(tgt)
        accum_pred.append(pred)
        
        accum_pred = np.concatenate(accum_pred, axis=0)
        accum_tgt = np.concatenate(accum_tgt, axis=0)
        acc_train = np.mean(accum_tgt == accum_pred)
        
        #if i % 50 == 0:
            #accum_pred = np.concatenate(accum_pred, axis=0)
            #accum_tgt = np.concatenate(accum_tgt, axis=0)
            
        #acc_train=np.mean(accum_tgt == accum_pred)
        with open("results/gr_train_ceiling.txt",'a') as fp:
            fp.write(str(acc_train)+'\n')
            
        print("epoch{}: \nacc = {}, loss A = {}, loss B = {}".format(i, acc_train, total_loss_A, total_loss_B))
        #print("epoch{}: \nacc = {}".format(i, np.mean(accum_tgt == accum_pred)))
        accum_pred = []; accum_tgt = []
        acc_eval=guess_room_evaluate(agent_a,agent_b)
            
        # 检查验证集准确率是否提高
        if acc_eval > best_val_acc:
            best_val_acc = acc_eval
            torch.save(agent_a.state_dict(), 'agent_a_cnn.pth')
            torch.save(agent_b.state_dict(), 'agent_b_cnn.pth')
            counter = 0
        else:
            counter += 1

        # 检查耐心值，如果连续多个周期准确率没有提高，则停止训练
        if counter >= patience or acc_eval==1.0:
            print("Training stopped due to early stopping.")
            
            break
        
    
            #guess_room_evaluate(conv_net)
        #print(f"Epoch {i+1}/{num_epochs}, Agent A Loss: {running_loss_a/len(train_dataloader)}, Agent B Loss: {running_loss_b/len(train_dataloader)}, Accuracy: {100*correct/total}%")

    print("训练完成")

def guess_room_evaluate(agent_a,agent_b):
    # 使用训练好的模型进行预测和计算准确率
    eval_dataset = EnvDataset(Param.eval_env_dir) # 请提供测试数据集
    eval_dataloader = DataLoader(eval_dataset, batch_size=Param.batch_size)

    agent_a.eval()
    agent_b.eval()
    #conv_net.eval()
    
    total = 0
    correct = 0
    tgt = []
    pred = []

    for step,images in enumerate(eval_dataloader):
        tgt_rooms = []
        tgt_rooms = np.random.randint(np.zeros(images.shape[0]), Param.room_num)
        tgt.append(tgt_rooms)
            
        images=images.to(dtype=torch.float32)

        with torch.no_grad():
            features_a= agent_a(images)[np.arange(tgt_rooms.shape[0]), tgt_rooms, :]
            features_b = agent_b(images)            
            outputs = classify_room(features_a, features_b)

        tgt_rooms=torch.tensor(tgt_rooms).long()
        _, predicted = torch.max(outputs.data, 1)
        pred.append(predicted)
        total += tgt_rooms.size(0)
        correct += (predicted == tgt_rooms).sum().item()
    tgt = np.concatenate(tgt, axis=0)
    pred = np.concatenate(pred, axis=0)
    
    acc_eval=np.mean(tgt == pred)
    with open("results/gr_test_ceiling.txt",'a') as f:
        f.write(str(acc_eval)+'\n')
    print("eval acc = {}".format(np.mean(tgt == pred)))
    accuracy = 100 * correct / total
    print("准确率:", accuracy)
    return acc_eval

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(1)
    guess_room_train()
