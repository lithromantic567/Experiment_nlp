import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import EnvDataset
from Param import *
import numpy as np
from torchvision import models
import random
from ceiling import *
# 定义自编码器模型
class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(Param.room_emb_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, (2, 2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, (2, 2), stride=2)
        )
        
        

    def forward(self, x):
        x = self.fc_layers(x)
        #(50,9,32)
        x = x.view(x.size(0)*x.size(1), 32, 1, 1)
        #(450,32,1,1)
        x = self.deconv_layers(x)
        #(450,3,8,8)
        return x
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
                
        self.encoder = ConvNet()
        self.decoder = DeconvNet()

    def forward(self, env_info):
        x=env_info
        #(50,9,8,8,3)
        encoded = self.encoder(x)
        #(50,9,50)
        decoded = self.decoder(encoded)
        #(450,3,8,8)
        decoded=decoded.transpose(1,2).transpose(2,3)
        decoded= decoded.reshape(env_info.shape[0],env_info.shape[1],env_info.shape[2],env_info.shape[3],env_info.shape[4])
        #(50,9,8,8,3)
        return decoded
'''
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
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

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, env_info):
        x=env_info
        #(20,9,8,8,3)
        x= x.reshape(x.shape[0]*x.shape[1],x.shape[2],x.shape[3],x.shape[4])
        #(180,8,8,3)
        x=x.transpose(1,3).transpose(2,3)
        #(180,3,8,8)
        encoded = self.encoder(x)
        #(180,64,1,1)
        decoded = self.decoder(encoded)
        #(180,3,8,8)
        decoded=decoded.transpose(1,2).transpose(2,3)
        decoded= decoded.reshape(env_info.shape[0],env_info.shape[1],env_info.shape[2],env_info.shape[3],env_info.shape[4])
        #(20,9,8,8,3)
        return decoded
'''
# 定义自编码器的训练函数
def train_autoencoder(autoencoder, train_dataloader, criterion, optimizer):
    autoencoder.train()
    running_loss = 0.0

    for images in train_dataloader:
        optimizer.zero_grad()

        images = images.to(dtype=torch.float32)
        reconstructed = autoencoder(images)
        loss = criterion(reconstructed, images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss

# 定义自编码器的验证函数
def eval_autoencoder(autoencoder, eval_dataloader, criterion):
    autoencoder.eval()
    running_loss = 0.0

    for images in eval_dataloader:
        images = images.to(dtype=torch.float32)
        reconstructed = autoencoder(images)
        loss = criterion(reconstructed, images)

        running_loss += loss.item()

    return running_loss

# 创建自编码器模型
autoencoder = Autoencoder()

# 加载训练数据集
train_dataset = EnvDataset(Param.env_dir)
train_dataloader = DataLoader(train_dataset, batch_size=Param.batch_size, shuffle=True)
#加载验证集
eval_dataset = EnvDataset(Param.eval_env_dir)
eval_dataloader = DataLoader(eval_dataset, batch_size=Param.batch_size, shuffle=True)


# 定义自编码器的损失函数和优化器
criterion_autoencoder = nn.MSELoss()
optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=Param.lr_task)

# 预训练自编码器
best_loss=1000000
epoch=0
while True:
        train_autoencoder(autoencoder, train_dataloader, criterion_autoencoder, optimizer_autoencoder)
        epoch=epoch+1
        loss_eval=eval_autoencoder(autoencoder, eval_dataloader, criterion_autoencoder)
        if loss_eval < best_loss:
            best_loss = loss_eval
            counter = 0
            print("Autoencoder Epoch {}: Loss = {}".format(epoch, loss_eval))
            torch.save(autoencoder.encoder.state_dict(), 'autoencoder_encoder.pth')
        else:
            
            counter += 1

        # 检查耐心值，如果连续多个周期准确率没有提高，则停止训练
        if counter >= 10 :
            print("Training stopped due to early stopping.")
            break

    


'''           
# 将预训练的自编码器加载到 agent A 和 agent B 中
agent_a = ConvNet()
agent_b = ConvNet()
agent_a.load_state_dict(autoencoder.encoder.state_dict())
agent_b.load_state_dict(autoencoder.encoder.state_dict())

# 修改训练过程，使用预训练的自编码器进行特征提取
for i in range(Param.epoch):
    # ...
    # 在每个训练步骤中，使用 agent A 和 agent B 进行前向传播和反向传播优化
    # ...

# 完成预训练后，进行后续的分类任务训练
# ...
# 在后续训练过程中，使用 agent A 和 agent B 进行前向传播和反向传播优化
# ...
''' 