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

def classify_room(query, features):
    # 计算余弦相似度
    similarities=nn.functional.cosine_similarity(query.unsqueeze(1), features,dim=2)
    #max_index=torch.max(similarities,dim=1)
    #prob=nn.functional.softmax(similarities,dim=1)
    return similarities

def guess_room_test(agent_a,agent_b):
    # 使用训练好的模型进行预测和计算准确率
    test_dataset = EnvDataset(Param.test_env_dir) # 请提供测试数据集
    test_dataloader = DataLoader(test_dataset, batch_size=Param.batch_size)

    agent_a.eval()
    agent_b.eval()
    #conv_net.eval()
    
    total = 0
    correct = 0
    tgt = []
    pred = []

    for step,images in enumerate(test_dataloader):
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
    tgt = np.concatenate(tgt, axis=0)
    pred = np.concatenate(pred, axis=0)
    
    acc_test=np.mean(tgt == pred)
    print("test acc = {}".format(acc_test))
    #accuracy = 100 * correct / total
    #print("准确率:", accuracy)
    

if __name__ == "__main__":
    agent_a = ConvNet()
    agent_b = ConvNet()
    agent_a.load_state_dict(torch.load('agent_a_cnn.pth'))
    agent_b.load_state_dict(torch.load('agent_b_cnn.pth'))

    guess_room_test(agent_a,agent_b)
