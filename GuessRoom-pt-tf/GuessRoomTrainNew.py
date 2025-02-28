import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from Agents_old import *
from Dataset import EnvDataset
from Param import *
import random
from GuessRoom import *


def guess_room_train():
    """
    train
    :return:
    """
    dataset = EnvDataset(Param.env_dir)
    loader = DataLoader(dataset, batch_size=Param.batch_size)
    task = GuessRoom()
    # if Param.is_gpu: task = task.to(Param.gpu_device)
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # opt = SGD(task.parameters(), lr=Param.lr, momentum=0.9)
    accum_tgt = []; accum_pred = []
    best_eval=0
    for i in range(Param.epoch):
        tgt = []
        pred = []
        cur_sent = None
        total_loss_A = 0; total_loss_B = 0
        total_reward=0
        opt.zero_grad()
        for step, data in enumerate(loader):
            #print(step)
            task.train()
            task.agentA.train()
            task.agentB.train()
            # --- FORWARD ----
            tgt_rooms = np.random.randint(np.zeros(data.shape[0]), Param.room_num)
            tgt.append(tgt_rooms)
            # num_room = num_room.to(torcht.float32); num_obs = num_obs.to(torch.float32)
            # cur_env_info = cur_env_info.to(torch.float32)
            cur_obs_info = data.to(torch.float32)
            room_idxes, token_probs, room_probs, sent = task(cur_obs_info,  tgt_rooms)
            cur_sent = sent
            pred.append(room_idxes)
            # --- BACKWARD ---
            reward = np.ones_like(room_idxes)
            reward[room_idxes.numpy() != tgt_rooms] = -1

            reward *= Param.reward
            positive_numbers = [x for x in reward if x > 0]
            #print(positive_numbers)
            cur_loss_A, cur_loss_B = task.backward(token_probs, room_probs[0], reward.tolist(), step)
            total_loss_A += cur_loss_A; total_loss_B += cur_loss_B
            total_reward += sum(reward)
            
        opt.step()
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        accum_tgt.append(tgt)
        accum_pred.append(pred)
        print("|",end='',flush=True)
        if i % 10 == 0:
            task.eval()
            task.agentA.eval()
            task.agentB.eval()
            accum_pred = np.concatenate(accum_pred, axis=0)
            accum_tgt = np.concatenate(accum_tgt, axis=0)
            
            acc_train=np.mean(accum_tgt == accum_pred)
            with open("results/gr_train_npt.txt",'a') as fp:
                fp.write(str(acc_train)+'\n')
            print()
            print("epoch{}: \nacc = {}, loss A = {}, loss B = {}, reward={}".format(i, acc_train, total_loss_A, total_loss_B,total_reward),flush=True) 
            #print("epoch{}: \nacc = {}, loss A = {}, loss B = {}".format(i, np.mean(accum_tgt == accum_pred), total_loss_A, total_loss_B))           
            accum_pred = []; accum_tgt = []
            acc_eval=guess_room_evaluate(task)
            if acc_eval>best_eval:
                best_eval=acc_eval
                torch.save(task.state_dict(), 'gr_model_npt.pth')


def guess_room_evaluate(model):
    """
    evaluation
    :param model:
    :return:
    """
    model.eval()
    
    with torch.no_grad():
        tgt = []; pred = []
        acc_eval=[]
        dataset = EnvDataset(Param.eval_env_dir)
        loader = DataLoader(dataset, batch_size=Param.batch_size)
        for step, data in enumerate(loader):
            # TODO maybe it is better to use this random way
            # tgt_rooms = np.random.randint(np.zeros(cur_obs_info.shape[0]), num_room)
            tgt_rooms = np.random.randint(np.zeros(data.shape[0]), Param.room_num)
            tgt.append(tgt_rooms)
            cur_obs_info = data.to(torch.float32)
            room_idxes, token_probs, room_probs, sent = model(cur_obs_info,tgt_rooms, choose_method="greedy")
            pred.append(room_idxes)
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        acc_eval=np.mean(tgt == pred)
        with open("results/gr_eval_npt.txt",'a') as f:
            f.write(str(acc_eval)+'\n')
        print("eval acc = {}".format(acc_eval))
        return acc_eval


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    setup_seed(50)
    guess_room_train()

# 预训练的CNN
# 全连接网络编码 参数为192*640，640*50