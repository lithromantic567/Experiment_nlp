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
    #[50,9,8,8,3]
    #for step, data in enumerate(loader):
    #    print(data)
    task = GuessRoom()
    # if Param.is_gpu: task = task.to(Param.gpu_ce)
    opt = Adam(task.parameters(), lr=Param.lr_task, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    # opt = SGD(task.parameters(), lr=Param.lr, momentum=0.9)
    accum_tgt = []; accum_pred = []
    for i in range(Param.epoch):
        tgt = []
        pred = []
        cur_sent = None
        total_loss_A = 0; total_loss_B = 0
        opt.zero_grad()
        for step, data in enumerate(loader):
            #cur_obs_info:50*9*4*4*4(batch_size,room_num,sub_num,max_obs_num,obs_feat_in)
            #cur_gate_info:50*9*4*2*3(batch_size,room_num,sub_num,max_gate_num,gate_feat_in)
            task.train()
            task.agentA.train()
            task.agentB.train()
            # --- FORWARD ----
            tgt_rooms = np.random.randint(np.zeros(data.shape[0]), Param.room_num)
            tgt.append(tgt_rooms)
            #add the direaction of agent
            #agent_dir = np.random.randint(np.zeros(data.shape[0]),Param.num_dir)
            # num_room = num_room.to(torcht.float32); num_obs = num_obs.to(torch.float32)
            # cur_env_info = cur_env_info.to(torch.float32)
            cur_env_info=data.to(torch.float32)
            obs_info= cur_env_info[:,:,2:5,2:5,:]
            room_idxes, token_probs, room_probs, sent = task(cur_env_info,tgt_rooms,obs_info)
            cur_sent = sent
            pred.append(room_idxes)
            # --- BACKWARD ---
            reward = np.ones_like(room_idxes)
            reward[room_idxes.numpy() != tgt_rooms] = -1

            reward *= Param.reward
            cur_loss_A, cur_loss_B = task.backward(token_probs, room_probs[0], reward.tolist(), step)
            total_loss_A += cur_loss_A; total_loss_B += cur_loss_B
        opt.step()
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        accum_tgt.append(tgt)
        accum_pred.append(pred)
        if i % 50 == 0:
            task.eval()
            task.agentA.eval()
            task.agentB.eval()
            accum_pred = np.concatenate(accum_pred, axis=0)
            accum_tgt = np.concatenate(accum_tgt, axis=0)
            
            acc_train=np.mean(accum_tgt == accum_pred)
            with open("guessRoom_acctrain_cnn.txt",'a') as fp:
                fp.write(str(acc_train)+'\n')
                
            print("epoch{}: \nacc = {}, loss A = {}, loss B = {}".format(i, np.mean(accum_tgt == accum_pred), total_loss_A, total_loss_B))
            accum_pred = []; accum_tgt = []
            guess_room_evaluate(task)


def guess_room_evaluate(model):
    """
    evaluation
    :param model:
    :return:
    """
    model.eval()
    with torch.no_grad():
        tgt = []; pred = []
        dataset = EnvDataset(Param.eval_env_dir)
        loader = DataLoader(dataset, batch_size=Param.batch_size)
        for step, data in enumerate(loader):
            # TODO maybe it is better to use this random way
            # tgt_rooms = np.random.randint(np.zeros(cur_obs_info.shape[0]), num_room)
            #tgt_rooms = np.ones(data.shape[0])???
            tgt_rooms = np.random.randint(np.zeros(data.shape[0]), Param.num_room)
            tgt.append(tgt_rooms)
            #agent_dir = np.random.randint(Param.subroom_num,size=cur_obs_info.shape[0])
            #cur_obs_info = cur_obs_info.to(torch.float32); cur_gate_info = cur_gate_info.to(torch.float32)
            #cur_sub_obs_info = cur_sub_obs_info.to(torch.float32); cur_sub_gate_info = cur_sub_gate_info.to(torch.float32)
            cur_env_info=data.to(torch.float32)
            obs_info= cur_env_info[:,:,2:5,2:5,:]
            room_idxes, token_probs, room_probs, sent = model(cur_env_info, tgt_rooms,obs_info,choose_method="greedy")
            pred.append(room_idxes)
        tgt = np.concatenate(tgt, axis=0)
        pred = np.concatenate(pred, axis=0)
        
        acc_eval=np.mean(tgt == pred)
        with open("guessRoom_acc_cnn.txt",'a') as f:
            f.write(str(acc_eval)+'\n')
        print("eval acc = {}".format(np.mean(tgt == pred)))


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
